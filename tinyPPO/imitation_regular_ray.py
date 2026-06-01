from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent

from tinyPPO.agents import ACTION_SLOTS, SHIP_BUCKET_MULTIPLIERS, TinyPPOAgent
from tinyPPO.eval import run_matchups
from tinyPPO.imitation_regular import _collect_one_game, bc_loss, row_from_regular_action, stack_rows, unpack
from tinyPPO.model import TinyPolicyValueNet


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _players_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--players-list must not be empty")
    return values


def _int_list(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _path_list(raw: str) -> list[Path]:
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def _partial_cache_paths(cache_path: Path) -> list[Path]:
    prefix = cache_path.name + ".partial_"
    return sorted(cache_path.parent.glob(prefix + "*"), key=lambda path: path.name)


def _prune_partial_caches(cache_path: Path, keep: int) -> None:
    keep = max(0, keep)
    paths = _partial_cache_paths(cache_path)
    if keep:
        paths = paths[:-keep]
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


@ray.remote(num_cpus=1)
def collect_game_task(
    seed: int,
    players: int,
    episode_steps: int,
    keep_noop_prob: float,
    sample_stride: int,
    rows_per_game: int,
    use_numba: bool,
    target_mask_mode: str,
) -> tuple[list[Any], dict[str, float]]:
    return _collect_one_game(seed, players, episode_steps, keep_noop_prob, sample_stride, rows_per_game, use_numba, target_mask_mode)


@ray.remote(num_cpus=1)
def collect_games_task(
    jobs: list[tuple[int, int]],
    episode_steps: int,
    keep_noop_prob: float,
    sample_stride: int,
    rows_per_game: int,
    use_numba: bool,
    target_mask_mode: str,
) -> tuple[list[Any], dict[str, float]]:
    rows: list[Any] = []
    labelled_actions = 0.0
    skipped_actions = 0.0
    completed = 0
    for seed, players in jobs:
        game_rows, metrics = _collect_one_game(seed, players, episode_steps, keep_noop_prob, sample_stride, rows_per_game, use_numba, target_mask_mode)
        rows.extend(game_rows)
        labelled_actions += float(metrics.get("labelled_actions", 0.0))
        skipped_actions += float(metrics.get("skipped_actions", 0.0))
        completed += 1
    return rows, {
        "games": float(completed),
        "samples": float(len(rows)),
        "labelled_actions": labelled_actions,
        "skipped_actions": skipped_actions,
        "row_target_mask_mode": target_mask_mode,
    }


@ray.remote
class DaggerCollectActor:
    def __init__(
        self,
        checkpoint: str,
        device: str,
        deterministic: bool,
        launch_bias: float,
        ship_bias: float,
        launch_temperature: float,
        target_mask_mode: str,
        target_pair_weight: float,
        collect_model_seat_only: bool,
    ):
        resolved_device = _resolve_actor_device(device)
        self.model_agent = TinyPPOAgent(
            checkpoint,
            device=str(resolved_device),
            deterministic=deterministic,
            launch_bias=launch_bias,
            ship_bias=ship_bias,
            launch_temperature=launch_temperature,
            target_mask_mode=target_mask_mode,
            target_pair_weight=target_pair_weight,
        )
        self.collect_model_seat_only = collect_model_seat_only
        self.checkpoint = checkpoint
        self.target_mask_mode = target_mask_mode
        self.target_pair_weight = float(target_pair_weight)

    def collect_games(
        self,
        jobs: list[tuple[int, int, int]],
        episode_steps: int,
        keep_noop_prob: float,
        sample_stride: int,
        rows_per_game: int,
        use_numba: bool,
        target_mask_mode: str,
    ) -> tuple[list[Any], dict[str, float]]:
        import random

        rows: list[Any] = []
        labelled_actions = 0
        skipped_actions = 0
        model_actions = 0
        regular_label_actions = 0
        for job_index, (seed, players, model_seat) in enumerate(jobs, start=1):
            random.seed(seed)
            np.random.seed(seed)
            raw_rows: list[dict[str, Any]] = []
            step_box = {"current": -1}
            label_agent = make_rulebase_agent("regular")
            regular_agents = [make_rulebase_agent("regular") for _ in range(players)]
            agents = []
            for pid in range(players):
                if pid == model_seat:
                    def model_logged(obs: dict[str, Any], configuration=None, pid: int = pid) -> list[list]:
                        del configuration
                        if "step" in obs:
                            step_box["current"] = int(obs["step"])
                        turn_index = int(obs.get("step", step_box["current"]))
                        row_obs = copy.deepcopy(obs)
                        action = self.model_agent(copy.deepcopy(row_obs)) or []
                        label = label_agent(copy.deepcopy(row_obs)) or []
                        if label or random.random() < keep_noop_prob:
                            raw_rows.append(
                                {
                                    "obs": row_obs,
                                    "player": pid,
                                    "label_action": label,
                                    "model_action_count": len(action),
                                    "label_action_count": len(label),
                                    "model_seat": model_seat,
                                    "seed": seed,
                                    "players": players,
                                    "checkpoint": self.checkpoint,
                                    "raw_index": len(raw_rows),
                                    "turn_index": turn_index,
                                }
                            )
                        return action

                    agents.append(model_logged)
                else:
                    regular_agent = regular_agents[pid]

                    def regular_logged(obs: dict[str, Any], configuration=None, pid: int = pid, regular_agent=regular_agent) -> list[list]:
                        del configuration
                        if "step" in obs:
                            step_box["current"] = int(obs["step"])
                        turn_index = int(obs.get("step", step_box["current"]))
                        action = regular_agent(obs) or []
                        if not self.collect_model_seat_only and (action or random.random() < keep_noop_prob):
                            raw_rows.append(
                                {
                                    "obs": copy.deepcopy(obs),
                                    "player": pid,
                                    "label_action": action,
                                    "model_action_count": 0,
                                    "label_action_count": len(action),
                                    "model_seat": model_seat,
                                    "seed": seed,
                                    "players": players,
                                    "checkpoint": self.checkpoint,
                                    "raw_index": len(raw_rows),
                                    "turn_index": turn_index,
                                }
                            )
                        return action

                    agents.append(regular_logged)
            env = make_fast_orbit_wars(
                {"episodeSteps": episode_steps, "seed": seed},
                keep_history=False,
                use_numba=use_numba,
            )
            env.run(agents)
            final_frame = env.steps[-1] if env.steps else []
            game_length = int(getattr(env, "_step", 0))
            if game_length <= 0 and raw_rows:
                game_length = max(int(raw["turn_index"]) for raw in raw_rows) + 1
            final_reward = float(final_frame[model_seat].get("reward", 0.0)) if final_frame else 0.0
            final_status = str(final_frame[model_seat].get("status", "")) if final_frame else ""
            sampled = raw_rows[:: max(1, sample_stride)]
            if rows_per_game > 0 and len(sampled) > rows_per_game:
                sampled = random.sample(sampled, rows_per_game)
            for sampled_index, raw in enumerate(sampled):
                obs = raw["obs"]
                player = int(raw["player"])
                label_action = raw["label_action"]
                row = row_from_regular_action(obs, player, label_action, players=players, target_mask_mode=target_mask_mode)
                if row is None:
                    continue
                row.dagger_checkpoint = raw["checkpoint"]  # type: ignore[attr-defined]
                row.dagger_seed = int(raw["seed"])  # type: ignore[attr-defined]
                row.dagger_players = int(raw["players"])  # type: ignore[attr-defined]
                row.dagger_model_seat = int(raw["model_seat"])  # type: ignore[attr-defined]
                row.dagger_player = player  # type: ignore[attr-defined]
                row.dagger_raw_index = int(raw["raw_index"])  # type: ignore[attr-defined]
                row.dagger_sampled_index = int(sampled_index)  # type: ignore[attr-defined]
                row.dagger_turn_index = int(raw["turn_index"])  # type: ignore[attr-defined]
                row.dagger_obs_step = int(raw["turn_index"])  # type: ignore[attr-defined]
                row.dagger_model_action_count = int(raw["model_action_count"])  # type: ignore[attr-defined]
                row.dagger_regular_label_action_count = int(raw["label_action_count"])  # type: ignore[attr-defined]
                row.dagger_game_length = int(game_length)  # type: ignore[attr-defined]
                row.dagger_model_final_reward = float(final_reward)  # type: ignore[attr-defined]
                row.dagger_model_final_status = final_status  # type: ignore[attr-defined]
                rows.append(row)
                labelled_actions += int(getattr(row, "labelled", 0))
                skipped_actions += int(getattr(row, "skipped", 0))
                regular_label_actions += len(label_action)
                if player == model_seat:
                    model_actions += int(raw["model_action_count"])
            if job_index % 10 == 0 or job_index == len(jobs):
                print(
                    json.dumps(
                        {
                            "event": "dagger_actor_progress",
                            "pid": os.getpid(),
                            "done": job_index,
                            "total": len(jobs),
                            "rows": len(rows),
                            "labelled_actions": labelled_actions,
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
        return rows, {
            "games": float(len(jobs)),
            "samples": float(len(rows)),
            "labelled_actions": float(labelled_actions),
            "skipped_actions": float(skipped_actions),
            "regular_label_actions": float(regular_label_actions),
            "model_seat_label_actions": float(model_actions),
            "row_target_mask_mode": target_mask_mode,
            "model_target_mask_mode": self.target_mask_mode,
            "model_target_pair_weight": float(self.target_pair_weight),
        }


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _average_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in states[0]:
        acc = None
        for state in states:
            value = state[key].float()
            acc = value.clone() if acc is None else acc.add_(value)
        out[key] = (acc / float(len(states))).to(states[0][key].dtype)
    return out


def _resolve_actor_device(device_override: str) -> torch.device:
    if device_override.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device_override.split(":", 1)[1]
        return torch.device("cuda:0")
    return torch.device(device_override if device_override else ("cuda" if torch.cuda.is_available() else "cpu"))


def _regular_anchor_loss(
    model: TinyPolicyValueNet,
    anchor_model: TinyPolicyValueNet | None,
    batch: dict[str, torch.Tensor],
    source_weight: float,
    pair_weight: float,
    count_weight: float,
    target_weight: float,
    min_sample_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = torch.tensor(0.0, device=batch["planets"].device)
    if anchor_model is None or (
        source_weight <= 0.0 and pair_weight <= 0.0 and count_weight <= 0.0 and target_weight <= 0.0
    ):
        return zero, {
            "anchor_loss": 0.0,
            "anchor_source_loss": 0.0,
            "anchor_pair_loss": 0.0,
            "anchor_count_loss": 0.0,
            "anchor_target_loss": 0.0,
            "anchor_rows": 0.0,
        }

    row_weights = batch.get("sample_weight")
    if row_weights is None:
        row_mask = torch.ones(batch["planets"].shape[0], dtype=torch.bool, device=batch["planets"].device)
    else:
        row_mask = row_weights.float() >= float(min_sample_weight)
    anchor_rows = float(row_mask.float().sum().detach().cpu())
    if not bool(row_mask.any()):
        return zero, {
            "anchor_loss": 0.0,
            "anchor_source_loss": 0.0,
            "anchor_pair_loss": 0.0,
            "anchor_count_loss": 0.0,
            "anchor_target_loss": 0.0,
            "anchor_rows": 0.0,
        }

    out = model(batch["planets"], batch["pair_features"], batch["global_features"], batch["planet_mask"], batch["own_mask"])
    with torch.no_grad():
        target = anchor_model(batch["planets"], batch["pair_features"], batch["global_features"], batch["planet_mask"], batch["own_mask"])

    source_loss = zero
    own_slots = batch["own_mask"][:, :, None].expand_as(batch["launch_actions"]) & row_mask[:, None, None]
    if source_weight > 0.0 and bool(own_slots.any()):
        source_loss_parts = F.kl_div(
            F.log_softmax(out["source_logits"], dim=-1),
            F.softmax(target["source_logits"], dim=-1),
            reduction="none",
        ).sum(dim=-1)
        source_loss = source_loss_parts.masked_select(own_slots).mean()

    pair_loss = zero
    if pair_weight > 0.0:
        pair_logits = out.get("target_pair_logits")
        target_pair_logits = target.get("target_pair_logits")
        if pair_logits is None:
            pair_logits = out["target_logits"].max(dim=2).values
        if target_pair_logits is None:
            target_pair_logits = target["target_logits"].max(dim=2).values
        pair_valid = batch["own_mask"][:, :, None] & batch["planet_mask"][:, None, :] & row_mask[:, None, None]
        eye = torch.eye(pair_valid.shape[1], dtype=torch.bool, device=pair_valid.device)[None, :, :]
        pair_valid = pair_valid & ~eye
        source_valid = pair_valid.any(dim=-1)
        if bool(source_valid.any()):
            masked_logits = pair_logits.masked_fill(~pair_valid, -1e9)
            masked_target_logits = target_pair_logits.masked_fill(~pair_valid, -1e9)
            pair_parts = F.kl_div(
                F.log_softmax(masked_logits, dim=-1),
                F.softmax(masked_target_logits, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            pair_loss = pair_parts.masked_select(source_valid).mean()

    count_loss = zero
    if count_weight > 0.0:
        slot_valid = batch["own_mask"][:, :, None].expand_as(batch["launch_actions"])
        launch_prob = F.softmax(out["source_logits"], dim=-1)[..., 1].masked_fill(~slot_valid, 0.0)
        target_launch_prob = F.softmax(target["source_logits"], dim=-1)[..., 1].masked_fill(~slot_valid, 0.0)
        pred_count = launch_prob.sum(dim=(1, 2))
        target_count = target_launch_prob.sum(dim=(1, 2))
        count_parts = F.smooth_l1_loss(pred_count, target_count, reduction="none")
        count_loss = count_parts.masked_select(row_mask).mean()

    target_loss = zero
    if target_weight > 0.0:
        source_count = int(batch["own_mask"].shape[1])
        target_valid = (
            batch["own_mask"][:, :, None, None]
            & batch["planet_mask"][:, None, None, :]
            & row_mask[:, None, None, None]
        )
        eye = torch.eye(source_count, dtype=torch.bool, device=target_valid.device)[None, :, None, :]
        target_valid = target_valid & ~eye
        target_valid = target_valid.expand_as(out["target_logits"])
        slot_valid = target_valid.any(dim=-1)
        if bool(slot_valid.any()):
            masked_logits = out["target_logits"].masked_fill(~target_valid, -1e9)
            masked_target_logits = target["target_logits"].masked_fill(~target_valid, -1e9)
            target_parts = F.kl_div(
                F.log_softmax(masked_logits, dim=-1),
                F.softmax(masked_target_logits, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            target_loss = target_parts.masked_select(slot_valid).mean()

    loss = (
        float(source_weight) * source_loss
        + float(pair_weight) * pair_loss
        + float(count_weight) * count_loss
        + float(target_weight) * target_loss
    )
    return loss, {
        "anchor_loss": float(loss.detach().cpu()),
        "anchor_source_loss": float(source_loss.detach().cpu()),
        "anchor_pair_loss": float(pair_loss.detach().cpu()),
        "anchor_count_loss": float(count_loss.detach().cpu()),
        "anchor_target_loss": float(target_loss.detach().cpu()),
        "anchor_rows": anchor_rows,
    }


@ray.remote
class BCTrainEvalActor:
    def __init__(
        self,
        rows: list[Any],
        sample_weights: list[float] | None,
        model_cfg: dict[str, int],
        batch_size: int,
        val_frac: float,
        lr: float,
        weight_decay: float,
        max_grad_norm: float,
        launch_pos_weight: float,
        target_loss_weight: float,
        ship_loss_weight: float,
        target_ship_joint_loss_weight: float,
        target_ship_joint_dagger_only: bool,
        critical_action_weight: float,
        target_loss_mask: str,
        target_margin_loss_weight: float,
        target_margin: float,
        target_margin_top_k: int,
        slot_set_loss: bool,
        target_binary_loss_weight: float,
        target_binary_pos_weight: float,
        target_pair_softmax_loss_weight: float,
        target_pair_margin_loss_weight: float,
        target_pair_owner_loss_weight: float,
        target_pair_within_owner_loss_weight: float,
        launch_count_loss_weight: float,
        sample_weight_launch_scale: float,
        sample_weight_target_scale: float,
        sample_weight_ship_scale: float,
        sample_weight_pair_scale: float,
        sample_weight_count_scale: float,
        dagger_launch_negative_weight_scale: float,
        dagger_launch_positive_weight_scale: float,
        regular_anchor_state: dict[str, torch.Tensor] | None,
        regular_anchor_source_weight: float,
        regular_anchor_pair_weight: float,
        regular_anchor_count_weight: float,
        regular_anchor_target_weight: float,
        regular_anchor_min_sample_weight: float,
        trainable_modules: str,
        seed: int,
        device_override: str = "",
    ):
        torch.set_num_threads(1)
        self.device = _resolve_actor_device(device_override)
        self.model_cfg = dict(model_cfg)
        self.model = TinyPolicyValueNet(**self.model_cfg).to(self.device)
        self.anchor_model: TinyPolicyValueNet | None = None
        if regular_anchor_state is not None and (
            regular_anchor_source_weight > 0.0
            or regular_anchor_pair_weight > 0.0
            or regular_anchor_count_weight > 0.0
            or regular_anchor_target_weight > 0.0
        ):
            self.anchor_model = TinyPolicyValueNet(**self.model_cfg).to(self.device)
            self.anchor_model.load_state_dict({key: value.to(self.device) for key, value in regular_anchor_state.items()})
            self.anchor_model.eval()
            for param in self.anchor_model.parameters():
                param.requires_grad_(False)
        trainable_params = list(self.model.parameters())
        if trainable_modules == "target_head":
            for param in self.model.parameters():
                param.requires_grad_(False)
            for param in self.model.target_head.parameters():
                param.requires_grad_(True)
            trainable_params = list(self.model.target_head.parameters())
        elif trainable_modules == "source_target_heads_no_slot":
            for param in self.model.parameters():
                param.requires_grad_(False)
            for param in self.model.source_head.parameters():
                param.requires_grad_(True)
            for param in self.model.target_head.parameters():
                param.requires_grad_(True)
            trainable_params = list(self.model.source_head.parameters()) + list(self.model.target_head.parameters())
        elif trainable_modules == "source_target_pair_heads_no_slot":
            if self.model.target_pair_head is None:
                raise ValueError("--trainable-modules source_target_pair_heads_no_slot requires --target-pair-head")
            for param in self.model.parameters():
                param.requires_grad_(False)
            for param in self.model.source_head.parameters():
                param.requires_grad_(True)
            for param in self.model.target_head.parameters():
                param.requires_grad_(True)
            for param in self.model.target_pair_head.parameters():
                param.requires_grad_(True)
            trainable_params = (
                list(self.model.source_head.parameters())
                + list(self.model.target_head.parameters())
                + list(self.model.target_pair_head.parameters())
            )
            if self.model.target_pair_owner_head is not None:
                for param in self.model.target_pair_owner_head.parameters():
                    param.requires_grad_(True)
                trainable_params += list(self.model.target_pair_owner_head.parameters())
        elif trainable_modules == "source_head":
            for param in self.model.parameters():
                param.requires_grad_(False)
            for param in self.model.source_head.parameters():
                param.requires_grad_(True)
            self.model.slot_embed.requires_grad_(True)
            trainable_params = list(self.model.source_head.parameters()) + [self.model.slot_embed]
        elif trainable_modules == "source_head_no_slot":
            for param in self.model.parameters():
                param.requires_grad_(False)
            for param in self.model.source_head.parameters():
                param.requires_grad_(True)
            trainable_params = list(self.model.source_head.parameters())
        elif trainable_modules == "target_pair_head":
            if self.model.target_pair_head is None:
                raise ValueError("--trainable-modules target_pair_head requires --target-pair-head")
            for param in self.model.parameters():
                param.requires_grad_(False)
            for param in self.model.target_pair_head.parameters():
                param.requires_grad_(True)
            trainable_params = list(self.model.target_pair_head.parameters())
            if self.model.target_pair_owner_head is not None:
                for param in self.model.target_pair_owner_head.parameters():
                    param.requires_grad_(True)
                trainable_params += list(self.model.target_pair_owner_head.parameters())
        elif trainable_modules == "target_ranking":
            if self.model.target_pair_head is None:
                raise ValueError("--trainable-modules target_ranking requires --target-pair-head")
            for param in self.model.parameters():
                param.requires_grad_(False)
            for param in self.model.edge.parameters():
                param.requires_grad_(True)
            for param in self.model.target_pair_head.parameters():
                param.requires_grad_(True)
            trainable_params = list(self.model.edge.parameters()) + list(self.model.target_pair_head.parameters())
            if self.model.target_pair_owner_head is not None:
                for param in self.model.target_pair_owner_head.parameters():
                    param.requires_grad_(True)
                trainable_params += list(self.model.target_pair_owner_head.parameters())
        elif trainable_modules == "target_pair_adapter":
            if self.model.target_pair_head is None or self.model.target_pair_edge is None:
                raise ValueError("--trainable-modules target_pair_adapter requires --target-pair-head --target-pair-adapter")
            for param in self.model.parameters():
                param.requires_grad_(False)
            for param in self.model.target_pair_edge.parameters():
                param.requires_grad_(True)
            for param in self.model.target_pair_head.parameters():
                param.requires_grad_(True)
            trainable_params = list(self.model.target_pair_edge.parameters()) + list(self.model.target_pair_head.parameters())
            if self.model.target_pair_owner_head is not None:
                for param in self.model.target_pair_owner_head.parameters():
                    param.requires_grad_(True)
                trainable_params += list(self.model.target_pair_owner_head.parameters())
        elif trainable_modules != "all":
            raise ValueError(f"unsupported trainable_modules: {trainable_modules!r}")
        self.opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.launch_pos_weight = float(launch_pos_weight)
        self.target_loss_weight = float(target_loss_weight)
        self.ship_loss_weight = float(ship_loss_weight)
        self.target_ship_joint_loss_weight = float(target_ship_joint_loss_weight)
        self.target_ship_joint_dagger_only = bool(target_ship_joint_dagger_only)
        self.critical_action_weight = float(critical_action_weight)
        self.target_loss_mask = str(target_loss_mask)
        self.target_margin_loss_weight = float(target_margin_loss_weight)
        self.target_margin = float(target_margin)
        self.target_margin_top_k = int(target_margin_top_k)
        self.slot_set_loss = bool(slot_set_loss)
        self.target_binary_loss_weight = float(target_binary_loss_weight)
        self.target_binary_pos_weight = float(target_binary_pos_weight)
        self.target_pair_softmax_loss_weight = float(target_pair_softmax_loss_weight)
        self.target_pair_margin_loss_weight = float(target_pair_margin_loss_weight)
        self.target_pair_owner_loss_weight = float(target_pair_owner_loss_weight)
        self.target_pair_within_owner_loss_weight = float(target_pair_within_owner_loss_weight)
        self.launch_count_loss_weight = float(launch_count_loss_weight)
        self.sample_weight_launch_scale = float(sample_weight_launch_scale)
        self.sample_weight_target_scale = float(sample_weight_target_scale)
        self.sample_weight_ship_scale = float(sample_weight_ship_scale)
        self.sample_weight_pair_scale = float(sample_weight_pair_scale)
        self.sample_weight_count_scale = float(sample_weight_count_scale)
        self.dagger_launch_negative_weight_scale = float(dagger_launch_negative_weight_scale)
        self.dagger_launch_positive_weight_scale = float(dagger_launch_positive_weight_scale)
        self.regular_anchor_source_weight = float(regular_anchor_source_weight)
        self.regular_anchor_pair_weight = float(regular_anchor_pair_weight)
        self.regular_anchor_count_weight = float(regular_anchor_count_weight)
        self.regular_anchor_target_weight = float(regular_anchor_target_weight)
        self.regular_anchor_min_sample_weight = float(regular_anchor_min_sample_weight)
        dataset = stack_rows(rows, sample_weights)
        generator = torch.Generator().manual_seed(seed)
        val_size = max(1, int(len(dataset) * val_frac))
        train_size = max(1, len(dataset) - val_size)
        train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)
        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
        self.samples = len(dataset)

    def train_epoch(self, state: dict[str, torch.Tensor], epoch: int) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        del epoch
        self.model.load_state_dict({key: value.to(self.device) for key, value in state.items()})
        self.model.train()
        sums: dict[str, float] = {}
        count = 0
        skipped_no_grad = 0
        for batch in self.train_loader:
            self.opt.zero_grad(set_to_none=True)
            loss, metrics = bc_loss(
                self.model,
                unpack(batch, self.device),
                self.launch_pos_weight,
                self.target_loss_weight,
                self.ship_loss_weight,
                self.target_ship_joint_loss_weight,
                self.target_ship_joint_dagger_only,
                self.critical_action_weight,
                self.target_loss_mask,
                self.target_margin_loss_weight,
                self.target_margin,
                self.target_margin_top_k,
                self.slot_set_loss,
                self.target_binary_loss_weight,
                self.target_binary_pos_weight,
                self.target_pair_softmax_loss_weight,
                self.target_pair_margin_loss_weight,
                self.target_pair_owner_loss_weight,
                self.target_pair_within_owner_loss_weight,
                self.launch_count_loss_weight,
                self.sample_weight_launch_scale,
                self.sample_weight_target_scale,
                self.sample_weight_ship_scale,
                self.sample_weight_pair_scale,
                self.sample_weight_count_scale,
                self.dagger_launch_negative_weight_scale,
                self.dagger_launch_positive_weight_scale,
            )
            anchor_loss, anchor_metrics = _regular_anchor_loss(
                self.model,
                self.anchor_model,
                unpack(batch, self.device),
                self.regular_anchor_source_weight,
                self.regular_anchor_pair_weight,
                self.regular_anchor_count_weight,
                self.regular_anchor_target_weight,
                self.regular_anchor_min_sample_weight,
            )
            loss = loss + anchor_loss
            metrics.update(anchor_metrics)
            n = int(batch[0].shape[0])
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value) * n
            count += n
            if not loss.requires_grad:
                skipped_no_grad += n
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()
        metrics = {f"train_{key}": value / max(1, count) for key, value in sums.items()}
        metrics["samples"] = float(self.samples)
        metrics["train_skipped_no_grad_samples"] = float(skipped_no_grad)
        metrics.update({f"val_{key}": value for key, value in self._eval_loader().items()})
        return _cpu_state_dict(self.model), metrics

    @torch.no_grad()
    def _eval_loader(self) -> dict[str, float]:
        self.model.eval()
        sums: dict[str, float] = {}
        count = 0
        for batch in self.val_loader:
            _, metrics = bc_loss(
                self.model,
                unpack(batch, self.device),
                self.launch_pos_weight,
                self.target_loss_weight,
                self.ship_loss_weight,
                self.target_ship_joint_loss_weight,
                self.target_ship_joint_dagger_only,
                self.critical_action_weight,
                self.target_loss_mask,
                self.target_margin_loss_weight,
                self.target_margin,
                self.target_margin_top_k,
                self.slot_set_loss,
                self.target_binary_loss_weight,
                self.target_binary_pos_weight,
                self.target_pair_softmax_loss_weight,
                self.target_pair_margin_loss_weight,
                self.target_pair_owner_loss_weight,
                self.target_pair_within_owner_loss_weight,
                self.launch_count_loss_weight,
                self.sample_weight_launch_scale,
                self.sample_weight_target_scale,
                self.sample_weight_ship_scale,
                self.sample_weight_pair_scale,
                self.sample_weight_count_scale,
                self.dagger_launch_negative_weight_scale,
                self.dagger_launch_positive_weight_scale,
            )
            anchor_loss, anchor_metrics = _regular_anchor_loss(
                self.model,
                self.anchor_model,
                unpack(batch, self.device),
                self.regular_anchor_source_weight,
                self.regular_anchor_pair_weight,
                self.regular_anchor_count_weight,
                self.regular_anchor_target_weight,
                self.regular_anchor_min_sample_weight,
            )
            del anchor_loss
            metrics.update(anchor_metrics)
            n = int(batch[0].shape[0])
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value) * n
            count += n
        return {key: value / max(1, count) for key, value in sums.items()}

    def eval_vs_regular(
        self,
        state: dict[str, torch.Tensor],
        games: int,
        seed: int,
        stochastic: bool,
        launch_bias: float,
        ship_bias: float,
        launch_temperature: float,
        target_mask_mode: str,
    ) -> dict[str, float]:
        self.model.load_state_dict({key: value.to(self.device) for key, value in state.items()})
        fd, path = tempfile.mkstemp(prefix="regular_bc_ray_eval_", suffix=".pt")
        os.close(fd)
        torch.save({"state_dict": _cpu_state_dict(self.model), "model": self.model_cfg, "update": 0, "metrics": {}}, path)
        try:
            return run_matchups(
                lambda: TinyPPOAgent(
                    path,
                    device=str(self.device),
                    deterministic=not stochastic,
                    launch_bias=launch_bias,
                    ship_bias=ship_bias,
                    launch_temperature=launch_temperature,
                    target_mask_mode=target_mask_mode,
                ),
                lambda: make_rulebase_agent("regular"),
                games=games,
                seed=seed,
                episode_steps=500,
                use_numba=True,
                progress=False,
            )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


@ray.remote
class BCEvalActor:
    def __init__(self, model_cfg: dict[str, int], device_override: str = ""):
        torch.set_num_threads(1)
        self.device = _resolve_actor_device(device_override)
        self.model_cfg = dict(model_cfg)
        self.model = TinyPolicyValueNet(**self.model_cfg).to(self.device)

    def eval_vs_regular(
        self,
        state: dict[str, torch.Tensor],
        games: int,
        seed: int,
        stochastic: bool,
        launch_bias: float,
        ship_bias: float,
        launch_temperature: float,
        target_mask_mode: str,
    ) -> dict[str, float]:
        self.model.load_state_dict({key: value.to(self.device) for key, value in state.items()})
        fd, path = tempfile.mkstemp(prefix="regular_bc_ray_eval_", suffix=".pt")
        os.close(fd)
        torch.save({"state_dict": _cpu_state_dict(self.model), "model": self.model_cfg, "update": 0, "metrics": {}}, path)
        try:
            return run_matchups(
                lambda: TinyPPOAgent(
                    path,
                    device=str(self.device),
                    deterministic=not stochastic,
                    launch_bias=launch_bias,
                    ship_bias=ship_bias,
                    launch_temperature=launch_temperature,
                    target_mask_mode=target_mask_mode,
                ),
                lambda: make_rulebase_agent("regular"),
                games=games,
                seed=seed,
                episode_steps=500,
                use_numba=True,
                progress=True,
                desc="regular-bc eval",
            )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


def _weighted_mean(parts: list[dict[str, float]]) -> dict[str, float]:
    total = sum(float(p.get("samples", 1.0)) for p in parts)
    out: dict[str, float] = {}
    for part in parts:
        weight = float(part.get("samples", 1.0)) / max(1.0, total)
        for key, value in part.items():
            if key == "samples":
                continue
            out[key] = out.get(key, 0.0) + float(value) * weight
    out["samples"] = total
    return out


def _merge_eval(parts: list[dict[str, float]]) -> dict[str, float]:
    games = sum(float(p["games"]) for p in parts)
    wins = sum(float(p["wins"]) for p in parts)
    losses = sum(float(p["losses"]) for p in parts)
    draws = sum(float(p["draws"]) for p in parts)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": wins / max(1.0, games),
        "nonloss": (wins + draws) / max(1.0, games),
        "mean_reward": (wins - losses) / max(1.0, games),
    }


def _imitation_score(metrics: dict[str, float]) -> float:
    launch_f1 = float(metrics.get("val_launch_f1", 0.0))
    target_acc = float(metrics.get("val_target_acc", 0.0))
    target_pair_acc = float(metrics.get("val_target_pair_acc", 0.0))
    target_quality = max(target_acc, target_pair_acc)
    ship_acc = float(metrics.get("val_ship_acc", 0.0))
    count_mae = min(3.0, float(metrics.get("val_action_count_mae", 3.0)))
    pred_rate = max(1e-6, float(metrics.get("val_launch_pred_rate", 0.0)))
    true_rate = max(1e-6, float(metrics.get("val_launch_true_rate", 0.0)))
    density_penalty = min(3.0, abs(math.log(pred_rate / true_rate)))
    density_alignment = max(0.0, 1.0 - density_penalty / 3.0)
    metrics["val_launch_density_penalty"] = density_penalty
    metrics["val_launch_density_alignment"] = density_alignment
    metrics["val_target_quality"] = target_quality
    return (
        0.45 * launch_f1
        + 0.20 * target_quality
        + 0.10 * ship_acc
        + 0.20 * density_alignment
        - 0.08 * count_mae
        - 0.04 * density_penalty
    )


def save_checkpoint(path: Path, state: dict[str, torch.Tensor], model_cfg: dict[str, int], epoch: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "model": model_cfg, "update": epoch, "metrics": metrics}, path)


def _flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_metrics(value, name))
        elif isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
            out[name] = float(value)
    return out


def _init_swanlab(args: argparse.Namespace) -> Any | None:
    if args.no_swanlab:
        return None
    key = os.environ.get("SWANLAB_API_KEY")
    if not key:
        key_file = Path("training/config/swanlab_key.txt")
        if key_file.exists():
            os.environ["SWANLAB_API_KEY"] = key_file.read_text().strip()
    os.environ.setdefault("SWANLAB_NO_INTERACTIVE", "1")
    os.environ.setdefault("SWANLAB_DISABLE_INTERACTIVE", "1")
    try:
        import swanlab

        run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_experiment,
            config={"tinyPPO_regular_bc_ray": vars(args)},
            mode=args.swanlab_mode,
            public=False,
            launcher=False,
        )
        swanlab.log({"bc/swanlab_initialized": 1}, step=0)
        return swanlab if run is None else run
    except Exception as exc:
        if not args.allow_no_swanlab:
            raise
        print(json.dumps({"event": "swanlab_init_failed", "error": str(exc)}, ensure_ascii=True), flush=True)
        return None


def _log_swanlab(swan: Any | None, payload: dict[str, Any], step: int) -> None:
    if swan is None:
        return
    metrics = _flatten_metrics(payload)
    if not metrics:
        return
    try:
        swan.log(metrics, step=step)
    except Exception as exc:
        print(json.dumps({"event": "swanlab_log_failed", "step": step, "error": str(exc)}, ensure_ascii=True), flush=True)


def collect_dataset(args: argparse.Namespace) -> tuple[list[Any], dict[str, float], list[float] | None]:
    cache_path = Path(args.dataset_cache) if args.dataset_cache else None
    if cache_path is not None and cache_path.exists() and not args.refresh_dataset:
        with cache_path.open("rb") as fh:
            payload = pickle.load(fh)
        rows = payload["rows"]
        metrics = dict(payload.get("metrics", {}))
        metrics["cache_loaded"] = 1.0
        metrics["cache_path"] = str(cache_path)
    else:
        rows, metrics = collect_regular_dataset(args, cache_path)
    rows, metrics = _subsample_regular_rows(rows, metrics, args)
    sample_weights: list[float] | None = None
    if args.dagger_checkpoint or args.dagger_checkpoints or args.dagger_cache:
        regular_count = len(rows)
        regular_labelled_actions = float(sum(_row_action_count(row) for row in rows))
        dagger_rows, dagger_metrics = collect_dagger_dataset(args)
        dagger_weights, dagger_weight_metrics = _dagger_sample_weights(dagger_rows, args)
        dagger_rows, dagger_weights, dagger_repeat_metrics = _repeat_dagger_rows(dagger_rows, dagger_weights, args)
        dagger_labelled_actions = float(sum(_row_action_count(row) for row in dagger_rows))
        rows.extend(dagger_rows)
        if dagger_weights is not None:
            sample_weights = [1.0] * regular_count + dagger_weights
        total_labelled_actions = regular_labelled_actions + dagger_labelled_actions
        metrics = {
            **metrics,
            "samples": float(len(rows)),
            "labelled_actions": total_labelled_actions,
            "regular_samples": float(regular_count),
            "regular_labelled_actions": regular_labelled_actions,
            "dagger_samples": float(len(dagger_rows)),
            "dagger_labelled_actions": dagger_labelled_actions,
            "dagger_action_fraction": (
                float(dagger_labelled_actions / total_labelled_actions) if total_labelled_actions > 0 else 0.0
            ),
            "regular_action_fraction": (
                float(regular_labelled_actions / total_labelled_actions) if total_labelled_actions > 0 else 0.0
            ),
            "dagger": {**dagger_metrics, **dagger_weight_metrics, **dagger_repeat_metrics},
        }
    return rows, metrics, sample_weights


def _subsample_regular_rows(rows: list[Any], metrics: dict[str, float], args: argparse.Namespace) -> tuple[list[Any], dict[str, float]]:
    max_rows = int(getattr(args, "regular_max_rows", 0))
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows, metrics
    rng = random.Random(int(args.seed) + 27_182_818)
    indices = sorted(rng.sample(range(len(rows)), max_rows))
    sampled_rows = [rows[index] for index in indices]
    before_actions = float(sum(_row_action_count(row) for row in rows))
    sampled_actions = float(sum(_row_action_count(row) for row in sampled_rows))
    sampled_metrics = dict(metrics)
    sampled_metrics["regular_samples_before_subsample"] = float(len(rows))
    sampled_metrics["regular_labelled_actions_before_subsample"] = before_actions
    sampled_metrics["samples"] = float(len(sampled_rows))
    sampled_metrics["labelled_actions"] = sampled_actions
    sampled_metrics["regular_subsampled"] = 1.0
    sampled_metrics["regular_max_rows"] = float(max_rows)
    sampled_metrics["regular_subsample_keep_frac"] = float(len(sampled_rows) / len(rows))
    sampled_metrics["regular_action_keep_frac"] = float(sampled_actions / before_actions) if before_actions > 0 else 0.0
    return sampled_rows, sampled_metrics


def _dagger_sample_weights(rows: list[Any], args: argparse.Namespace) -> tuple[list[float] | None, dict[str, float]]:
    base_weight = float(args.dagger_loss_weight)
    action_cap = float(getattr(args, "dagger_action_weight_cap", 0.0))
    has_row_overrides = any(
        abs(float(getattr(row, "dagger_loss_weight_override", getattr(row, "sample_weight", 1.0))) - 1.0) > 1e-9
        for row in rows
    )
    use_weights = abs(base_weight - 1.0) > 1e-9 or action_cap > 0.0 or has_row_overrides
    if not use_weights:
        return None, {"dagger_loss_weight": 1.0, "dagger_action_weight_cap": 0.0}

    weights: list[float] = []
    raw_actions = 0.0
    weighted_actions = 0.0
    for row in rows:
        action_count = _row_action_count(row)
        raw_actions += float(action_count)
        row_weight = base_weight * float(getattr(row, "dagger_loss_weight_override", getattr(row, "sample_weight", 1.0)))
        if action_cap > 0.0 and action_count > action_cap:
            row_weight *= action_cap / float(action_count)
        weights.append(float(row_weight))
        weighted_actions += float(action_count) * float(row_weight)

    return weights, {
        "dagger_loss_weight": base_weight,
        "dagger_action_weight_cap": action_cap,
        "dagger_weight_mean": float(sum(weights) / len(weights)) if weights else 0.0,
        "dagger_weight_min": float(min(weights)) if weights else 0.0,
        "dagger_weight_max": float(max(weights)) if weights else 0.0,
        "dagger_unweighted_labelled_actions": raw_actions,
        "dagger_weighted_labelled_actions": weighted_actions,
    }


def _repeat_dagger_rows(
    rows: list[Any], weights: list[float] | None, args: argparse.Namespace
) -> tuple[list[Any], list[float] | None, dict[str, float]]:
    repeat = float(getattr(args, "dagger_repeat", 1.0))
    if repeat <= 0.0:
        return [], [] if weights is not None else None, {
            "dagger_repeat": repeat,
            "dagger_repeat_base_rows": float(len(rows)),
            "dagger_repeat_rows": 0.0,
        }
    if not rows or abs(repeat - 1.0) <= 1e-9:
        return rows, weights, {
            "dagger_repeat": repeat,
            "dagger_repeat_base_rows": float(len(rows)),
            "dagger_repeat_rows": float(len(rows)),
        }

    full = int(math.floor(repeat))
    frac = repeat - float(full)
    repeated_rows: list[Any] = []
    repeated_weights: list[float] | None = [] if weights is not None else None
    for _ in range(full):
        repeated_rows.extend(rows)
        if repeated_weights is not None and weights is not None:
            repeated_weights.extend(weights)

    fractional_count = int(round(frac * len(rows)))
    if fractional_count > 0:
        rng = random.Random(int(args.seed) + 87_654_321)
        indices = sorted(rng.sample(range(len(rows)), min(fractional_count, len(rows))))
        repeated_rows.extend(rows[index] for index in indices)
        if repeated_weights is not None and weights is not None:
            repeated_weights.extend(weights[index] for index in indices)

    raw_actions = float(sum(_row_action_count(row) for row in rows))
    repeated_actions = float(sum(_row_action_count(row) for row in repeated_rows))
    return repeated_rows, repeated_weights, {
        "dagger_repeat": repeat,
        "dagger_repeat_base_rows": float(len(rows)),
        "dagger_repeat_rows": float(len(repeated_rows)),
        "dagger_repeat_base_labelled_actions": raw_actions,
        "dagger_repeat_labelled_actions": repeated_actions,
        "dagger_repeat_action_mult": float(repeated_actions / raw_actions) if raw_actions > 0 else 0.0,
    }


def collect_regular_dataset(args: argparse.Namespace, cache_path: Path | None) -> tuple[list[Any], dict[str, float]]:
    players_values = _players_list(args.players_list)
    jobs = [
        (args.seed + players * 1_000_000 + game, players)
        for players in players_values
        for game in range(args.games_per_players)
    ]
    games_per_task = max(1, int(args.collect_games_per_task))
    job_shards = [jobs[i : i + games_per_task] for i in range(0, len(jobs), games_per_task)]
    futures = [
        collect_games_task.remote(
            shard,
            args.episode_steps,
            args.keep_noop_prob,
            args.sample_stride,
            args.rows_per_game,
            not args.no_numba,
            args.row_target_mask_mode,
        )
        for shard in job_shards
    ]
    rows: list[Any] = []
    labelled = 0.0
    skipped = 0.0
    completed = 0
    next_partial_save = args.partial_cache_interval if args.partial_cache_interval > 0 else 0
    progress = tqdm(total=len(jobs), desc="ray collect regular BC", dynamic_ncols=True) if tqdm is not None else None
    pending = list(futures)
    while pending:
        done, pending = ray.wait(pending, num_returns=min(64, len(pending)))
        for ref in done:
            game_rows, metrics = ray.get(ref)
            rows.extend(game_rows)
            labelled += float(metrics.get("labelled_actions", 0.0))
            skipped += float(metrics.get("skipped_actions", 0.0))
            completed += int(metrics.get("games", 0.0))
        if progress is not None:
            progress.n = completed
            progress.refresh()
            progress.set_postfix(samples=len(rows), labelled=int(labelled))
        if cache_path is not None and next_partial_save > 0 and completed >= next_partial_save:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            partial_metrics = {
                "games": float(completed),
                "samples": float(len(rows)),
                "labelled_actions": labelled,
                "skipped_actions": skipped,
                "players_modes": float(len(players_values)),
                "row_target_mask_mode": args.row_target_mask_mode,
                "partial": 1.0,
            }
            partial_path = cache_path.with_suffix(cache_path.suffix + f".partial_{completed:05d}")
            tmp_partial_path = partial_path.with_suffix(partial_path.suffix + ".tmp")
            with tmp_partial_path.open("wb") as fh:
                pickle.dump({"rows": rows, "metrics": partial_metrics, "args": vars(args)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_partial_path.replace(partial_path)
            _prune_partial_caches(cache_path, args.partial_cache_keep)
            while next_partial_save <= completed:
                next_partial_save += args.partial_cache_interval
    if progress is not None:
        progress.close()
    metrics = {
        "games": float(completed),
        "samples": float(len(rows)),
        "labelled_actions": labelled,
        "skipped_actions": skipped,
        "players_modes": float(len(players_values)),
        "collect_games_per_task": float(games_per_task),
        "row_target_mask_mode": args.row_target_mask_mode,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _prune_partial_caches(cache_path, 0)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump({"rows": rows, "metrics": metrics, "args": vars(args)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        metrics["cache_saved"] = 1.0
        metrics["cache_path"] = str(cache_path)
    return rows, metrics


def collect_dagger_dataset(args: argparse.Namespace) -> tuple[list[Any], dict[str, float]]:
    cache_paths = _path_list(args.dagger_cache) if args.dagger_cache else []
    if cache_paths and not args.refresh_dataset:
        missing = [str(path) for path in cache_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing DAgger caches: {missing}")
        rows: list[Any] = []
        metrics: dict[str, float] = {
            "cache_loaded": 1.0,
            "cache_count": float(len(cache_paths)),
        }
        loaded_paths: list[str] = []
        for cache_path in cache_paths:
            with cache_path.open("rb") as fh:
                payload = pickle.load(fh)
            part_rows = list(payload["rows"])
            part_metrics = dict(payload.get("metrics", {}))
            rows.extend(part_rows)
            loaded_paths.append(str(cache_path))
            for key, value in part_metrics.items():
                if isinstance(value, (int, float)) and key not in {"cache_loaded", "cache_saved"}:
                    metrics[key] = float(metrics.get(key, 0.0)) + float(value)
        metrics["samples"] = float(len(rows))
        metrics["cache_paths_count"] = float(len(loaded_paths))
        rows, metrics = _filter_dagger_rows(rows, metrics, args)
        metrics["cache_paths"] = ",".join(loaded_paths)  # type: ignore[assignment]
        rows, metrics = _subsample_dagger_rows(rows, metrics, args)
        return rows, metrics

    checkpoints = [item.strip() for item in args.dagger_checkpoints.split(",") if item.strip()]
    if args.dagger_checkpoint:
        checkpoints.insert(0, args.dagger_checkpoint)
    checkpoints = list(dict.fromkeys(checkpoints))
    if not checkpoints:
        return [], {"games": 0.0, "samples": 0.0, "checkpoints": 0.0}
    missing = [checkpoint for checkpoint in checkpoints if not Path(checkpoint).exists()]
    if missing:
        raise FileNotFoundError(f"missing DAgger checkpoints: {missing}")

    players_values = _players_list(args.players_list)
    jobs = [
        (args.seed + 20_000_000 + players * 1_000_000 + game, players, game % players)
        for players in players_values
        for game in range(args.dagger_games_per_players)
    ]
    if not jobs:
        return [], {"games": 0.0, "samples": 0.0}
    actor_count = max(1, min(args.dagger_actors, len(jobs)))
    shards = [jobs[i::actor_count] for i in range(actor_count)]
    dagger_gpu_ids = _int_list(args.dagger_gpu_ids_manual)
    if dagger_gpu_ids:
        print(
            json.dumps(
                {
                    "event": "dagger_manual_gpu_ids",
                    "dagger_gpu_ids": dagger_gpu_ids,
                    "actors": actor_count,
                    "checkpoints": checkpoints,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    actors = [
        DaggerCollectActor.options(
            num_cpus=args.dagger_cpus_per_actor,
            num_gpus=0.0 if dagger_gpu_ids else args.dagger_gpus_per_actor,
        ).remote(
            checkpoints[i % len(checkpoints)],
            f"cuda:{dagger_gpu_ids[i % len(dagger_gpu_ids)]}" if dagger_gpu_ids else args.dagger_device,
            not args.dagger_stochastic,
            args.launch_bias,
            args.ship_bias,
            args.launch_temperature,
            args.dagger_model_target_mask_mode,
            args.dagger_target_pair_weight,
            args.dagger_model_seat_only,
        )
        for i in range(actor_count)
    ]
    futures = [
        actor.collect_games.remote(
            shard,
            args.episode_steps,
            args.keep_noop_prob,
            args.sample_stride,
            args.rows_per_game,
            not args.no_numba,
            args.row_target_mask_mode,
        )
        for actor, shard in zip(actors, shards, strict=True)
        if shard
    ]
    rows: list[Any] = []
    labelled = 0.0
    skipped = 0.0
    model_seat_label_actions = 0.0
    regular_label_actions = 0.0
    completed_games = 0.0
    progress = tqdm(total=len(futures), desc="ray collect regular DAgger", dynamic_ncols=True) if tqdm is not None else None
    pending = list(futures)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        for ref in done:
            part_rows, part_metrics = ray.get(ref)
            rows.extend(part_rows)
            completed_games += float(part_metrics.get("games", 0.0))
            labelled += float(part_metrics.get("labelled_actions", 0.0))
            skipped += float(part_metrics.get("skipped_actions", 0.0))
            model_seat_label_actions += float(part_metrics.get("model_seat_label_actions", 0.0))
            regular_label_actions += float(part_metrics.get("regular_label_actions", 0.0))
        if progress is not None:
            progress.update(len(done))
            progress.set_postfix(samples=len(rows), games=int(completed_games))
    if progress is not None:
        progress.close()
    metrics = {
        "games": completed_games,
        "samples": float(len(rows)),
        "labelled_actions": labelled,
        "skipped_actions": skipped,
        "model_seat_label_actions": model_seat_label_actions,
        "regular_label_actions": regular_label_actions,
        "actors": float(actor_count),
        "checkpoints": float(len(checkpoints)),
        "model_seat_only": float(bool(args.dagger_model_seat_only)),
        "row_target_mask_mode": args.row_target_mask_mode,
    }
    cache_path = cache_paths[0] if len(cache_paths) == 1 else None
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump({"rows": rows, "metrics": metrics, "args": vars(args)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        metrics["cache_saved"] = 1.0
        metrics["cache_path"] = str(cache_path)
    rows, metrics = _filter_dagger_rows(rows, metrics, args)
    rows, metrics = _subsample_dagger_rows(rows, metrics, args)
    return rows, metrics


def _row_action_count(row: Any) -> int:
    return int(np.asarray(row.launch_mask).sum())


def _filter_dagger_rows(rows: list[Any], metrics: dict[str, float], args: argparse.Namespace) -> tuple[list[Any], dict[str, float]]:
    min_actions = int(getattr(args, "dagger_min_actions_per_row", -1))
    max_actions = int(getattr(args, "dagger_max_actions_per_row", -1))
    min_turn = int(getattr(args, "dagger_min_turn", -1))
    max_turn = int(getattr(args, "dagger_max_turn", -1))
    min_final_reward = float(getattr(args, "dagger_min_final_reward", -999.0))
    max_final_reward = float(getattr(args, "dagger_max_final_reward", 999.0))
    min_abs_action_gap = int(getattr(args, "dagger_min_abs_action_gap", -1))
    keep_outcomes = {item.strip() for item in str(getattr(args, "dagger_keep_outcomes", "")).split(",") if item.strip()}
    if (
        min_actions < 0
        and max_actions < 0
        and min_turn < 0
        and max_turn < 0
        and min_final_reward <= -999.0
        and max_final_reward >= 999.0
        and min_abs_action_gap < 0
        and not keep_outcomes
    ):
        return rows, metrics

    filtered: list[Any] = []
    kept_actions = 0
    before_actions = 0
    missing_turn = 0
    missing_final_reward = 0
    missing_action_gap = 0
    missing_outcome = 0
    for row in rows:
        action_count = _row_action_count(row)
        before_actions += action_count
        if keep_outcomes:
            outcome = getattr(row, "dagger_model_outcome", None)
            if outcome is None:
                missing_outcome += 1
                continue
            if str(outcome) not in keep_outcomes:
                continue
        if min_actions >= 0 and action_count < min_actions:
            continue
        if max_actions >= 0 and action_count > max_actions:
            continue
        if min_turn >= 0 or max_turn >= 0:
            turn = getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", None))
            if turn is None:
                missing_turn += 1
                continue
            turn = int(turn)
            if min_turn >= 0 and turn < min_turn:
                continue
            if max_turn >= 0 and turn > max_turn:
                continue
        if min_final_reward > -999.0 or max_final_reward < 999.0:
            final_reward = getattr(row, "dagger_model_final_reward", None)
            if final_reward is None:
                missing_final_reward += 1
                continue
            final_reward = float(final_reward)
            if final_reward < min_final_reward or final_reward > max_final_reward:
                continue
        if min_abs_action_gap >= 0:
            model_count = getattr(row, "dagger_model_action_count", None)
            label_count = getattr(row, "dagger_regular_label_action_count", None)
            if model_count is None or label_count is None:
                missing_action_gap += 1
                continue
            if abs(int(label_count) - int(model_count)) < min_abs_action_gap:
                continue
        filtered.append(row)
        kept_actions += action_count

    filtered_metrics = dict(metrics)
    filtered_metrics["samples_before_filter"] = float(len(rows))
    filtered_metrics["samples"] = float(len(filtered))
    filtered_metrics["labelled_actions_before_filter"] = float(before_actions)
    filtered_metrics["labelled_actions"] = float(kept_actions)
    filtered_metrics["action_filter_min"] = float(min_actions)
    filtered_metrics["action_filter_max"] = float(max_actions)
    filtered_metrics["turn_filter_min"] = float(min_turn)
    filtered_metrics["turn_filter_max"] = float(max_turn)
    filtered_metrics["final_reward_filter_min"] = float(min_final_reward)
    filtered_metrics["final_reward_filter_max"] = float(max_final_reward)
    filtered_metrics["abs_action_gap_filter_min"] = float(min_abs_action_gap)
    filtered_metrics["outcome_filter_enabled"] = float(bool(keep_outcomes))
    filtered_metrics["missing_outcome_for_filter"] = float(missing_outcome)
    filtered_metrics["missing_turn_for_filter"] = float(missing_turn)
    filtered_metrics["missing_final_reward_for_filter"] = float(missing_final_reward)
    filtered_metrics["missing_action_gap_for_filter"] = float(missing_action_gap)
    filtered_metrics["dagger_filtered"] = 1.0
    if rows:
        filtered_metrics["filter_keep_frac"] = float(len(filtered) / len(rows))
    if filtered:
        filtered_metrics["filter_mean_actions"] = float(kept_actions / len(filtered))
    return filtered, filtered_metrics


def _subsample_dagger_rows(rows: list[Any], metrics: dict[str, float], args: argparse.Namespace) -> tuple[list[Any], dict[str, float]]:
    max_rows = int(getattr(args, "dagger_max_rows", 0))
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows, metrics
    rng = random.Random(int(args.seed) + 31_415_927)
    indices = sorted(rng.sample(range(len(rows)), max_rows))
    sampled_rows = [rows[index] for index in indices]
    before_actions = float(sum(_row_action_count(row) for row in rows))
    sampled_actions = float(sum(_row_action_count(row) for row in sampled_rows))
    sampled_metrics = dict(metrics)
    sampled_metrics["samples_before_subsample"] = float(len(rows))
    sampled_metrics["labelled_actions_before_subsample"] = before_actions
    sampled_metrics["samples"] = float(len(sampled_rows))
    sampled_metrics["labelled_actions"] = sampled_actions
    sampled_metrics["subsampled"] = 1.0
    sampled_metrics["max_rows"] = float(max_rows)
    sampled_metrics["subsample_keep_frac"] = float(len(sampled_rows) / len(rows))
    sampled_metrics["subsample_action_keep_frac"] = float(sampled_actions / before_actions) if before_actions > 0 else 0.0
    return sampled_rows, sampled_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-temp-dir", default="", help="Optional Ray temp/session directory, useful when /tmp is low on space.")
    parser.add_argument("--ray-no-runtime-env", action="store_true", help="Start Ray workers in the current environment without packaging a runtime_env.")
    parser.add_argument("--out", default="tinyPPO/regular_bc_ray.pt")
    parser.add_argument("--best-out", default="tinyPPO/regular_bc.pt")
    parser.add_argument("--best-gate-out", default="", help="Optional checkpoint path for the best same-state Phase1 gate result.")
    parser.add_argument("--best-online-out", default="", help="Optional checkpoint path for the best online vs-regular nonloss result.")
    parser.add_argument("--resume", default="", help="Resume model weights from a regular BC checkpoint. Epoch numbering continues from checkpoint update.")
    parser.add_argument("--resume-compatible", action="store_true", help="Warm-start only checkpoint tensors whose names and shapes match the requested model config.")
    parser.add_argument("--players-list", default="2")
    parser.add_argument("--games-per-players", type=int, default=2000)
    parser.add_argument("--dataset-cache", default="", help="Pickle cache for collected BC rows. Existing cache is reused unless --refresh-dataset is set.")
    parser.add_argument("--regular-max-rows", type=int, default=0, help="Deterministically subsample original regular BC rows after cache load/collection. 0 keeps all regular rows.")
    parser.add_argument("--refresh-dataset", action="store_true")
    parser.add_argument("--partial-cache-interval", type=int, default=0, help="Save partial regular BC dataset caches every N completed games.")
    parser.add_argument("--partial-cache-keep", type=int, default=1, help="Keep only the newest N partial regular BC caches. Final cache save removes all partials first.")
    parser.add_argument("--collect-games-per-task", type=int, default=1, help="Batch this many regular games into one Ray collection task.")
    parser.add_argument("--dagger-checkpoint", default="", help="Optional policy checkpoint used to generate on-policy states labelled by regular.")
    parser.add_argument("--dagger-checkpoints", default="", help="Comma list of policy checkpoints used round-robin to generate on-policy states labelled by regular.")
    parser.add_argument("--dagger-cache", default="", help="Pickle cache for DAgger rows. Existing cache is reused unless --refresh-dataset is set.")
    parser.add_argument("--dagger-games-per-players", type=int, default=0)
    parser.add_argument("--dagger-max-rows", type=int, default=0, help="Deterministically subsample DAgger rows for training after cache load/collection. 0 keeps all DAgger rows.")
    parser.add_argument("--dagger-min-actions-per-row", type=int, default=-1, help="Keep only DAgger rows with at least this many regular-labelled launch actions. <0 disables the lower bound.")
    parser.add_argument("--dagger-max-actions-per-row", type=int, default=-1, help="Keep only DAgger rows with at most this many regular-labelled launch actions. <0 disables the upper bound.")
    parser.add_argument("--dagger-min-turn", type=int, default=-1, help="Keep only metadata DAgger rows at or after this turn. <0 disables.")
    parser.add_argument("--dagger-max-turn", type=int, default=-1, help="Keep only metadata DAgger rows at or before this turn. <0 disables.")
    parser.add_argument("--dagger-min-final-reward", type=float, default=-999.0, help="Keep only metadata DAgger rows whose model rollout final reward is at least this value.")
    parser.add_argument("--dagger-max-final-reward", type=float, default=999.0, help="Keep only metadata DAgger rows whose model rollout final reward is at most this value. Use 0 or -1 to focus losing rollouts.")
    parser.add_argument("--dagger-keep-outcomes", default="", help="Comma list of score-based DAgger outcomes to keep, e.g. loss,draw. Requires rows with dagger_model_outcome metadata.")
    parser.add_argument("--dagger-min-abs-action-gap", type=int, default=-1, help="Keep only metadata DAgger rows where abs(regular_label_actions - model_actions) is at least this value.")
    parser.add_argument("--dagger-repeat", type=float, default=1.0, help="Repeat/oversample DAgger rows after filtering without dropping regular rows. 2.0 roughly doubles DAgger row exposure.")
    parser.add_argument("--dagger-loss-weight", type=float, default=1.0, help="Per-row loss weight for DAgger rows. Use <1 to expose model-rollout states while keeping original regular cadence dominant.")
    parser.add_argument("--dagger-action-weight-cap", type=float, default=0.0, help="If >0, scale DAgger rows with more labelled actions than this by cap/action_count. Keeps high-action states while capping their active-action loss mass.")
    parser.add_argument("--dagger-actors", type=int, default=8)
    parser.add_argument("--dagger-cpus-per-actor", type=float, default=2.0)
    parser.add_argument("--dagger-gpus-per-actor", type=float, default=0.0)
    parser.add_argument("--dagger-gpu-ids-manual", default="", help="Comma list of physical CUDA ids for DAgger actors. If set, Ray GPU resources are not reserved for DAgger actors.")
    parser.add_argument("--dagger-device", default="cpu")
    parser.add_argument("--dagger-stochastic", action="store_true")
    parser.add_argument("--dagger-model-seat-only", action="store_true", help="Collect only the model-controlled seat states from DAgger rollouts; regular opponents still act but are not added as rows.")
    parser.add_argument("--dagger-model-target-mask-mode", choices=["candidate", "safe", "all_planets"], default="candidate", help="Runtime target mask used by the model while generating DAgger states.")
    parser.add_argument("--dagger-target-pair-weight", type=float, default=1.0, help="Runtime target-pair logit weight used by the model while generating DAgger states.")
    parser.add_argument("--rows-per-game", type=int, default=12)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--keep-noop-prob", type=float, default=0.15)
    parser.add_argument(
        "--row-target-mask-mode",
        choices=["candidate", "safe", "all_planets"],
        default="candidate",
        help="Target mask stored in newly collected regular/DAgger BC rows. Loaded caches keep their saved mask.",
    )
    parser.add_argument("--trainers", type=int, default=8)
    parser.add_argument("--cpus-per-trainer", type=float, default=2.0)
    parser.add_argument("--gpus-per-trainer", type=float, default=1.0)
    parser.add_argument("--trainer-gpu-ids", default="", help="Comma list of physical CUDA ids for trainer actors. If set, Ray GPU resources are not reserved for trainers.")
    parser.add_argument("--eval-actors", type=int, default=1)
    parser.add_argument("--eval-cpus-per-actor", type=float, default=4.0)
    parser.add_argument("--gpus-per-eval-actor", type=float, default=1.0)
    parser.add_argument("--eval-gpu-ids-manual", default="", help="Comma list of physical CUDA ids for eval actors. If set, Ray GPU resources are not reserved for eval actors.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--launch-pos-weight", type=float, default=4.0)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--ship-loss-weight", type=float, default=0.5)
    parser.add_argument("--target-ship-joint-loss-weight", type=float, default=0.0, help="Auxiliary CE over the joint target x ship-bucket choice for each labelled launch.")
    parser.add_argument("--target-ship-joint-dagger-only", action="store_true", help="Apply target-ship joint auxiliary loss only to DAgger-like rows (sample_weight < 1).")
    parser.add_argument("--critical-action-weight", type=float, default=0.0)
    parser.add_argument("--target-loss-mask", choices=["dataset", "all_planets"], default="dataset")
    parser.add_argument("--target-margin-loss-weight", type=float, default=0.0)
    parser.add_argument("--target-margin", type=float, default=0.20)
    parser.add_argument("--target-margin-top-k", type=int, default=3)
    parser.add_argument("--slot-set-loss", action="store_true")
    parser.add_argument("--target-binary-loss-weight", type=float, default=0.0)
    parser.add_argument("--target-binary-pos-weight", type=float, default=1.0)
    parser.add_argument("--target-pair-softmax-loss-weight", type=float, default=0.0)
    parser.add_argument("--target-pair-margin-loss-weight", type=float, default=0.0, help="Auxiliary hard-negative margin loss over source-target pair logits for regular targets.")
    parser.add_argument("--target-pair-owner-loss-weight", type=float, default=0.0, help="Auxiliary CE over target owner groups aggregated from source-target pair logits.")
    parser.add_argument("--target-pair-within-owner-loss-weight", type=float, default=0.0, help="Auxiliary CE over same-owner target candidates for each regular-labelled source-target pair.")
    parser.add_argument("--launch-count-loss-weight", type=float, default=0.0, help="Auxiliary SmoothL1 loss matching predicted launch-count probability sum to the regular action count per row.")
    parser.add_argument("--sample-weight-launch-scale", type=float, default=1.0, help="Scale how much per-row sample_weight amplifies launch/source loss. 1 keeps historical behavior.")
    parser.add_argument("--sample-weight-target-scale", type=float, default=1.0, help="Scale how much per-row sample_weight amplifies slot target loss. 0 makes weighted rows count like normal rows for this component.")
    parser.add_argument("--sample-weight-ship-scale", type=float, default=1.0, help="Scale how much per-row sample_weight amplifies ship bucket loss.")
    parser.add_argument("--sample-weight-pair-scale", type=float, default=1.0, help="Scale how much per-row sample_weight amplifies target-pair auxiliary losses.")
    parser.add_argument("--sample-weight-count-scale", type=float, default=1.0, help="Scale how much per-row sample_weight amplifies launch-count loss.")
    parser.add_argument("--dagger-launch-negative-weight-scale", type=float, default=1.0, help="Extra multiplier for no-launch CE slots on DAgger-like rows (sample_weight < 1). Regular rows are unchanged.")
    parser.add_argument("--dagger-launch-positive-weight-scale", type=float, default=1.0, help="Extra multiplier for launch CE slots on DAgger-like rows (sample_weight < 1). Regular rows are unchanged.")
    parser.add_argument("--regular-anchor-source-weight", type=float, default=0.0, help="KL anchor against --resume source logits on regular rows (sample_weight above threshold).")
    parser.add_argument("--regular-anchor-pair-weight", type=float, default=0.0, help="KL anchor against --resume source-target pair logits on regular rows (sample_weight above threshold).")
    parser.add_argument("--regular-anchor-count-weight", type=float, default=0.0, help="SmoothL1 anchor against --resume soft launch-count on regular rows.")
    parser.add_argument("--regular-anchor-target-weight", type=float, default=0.0, help="KL anchor against --resume slot target logits on regular rows.")
    parser.add_argument("--regular-anchor-min-sample-weight", type=float, default=0.999, help="Rows with sample_weight >= this value receive regular-anchor loss; use DAgger loss weights below this to leave DAgger rows unanchored.")
    parser.add_argument(
        "--trainable-modules",
        choices=[
            "all",
            "target_head",
            "source_target_heads_no_slot",
            "source_target_pair_heads_no_slot",
            "source_head",
            "source_head_no_slot",
            "target_pair_head",
            "target_ranking",
            "target_pair_adapter",
        ],
        default="all",
    )
    parser.add_argument("--val-frac", type=float, default=0.08)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--source-target-summary", action="store_true", help="Let the launch/source head see a pooled summary of source-target edge features.")
    parser.add_argument("--target-pair-head", action="store_true", help="Add a source-target pair head for direct target imitation.")
    parser.add_argument("--target-pair-adapter", action="store_true", help="Use a separate edge MLP for target-pair logits so target-ranking updates do not perturb source/ship heads.")
    parser.add_argument("--target-pair-owner-head", action="store_true", help="Add a zero-initialized target-owner bias head on target-pair logits.")
    parser.add_argument("--eval-interval", type=int, default=40)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument(
        "--eval-games-per-actor",
        type=int,
        default=0,
        help="If >0, submit this many games to each eval actor and derive total eval games from the selected actors.",
    )
    parser.add_argument("--max-pending-evals", type=int, default=2)
    parser.add_argument("--eval-min-launch-recall", type=float, default=0.0, help="Skip slow online eval until validation launch recall reaches this value.")
    parser.add_argument("--target-imitation-score", type=float, default=0.0, help="Stop BC once validation imitation score reaches this value; <=0 disables.")
    parser.add_argument("--imitation-patience", type=int, default=0, help="Stop BC after this many epochs without a validation imitation-score improvement; <=0 disables.")
    parser.add_argument("--min-epochs", type=int, default=0, help="Minimum epochs before imitation-score early stopping can trigger.")
    parser.add_argument("--gate-min-launch-f1", type=float, default=0.0, help="If >0, require this validation launch F1 for same-state gate checkpointing.")
    parser.add_argument("--gate-min-target-acc", type=float, default=0.0, help="If >0, require this validation target accuracy for same-state gate checkpointing.")
    parser.add_argument("--gate-min-target-pair-acc", type=float, default=0.0, help="If >0, require this validation target-pair accuracy for same-state gate checkpointing.")
    parser.add_argument("--enable-online-stop", action="store_true", help="Allow online vs-regular eval metrics to stop BC. Disabled by default because BC phase is imitation-only.")
    parser.add_argument("--target-nonloss", type=float, default=0.50)
    parser.add_argument("--target-winrate", type=float, default=0.20)
    parser.add_argument("--eval-stochastic", action="store_true")
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--ship-bias", type=float, default=0.0)
    parser.add_argument("--launch-temperature", type=float, default=1.0)
    parser.add_argument("--eval-target-mask-mode", choices=["candidate", "safe", "all_planets"], default="candidate")
    parser.add_argument("--swanlab-project", default="orbit-wars")
    parser.add_argument("--swanlab-experiment", default="tinyppo-regular-bc-ray")
    parser.add_argument("--swanlab-mode", default="cloud")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--allow-no-swanlab", action="store_true")
    parser.add_argument("--seed", type=int, default=260525)
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()
    if (
        args.regular_anchor_source_weight > 0.0
        or args.regular_anchor_pair_weight > 0.0
        or args.regular_anchor_count_weight > 0.0
        or args.regular_anchor_target_weight > 0.0
    ) and not args.resume:
        raise ValueError("regular anchor requires --resume so the anchor teacher is well-defined")

    runtime_env = None
    if not args.ray_no_runtime_env:
        runtime_env = {
            "excludes": [
                "swanlog/**",
                "wandb/**",
                "tinyPPO/data/*.pkl",
                "tinyPPO/runs/**/*.pt",
                "tinyPPO/runs/**/*.pkl",
                "tinyPPO/runs/**/train.log",
            ]
        }
    ray_init_kwargs = {
        "address": args.ray_address,
        "ignore_reinit_error": True,
        "runtime_env": runtime_env,
    }
    if args.ray_temp_dir:
        ray_init_kwargs["_temp_dir"] = args.ray_temp_dir
    ray.init(**ray_init_kwargs)
    swan = _init_swanlab(args)
    rows, collect_metrics, sample_weights = collect_dataset(args)
    if not rows:
        raise RuntimeError("no rows collected")
    print(json.dumps({"collect": collect_metrics}, ensure_ascii=True), flush=True)
    _log_swanlab(swan, {"collect": collect_metrics}, 0)

    shards = [rows[i :: args.trainers] for i in range(args.trainers)]
    weight_shards = [sample_weights[i :: args.trainers] if sample_weights is not None else None for i in range(args.trainers)]
    resume_state: dict[str, torch.Tensor] | None = None
    resume_update = 0
    resume_model_cfg: dict[str, int] | None = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_state = {key: value.detach().cpu() for key, value in resume_payload["state_dict"].items()}
        resume_update = 0 if args.resume_compatible else int(resume_payload.get("update", 0))
        if isinstance(resume_payload.get("model"), dict) and not args.resume_compatible:
            resume_model_cfg = {
                key: bool(value) if key in {"source_target_summary", "target_pair_head", "target_pair_adapter", "target_pair_owner_head"} else int(value)
                for key, value in resume_payload["model"].items()
            }
        print(
            json.dumps(
                {"event": "resume_loaded", "path": args.resume, "resume_update": resume_update, "model": resume_model_cfg},
                ensure_ascii=True,
            ),
            flush=True,
        )

    model_cfg = resume_model_cfg or {
        "hidden": args.hidden,
        "heads": args.heads,
        "layers": args.layers,
        "ship_buckets": len(SHIP_BUCKET_MULTIPLIERS),
        "action_slots": ACTION_SLOTS,
        "source_target_summary": bool(args.source_target_summary),
        "target_pair_head": bool(args.target_pair_head),
        "target_pair_adapter": bool(args.target_pair_adapter),
        "target_pair_owner_head": bool(args.target_pair_owner_head),
    }
    init_model = TinyPolicyValueNet(**model_cfg)
    state = _cpu_state_dict(init_model)
    if resume_state:
        if args.resume_compatible:
            loaded_keys = []
            skipped_keys = []
            for key, value in resume_state.items():
                if key in state and tuple(state[key].shape) == tuple(value.shape):
                    state[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(key)
            adapter_copied = []
            if bool(model_cfg.get("target_pair_adapter", False)):
                for key, value in list(state.items()):
                    if not key.startswith("target_pair_edge."):
                        continue
                    if key in loaded_keys:
                        continue
                    edge_key = "edge." + key.removeprefix("target_pair_edge.")
                    if edge_key in state and tuple(state[edge_key].shape) == tuple(value.shape):
                        state[key] = state[edge_key].detach().clone()
                        adapter_copied.append(key)
            print(
                json.dumps(
                    {
                        "event": "resume_compatible_loaded",
                        "path": args.resume,
                        "loaded_tensors": len(loaded_keys),
                        "skipped_tensors": skipped_keys,
                        "adapter_copied_from_edge": adapter_copied,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
        else:
            state = resume_state
    trainer_gpu_ids = _int_list(args.trainer_gpu_ids)
    eval_gpu_ids = _int_list(args.eval_gpu_ids_manual)
    if trainer_gpu_ids or eval_gpu_ids:
        print(
            json.dumps(
                {"event": "manual_gpu_ids", "trainer_gpu_ids": trainer_gpu_ids, "eval_gpu_ids": eval_gpu_ids},
                ensure_ascii=True,
            ),
            flush=True,
        )

    actors = [
        BCTrainEvalActor.options(
            num_cpus=args.cpus_per_trainer,
            num_gpus=0.0 if trainer_gpu_ids else args.gpus_per_trainer,
        ).remote(
            shard,
            weight_shards[i],
            model_cfg,
            args.batch_size,
            args.val_frac,
            args.lr,
            args.weight_decay,
            args.max_grad_norm,
            args.launch_pos_weight,
            args.target_loss_weight,
            args.ship_loss_weight,
            args.target_ship_joint_loss_weight,
            args.target_ship_joint_dagger_only,
            args.critical_action_weight,
            args.target_loss_mask,
            args.target_margin_loss_weight,
            args.target_margin,
            args.target_margin_top_k,
            args.slot_set_loss,
            args.target_binary_loss_weight,
            args.target_binary_pos_weight,
            args.target_pair_softmax_loss_weight,
            args.target_pair_margin_loss_weight,
            args.target_pair_owner_loss_weight,
            args.target_pair_within_owner_loss_weight,
            args.launch_count_loss_weight,
            args.sample_weight_launch_scale,
            args.sample_weight_target_scale,
            args.sample_weight_ship_scale,
            args.sample_weight_pair_scale,
            args.sample_weight_count_scale,
            args.dagger_launch_negative_weight_scale,
            args.dagger_launch_positive_weight_scale,
            state
            if (
                args.regular_anchor_source_weight > 0.0
                or args.regular_anchor_pair_weight > 0.0
                or args.regular_anchor_count_weight > 0.0
                or args.regular_anchor_target_weight > 0.0
            )
            else None,
            args.regular_anchor_source_weight,
            args.regular_anchor_pair_weight,
            args.regular_anchor_count_weight,
            args.regular_anchor_target_weight,
            args.regular_anchor_min_sample_weight,
            args.trainable_modules,
            args.seed + i,
            f"cuda:{trainer_gpu_ids[i % len(trainer_gpu_ids)]}" if trainer_gpu_ids else "",
        )
        for i, shard in enumerate(shards)
        if shard
    ]
    eval_actors = [
        BCEvalActor.options(
            num_cpus=args.eval_cpus_per_actor,
            num_gpus=0.0 if eval_gpu_ids else args.gpus_per_eval_actor,
        ).remote(model_cfg, f"cuda:{eval_gpu_ids[i % len(eval_gpu_ids)]}" if eval_gpu_ids else "")
        for i in range(args.eval_actors)
    ]
    best_nonloss = -1.0
    best_imitation_score = -1e9
    best_imitation_epoch = 0
    best_gate_score = (-1e9, -1e9, -1e9)
    best_gate_epoch = 0
    best_metrics: dict[str, Any] = {}
    pending_eval_refs: dict[Any, int] = {}
    pending_eval_states: dict[int, dict[str, torch.Tensor]] = {}
    eval_parts_by_epoch: dict[int, list[dict[str, float]]] = {}
    eval_expected_parts: dict[int, int] = {}
    eval_completion_count = 0
    stop_after_epoch: int | None = None
    eval_actor_cursor = 0

    def maybe_finish_eval(eval_epoch: int) -> None:
        nonlocal best_nonloss, best_metrics, eval_completion_count, stop_after_epoch
        expected = eval_expected_parts.get(eval_epoch, 1)
        if len(eval_parts_by_epoch.get(eval_epoch, [])) < expected:
            return
        eval_metrics = _merge_eval(eval_parts_by_epoch.pop(eval_epoch))
        eval_expected_parts.pop(eval_epoch, None)
        eval_state = pending_eval_states.pop(eval_epoch, None)
        eval_completion_count += 1
        summary = {
            "event": "async_eval_done",
            "epoch": eval_epoch,
            "eval_completion": eval_completion_count,
            "eval_vs_regular": eval_metrics,
        }
        print(json.dumps(summary, ensure_ascii=True), flush=True)
        _log_swanlab(swan, summary, eval_epoch)
        if eval_state is not None:
            save_checkpoint(Path(args.out), eval_state, model_cfg, eval_epoch, {"collect": collect_metrics, **summary})
            if eval_metrics["nonloss"] > best_nonloss:
                best_nonloss = eval_metrics["nonloss"]
                best_metrics = dict(summary)
                online_path = Path(args.best_online_out) if args.best_online_out else Path(args.out).with_name("regular_bc_ray_online_best.pt")
                save_checkpoint(online_path, eval_state, model_cfg, eval_epoch, {"collect": collect_metrics, **summary, "best_online": True})
                save_checkpoint(
                    Path(args.out).with_name(f"regular_bc_ray_best_e{eval_epoch:04d}.pt"),
                    eval_state,
                    model_cfg,
                    eval_epoch,
                    {"collect": collect_metrics, **summary, "best": True},
                )
        if args.enable_online_stop and eval_metrics["nonloss"] >= args.target_nonloss and eval_metrics["winrate"] >= args.target_winrate:
            stop_after_epoch = eval_epoch
            print(json.dumps({"event": "online_target_reached", "epoch": eval_epoch, "metrics": eval_metrics}, ensure_ascii=True), flush=True)

    def drain_ready_evals() -> None:
        if not pending_eval_refs:
            return
        ready, _not_ready = ray.wait(list(pending_eval_refs), num_returns=len(pending_eval_refs), timeout=0.0)
        for ref in ready:
            eval_epoch = pending_eval_refs.pop(ref)
            part = ray.get(ref)
            eval_parts_by_epoch.setdefault(eval_epoch, []).append(part)
            maybe_finish_eval(eval_epoch)

    start_epoch = resume_update + 1 if resume_update > 0 else 1
    for epoch in range(start_epoch, args.epochs + 1):
        drain_ready_evals()
        if stop_after_epoch is not None:
            break
        state_refs = [actor.train_epoch.remote(state, epoch) for actor in actors]
        results = ray.get(state_refs)
        state = _average_states([item[0] for item in results])
        train_metrics = _weighted_mean([item[1] for item in results])
        imitation_score = _imitation_score(train_metrics)
        train_metrics["val_imitation_score"] = imitation_score
        gate_enabled = bool(args.best_gate_out) or any(
            threshold > 0.0
            for threshold in (args.gate_min_launch_f1, args.gate_min_target_acc, args.gate_min_target_pair_acc)
        )
        gate_passed = (
            float(train_metrics.get("val_launch_f1", 0.0)) >= float(args.gate_min_launch_f1)
            and float(train_metrics.get("val_target_acc", 0.0)) >= float(args.gate_min_target_acc)
            and float(train_metrics.get("val_target_pair_acc", 0.0)) >= float(args.gate_min_target_pair_acc)
        )
        gate_score = (
            float(train_metrics.get("val_target_pair_acc", 0.0)),
            float(train_metrics.get("val_target_acc", 0.0)),
            float(train_metrics.get("val_launch_f1", 0.0)),
        )
        if imitation_score > best_imitation_score:
            best_imitation_score = imitation_score
            best_imitation_epoch = epoch
            save_checkpoint(
                Path(args.best_out),
                state,
                model_cfg,
                epoch,
                {"collect": collect_metrics, "epoch": epoch, "train": train_metrics, "best_imitation": True},
            )
            if args.eval_interval > 0 and epoch % args.eval_interval == 0:
                save_checkpoint(
                    Path(args.out).with_name(f"regular_bc_ray_best_imitation_e{epoch:04d}.pt"),
                    state,
                    model_cfg,
                    epoch,
                    {"collect": collect_metrics, "epoch": epoch, "train": train_metrics, "best_imitation": True},
                )
        if gate_enabled and gate_passed and gate_score > best_gate_score:
            best_gate_score = gate_score
            best_gate_epoch = epoch
            gate_path = Path(args.best_gate_out) if args.best_gate_out else Path(args.out).with_name("regular_bc_gate_best.pt")
            save_checkpoint(
                gate_path,
                state,
                model_cfg,
                epoch,
                {
                    "collect": collect_metrics,
                    "epoch": epoch,
                    "train": train_metrics,
                    "best_gate": True,
                    "gate_score": {
                        "val_target_pair_acc": gate_score[0],
                        "val_target_acc": gate_score[1],
                        "val_launch_f1": gate_score[2],
                    },
                    "gate_thresholds": {
                        "val_launch_f1": float(args.gate_min_launch_f1),
                        "val_target_acc": float(args.gate_min_target_acc),
                        "val_target_pair_acc": float(args.gate_min_target_pair_acc),
                    },
                },
            )
        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            save_checkpoint(
                Path(args.out).with_name(f"regular_bc_ray_e{epoch:04d}.pt"),
                state,
                model_cfg,
                epoch,
                {"collect": collect_metrics, "epoch": epoch, "train": train_metrics},
            )
            save_checkpoint(Path(args.out), state, model_cfg, epoch, {"collect": collect_metrics, "epoch": epoch, "train": train_metrics})
        summary: dict[str, Any] = {"epoch": epoch, "train": train_metrics}
        if gate_enabled and gate_passed and best_gate_epoch == epoch:
            summary["best_gate"] = {
                "epoch": float(epoch),
                "score": {
                    "val_target_pair_acc": gate_score[0],
                    "val_target_acc": gate_score[1],
                    "val_launch_f1": gate_score[2],
                },
            }
        if args.target_imitation_score > 0.0 and epoch >= args.min_epochs and imitation_score >= args.target_imitation_score:
            stop_after_epoch = epoch
            summary["imitation_stop"] = {
                "reason": "target_imitation_score",
                "score": float(imitation_score),
                "target": float(args.target_imitation_score),
                "best_epoch": float(best_imitation_epoch),
            }
        elif args.imitation_patience > 0 and epoch >= args.min_epochs and epoch - best_imitation_epoch >= args.imitation_patience:
            stop_after_epoch = epoch
            summary["imitation_stop"] = {
                "reason": "patience",
                "score": float(imitation_score),
                "best_score": float(best_imitation_score),
                "best_epoch": float(best_imitation_epoch),
                "patience": float(args.imitation_patience),
            }

        should_eval = (
            args.eval_interval > 0
            and epoch % args.eval_interval == 0
            and float(train_metrics.get("val_launch_recall", 0.0)) >= args.eval_min_launch_recall
            and len(eval_expected_parts) < args.max_pending_evals
        )
        if should_eval:
            actors_for_eval = len(eval_actors) if args.eval_games_per_actor > 0 else min(len(eval_actors), max(1, args.eval_games))
            selected_eval_actors = [eval_actors[(eval_actor_cursor + i) % len(eval_actors)] for i in range(actors_for_eval)]
            eval_actor_cursor = (eval_actor_cursor + actors_for_eval) % len(eval_actors)
            if args.eval_games_per_actor > 0:
                games_parts = [args.eval_games_per_actor] * actors_for_eval
            else:
                games_parts = [args.eval_games // actors_for_eval] * actors_for_eval
                for i in range(args.eval_games % actors_for_eval):
                    games_parts[i] += 1
            total_eval_games = sum(games_parts)
            eval_state = {key: value.detach().cpu() for key, value in state.items()}
            eval_refs = []
            for i, (actor, games) in enumerate(zip(selected_eval_actors, games_parts, strict=True)):
                if games <= 0:
                    continue
                eval_refs.append(
                    actor.eval_vs_regular.remote(
                        eval_state,
                        games,
                        args.seed + 10_000 + epoch * 1_000 + i * 100,
                        args.eval_stochastic,
                        args.launch_bias,
                        args.ship_bias,
                        args.launch_temperature,
                        args.eval_target_mask_mode,
                    )
                )
            pending_eval_states[epoch] = eval_state
            for ref in eval_refs:
                pending_eval_refs[ref] = epoch
            eval_expected_parts[epoch] = len(eval_refs)
            summary["async_eval_submitted"] = {
                "parts": float(len(eval_refs)),
                "games": float(total_eval_games),
                "games_per_actor": float(args.eval_games_per_actor) if args.eval_games_per_actor > 0 else float(total_eval_games / max(1, len(eval_refs))),
                "pending_evals": float(len(eval_expected_parts)),
            }
            print(json.dumps(summary, ensure_ascii=True), flush=True)
            _log_swanlab(swan, summary, epoch)
        else:
            if args.eval_interval > 0 and epoch % args.eval_interval == 0:
                if float(train_metrics.get("val_launch_recall", 0.0)) < args.eval_min_launch_recall:
                    summary["eval_skipped"] = {
                        "reason": "val_launch_recall_below_threshold",
                        "val_launch_recall": float(train_metrics.get("val_launch_recall", 0.0)),
                        "threshold": args.eval_min_launch_recall,
                    }
                else:
                    summary["eval_skipped"] = {
                        "reason": "max_pending_evals",
                        "pending_evals": float(len(eval_expected_parts)),
                        "max_pending_evals": float(args.max_pending_evals),
                    }
            print(json.dumps(summary, ensure_ascii=True), flush=True)
            _log_swanlab(swan, summary, epoch)
        drain_ready_evals()

    if pending_eval_refs:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass

    while pending_eval_refs and stop_after_epoch is None:
        done, _not_ready = ray.wait(list(pending_eval_refs), num_returns=1, timeout=30.0)
        if not done:
            print(
                json.dumps(
                    {"event": "async_eval_wait", "pending_parts": len(pending_eval_refs), "pending_evals": len(eval_expected_parts)},
                    ensure_ascii=True,
                ),
                flush=True,
            )
            continue
        for ref in done:
            eval_epoch = pending_eval_refs.pop(ref)
            part = ray.get(ref)
            eval_parts_by_epoch.setdefault(eval_epoch, []).append(part)
            maybe_finish_eval(eval_epoch)
    print(
        json.dumps(
            {"event": "done", "best_nonloss": best_nonloss, "best_imitation_score": best_imitation_score, "best": best_metrics},
            ensure_ascii=True,
        ),
        flush=True,
    )
    if swan is not None:
        swan.finish()


if __name__ == "__main__":
    main()
