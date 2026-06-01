from __future__ import annotations

import argparse
import json
import math
import random
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from training.expert.action_labeling import infer_target_planet_id
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent

from tinyPPO.agents import (
    ACTION_SLOTS,
    SHIP_BUCKET_MULTIPLIERS,
    all_planets_target_mask,
    candidate_target_mask,
    required_ships,
    safe_target_mask,
)
from tinyPPO.features import MAX_PLANETS, encode_obs
from tinyPPO.model import TinyPolicyValueNet


@dataclass
class BCRow:
    planets: np.ndarray
    pair_features: np.ndarray
    global_features: np.ndarray
    planet_mask: np.ndarray
    own_mask: np.ndarray
    target_safety_mask: np.ndarray
    launch_actions: np.ndarray
    target_actions: np.ndarray
    ship_actions: np.ndarray
    launch_mask: np.ndarray


class LoggingRuleAgent:
    def __init__(self, player: int, rows: list[tuple[dict[str, Any], int, list[list]]], keep_noop_prob: float):
        self.player = player
        self.agent = make_rulebase_agent("regular")
        self.rows = rows
        self.keep_noop_prob = keep_noop_prob

    def __call__(self, obs: dict[str, Any], configuration=None) -> list[list]:
        del configuration
        action = self.agent(obs) or []
        if action or random.random() < self.keep_noop_prob:
            self.rows.append((obs, self.player, action))
        return action


def _bucket_for_action(obs: dict[str, Any], player: int, source: list, target: list, ships: int) -> int:
    needed = max(1, required_ships(obs, player, source, target))
    ratio = float(max(1, ships)) / float(needed)
    return int(min(range(len(SHIP_BUCKET_MULTIPLIERS)), key=lambda i: abs(SHIP_BUCKET_MULTIPLIERS[i] - ratio)))


def _row_target_mask(obs: dict[str, Any], player: int, mode: str) -> np.ndarray:
    if mode == "candidate":
        return candidate_target_mask(obs, player)
    if mode == "safe":
        return safe_target_mask(obs, player)
    if mode == "all_planets":
        return all_planets_target_mask(obs, player)
    raise ValueError(f"unsupported row target mask mode: {mode!r}")


def row_from_regular_action(
    obs: dict[str, Any],
    player: int,
    action: list[list],
    players: int,
    target_mask_mode: str = "candidate",
) -> BCRow | None:
    enc = encode_obs(obs, player, players=players)
    planets = list(obs.get("planets", []))[:MAX_PLANETS]
    id_to_idx = {int(p[0]): i for i, p in enumerate(planets)}

    launch_actions = np.zeros((MAX_PLANETS, ACTION_SLOTS), dtype=np.int64)
    target_actions = np.zeros((MAX_PLANETS, ACTION_SLOTS), dtype=np.int64)
    ship_actions = np.zeros((MAX_PLANETS, ACTION_SLOTS), dtype=np.int64)
    launch_mask = np.zeros((MAX_PLANETS, ACTION_SLOTS), dtype=np.bool_)
    used_slots = np.zeros(MAX_PLANETS, dtype=np.int64)
    target_mask = _row_target_mask(obs, player, target_mask_mode)

    labelled = 0
    skipped = 0
    for move in action:
        if not isinstance(move, list) or len(move) < 3:
            skipped += 1
            continue
        src_idx = id_to_idx.get(int(move[0]))
        if src_idx is None or src_idx >= len(planets):
            skipped += 1
            continue
        source = planets[src_idx]
        if int(source[1]) != player:
            skipped += 1
            continue
        slot = int(used_slots[src_idx])
        if slot >= ACTION_SLOTS:
            skipped += 1
            continue
        ships = max(1, int(float(move[2])))
        target_id = infer_target_planet_id(obs, int(move[0]), float(move[1]), ships)
        tgt_idx = id_to_idx.get(int(target_id)) if target_id is not None else None
        if tgt_idx is None or tgt_idx == src_idx or tgt_idx >= len(planets):
            skipped += 1
            continue
        target = planets[tgt_idx]
        launch_actions[src_idx, slot] = 1
        target_actions[src_idx, slot] = int(tgt_idx)
        ship_actions[src_idx, slot] = _bucket_for_action(obs, player, source, target, ships)
        launch_mask[src_idx, slot] = True
        target_mask[src_idx, tgt_idx] = True
        used_slots[src_idx] += 1
        labelled += 1

    if not action and not np.any(enc.own_mask):
        return None
    row = BCRow(
        planets=enc.planets.astype(np.float32, copy=False),
        pair_features=enc.pair_features.astype(np.float32, copy=False),
        global_features=enc.global_features.astype(np.float32, copy=False),
        planet_mask=enc.planet_mask.astype(np.bool_, copy=False),
        own_mask=enc.own_mask.astype(np.bool_, copy=False),
        target_safety_mask=target_mask.astype(np.bool_, copy=False),
        launch_actions=launch_actions,
        target_actions=target_actions,
        ship_actions=ship_actions,
        launch_mask=launch_mask,
    )
    row.labelled = labelled  # type: ignore[attr-defined]
    row.skipped = skipped  # type: ignore[attr-defined]
    return row


def _collect_one_game(
    seed: int,
    players: int,
    episode_steps: int,
    keep_noop_prob: float,
    sample_stride: int,
    rows_per_game: int,
    use_numba: bool,
    target_mask_mode: str = "candidate",
) -> tuple[list[BCRow], dict[str, float]]:
    random.seed(seed)
    np.random.seed(seed)
    raw_rows: list[tuple[dict[str, Any], int, list[list]]] = []
    env = make_fast_orbit_wars(
        {"episodeSteps": episode_steps, "seed": seed},
        keep_history=False,
        use_numba=use_numba,
    )
    agents = [LoggingRuleAgent(pid, raw_rows, keep_noop_prob) for pid in range(players)]
    env.run(agents)
    sampled = raw_rows[:: max(1, sample_stride)]
    if rows_per_game > 0 and len(sampled) > rows_per_game:
        sampled = random.sample(sampled, rows_per_game)

    rows: list[BCRow] = []
    labelled_actions = 0
    skipped_actions = 0
    for obs, player, action in sampled:
        row = row_from_regular_action(obs, player, action, players=players, target_mask_mode=target_mask_mode)
        if row is None:
            continue
        rows.append(row)
        labelled_actions += int(getattr(row, "labelled", 0))
        skipped_actions += int(getattr(row, "skipped", 0))
    return rows, {
        "games": 1.0,
        "samples": float(len(rows)),
        "labelled_actions": float(labelled_actions),
        "skipped_actions": float(skipped_actions),
        "row_target_mask_mode": target_mask_mode,
    }


def _players_list(raw: str) -> list[int]:
    players = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not players or any(p < 2 for p in players):
        raise ValueError(f"invalid players list: {raw!r}")
    return players


def collect_rows(args: argparse.Namespace) -> tuple[list[BCRow], dict[str, float]]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    rows: list[BCRow] = []
    labelled_actions = 0
    skipped_actions = 0

    players_values = _players_list(args.players_list)
    games_per_players = args.games_per_players if args.games_per_players > 0 else args.games
    jobs = [
        (args.seed + players * 1_000_000 + game, players)
        for players in players_values
        for game in range(games_per_players)
    ]

    def add_result(game_rows: list[BCRow], metrics: dict[str, float]) -> bool:
        nonlocal labelled_actions, skipped_actions
        rows.extend(game_rows)
        labelled_actions += int(metrics.get("labelled_actions", 0.0))
        skipped_actions += int(metrics.get("skipped_actions", 0.0))
        return bool(args.max_samples > 0 and len(rows) >= args.max_samples)

    iterator = jobs
    if args.progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="collect regular BC games", dynamic_ncols=True)

    if args.collect_workers <= 1:
        completed_games = 0
        for seed, players in iterator:
            game_rows, metrics = _collect_one_game(
                seed,
                players,
                args.episode_steps,
                args.keep_noop_prob,
                args.sample_stride,
                args.rows_per_game,
                not args.no_numba,
                args.row_target_mask_mode,
            )
            completed_games += 1
            if add_result(game_rows, metrics):
                break
        return rows[: args.max_samples if args.max_samples > 0 else None], {
            "games": float(completed_games),
            "samples": float(min(len(rows), args.max_samples) if args.max_samples > 0 else len(rows)),
            "labelled_actions": float(labelled_actions),
            "skipped_actions": float(skipped_actions),
            "players_modes": float(len(players_values)),
            "row_target_mask_mode": args.row_target_mask_mode,
        }

    completed_games = 0
    with ProcessPoolExecutor(max_workers=args.collect_workers) as pool:
        futures = [
            pool.submit(
                _collect_one_game,
                seed,
                players,
                args.episode_steps,
                args.keep_noop_prob,
                args.sample_stride,
                args.rows_per_game,
                not args.no_numba,
                args.row_target_mask_mode,
            )
            for seed, players in jobs
        ]
        future_iter = as_completed(futures)
        if args.progress:
            from tqdm.auto import tqdm

            future_iter = tqdm(future_iter, total=len(futures), desc="collect regular BC games", dynamic_ncols=True)
        for future in future_iter:
            game_rows, metrics = future.result()
            completed_games += 1
            if add_result(game_rows, metrics):
                for pending in futures:
                    pending.cancel()
                break
    return rows[: args.max_samples if args.max_samples > 0 else None], {
        "games": float(completed_games),
        "samples": float(min(len(rows), args.max_samples) if args.max_samples > 0 else len(rows)),
        "labelled_actions": float(labelled_actions),
        "skipped_actions": float(skipped_actions),
        "players_modes": float(len(players_values)),
        "row_target_mask_mode": args.row_target_mask_mode,
    }


def stack_rows(rows: list[BCRow], sample_weights: list[float] | None = None) -> TensorDataset:
    arrays = {
        "planets": np.stack([r.planets for r in rows]),
        "pair_features": np.stack([r.pair_features for r in rows]),
        "global_features": np.stack([r.global_features for r in rows]),
        "planet_mask": np.stack([r.planet_mask for r in rows]),
        "own_mask": np.stack([r.own_mask for r in rows]),
        "target_safety_mask": np.stack([r.target_safety_mask for r in rows]),
        "launch_actions": np.stack([r.launch_actions for r in rows]),
        "target_actions": np.stack([r.target_actions for r in rows]),
        "ship_actions": np.stack([r.ship_actions for r in rows]),
        "launch_mask": np.stack([r.launch_mask for r in rows]),
    }
    tensors = []
    for key in (
        "planets",
        "pair_features",
        "global_features",
        "planet_mask",
        "own_mask",
        "target_safety_mask",
        "launch_actions",
        "target_actions",
        "ship_actions",
        "launch_mask",
    ):
        arr = arrays[key]
        dtype = torch.float32 if arr.dtype.kind == "f" else torch.bool if arr.dtype == np.bool_ else torch.long
        tensors.append(torch.tensor(arr, dtype=dtype))
    if sample_weights is not None:
        if len(sample_weights) != len(rows):
            raise ValueError(f"sample_weights length {len(sample_weights)} does not match rows length {len(rows)}")
        tensors.append(torch.tensor(sample_weights, dtype=torch.float32))
    return TensorDataset(*tensors)


def unpack(batch: tuple[torch.Tensor, ...], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "planets",
        "pair_features",
        "global_features",
        "planet_mask",
        "own_mask",
        "target_safety_mask",
        "launch_actions",
        "target_actions",
        "ship_actions",
        "launch_mask",
    )
    if len(batch) == len(keys):
        return {k: v.to(device, non_blocking=True) for k, v in zip(keys, batch, strict=True)}
    if len(batch) == len(keys) + 1:
        out = {k: v.to(device, non_blocking=True) for k, v in zip(keys, batch[: len(keys)], strict=True)}
        out["sample_weight"] = batch[-1].to(device, non_blocking=True).float()
        return out
    raise ValueError(f"unexpected BC batch width {len(batch)}")


def _weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, targets, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def _slot_set_bc_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    launch_pos_weight: float,
    target_loss_weight: float,
    ship_loss_weight: float,
    target_loss_mask: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Permutation-invariant loss for the unordered action slots of each source."""
    source_logits = out["source_logits"]
    bsz, sources, slots, _classes = source_logits.shape
    own_slots = batch["own_mask"][:, :, None].expand_as(batch["launch_actions"])
    if target_loss_mask == "all_planets":
        target_mask = batch["planet_mask"][:, None, :].expand(-1, batch["target_safety_mask"].shape[1], -1).clone()
        eye = torch.eye(target_mask.shape[1], dtype=torch.bool, device=target_mask.device)[None, :, :]
        target_mask = target_mask & ~eye
    elif target_loss_mask == "dataset":
        target_mask = batch["target_safety_mask"]
    else:
        raise ValueError(f"unsupported target_loss_mask: {target_loss_mask!r}")
    target_logits = out["target_logits"].masked_fill(~target_mask[:, :, None, :], -1e9)
    perms = torch.tensor(list(itertools.permutations(range(slots))), dtype=torch.long, device=source_logits.device)
    per_perm_total: list[torch.Tensor] = []
    per_perm_launch: list[torch.Tensor] = []
    per_perm_target: list[torch.Tensor] = []
    per_perm_ship: list[torch.Tensor] = []
    for perm in perms:
        launch_labels = batch["launch_actions"][:, :, perm]
        active = (launch_labels == 1) & own_slots
        launch_part = F.cross_entropy(
            source_logits.reshape(-1, 2),
            launch_labels.reshape(-1),
            reduction="none",
            weight=torch.tensor([1.0, launch_pos_weight], dtype=torch.float32, device=source_logits.device),
        ).reshape(bsz, sources, slots)
        target_labels = batch["target_actions"][:, :, perm]
        target_part = F.cross_entropy(
            target_logits.reshape(-1, target_logits.shape[-1]),
            target_labels.reshape(-1),
            reduction="none",
        ).reshape(bsz, sources, slots).masked_fill(~active, 0.0)
        ship_part = torch.zeros_like(target_part)
        if "ship_logits" in out:
            ship_logits_all = out["ship_logits"]
            b_idx = torch.arange(bsz, device=source_logits.device)[:, None, None].expand(bsz, sources, slots)
            s_idx = torch.arange(sources, device=source_logits.device)[None, :, None].expand(bsz, sources, slots)
            slot_idx = torch.arange(slots, device=source_logits.device)[None, None, :].expand(bsz, sources, slots)
            target_idx = target_labels.clamp(0, target_logits.shape[-1] - 1)
            ship_logits = ship_logits_all[b_idx, s_idx, slot_idx, target_idx]
            ship_labels = batch["ship_actions"][:, :, perm]
            ship_part = F.cross_entropy(
                ship_logits.reshape(-1, ship_logits.shape[-1]),
                ship_labels.reshape(-1),
                reduction="none",
            ).reshape(bsz, sources, slots).masked_fill(~active, 0.0)
        launch_sum = launch_part.masked_fill(~own_slots, 0.0).sum(dim=-1)
        target_sum = target_part.sum(dim=-1)
        ship_sum = ship_part.sum(dim=-1)
        per_perm_launch.append(launch_sum)
        per_perm_target.append(target_sum)
        per_perm_ship.append(ship_sum)
        per_perm_total.append(launch_sum + target_loss_weight * target_sum + ship_loss_weight * ship_sum)

    totals = torch.stack(per_perm_total, dim=0)
    best_idx = totals.argmin(dim=0)
    selector = F.one_hot(best_idx, num_classes=perms.shape[0]).permute(2, 0, 1).float()
    own_sources = batch["own_mask"]
    launch_sum = (torch.stack(per_perm_launch, dim=0) * selector).sum(dim=0).masked_select(own_sources).sum()
    target_sum = (torch.stack(per_perm_target, dim=0) * selector).sum(dim=0).masked_select(own_sources).sum()
    ship_sum = (torch.stack(per_perm_ship, dim=0) * selector).sum(dim=0).masked_select(own_sources).sum()
    own_slot_count = own_slots.float().sum().clamp_min(1.0)
    active_count = (batch["launch_actions"].bool() & own_slots).float().sum().clamp_min(1.0)
    launch_loss = launch_sum / own_slot_count
    target_loss = target_sum / active_count
    ship_loss = ship_sum / active_count
    loss = launch_loss + target_loss_weight * target_loss + ship_loss_weight * ship_loss
    return loss, {
        "launch_loss": launch_loss,
        "target_loss": target_loss,
        "ship_loss": ship_loss,
        "target_margin_loss": torch.tensor(0.0, device=source_logits.device),
        "action_weight_mean": torch.tensor(1.0, device=source_logits.device),
    }


def _valid_pair_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    pair_valid = batch["own_mask"][:, :, None] & batch["planet_mask"][:, None, :]
    eye = torch.eye(pair_valid.shape[1], dtype=torch.bool, device=pair_valid.device)[None, :, :]
    return pair_valid & ~eye


def _target_owner_labels(batch: dict[str, torch.Tensor], target_indices: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
    owner_feats = batch["planets"][b_idx, target_indices, :3]
    return owner_feats.argmax(dim=-1)


def _owner_group_logits(pair_logits: torch.Tensor, batch: dict[str, torch.Tensor], pair_valid: torch.Tensor) -> torch.Tensor:
    target_owner = batch["planets"][:, :, :3].argmax(dim=-1)
    groups: list[torch.Tensor] = []
    for owner_idx in range(3):
        owner_mask = target_owner[:, None, :].eq(owner_idx) & pair_valid
        groups.append(pair_logits.masked_fill(~owner_mask, -1e9).logsumexp(dim=-1))
    return torch.stack(groups, dim=-1)


def bc_loss(
    model: TinyPolicyValueNet,
    batch: dict[str, torch.Tensor],
    launch_pos_weight: float,
    target_loss_weight: float = 1.0,
    ship_loss_weight: float = 0.5,
    target_ship_joint_loss_weight: float = 0.0,
    target_ship_joint_dagger_only: bool = False,
    critical_action_weight: float = 0.0,
    target_loss_mask: str = "dataset",
    target_margin_loss_weight: float = 0.0,
    target_margin: float = 0.5,
    target_margin_top_k: int = 8,
    slot_set_loss: bool = False,
    target_binary_loss_weight: float = 0.0,
    target_binary_pos_weight: float = 12.0,
    target_pair_softmax_loss_weight: float = 0.0,
    target_pair_margin_loss_weight: float = 0.0,
    target_pair_owner_loss_weight: float = 0.0,
    target_pair_within_owner_loss_weight: float = 0.0,
    launch_count_loss_weight: float = 0.0,
    sample_weight_launch_scale: float = 1.0,
    sample_weight_target_scale: float = 1.0,
    sample_weight_ship_scale: float = 1.0,
    sample_weight_pair_scale: float = 1.0,
    sample_weight_count_scale: float = 1.0,
    dagger_launch_negative_weight_scale: float = 1.0,
    dagger_launch_positive_weight_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(batch["planets"], batch["pair_features"], batch["global_features"], batch["planet_mask"], batch["own_mask"])
    own_slots = batch["own_mask"][:, :, None].expand_as(batch["launch_actions"])
    row_weights = batch.get("sample_weight")
    if row_weights is None:
        row_weights = torch.ones(batch["launch_actions"].shape[0], dtype=torch.float32, device=batch["launch_actions"].device)
    row_weights = row_weights.float().clamp_min(0.0)
    def _component_row_weights(scale: float) -> torch.Tensor:
        return (1.0 + (row_weights - 1.0) * float(scale)).clamp_min(0.0)

    launch_row_weights = _component_row_weights(sample_weight_launch_scale)
    target_row_weights = _component_row_weights(sample_weight_target_scale)
    ship_row_weights = _component_row_weights(sample_weight_ship_scale)
    pair_row_weights = _component_row_weights(sample_weight_pair_scale)
    count_row_weights = _component_row_weights(sample_weight_count_scale)
    row_slot_weights = launch_row_weights[:, None, None].expand_as(batch["launch_actions"])
    if slot_set_loss and out["source_logits"].shape[2] > 1:
        set_loss, set_parts = _slot_set_bc_loss(out, batch, launch_pos_weight, target_loss_weight, ship_loss_weight, target_loss_mask)
    else:
        set_loss = None
        set_parts = {}
    launch_logits = out["source_logits"][own_slots]
    launch_targets = batch["launch_actions"][own_slots]
    launch_prob = F.softmax(out["source_logits"], dim=-1)[..., 1].masked_fill(~own_slots, 0.0)
    true_count_soft_target = batch["launch_actions"].masked_fill(~own_slots, 0).sum(dim=(1, 2)).float()
    pred_count_soft = launch_prob.sum(dim=(1, 2))
    launch_count_parts = F.smooth_l1_loss(pred_count_soft, true_count_soft_target, reduction="none")
    launch_count_loss = (launch_count_parts * count_row_weights).sum() / count_row_weights.sum().clamp_min(1.0)
    launch_parts = F.cross_entropy(
        out["source_logits"].reshape(-1, 2),
        batch["launch_actions"].reshape(-1),
        reduction="none",
        weight=torch.tensor([1.0, launch_pos_weight], dtype=torch.float32, device=launch_logits.device),
    ).reshape_as(batch["launch_actions"])
    launch_weights = row_slot_weights.masked_fill(~own_slots, 0.0)
    if dagger_launch_negative_weight_scale != 1.0:
        dagger_like_rows = (row_weights < 0.999).to(dtype=torch.bool)[:, None, None]
        dagger_negative_slots = dagger_like_rows & (batch["launch_actions"] == 0) & own_slots
        launch_weights = torch.where(
            dagger_negative_slots,
            launch_weights * float(dagger_launch_negative_weight_scale),
            launch_weights,
        )
    if dagger_launch_positive_weight_scale != 1.0:
        dagger_like_rows = (row_weights < 0.999).to(dtype=torch.bool)[:, None, None]
        dagger_positive_slots = dagger_like_rows & (batch["launch_actions"] == 1) & own_slots
        launch_weights = torch.where(
            dagger_positive_slots,
            launch_weights * float(dagger_launch_positive_weight_scale),
            launch_weights,
        )
    launch_loss = set_parts.get(
        "launch_loss",
        (launch_parts * launch_weights).sum() / launch_weights.sum().clamp_min(1.0),
    )

    active = batch["launch_mask"] & own_slots
    target_loss = torch.tensor(0.0, device=launch_logits.device)
    ship_loss = torch.tensor(0.0, device=launch_logits.device)
    target_ship_joint_loss = torch.tensor(0.0, device=launch_logits.device)
    target_acc = torch.tensor(0.0, device=launch_logits.device)
    target_margin_loss = torch.tensor(0.0, device=launch_logits.device)
    ship_acc = torch.tensor(0.0, device=launch_logits.device)
    target_ship_joint_acc = torch.tensor(0.0, device=launch_logits.device)
    action_weight_mean = torch.tensor(0.0, device=launch_logits.device)
    target_binary_loss = torch.tensor(0.0, device=launch_logits.device)
    target_pair_softmax_loss = torch.tensor(0.0, device=launch_logits.device)
    target_pair_margin_loss = torch.tensor(0.0, device=launch_logits.device)
    target_pair_owner_loss = torch.tensor(0.0, device=launch_logits.device)
    target_pair_within_owner_loss = torch.tensor(0.0, device=launch_logits.device)
    target_pair_owner_acc = torch.tensor(0.0, device=launch_logits.device)
    target_pair_within_owner_acc = torch.tensor(0.0, device=launch_logits.device)
    target_pair_acc = torch.tensor(0.0, device=launch_logits.device)
    if active.any():
        if target_loss_mask == "all_planets":
            target_mask = batch["planet_mask"][:, None, :].expand(-1, batch["target_safety_mask"].shape[1], -1).clone()
            eye = torch.eye(target_mask.shape[1], dtype=torch.bool, device=target_mask.device)[None, :, :]
            target_mask = target_mask & ~eye
        elif target_loss_mask == "dataset":
            target_mask = batch["target_safety_mask"]
        else:
            raise ValueError(f"unsupported target_loss_mask: {target_loss_mask!r}")
        target_logits = out["target_logits"].masked_fill(~target_mask[:, :, None, :], -1e9)
        target_active_weights = target_row_weights[:, None, None].expand_as(batch["target_actions"])[active].float()
        ship_active_weights = ship_row_weights[:, None, None].expand_as(batch["target_actions"])[active].float()
        pair_active_weights = pair_row_weights[:, None, None].expand_as(batch["target_actions"])[active].float()
        if critical_action_weight > 0.0:
            b, s, slot = torch.where(active)
            del slot
            t = batch["target_actions"][active]
            source_ship = batch["planets"][b, s, 6].clamp(0.0, 1.5)
            source_prod = batch["planets"][b, s, 7].clamp(0.0, 2.0)
            target_prod = batch["planets"][b, t, 7].clamp(0.0, 2.0)
            target_enemy = batch["pair_features"][b, s, t, 10].clamp(0.0, 1.0)
            target_neutral = batch["pair_features"][b, s, t, 9].clamp(0.0, 1.0)
            incoming_enemy = batch["planets"][b, t, 14].clamp(0.0, 2.0)
            importance = 0.35 * source_ship + 0.25 * source_prod + 0.35 * target_prod + 0.35 * target_enemy + 0.15 * target_neutral + 0.30 * incoming_enemy
            target_active_weights = target_active_weights + critical_action_weight * importance.clamp(0.0, 2.0)
        action_weight_mean = target_active_weights.mean()
        target_loss = _weighted_cross_entropy(target_logits[active], batch["target_actions"][active], target_active_weights)
        target_pred = target_logits[active].argmax(dim=-1)
        target_acc = (target_pred == batch["target_actions"][active]).float().mean()
        if target_margin_loss_weight > 0.0:
            active_logits = target_logits[active]
            active_targets = batch["target_actions"][active]
            positive_logits = active_logits.gather(1, active_targets[:, None]).squeeze(1)
            negative_logits = active_logits.clone()
            negative_logits.scatter_(1, active_targets[:, None], -1e9)
            top_k = min(max(1, int(target_margin_top_k)), max(1, negative_logits.shape[1] - 1))
            hard_negatives = negative_logits.topk(top_k, dim=1).values
            margin_terms = torch.relu(float(target_margin) + hard_negatives - positive_logits[:, None])
            target_margin_loss = (margin_terms.mean(dim=1) * target_active_weights).sum() / target_active_weights.sum().clamp_min(1.0)
        if "ship_logits" in out:
            ship_logits_all = out["ship_logits"]
            b, s, slot = torch.where(active)
            t = batch["target_actions"][active]
            ship_logits = ship_logits_all[b, s, slot, t]
            ship_loss = _weighted_cross_entropy(ship_logits, batch["ship_actions"][active], ship_active_weights)
            ship_acc = (ship_logits.argmax(dim=-1) == batch["ship_actions"][active]).float().mean()
            if target_ship_joint_loss_weight > 0.0:
                active_target_logits = target_logits[b, s, slot]
                active_ship_logits = ship_logits_all[b, s, slot]
                joint_logits = active_target_logits[:, :, None] + active_ship_logits
                joint_logits = joint_logits.reshape(joint_logits.shape[0], -1)
                joint_targets = t * active_ship_logits.shape[-1] + batch["ship_actions"][active]
                joint_weights = ship_active_weights
                if target_ship_joint_dagger_only:
                    joint_weights = joint_weights * row_weights[b].lt(0.999).float()
                if joint_weights.sum() > 0.0:
                    target_ship_joint_loss = _weighted_cross_entropy(joint_logits, joint_targets, joint_weights)
                    target_ship_joint_acc = (joint_logits.argmax(dim=-1) == joint_targets).float().mean()

    if target_binary_loss_weight > 0.0:
        pair_valid = _valid_pair_mask(batch)
        pair_labels = torch.zeros_like(pair_valid, dtype=torch.float32)
        b, s, _slot = torch.where(active)
        if b.numel() > 0:
            t = batch["target_actions"][active].clamp(0, pair_valid.shape[-1] - 1)
            pair_labels[b, s, t] = 1.0
        pair_logits = out.get("target_pair_logits")
        if pair_logits is None:
            pair_logits = out["target_logits"].max(dim=2).values
        binary_losses = F.binary_cross_entropy_with_logits(
            pair_logits,
            pair_labels,
            reduction="none",
            pos_weight=torch.tensor(float(target_binary_pos_weight), dtype=torch.float32, device=launch_logits.device),
        )
        pair_weights = pair_row_weights[:, None, None].expand_as(pair_valid).float()
        target_binary_loss = (binary_losses * pair_weights).masked_select(pair_valid).sum() / pair_weights.masked_select(pair_valid).sum().clamp_min(1.0)

    pair_logits_for_metric = out.get("target_pair_logits")
    if pair_logits_for_metric is None:
        pair_logits_for_metric = out["target_logits"].max(dim=2).values
    if active.any():
        pair_valid = _valid_pair_mask(batch)
        masked_pair_logits = pair_logits_for_metric.masked_fill(~pair_valid, -1e9)
        b, s, _slot = torch.where(active)
        pair_targets = batch["target_actions"][active].clamp(0, masked_pair_logits.shape[-1] - 1)
        if target_pair_softmax_loss_weight > 0.0:
            target_pair_softmax_loss = _weighted_cross_entropy(masked_pair_logits[b, s], pair_targets, pair_active_weights)
        if target_pair_margin_loss_weight > 0.0:
            active_pair_logits = masked_pair_logits[b, s]
            positive_logits = active_pair_logits.gather(1, pair_targets[:, None]).squeeze(1)
            negative_logits = active_pair_logits.clone()
            negative_logits.scatter_(1, pair_targets[:, None], -1e9)
            top_k = min(max(1, int(target_margin_top_k)), max(1, negative_logits.shape[1] - 1))
            hard_negatives = negative_logits.topk(top_k, dim=1).values
            margin_terms = torch.relu(float(target_margin) + hard_negatives - positive_logits[:, None])
            target_pair_margin_loss = (margin_terms.mean(dim=1) * pair_active_weights).sum() / pair_active_weights.sum().clamp_min(1.0)
        if target_pair_owner_loss_weight > 0.0:
            owner_logits = _owner_group_logits(pair_logits_for_metric, batch, pair_valid)
            owner_targets = _target_owner_labels(batch, pair_targets, b)
            target_pair_owner_loss = _weighted_cross_entropy(owner_logits[b, s], owner_targets, pair_active_weights)
            target_pair_owner_acc = (owner_logits[b, s].argmax(dim=-1) == owner_targets).float().mean()
        if target_pair_within_owner_loss_weight > 0.0:
            target_owner = batch["planets"][:, :, :3].argmax(dim=-1)
            owner_targets = _target_owner_labels(batch, pair_targets, b)
            same_owner = target_owner[b].eq(owner_targets[:, None])
            active_same_owner_logits = masked_pair_logits[b, s].masked_fill(~same_owner, -1e9)
            target_pair_within_owner_loss = _weighted_cross_entropy(active_same_owner_logits, pair_targets, pair_active_weights)
            target_pair_within_owner_acc = (active_same_owner_logits.argmax(dim=-1) == pair_targets).float().mean()
        pair_pred = masked_pair_logits[b, s].argmax(dim=-1)
        target_pair_acc = (pair_pred == pair_targets).float().mean()

    if set_loss is not None:
        loss = (
            set_loss
            + target_binary_loss_weight * target_binary_loss
            + target_pair_softmax_loss_weight * target_pair_softmax_loss
            + target_pair_margin_loss_weight * target_pair_margin_loss
            + target_pair_owner_loss_weight * target_pair_owner_loss
            + target_pair_within_owner_loss_weight * target_pair_within_owner_loss
            + launch_count_loss_weight * launch_count_loss
            + target_ship_joint_loss_weight * target_ship_joint_loss
        )
        target_loss = set_parts["target_loss"]
        ship_loss = set_parts["ship_loss"]
        target_margin_loss = set_parts["target_margin_loss"]
        action_weight_mean = set_parts["action_weight_mean"]
    else:
        loss = (
            launch_loss
            + target_loss_weight * target_loss
            + ship_loss_weight * ship_loss
            + target_margin_loss_weight * target_margin_loss
            + target_binary_loss_weight * target_binary_loss
            + target_pair_softmax_loss_weight * target_pair_softmax_loss
            + target_pair_margin_loss_weight * target_pair_margin_loss
            + target_pair_owner_loss_weight * target_pair_owner_loss
            + target_pair_within_owner_loss_weight * target_pair_within_owner_loss
            + launch_count_loss_weight * launch_count_loss
            + target_ship_joint_loss_weight * target_ship_joint_loss
        )
    launch_pred = out["source_logits"][own_slots].argmax(dim=-1)
    launch_acc = (launch_pred == launch_targets).float().mean()
    pos = launch_targets == 1
    pred_pos = launch_pred == 1
    true_count = batch["launch_actions"].masked_fill(~own_slots, 0).sum(dim=(1, 2)).float()
    pred_count = (out["source_logits"].argmax(dim=-1) == 1).masked_fill(~own_slots, False).sum(dim=(1, 2)).float()
    tp = (pred_pos & pos).float().sum()
    fp = (pred_pos & ~pos).float().sum()
    fn = (~pred_pos & pos).float().sum()
    launch_precision = tp / (tp + fp).clamp_min(1.0)
    launch_recall = (launch_pred[pos] == 1).float().mean() if pos.any() else torch.tensor(0.0, device=launch_logits.device)
    launch_f1 = 2.0 * launch_precision * launch_recall / (launch_precision + launch_recall).clamp_min(1e-6)
    launch_pred_rate = pred_pos.float().mean()
    launch_true_rate = pos.float().mean()
    action_count_mae = (pred_count - true_count).abs().mean()
    soft_action_count_mae = (pred_count_soft - true_count_soft_target).abs().mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "launch_loss": float(launch_loss.detach().cpu()),
        "launch_count_loss": float(launch_count_loss.detach().cpu()),
        "target_loss": float(target_loss.detach().cpu()),
        "target_margin_loss": float(target_margin_loss.detach().cpu()),
        "target_binary_loss": float(target_binary_loss.detach().cpu()),
        "target_pair_softmax_loss": float(target_pair_softmax_loss.detach().cpu()),
        "target_pair_margin_loss": float(target_pair_margin_loss.detach().cpu()),
        "target_pair_owner_loss": float(target_pair_owner_loss.detach().cpu()),
        "target_pair_within_owner_loss": float(target_pair_within_owner_loss.detach().cpu()),
        "ship_loss": float(ship_loss.detach().cpu()),
        "target_ship_joint_loss": float(target_ship_joint_loss.detach().cpu()),
        "launch_acc": float(launch_acc.detach().cpu()),
        "launch_precision": float(launch_precision.detach().cpu()),
        "launch_recall": float(launch_recall.detach().cpu()),
        "launch_f1": float(launch_f1.detach().cpu()),
        "launch_pred_rate": float(launch_pred_rate.detach().cpu()),
        "launch_true_rate": float(launch_true_rate.detach().cpu()),
        "action_count_mae": float(action_count_mae.detach().cpu()),
        "soft_action_count_mae": float(soft_action_count_mae.detach().cpu()),
        "target_acc": float(target_acc.detach().cpu()),
        "target_pair_acc": float(target_pair_acc.detach().cpu()),
        "target_pair_owner_acc": float(target_pair_owner_acc.detach().cpu()),
        "target_pair_within_owner_acc": float(target_pair_within_owner_acc.detach().cpu()),
        "ship_acc": float(ship_acc.detach().cpu()),
        "target_ship_joint_acc": float(target_ship_joint_acc.detach().cpu()),
        "action_weight_mean": float(action_weight_mean.detach().cpu()),
        "sample_weight_mean": float(row_weights.mean().detach().cpu()),
        "launch_sample_weight_mean": float(launch_row_weights.mean().detach().cpu()),
        "target_sample_weight_mean": float(target_row_weights.mean().detach().cpu()),
        "ship_sample_weight_mean": float(ship_row_weights.mean().detach().cpu()),
        "pair_sample_weight_mean": float(pair_row_weights.mean().detach().cpu()),
        "count_sample_weight_mean": float(count_row_weights.mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate_loader(
    model: TinyPolicyValueNet,
    loader: DataLoader,
    device: torch.device,
    launch_pos_weight: float,
    target_loss_weight: float = 1.0,
    ship_loss_weight: float = 0.5,
    target_ship_joint_loss_weight: float = 0.0,
    target_ship_joint_dagger_only: bool = False,
    critical_action_weight: float = 0.0,
    target_loss_mask: str = "dataset",
    target_margin_loss_weight: float = 0.0,
    target_margin: float = 0.5,
    target_margin_top_k: int = 8,
    slot_set_loss: bool = False,
    target_binary_loss_weight: float = 0.0,
    target_binary_pos_weight: float = 12.0,
    target_pair_softmax_loss_weight: float = 0.0,
    target_pair_margin_loss_weight: float = 0.0,
    target_pair_owner_loss_weight: float = 0.0,
    target_pair_within_owner_loss_weight: float = 0.0,
    launch_count_loss_weight: float = 0.0,
    sample_weight_launch_scale: float = 1.0,
    sample_weight_target_scale: float = 1.0,
    sample_weight_ship_scale: float = 1.0,
    sample_weight_pair_scale: float = 1.0,
    sample_weight_count_scale: float = 1.0,
    dagger_launch_negative_weight_scale: float = 1.0,
    dagger_launch_positive_weight_scale: float = 1.0,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        _, metrics = bc_loss(
            model,
            unpack(batch, device),
            launch_pos_weight,
            target_loss_weight,
            ship_loss_weight,
            target_ship_joint_loss_weight,
            target_ship_joint_dagger_only,
            critical_action_weight,
            target_loss_mask,
            target_margin_loss_weight,
            target_margin,
            target_margin_top_k,
            slot_set_loss,
            target_binary_loss_weight,
            target_binary_pos_weight,
            target_pair_softmax_loss_weight,
            target_pair_margin_loss_weight,
            target_pair_owner_loss_weight,
            target_pair_within_owner_loss_weight,
            launch_count_loss_weight,
            sample_weight_launch_scale,
            sample_weight_target_scale,
            sample_weight_ship_scale,
            sample_weight_pair_scale,
            sample_weight_count_scale,
            dagger_launch_negative_weight_scale,
            dagger_launch_positive_weight_scale,
        )
        n = int(batch[0].shape[0])
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value * n
        count += n
    return {key: value / max(1, count) for key, value in sums.items()}


def train_bc(args: argparse.Namespace, dataset: TensorDataset) -> tuple[TinyPolicyValueNet, dict[str, float]]:
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = max(1, len(dataset) - val_size)
    train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.loader_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.loader_workers, pin_memory=device.type == "cuda")

    model_cfg = {
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
    model = TinyPolicyValueNet(**model_cfg).to(device)
    trainable_params = list(model.parameters())
    if args.trainable_modules == "target_head":
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.target_head.parameters():
            param.requires_grad_(True)
        trainable_params = list(model.target_head.parameters())
    elif args.trainable_modules == "source_target_heads_no_slot":
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.source_head.parameters():
            param.requires_grad_(True)
        for param in model.target_head.parameters():
            param.requires_grad_(True)
        trainable_params = list(model.source_head.parameters()) + list(model.target_head.parameters())
    elif args.trainable_modules == "source_target_pair_heads_no_slot":
        if model.target_pair_head is None:
            raise ValueError("--trainable-modules source_target_pair_heads_no_slot requires --target-pair-head")
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.source_head.parameters():
            param.requires_grad_(True)
        for param in model.target_head.parameters():
            param.requires_grad_(True)
        for param in model.target_pair_head.parameters():
            param.requires_grad_(True)
        trainable_params = (
            list(model.source_head.parameters())
            + list(model.target_head.parameters())
            + list(model.target_pair_head.parameters())
        )
        if model.target_pair_owner_head is not None:
            for param in model.target_pair_owner_head.parameters():
                param.requires_grad_(True)
            trainable_params += list(model.target_pair_owner_head.parameters())
    elif args.trainable_modules == "source_head":
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.source_head.parameters():
            param.requires_grad_(True)
        model.slot_embed.requires_grad_(True)
        trainable_params = list(model.source_head.parameters()) + [model.slot_embed]
    elif args.trainable_modules == "target_pair_head":
        if model.target_pair_head is None:
            raise ValueError("--trainable-modules target_pair_head requires --target-pair-head")
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.target_pair_head.parameters():
            param.requires_grad_(True)
        trainable_params = list(model.target_pair_head.parameters())
        if model.target_pair_owner_head is not None:
            for param in model.target_pair_owner_head.parameters():
                param.requires_grad_(True)
            trainable_params += list(model.target_pair_owner_head.parameters())
    elif args.trainable_modules == "target_ranking":
        if model.target_pair_head is None:
            raise ValueError("--trainable-modules target_ranking requires --target-pair-head")
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.edge.parameters():
            param.requires_grad_(True)
        for param in model.target_pair_head.parameters():
            param.requires_grad_(True)
        trainable_params = list(model.edge.parameters()) + list(model.target_pair_head.parameters())
        if model.target_pair_owner_head is not None:
            for param in model.target_pair_owner_head.parameters():
                param.requires_grad_(True)
            trainable_params += list(model.target_pair_owner_head.parameters())
    elif args.trainable_modules == "target_pair_adapter":
        if model.target_pair_head is None or model.target_pair_edge is None:
            raise ValueError("--trainable-modules target_pair_adapter requires --target-pair-head --target-pair-adapter")
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.target_pair_edge.parameters():
            param.requires_grad_(True)
        for param in model.target_pair_head.parameters():
            param.requires_grad_(True)
        trainable_params = list(model.target_pair_edge.parameters()) + list(model.target_pair_head.parameters())
        if model.target_pair_owner_head is not None:
            for param in model.target_pair_owner_head.parameters():
                param.requires_grad_(True)
            trainable_params += list(model.target_pair_owner_head.parameters())
    elif args.trainable_modules != "all":
        raise ValueError(f"unsupported trainable_modules: {args.trainable_modules!r}")
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_metrics: dict[str, float] = {}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sums: dict[str, float] = {}
        train_count = 0
        skipped_no_grad = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, metrics = bc_loss(
                model,
                unpack(batch, device),
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
            )
            n = int(batch[0].shape[0])
            for key, value in metrics.items():
                train_sums[key] = train_sums.get(key, 0.0) + value * n
            train_count += n
            if not loss.requires_grad:
                skipped_no_grad += n
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
        train_metrics = {f"train_{key}": value / max(1, train_count) for key, value in train_sums.items()}
        train_metrics["train_skipped_no_grad_samples"] = float(skipped_no_grad)
        val_metrics = {
            f"val_{key}": value
            for key, value in evaluate_loader(
                model,
                val_loader,
                device,
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
            ).items()
        }
        merged = {"epoch": float(epoch), **train_metrics, **val_metrics}
        print(json.dumps(merged, ensure_ascii=True), flush=True)
        if best_state is None or merged["val_loss"] < best_metrics.get("val_loss", math.inf):
            best_metrics = merged
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.model_config = model_cfg  # type: ignore[attr-defined]
    return model, best_metrics


def save_checkpoint(path: Path, model: TinyPolicyValueNet, args: argparse.Namespace, collect_metrics: dict[str, float], train_metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = getattr(model, "model_config")
    payload = {
        "state_dict": model.state_dict(),
        "model": cfg,
        "update": 0,
        "metrics": {
            "kind": "regular_bc",
            "collect": collect_metrics,
            "train": train_metrics,
        },
        "args": vars(args),
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tinyPPO/regular_bc.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--games-per-players", type=int, default=0, help="Games for each value in --players-list. Defaults to --games.")
    parser.add_argument("--players-list", default="2", help="Comma list of player counts to collect, e.g. 2,4.")
    parser.add_argument("--collect-workers", type=int, default=1)
    parser.add_argument("--rows-per-game", type=int, default=0, help="Randomly cap labelled observation rows per game. 0 keeps all sampled rows.")
    parser.add_argument("--max-samples", type=int, default=1536, help="Global cap after collection. 0 disables the cap.")
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--keep-noop-prob", type=float, default=0.08)
    parser.add_argument(
        "--row-target-mask-mode",
        choices=["candidate", "safe", "all_planets"],
        default="candidate",
        help="Target mask stored in collected BC rows. Existing caches keep their saved mask.",
    )
    parser.add_argument("--seed", type=int, default=260525)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--launch-pos-weight", type=float, default=8.0)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--ship-loss-weight", type=float, default=0.5)
    parser.add_argument("--target-ship-joint-loss-weight", type=float, default=0.0, help="Auxiliary CE over the joint target x ship-bucket choice for each labelled launch.")
    parser.add_argument("--target-ship-joint-dagger-only", action="store_true", help="Apply target-ship joint auxiliary loss only to DAgger-like rows (sample_weight < 1).")
    parser.add_argument("--critical-action-weight", type=float, default=0.0)
    parser.add_argument("--target-loss-mask", choices=["dataset", "all_planets"], default="dataset")
    parser.add_argument("--target-margin-loss-weight", type=float, default=0.0)
    parser.add_argument("--target-margin", type=float, default=0.5)
    parser.add_argument("--target-margin-top-k", type=int, default=8)
    parser.add_argument("--slot-set-loss", action="store_true", help="Treat same-source action slots as an unordered set during BC loss.")
    parser.add_argument("--target-binary-loss-weight", type=float, default=0.0, help="Auxiliary BCE over source-target edges labelled by regular actions.")
    parser.add_argument("--target-binary-pos-weight", type=float, default=12.0)
    parser.add_argument("--target-pair-softmax-loss-weight", type=float, default=0.0, help="Auxiliary CE over each source's target-pair logits for regular targets.")
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
    parser.add_argument(
        "--trainable-modules",
        choices=[
            "all",
            "target_head",
            "source_target_heads_no_slot",
            "source_target_pair_heads_no_slot",
            "source_head",
            "target_pair_head",
            "target_ranking",
            "target_pair_adapter",
        ],
        default="all",
    )
    parser.add_argument("--val-frac", type=float, default=0.12)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--source-target-summary", action="store_true", help="Let the launch/source head see a pooled summary of source-target edge features.")
    parser.add_argument("--target-pair-head", action="store_true", help="Add an auxiliary source-target edge head for regular target selection.")
    parser.add_argument("--target-pair-adapter", action="store_true", help="Use a separate edge MLP for target-pair logits so target-ranking updates do not perturb source/ship heads.")
    parser.add_argument("--target-pair-owner-head", action="store_true", help="Add a zero-initialized target-owner bias head on target-pair logits.")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    rows, collect_metrics = collect_rows(args)
    if not rows:
        raise RuntimeError("no BC rows collected")
    print(json.dumps({"collect": collect_metrics}, ensure_ascii=True), flush=True)
    dataset = stack_rows(rows)
    model, train_metrics = train_bc(args, dataset)
    save_checkpoint(Path(args.out), model, args, collect_metrics, train_metrics)
    print(json.dumps({"saved": args.out, "metrics": train_metrics}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
