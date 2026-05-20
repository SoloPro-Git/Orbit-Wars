"""Ray stage-1.5 training: make model proposals safe under rollout feedback.

Stage1 teaches the ranker to imitate the regular rulebase candidate. Stage1.5
lets model-generated proposals enter the candidate set, then uses real game
outcomes to either reinforce the selected action or fall back to the rulebase
oracle when that selected action led to a loss.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KAGGLE_ENGINES_LOG_LEVEL"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import ray
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from training2.batching import pad_planets
from training2.candidates import _canonical, build_candidates
from training2.envs import make_orbit_wars_env
from training2.features import encode_position, result_value
from training2.model import CandidatePolicyValueNet, load_compatible_state_dict
from training2.proposal import ProposalConfig, proposals_from_model
from training2.rulebase_bridge import make_rulebase_agent


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        p = PROJECT_ROOT / path
    with p.open() as f:
        return yaml.safe_load(f) or {}


def _resolve_project_path(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


def _raw_obs(env, player: int) -> dict:
    return env.steps[-1][player]["observation"]


def _score(obs: dict, player: int) -> float:
    return float(
        sum(p[5] for p in obs.get("planets", []) if int(p[1]) == player)
        + sum(f[6] for f in obs.get("fleets", []) if int(f[1]) == player)
    )


def _make_model(model_cfg: dict, device: str) -> CandidatePolicyValueNet:
    return CandidatePolicyValueNet(
        d_model=int(model_cfg.get("d_model", 192)),
        nhead=int(model_cfg.get("nhead", 6)),
        layers=int(model_cfg.get("layers", 4)),
        dropout=float(model_cfg.get("dropout", 0.10)),
    ).to(device)


def _init_swanlab(config: dict, args) -> Any | None:
    if args.no_swanlab:
        return None
    import swanlab

    key = os.environ.get("SWANLAB_API_KEY")
    key_path = PROJECT_ROOT / "training/config/swanlab_key.txt"
    if not key and key_path.exists():
        key = key_path.read_text().strip()
    if key:
        os.environ["SWANLAB_API_KEY"] = key
    return swanlab.init(
        project=str(config.get("swanlab_project", "orbit-wars")),
        experiment_name=str(config.get("swanlab_experiment", "training2-stage15-proposal")),
        mode=str(config.get("swanlab_mode", "cloud")),
        config={"training2_stage15": config, "args": vars(args)},
    )


def _train_batch(model, opt, rows: list[dict], device: str, value_weight: float, entropy_weight: float) -> dict:
    t_data = time.time()
    planets = pad_planets([r["planets"] for r in rows]).to(device)
    glob = torch.tensor([r["global"] for r in rows], dtype=torch.float32, device=device)
    candidates = torch.tensor([r["candidates"] for r in rows], dtype=torch.float32, device=device)
    mask = torch.tensor([r["candidate_mask"] for r in rows], dtype=torch.float32, device=device)
    target = torch.tensor([r["target"] for r in rows], dtype=torch.long, device=device)
    value_target = torch.tensor([r["value"] for r in rows], dtype=torch.float32, device=device)
    weight = torch.tensor([r.get("policy_weight", 1.0) for r in rows], dtype=torch.float32, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    data_sec = time.time() - t_data

    t_forward = time.time()
    logits, value = model(planets, glob, candidates, mask)
    ce = F.cross_entropy(logits, target, reduction="none")
    policy_loss = (ce * weight).sum() / weight.sum().clamp(min=1.0)
    value_loss = F.mse_loss(value, value_target)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    loss = policy_loss + value_weight * value_loss - entropy_weight * entropy
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    forward_sec = time.time() - t_forward

    t_backward = time.time()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    backward_sec = time.time() - t_backward

    selected = torch.tensor([r.get("selected_idx", r["target"]) for r in rows], dtype=torch.long, device=device)
    return {
        "train/loss": float(loss.item()),
        "train/policy_loss": float(policy_loss.item()),
        "train/value_loss": float(value_loss.item()),
        "train/entropy": float(entropy.item()),
        "train/target_top1": float((logits.argmax(-1) == target).float().mean().item()),
        "train/selected_top1": float((logits.argmax(-1) == selected).float().mean().item()),
        "train/value_mean": float(value.mean().item()),
        "train/batch_data_sec": data_sec,
        "train/batch_forward_sec": forward_sec,
        "train/batch_backward_sec": backward_sec,
    }


def _train_batch_ppo(
    model,
    opt,
    rows: list[dict],
    device: str,
    value_weight: float,
    entropy_weight: float,
    clip_coef: float,
    value_clip_coef: float,
    bc_anchor_weight: float,
    policy_temperature: float,
) -> dict:
    t_data = time.time()
    planets = pad_planets([r["planets"] for r in rows]).to(device)
    glob = torch.tensor([r["global"] for r in rows], dtype=torch.float32, device=device)
    candidates = torch.tensor([r["candidates"] for r in rows], dtype=torch.float32, device=device)
    mask = torch.tensor([r["candidate_mask"] for r in rows], dtype=torch.float32, device=device)
    action = torch.tensor([r.get("selected_idx", r["target"]) for r in rows], dtype=torch.long, device=device)
    old_logprob = torch.tensor([r["old_logprob"] for r in rows], dtype=torch.float32, device=device)
    old_value = torch.tensor([r["old_value"] for r in rows], dtype=torch.float32, device=device)
    returns = torch.tensor([r["return"] for r in rows], dtype=torch.float32, device=device)
    advantages = torch.tensor([r["advantage"] for r in rows], dtype=torch.float32, device=device)
    oracle = torch.tensor([r.get("oracle_idx", r.get("target", 0)) for r in rows], dtype=torch.long, device=device)
    selected_is_proposal = torch.tensor(
        [r.get("selected_is_proposal", 0.0) for r in rows],
        dtype=torch.float32,
        device=device,
    )
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp(min=1e-6)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    data_sec = time.time() - t_data

    t_forward = time.time()
    logits, value = model(planets, glob, candidates, mask)
    behavior_logits = logits / max(policy_temperature, 1e-6)
    logprob = torch.log_softmax(behavior_logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)
    ratio = torch.exp(logprob - old_logprob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    if value_clip_coef > 0.0:
        value_clipped = old_value + torch.clamp(value - old_value, -value_clip_coef, value_clip_coef)
        value_loss = torch.max((value - returns).pow(2), (value_clipped - returns).pow(2)).mean()
    else:
        value_loss = F.mse_loss(value, returns)
    probs = torch.softmax(behavior_logits, dim=-1)
    entropy = -(probs * torch.log_softmax(behavior_logits, dim=-1)).sum(dim=-1).mean()
    bc_anchor_loss = F.cross_entropy(logits, oracle) if bc_anchor_weight > 0.0 else torch.tensor(0.0, device=device)
    loss = policy_loss + value_weight * value_loss - entropy_weight * entropy + bc_anchor_weight * bc_anchor_loss
    approx_kl = (old_logprob - logprob).mean()
    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    forward_sec = time.time() - t_forward

    t_backward = time.time()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    backward_sec = time.time() - t_backward

    with torch.no_grad():
        top1 = logits.argmax(-1)
        proposal_mask = selected_is_proposal > 0.5
        if proposal_mask.any():
            proposal_adv = advantages[proposal_mask].mean()
            proposal_top1 = (top1[proposal_mask] == action[proposal_mask]).float().mean()
        else:
            proposal_adv = torch.tensor(0.0, device=device)
            proposal_top1 = torch.tensor(0.0, device=device)
    return {
        "train/loss": float(loss.item()),
        "train/policy_loss": float(policy_loss.item()),
        "train/value_loss": float(value_loss.item()),
        "train/entropy": float(entropy.item()),
        "train/bc_anchor_loss": float(bc_anchor_loss.item()),
        "train/approx_kl": float(approx_kl.item()),
        "train/clipfrac": float(clipfrac.item()),
        "train/ratio_mean": float(ratio.mean().item()),
        "train/advantage_mean": float(advantages.mean().item()),
        "train/return_mean": float(returns.mean().item()),
        "train/target_top1": float((top1 == action).float().mean().item()),
        "train/oracle_top1": float((top1 == oracle).float().mean().item()),
        "train/proposal_advantage_mean": float(proposal_adv.item()),
        "train/proposal_selected_top1": float(proposal_top1.item()),
        "train/value_mean": float(value.mean().item()),
        "train/batch_data_sec": data_sec,
        "train/batch_forward_sec": forward_sec,
        "train/batch_backward_sec": backward_sec,
    }


@ray.remote
class Stage15RolloutActor:
    def __init__(
        self,
        worker_id: int,
        model_cfg: dict,
        oracle: str,
        max_candidates: int,
        proposal_cfg: dict,
        device: str,
        env_backend: str = "kaggle",
        env_use_numba: bool = False,
        torch_threads: int | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.device = device
        if torch_threads is not None and torch_threads > 0:
            torch.set_num_threads(int(torch_threads))
            torch.set_num_interop_threads(int(torch_threads))
        self.model = _make_model(model_cfg, device)
        self.model.eval()
        self.oracle = oracle
        self.rulebase = make_rulebase_agent(oracle)
        self.max_candidates = max_candidates
        self.proposal_cfg = ProposalConfig(**proposal_cfg)
        self.env_backend = env_backend
        self.env_use_numba = env_use_numba
        self.torch_threads = torch.get_num_threads()

    def _rulebase_margin(self, seed: int, players: int, model_pid: int) -> tuple[float, float]:
        env = make_orbit_wars_env(
            {"episodeSteps": 500, "seed": seed},
            backend=self.env_backend,
            debug=True,
            use_numba=self.env_use_numba,
        )
        env.reset(players)
        agents = {pid: make_rulebase_agent(self.oracle) for pid in range(players)}
        for _ in range(500):
            actions = [agents[pid](_raw_obs(env, pid)) or [] for pid in range(players)]
            env.step(actions)
            if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                break
        final_obs = _raw_obs(env, model_pid)
        my_score = _score(final_obs, model_pid)
        opp_best = max(_score(final_obs, pid) for pid in range(players) if pid != model_pid)
        return my_score - opp_best, result_value(final_obs, model_pid)

    def rollout(
        self,
        state_dict: dict,
        games: int,
        seed_offset: int,
        four_player_prob: float,
        temperature: float,
        epsilon: float,
        reward_mode: str,
        advantage_margin_threshold: float,
        advantage_value_scale: float,
        min_policy_weight: float,
        max_policy_weight: float,
        positive_policy_weight: float,
        negative_policy_weight: float,
        training_mode: str,
        gamma: float,
        gae_lambda: float,
    ) -> dict:
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        rows: list[dict] = []
        wins = losses = draws = 0
        games_2p = games_4p = 0
        selected_proposal = 0
        selected_oracle = 0
        fallback_targets = 0
        game_lengths: list[int] = []
        margins: list[float] = []
        baseline_margins: list[float] = []
        advantages: list[float] = []
        positive_advantage_games = 0
        model_game_sec = 0.0
        opponent_rulebase_sec = 0.0
        candidate_sec = 0.0
        encode_sec = 0.0
        model_forward_sec = 0.0
        env_step_sec = 0.0
        baseline_sec = 0.0

        for game in range(games):
            t_model_game = time.time()
            seed = seed_offset + self.worker_id * 1_000_000 + game
            rng = random.Random(seed)
            players = 4 if rng.random() < four_player_prob else 2
            games_4p += int(players == 4)
            games_2p += int(players == 2)
            env = make_orbit_wars_env(
                {"episodeSteps": 500, "seed": seed},
                backend=self.env_backend,
                debug=True,
                use_numba=self.env_use_numba,
            )
            env.reset(players)
            model_pid = rng.randrange(players)
            agents = {pid: make_rulebase_agent("rl_informed_regular") for pid in range(players)}
            candidate_rulebase = make_rulebase_agent(self.oracle)
            pending: list[dict] = []
            steps = 0

            for steps in range(500):
                actions = []
                for pid in range(players):
                    obs = _raw_obs(env, pid)
                    if pid != model_pid:
                        t_rule = time.time()
                        actions.append(agents[pid](obs) or [])
                        opponent_rulebase_sec += time.time() - t_rule
                        continue

                    t_candidate = time.time()
                    extra = proposals_from_model(obs, pid, self.model, self.device, self.proposal_cfg)
                    extra_keys = {_canonical(action) for action in extra}
                    candidates, oracle_idx = build_candidates(
                        obs,
                        candidate_rulebase,
                        max_candidates=self.max_candidates,
                        extra_candidates=extra,
                    )
                    candidate_sec += time.time() - t_candidate
                    if not candidates:
                        actions.append([])
                        continue

                    t_encode = time.time()
                    enc = encode_position(obs, pid, candidates, max_candidates=self.max_candidates)
                    encode_sec += time.time() - t_encode
                    t_forward = time.time()
                    with torch.no_grad():
                        logits, value_pred = self.model(
                            torch.tensor(enc.planet_features, dtype=torch.float32, device=self.device).unsqueeze(0),
                            torch.tensor(enc.global_features, dtype=torch.float32, device=self.device).unsqueeze(0),
                            torch.tensor(enc.candidate_features, dtype=torch.float32, device=self.device).unsqueeze(0),
                            torch.tensor(enc.candidate_mask, dtype=torch.float32, device=self.device).unsqueeze(0),
                        )
                        behavior_probs = torch.softmax(logits[0] / max(temperature, 1e-6), dim=-1)
                        valid = int(enc.candidate_mask.sum())
                        if epsilon > 0.0 and valid > 0:
                            uniform = torch.zeros_like(behavior_probs)
                            uniform[:valid] = 1.0 / float(valid)
                            behavior_probs = (1.0 - epsilon) * behavior_probs + epsilon * uniform
                        behavior_log_probs = torch.log(behavior_probs.clamp(min=1e-12))
                        if rng.random() < epsilon:
                            selected_idx = rng.randrange(max(valid, 1))
                        else:
                            selected_idx = int(torch.multinomial(behavior_probs, 1).item())
                    model_forward_sec += time.time() - t_forward
                    selected_idx = min(selected_idx, len(candidates) - 1)
                    is_proposal = _canonical(candidates[selected_idx]) in extra_keys
                    selected_proposal += int(is_proposal)
                    selected_oracle += int(selected_idx == oracle_idx)
                    pending.append(
                        {
                            "player": pid,
                            "planets": enc.planet_features.tolist(),
                            "global": enc.global_features.tolist(),
                            "candidates": enc.candidate_features.tolist(),
                            "candidate_mask": enc.candidate_mask.tolist(),
                            "selected_idx": int(selected_idx),
                            "oracle_idx": int(oracle_idx),
                            "selected_is_proposal": float(is_proposal),
                            "old_logprob": float(behavior_log_probs[selected_idx].item()),
                            "old_value": float(value_pred[0].item()),
                        }
                    )
                    actions.append(candidates[selected_idx])
                t_env_step = time.time()
                env.step(actions)
                env_step_sec += time.time() - t_env_step
                if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                    break

            model_game_sec += time.time() - t_model_game
            final_obs = _raw_obs(env, model_pid)
            result = result_value(final_obs, model_pid)
            my_score = _score(final_obs, model_pid)
            opp_best = max(_score(final_obs, pid) for pid in range(players) if pid != model_pid)
            margin = my_score - opp_best
            if reward_mode == "paired_margin_advantage":
                t_baseline = time.time()
                baseline_margin, _ = self._rulebase_margin(seed, players, model_pid)
                baseline_sec += time.time() - t_baseline
                advantage = margin - baseline_margin
                value = float(np.tanh(advantage / max(advantage_value_scale, 1e-6)))
                positive_advantage = advantage >= advantage_margin_threshold
            else:
                baseline_margin = 0.0
                advantage = margin
                value = float(result)
                positive_advantage = value >= advantage_margin_threshold
            margins.append(margin)
            baseline_margins.append(baseline_margin)
            advantages.append(advantage)
            positive_advantage_games += int(positive_advantage)
            wins += int(margin > 0)
            losses += int(margin < 0)
            draws += int(margin == 0)
            if training_mode in {"ppo_lite", "stage2", "stage3", "ppo"}:
                next_value = 0.0
                next_advantage = 0.0
                for row_idx in range(len(pending) - 1, -1, -1):
                    row = pending[row_idx]
                    reward = float(value) if row_idx == len(pending) - 1 else 0.0
                    old_value = float(row["old_value"])
                    delta = reward + gamma * next_value - old_value
                    gae = delta + gamma * gae_lambda * next_advantage
                    row["target"] = int(row["selected_idx"])
                    row["value"] = float(gae + old_value)
                    row["return"] = float(gae + old_value)
                    row["advantage"] = float(gae)
                    row["fallback_to_oracle"] = 0.0
                    row["advantage_margin"] = float(advantage)
                    row["baseline_margin"] = float(baseline_margin)
                    row["model_margin"] = float(margin)
                    next_value = old_value
                    next_advantage = gae
            else:
                for row in pending:
                    selected_idx = int(row["selected_idx"])
                    oracle_idx = int(row["oracle_idx"])
                    use_oracle = (not positive_advantage) and selected_idx != oracle_idx
                    row["target"] = oracle_idx if use_oracle else selected_idx
                    row["value"] = float(value)
                    base_weight = positive_policy_weight if positive_advantage else negative_policy_weight
                    advantage_weight = max(abs(float(value)), min_policy_weight)
                    row["policy_weight"] = float(
                        min(max(base_weight * advantage_weight, min_policy_weight), max_policy_weight)
                    )
                    row["fallback_to_oracle"] = float(use_oracle)
                    row["advantage_margin"] = float(advantage)
                    row["baseline_margin"] = float(baseline_margin)
                    row["model_margin"] = float(margin)
                    fallback_targets += int(use_oracle)
            rows.extend(pending)
            game_lengths.append(steps + 1)

        return {
            "rows": rows,
            "games": games,
            "games_2p": games_2p,
            "games_4p": games_4p,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "avg_margin": float(np.mean(margins)) if margins else 0.0,
            "avg_baseline_margin": float(np.mean(baseline_margins)) if baseline_margins else 0.0,
            "avg_advantage_margin": float(np.mean(advantages)) if advantages else 0.0,
            "positive_advantage_games": positive_advantage_games,
            "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
            "selected_proposal": selected_proposal,
            "selected_oracle": selected_oracle,
            "fallback_targets": fallback_targets,
            "model_game_sec": model_game_sec,
            "baseline_sec": baseline_sec,
            "opponent_rulebase_sec": opponent_rulebase_sec,
            "candidate_sec": candidate_sec,
            "encode_sec": encode_sec,
            "model_forward_sec": model_forward_sec,
            "env_step_sec": env_step_sec,
            "torch_threads": self.torch_threads,
        }


@ray.remote
class Stage15TrainerActor:
    def __init__(self, model_cfg: dict, train_cfg: dict, device: str, resume: str | None = None) -> None:
        self.device = device
        self.model = _make_model(model_cfg, device)
        if resume:
            ckpt = torch.load(resume, map_location=device, weights_only=False)
            load_compatible_state_dict(self.model, ckpt["model_state_dict"])
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        self.replay: list[dict] = []

    def add_rows(self, rows: list[dict], replay_capacity: int) -> dict:
        self.replay.extend(rows)
        self.replay = self.replay[-replay_capacity:]
        return {"replay_size": len(self.replay)}

    def update(
        self,
        updates: int,
        batch_size: int,
        value_weight: float,
        entropy_weight: float,
        training_mode: str = "outcome_bc",
        clip_coef: float = 0.2,
        value_clip_coef: float = 0.2,
        bc_anchor_weight: float = 0.0,
        policy_temperature: float = 1.0,
    ) -> dict:
        if not self.replay:
            return {"train/replay_size": 0.0}
        self.model.train()
        accum: dict[str, float] = {}
        for _ in range(updates):
            batch = random.sample(self.replay, k=min(batch_size, len(self.replay)))
            if training_mode in {"ppo_lite", "stage2", "stage3", "ppo"} and all("advantage" in row for row in batch):
                metrics = _train_batch_ppo(
                    self.model,
                    self.opt,
                    batch,
                    self.device,
                    value_weight,
                    entropy_weight,
                    clip_coef,
                    value_clip_coef,
                    bc_anchor_weight,
                    policy_temperature,
                )
            else:
                metrics = _train_batch(self.model, self.opt, batch, self.device, value_weight, entropy_weight)
            for key, value in metrics.items():
                accum[key] = accum.get(key, 0.0) + value
        out = {key: value / max(updates, 1) for key, value in accum.items()}
        out["train/replay_size"] = float(len(self.replay))
        out["train/lr"] = self.opt.param_groups[0]["lr"]
        if self.device.startswith("cuda"):
            out["gpu/memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            out["gpu/max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        return out

    def state_dict_cpu(self) -> dict:
        return {key: value.detach().cpu() for key, value in self.model.state_dict().items()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict, strict=False)

    def save(self, path: str, iteration: int, config: dict, best_win_rate: float) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "iteration": iteration,
                "config": config,
                "best_win_rate": best_win_rate,
            },
            path,
        )


@ray.remote
class Stage15EvalActor:
    def __init__(
        self,
        worker_id: int,
        model_cfg: dict,
        oracle: str,
        max_candidates: int,
        proposal_cfg: dict,
        env_backend: str = "kaggle",
        env_use_numba: bool = False,
    ) -> None:
        self.worker_id = worker_id
        self.model_cfg = model_cfg
        self.oracle = oracle
        self.max_candidates = max_candidates
        self.proposal_cfg = ProposalConfig(**proposal_cfg)
        self.env_backend = env_backend
        self.env_use_numba = env_use_numba

    def evaluate(self, state_dict: dict, games: int, seed_offset: int, four_player_prob: float, device: str) -> dict:
        model = _make_model(self.model_cfg, device)
        load_compatible_state_dict(model, state_dict)
        model.eval()
        wins = losses = draws = 0
        margins: list[float] = []
        games_2p = games_4p = 0
        for game in range(games):
            seed = seed_offset + self.worker_id * 100_000 + game
            rng = random.Random(seed)
            players = 4 if rng.random() < four_player_prob else 2
            games_4p += int(players == 4)
            games_2p += int(players == 2)
            env = make_orbit_wars_env(
                {"episodeSteps": 500, "seed": seed},
                backend=self.env_backend,
                debug=True,
                use_numba=self.env_use_numba,
            )
            env.reset(players)
            model_pid = game % players
            rulebases = {pid: make_rulebase_agent(self.oracle) for pid in range(players)}
            for _ in range(500):
                actions = []
                for pid in range(players):
                    obs = _raw_obs(env, pid)
                    if pid != model_pid:
                        actions.append(rulebases[pid](obs) or [])
                        continue
                    extra = proposals_from_model(obs, pid, model, device, self.proposal_cfg)
                    candidates, _ = build_candidates(obs, rulebases[pid], self.max_candidates, extra_candidates=extra)
                    enc = encode_position(obs, pid, candidates, max_candidates=self.max_candidates)
                    with torch.no_grad():
                        logits, _ = model(
                            torch.tensor(enc.planet_features, dtype=torch.float32, device=device).unsqueeze(0),
                            torch.tensor(enc.global_features, dtype=torch.float32, device=device).unsqueeze(0),
                            torch.tensor(enc.candidate_features, dtype=torch.float32, device=device).unsqueeze(0),
                            torch.tensor(enc.candidate_mask, dtype=torch.float32, device=device).unsqueeze(0),
                        )
                    idx = min(int(logits.argmax(-1).item()), len(candidates) - 1)
                    actions.append(candidates[idx] if candidates else [])
                env.step(actions)
                if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                    break
            final_obs = _raw_obs(env, model_pid)
            margin = _score(final_obs, model_pid) - max(_score(final_obs, pid) for pid in range(players) if pid != model_pid)
            margins.append(margin)
            wins += int(margin > 0)
            losses += int(margin < 0)
            draws += int(margin == 0)
        return {
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games_2p": games_2p,
            "games_4p": games_4p,
            "avg_margin": float(np.mean(margins)) if margins else 0.0,
        }


def _average_state_dicts(states: list[dict]) -> dict:
    if len(states) == 1:
        return states[0]
    avg = {}
    for key in states[0]:
        value = states[0][key]
        if torch.is_floating_point(value):
            avg[key] = torch.stack([state[key].float() for state in states], dim=0).mean(dim=0).to(value.dtype)
        else:
            avg[key] = value
    return avg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training2/config/default.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--stage-key", default="stage15", help="Config section to run: stage15, stage2, or stage3.")
    parser.add_argument("--no-swanlab", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    model_cfg = cfg.get("model", {})
    proposal_cfg = dict(cfg.get("proposal", {}))
    stage = cfg.get(args.stage_key, {})
    if not stage:
        raise KeyError(f"Missing stage config section: {args.stage_key}")
    proposal_cfg.update(dict(stage.get("proposal_overrides", {}) or {}))
    ray_cfg = cfg.get("ray", {})
    train_cfg = cfg.get("training", {})

    resume = _resolve_project_path(args.resume or stage.get("resume_from", "training2/checkpoints/stage1_regular/latest.pt"))
    iterations = args.iterations or int(stage.get("max_iterations", 200))
    max_candidates = int(model_cfg.get("max_candidates", 32))
    env_cfg = cfg.get("env", {})
    env_backend = str(stage.get("env_backend", env_cfg.get("backend", "kaggle")))
    env_use_numba = bool(stage.get("env_use_numba", env_cfg.get("use_numba", False)))
    rollout_workers = int(stage.get("num_rollout_workers", ray_cfg.get("num_data_workers", 16)))
    trainer_workers = int(stage.get("num_trainer_workers", ray_cfg.get("num_trainer_workers", 1)))
    eval_workers = int(stage.get("num_eval_workers", ray_cfg.get("num_eval_workers", 4)))
    games_per_iteration = int(stage.get("games_per_iteration", 64))
    games_per_rollout_task = max(1, int(stage.get("games_per_rollout_task", 1)))
    batch_size = int(stage.get("batch_size", train_cfg.get("batch_size", 512)))
    updates_per_iteration = int(stage.get("updates_per_iteration", train_cfg.get("updates_per_iteration", 100)))
    replay_capacity = int(stage.get("replay_capacity", 200000))
    device = str(stage.get("device", train_cfg.get("device", "cuda")))
    rollout_device = str(stage.get("rollout_device", "cpu"))
    eval_device = str(stage.get("eval_device", "cpu"))
    rollout_torch_threads = stage.get("rollout_torch_threads", 1 if rollout_device == "cpu" else None)
    rollout_torch_threads = int(rollout_torch_threads) if rollout_torch_threads is not None else None
    gpus_per_trainer = float(stage.get("gpus_per_trainer", ray_cfg.get("gpus_per_trainer", 1.0)))
    cpus_per_trainer = float(stage.get("cpus_per_trainer", 1.0))
    gpus_per_rollout = float(stage.get("gpus_per_rollout", 0.0 if rollout_device == "cpu" else 0.25))
    eval_gpus_per_worker = float(stage.get("eval_gpus_per_worker", 0.0 if eval_device == "cpu" else 0.25))
    cpus_per_rollout = float(stage.get("cpus_per_rollout", 1.0))
    rollout_resource = str(stage.get("rollout_resource", ray_cfg.get("rollout_resource") or "") or "")
    four_player_prob = float(stage.get("four_player_prob", 0.3))
    temperature = float(stage.get("temperature", 0.7))
    epsilon = float(stage.get("epsilon", 0.05))
    eval_interval = int(stage.get("eval_interval", 20))
    eval_games = int(stage.get("eval_games", 100))
    async_pipeline = bool(stage.get("async_pipeline", False))
    training_mode = str(stage.get("training_mode", "outcome_bc"))
    ppo_clip_coef = float(stage.get("ppo_clip_coef", 0.2))
    ppo_value_clip_coef = float(stage.get("ppo_value_clip_coef", 0.2))
    ppo_gamma = float(stage.get("ppo_gamma", 0.999))
    ppo_gae_lambda = float(stage.get("ppo_gae_lambda", 0.95))
    bc_anchor_weight = float(stage.get("bc_anchor_weight", 0.0))
    reward_mode = str(stage.get("reward_mode", "paired_margin_advantage"))
    advantage_margin_threshold = float(stage.get("advantage_margin_threshold", 0.0))
    advantage_value_scale = float(stage.get("advantage_value_scale", 2000.0))
    min_policy_weight = float(stage.get("min_policy_weight", 0.25))
    max_policy_weight = float(stage.get("max_policy_weight", 2.0))
    positive_policy_weight = float(stage.get("positive_policy_weight", 1.25))
    negative_policy_weight = float(stage.get("negative_policy_weight", 1.0))
    value_weight = float(stage.get("value_loss_weight", 0.5))
    entropy_weight = float(stage.get("entropy_weight", 0.01))
    output_dir = Path(stage.get("output_dir", "training2/checkpoints/stage15_proposal"))
    output_dir.mkdir(parents=True, exist_ok=True)

    swan = _init_swanlab({**train_cfg, **ray_cfg, **stage, "resume_from": resume}, args)

    runtime_env: dict[str, Any] = {}
    runtime_env_vars = dict(ray_cfg.get("runtime_env_vars", {}) or {})
    runtime_env_vars.update(dict(stage.get("runtime_env_vars", {}) or {}))
    runtime_env_id = stage.get("runtime_env_id")
    if runtime_env_id:
        runtime_env_vars["TRAINING2_RUNTIME_ENV_ID"] = str(runtime_env_id)
    if runtime_env_vars:
        runtime_env["env_vars"] = {str(k): str(v) for k, v in runtime_env_vars.items()}

    ray_address = stage.get("ray_address", ray_cfg.get("address"))
    if ray_address:
        ray.init(address=str(ray_address), ignore_reinit_error=True, runtime_env=runtime_env or None)
    else:
        ray_temp = Path(ray_cfg.get("temp_dir", "training2/.ray_temp")).resolve()
        ray_temp.mkdir(parents=True, exist_ok=True)
        ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=str(ray_temp), runtime_env=runtime_env or None)

    trainers = [
        Stage15TrainerActor.options(num_cpus=cpus_per_trainer, num_gpus=gpus_per_trainer).remote(
            model_cfg, {**train_cfg, **stage}, device, resume
        )
        for _ in range(trainer_workers)
    ]
    rollout_options: dict[str, Any] = {"num_cpus": cpus_per_rollout, "num_gpus": gpus_per_rollout}
    if rollout_resource:
        rollout_options["resources"] = {rollout_resource: cpus_per_rollout}
    rollouts = [
        Stage15RolloutActor.options(**rollout_options).remote(
            i,
            model_cfg,
            str(stage.get("oracle", "rl_informed_regular")),
            max_candidates,
            proposal_cfg,
            rollout_device,
            env_backend,
            env_use_numba,
            rollout_torch_threads,
        )
        for i in range(rollout_workers)
    ]
    evals = [
        Stage15EvalActor.options(num_gpus=eval_gpus_per_worker).remote(
            i,
            model_cfg,
            str(stage.get("oracle", "rl_informed_regular")),
            max_candidates,
            proposal_cfg,
            env_backend,
            env_use_numba,
        )
        for i in range(eval_workers)
    ]

    best_win_rate = -1.0
    print(
        json.dumps(
            {
                "stage": args.stage_key,
                "resume_from": resume,
                "iterations": iterations,
                "rollout_workers": rollout_workers,
                "trainer_workers": trainer_workers,
                "games_per_iteration": games_per_iteration,
                "games_per_rollout_task": games_per_rollout_task,
                "eval_interval": eval_interval,
                "eval_games": eval_games,
                "device": device,
                "rollout_device": rollout_device,
                "eval_device": eval_device,
                "cpus_per_rollout": cpus_per_rollout,
                "cpus_per_trainer": cpus_per_trainer,
                "rollout_resource": rollout_resource,
                "gpus_per_trainer": gpus_per_trainer,
                "gpus_per_rollout": gpus_per_rollout,
                "eval_gpus_per_worker": eval_gpus_per_worker,
                "rollout_torch_threads": rollout_torch_threads,
                "runtime_env_id": runtime_env_id,
                "async_pipeline": async_pipeline,
                "training_mode": training_mode,
                "ppo_clip_coef": ppo_clip_coef,
                "ppo_value_clip_coef": ppo_value_clip_coef,
                "ppo_gamma": ppo_gamma,
                "ppo_gae_lambda": ppo_gae_lambda,
                "bc_anchor_weight": bc_anchor_weight,
                "reward_mode": reward_mode,
                "advantage_margin_threshold": advantage_margin_threshold,
                "advantage_value_scale": advantage_value_scale,
                "proposal_enabled": proposal_cfg.get("enabled", True),
                "proposal_num_candidates": proposal_cfg.get("num_candidates"),
                "proposal_send_threshold": proposal_cfg.get("send_threshold"),
                "env_backend": env_backend,
                "env_use_numba": env_use_numba,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    try:
        progress = tqdm(range(iterations), desc="[Stage1.5]", unit="iter")
        for iteration in progress:
            t0 = time.time()
            t_phase = time.time()
            states = ray.get([trainer.state_dict_cpu.remote() for trainer in trainers])
            model_state = _average_state_dicts(states)
            sync_before_rollout_sec = time.time() - t_phase

            t_phase = time.time()
            parts = []
            remaining_games = games_per_iteration
            next_task_id = 0
            in_flight: dict[ray.ObjectRef, int] = {}
            train_refs: list[ray.ObjectRef] | None = None
            train_submit_time = 0.0
            train_submit_to_done_sec = 0.0
            async_train_started = False

            def submit_rollout(actor_idx: int) -> None:
                nonlocal remaining_games, next_task_id
                games = min(games_per_rollout_task, remaining_games)
                if games <= 0:
                    return
                task_id = next_task_id
                next_task_id += 1
                remaining_games -= games
                ref = rollouts[actor_idx].rollout.remote(
                    model_state,
                    games,
                    seed_offset=50_000_000 + iteration * 1_000_000 + task_id * 10_000,
                    four_player_prob=four_player_prob,
                    temperature=temperature,
                    epsilon=epsilon,
                    reward_mode=reward_mode,
                    advantage_margin_threshold=advantage_margin_threshold,
                    advantage_value_scale=advantage_value_scale,
                    min_policy_weight=min_policy_weight,
                    max_policy_weight=max_policy_weight,
                    positive_policy_weight=positive_policy_weight,
                    negative_policy_weight=negative_policy_weight,
                    training_mode=training_mode,
                    gamma=ppo_gamma,
                    gae_lambda=ppo_gae_lambda,
                )
                in_flight[ref] = actor_idx

            for actor_idx in range(min(rollout_workers, games_per_iteration)):
                submit_rollout(actor_idx)
            if async_pipeline and iteration > 0:
                train_submit_time = time.time()
                train_refs = [
                    trainer.update.remote(
                        updates_per_iteration,
                        batch_size,
                        value_weight,
                        entropy_weight,
                        training_mode,
                        ppo_clip_coef,
                        ppo_value_clip_coef,
                        bc_anchor_weight,
                        temperature,
                    )
                    for trainer in trainers
                ]
                async_train_started = True
            while in_flight:
                done_refs, _ = ray.wait(list(in_flight), num_returns=1)
                for ref in done_refs:
                    actor_idx = in_flight.pop(ref)
                    parts.append(ray.get(ref))
                    submit_rollout(actor_idx)
            rollout_sec = time.time() - t_phase

            train_parts: list[dict[str, float]] = []
            train_sec = 0.0
            if train_refs is not None:
                t_phase = time.time()
                train_parts = ray.get(train_refs)
                train_sec = time.time() - t_phase
                train_submit_to_done_sec = time.time() - train_submit_time

            t_phase = time.time()
            rows = [row for part in parts for row in part["rows"]]
            split_rows = [rows[i::trainer_workers] for i in range(trainer_workers)]
            ray.get([trainer.add_rows.remote(split_rows[i], replay_capacity) for i, trainer in enumerate(trainers)])
            add_rows_sec = time.time() - t_phase

            if train_refs is None:
                t_phase = time.time()
                train_parts = ray.get(
                    [
                        trainer.update.remote(
                            updates_per_iteration,
                            batch_size,
                            value_weight,
                            entropy_weight,
                            training_mode,
                            ppo_clip_coef,
                            ppo_value_clip_coef,
                            bc_anchor_weight,
                            temperature,
                        )
                        for trainer in trainers
                    ]
                )
                train_sec = time.time() - t_phase
                train_submit_to_done_sec = train_sec

            t_phase = time.time()
            states = ray.get([trainer.state_dict_cpu.remote() for trainer in trainers])
            model_state = _average_state_dicts(states)
            ray.get([trainer.load_state_dict.remote(model_state) for trainer in trainers])
            sync_after_train_sec = time.time() - t_phase

            log: dict[str, float | int] = {
                "iteration": iteration,
                "stage": args.stage_key,
                "time/iteration_sec": time.time() - t0,
                "time/sync_before_rollout_sec": sync_before_rollout_sec,
                "time/rollout_sec": rollout_sec,
                "time/add_rows_sec": add_rows_sec,
                "time/train_sec": train_sec,
                "time/train_submit_to_done_sec": train_submit_to_done_sec,
                "time/sync_after_train_sec": sync_after_train_sec,
                "runtime/async_pipeline": float(async_pipeline),
                "runtime/async_started": float(async_train_started),
                "rollout/games": sum(p["games"] for p in parts),
                "rollout/rows": len(rows),
                "rollout/wins": sum(p["wins"] for p in parts),
                "rollout/losses": sum(p["losses"] for p in parts),
                "rollout/draws": sum(p["draws"] for p in parts),
                "rollout/win_rate": sum(p["wins"] for p in parts) / max(sum(p["games"] for p in parts), 1),
                "rollout/avg_margin": float(np.mean([p["avg_margin"] for p in parts])) if parts else 0.0,
                "rollout/avg_baseline_margin": float(np.mean([p["avg_baseline_margin"] for p in parts])) if parts else 0.0,
                "rollout/avg_advantage_margin": float(np.mean([p["avg_advantage_margin"] for p in parts])) if parts else 0.0,
                "rollout/positive_advantage_rate": sum(p["positive_advantage_games"] for p in parts)
                / max(sum(p["games"] for p in parts), 1),
                "rollout/selected_proposal": sum(p["selected_proposal"] for p in parts),
                "rollout/selected_oracle": sum(p["selected_oracle"] for p in parts),
                "rollout/fallback_targets": sum(p["fallback_targets"] for p in parts),
                "rollout/workers": rollout_workers,
                "rollout/tasks": len(parts),
                "rollout/games_per_task": games_per_rollout_task,
                "time/rollout_actor_model_game_sec": float(np.mean([p["model_game_sec"] for p in parts]))
                if parts
                else 0.0,
                "time/rollout_actor_baseline_sec": float(np.mean([p["baseline_sec"] for p in parts]))
                if parts
                else 0.0,
                "time/rollout_actor_opponent_rulebase_sec": float(
                    np.mean([p["opponent_rulebase_sec"] for p in parts])
                )
                if parts
                else 0.0,
                "time/rollout_actor_candidate_sec": float(np.mean([p["candidate_sec"] for p in parts]))
                if parts
                else 0.0,
                "time/rollout_actor_encode_sec": float(np.mean([p["encode_sec"] for p in parts]))
                if parts
                else 0.0,
                "time/rollout_actor_model_forward_sec": float(np.mean([p["model_forward_sec"] for p in parts]))
                if parts
                else 0.0,
                "time/rollout_actor_env_step_sec": float(np.mean([p["env_step_sec"] for p in parts]))
                if parts
                else 0.0,
                "runtime/rollout_torch_threads": float(np.mean([p["torch_threads"] for p in parts])) if parts else 0.0,
            }
            for part in train_parts:
                for key, value in part.items():
                    log[key] = float(log.get(key, 0.0)) + float(value)
            for key in list(log):
                if key.startswith("train/") or key.startswith("gpu/"):
                    log[key] = float(log[key]) / max(len(train_parts), 1)
            log.update(
                {
                    "train/trainer_workers": trainer_workers,
                    "train/gpus_per_trainer": gpus_per_trainer,
                    "train/cpus_per_trainer": cpus_per_trainer,
                    "train/weight_synced": 1.0,
                }
            )

            if (iteration + 1) % eval_interval == 0:
                eval_games_per_actor = [eval_games // eval_workers] * eval_workers
                for i in range(eval_games % eval_workers):
                    eval_games_per_actor[i] += 1
                eval_parts = ray.get(
                    [
                        actor.evaluate.remote(
                            model_state,
                            eval_games_per_actor[i],
                            seed_offset=80_000_000 + iteration * 1_000_000,
                            four_player_prob=four_player_prob,
                            device=eval_device,
                        )
                        for i, actor in enumerate(evals)
                        if eval_games_per_actor[i] > 0
                    ]
                )
                games = sum(p["games"] for p in eval_parts)
                wins = sum(p["wins"] for p in eval_parts)
                losses = sum(p["losses"] for p in eval_parts)
                draws = sum(p["draws"] for p in eval_parts)
                win_rate = wins / max(games, 1)
                log.update(
                    {
                        "eval/games": games,
                        "eval/wins": wins,
                        "eval/losses": losses,
                        "eval/draws": draws,
                        "eval/win_rate_vs_regular": win_rate,
                        "eval/avg_margin_vs_regular": float(np.mean([p["avg_margin"] for p in eval_parts])),
                    }
                )
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    ray.get(trainers[0].save.remote(str(output_dir / "best.pt"), iteration, cfg, best_win_rate))

            ray.get(trainers[0].save.remote(str(output_dir / "latest.pt"), iteration, cfg, best_win_rate))
            if swan:
                swan.log(log, step=iteration)
            print(json.dumps(log, ensure_ascii=False), flush=True)
            progress.set_postfix(
                wr=f"{log.get('rollout/win_rate', 0.0):.3f}",
                loss=f"{log.get('train/loss', 0.0):.3f}",
                eval=f"{log.get('eval/win_rate_vs_regular', -1.0):.3f}",
            )
    finally:
        if swan:
            swan.finish()


if __name__ == "__main__":
    main()
