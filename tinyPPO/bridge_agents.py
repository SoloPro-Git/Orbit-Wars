from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyPPO.features import encode_obs
from tinyPPO.model import TinyPolicyValueNet
from training2.rulebase_bridge import make_rulebase_agent


class RegularSourceBridgeAgent:
    """Use a learned source gate while keeping regular target/ship decisions.

    The BC/PPO model is currently much more trustworthy on "which source should
    act" than on target and ship amount. This bridge lets the rulebase produce
    legal actions, then only allows the model to remove low-confidence source
    planets from that action set.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        threshold: float = 0.30,
        apply_prob: float = 0.02,
        max_source_drops: int | None = None,
        max_drop_frac: float = 1.0,
        min_keep_actions: int = 1,
        min_anchor_actions_to_filter: int = 1,
        launch_bias: float = 0.0,
        launch_temperature: float = 1.0,
        reduce: str = "noisy_or",
        anchor: str = "regular",
    ):
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        cfg = payload.get("model", {})
        self.model = TinyPolicyValueNet(**cfg).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device = torch.device(device)
        self.anchor = make_rulebase_agent(anchor)
        self.threshold = float(threshold)
        self.apply_prob = max(0.0, min(1.0, float(apply_prob)))
        self.max_source_drops = None if max_source_drops is None or max_source_drops < 0 else int(max_source_drops)
        self.max_drop_frac = max(0.0, min(1.0, float(max_drop_frac)))
        self.min_keep_actions = max(0, int(min_keep_actions))
        self.min_anchor_actions_to_filter = max(1, int(min_anchor_actions_to_filter))
        self.launch_bias = float(launch_bias)
        self.launch_temperature = max(1e-4, float(launch_temperature))
        if reduce not in {"noisy_or", "max", "mean"}:
            raise ValueError(f"unknown source probability reducer: {reduce}")
        self.reduce = reduce
        self.stats = {
            "calls": 0,
            "attempted": 0,
            "anchor_actions": 0,
            "kept_actions": 0,
            "dropped_actions": 0,
        }

    @torch.no_grad()
    def source_probs(self, obs: dict[str, Any], player: int) -> dict[int, float]:
        enc = encode_obs(obs, player, players=2)
        batch = {
            "planets": torch.tensor(enc.planets[None], dtype=torch.float32, device=self.device),
            "pair_features": torch.tensor(enc.pair_features[None], dtype=torch.float32, device=self.device),
            "global_features": torch.tensor(enc.global_features[None], dtype=torch.float32, device=self.device),
            "planet_mask": torch.tensor(enc.planet_mask[None], dtype=torch.bool, device=self.device),
            "own_mask": torch.tensor(enc.own_mask[None], dtype=torch.bool, device=self.device),
        }
        out = self.model(**batch)
        logits = out["source_logits"][0] / self.launch_temperature
        if self.launch_bias:
            logits = logits.clone()
            logits[..., 1] += self.launch_bias
        slot_probs = torch.softmax(logits, dim=-1)[..., 1].clamp(0.0, 1.0)
        if self.reduce == "max":
            probs = slot_probs.max(dim=-1).values
        elif self.reduce == "mean":
            probs = slot_probs.mean(dim=-1)
        else:
            probs = 1.0 - torch.prod(1.0 - slot_probs, dim=-1)
        return {pid: float(probs[i].item()) for i, pid in enumerate(enc.planet_ids)}

    def __call__(self, obs: dict[str, Any], configuration=None) -> list[list]:
        del configuration
        self.stats["calls"] += 1
        actions = list(self.anchor(obs) or [])
        self.stats["anchor_actions"] += len(actions)
        if (
            not actions
            or len(actions) < self.min_anchor_actions_to_filter
            or self.apply_prob <= 0.0
            or random.random() >= self.apply_prob
        ):
            self.stats["kept_actions"] += len(actions)
            return actions

        self.stats["attempted"] += 1
        player = int(obs.get("player", 0))
        probs = self.source_probs(obs, player)
        action_scores = [(idx, probs.get(int(action[0]), 0.0)) for idx, action in enumerate(actions)]
        drop = {idx for idx, score in action_scores if score < self.threshold}

        if self.max_source_drops is not None and len(drop) > self.max_source_drops:
            ranked = sorted((score, idx) for idx, score in action_scores if idx in drop)
            drop = {idx for _score, idx in ranked[: self.max_source_drops]}

        max_frac_drops = int(np.floor(len(actions) * self.max_drop_frac))
        if len(drop) > max_frac_drops:
            ranked = sorted((score, idx) for idx, score in action_scores if idx in drop)
            drop = {idx for _score, idx in ranked[:max_frac_drops]}

        max_droppable = max(0, len(actions) - self.min_keep_actions)
        if len(drop) > max_droppable:
            ranked = sorted((score, idx) for idx, score in action_scores if idx in drop)
            drop = {idx for _score, idx in ranked[:max_droppable]}

        kept = [action for idx, action in enumerate(actions) if idx not in drop]
        self.stats["kept_actions"] += len(kept)
        self.stats["dropped_actions"] += len(actions) - len(kept)
        return kept


def aggregate_bridge_stats(agents: list[RegularSourceBridgeAgent]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for agent in agents:
        for key, value in agent.stats.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    anchor = max(1.0, totals.get("anchor_actions", 0.0))
    calls = max(1.0, totals.get("calls", 0.0))
    totals["kept_frac"] = totals.get("kept_actions", 0.0) / anchor
    totals["dropped_frac"] = totals.get("dropped_actions", 0.0) / anchor
    totals["attempt_frac"] = totals.get("attempted", 0.0) / calls
    return totals


class RegularSourceDropGateAgent:
    """Trainable categorical source-delete bridge.

    At each step the rulebase proposes legal actions. The policy chooses one of:
    keep all actions, or drop exactly one proposed action. This keeps the action
    space tiny and lets the model make a real tactical choice without exposing
    the immature target/ship heads.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        no_drop_bias: float = 2.0,
        min_anchor_actions_to_filter: int = 4,
        deterministic: bool = False,
        anchor: str = "regular",
    ):
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        cfg = payload.get("model", {})
        self.model = TinyPolicyValueNet(**cfg).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device = torch.device(device)
        self.anchor = make_rulebase_agent(anchor)
        self.no_drop_bias = float(no_drop_bias)
        self.min_anchor_actions_to_filter = max(1, int(min_anchor_actions_to_filter))
        self.deterministic = bool(deterministic)
        self.stats = {
            "calls": 0,
            "attempted": 0,
            "anchor_actions": 0,
            "kept_actions": 0,
            "dropped_actions": 0,
        }

    @torch.no_grad()
    def __call__(self, obs: dict[str, Any], configuration=None) -> list[list]:
        del configuration
        self.stats["calls"] += 1
        actions = list(self.anchor(obs) or [])
        self.stats["anchor_actions"] += len(actions)
        if len(actions) < self.min_anchor_actions_to_filter:
            self.stats["kept_actions"] += len(actions)
            return actions
        player = int(obs.get("player", 0))
        enc = encode_obs(obs, player, players=2)
        id_to_idx = {pid: idx for idx, pid in enumerate(enc.planet_ids)}
        source_indices = [id_to_idx.get(int(action[0]), -1) for action in actions]
        if any(idx < 0 for idx in source_indices):
            self.stats["kept_actions"] += len(actions)
            return actions

        batch = {
            "planets": torch.tensor(enc.planets[None], dtype=torch.float32, device=self.device),
            "pair_features": torch.tensor(enc.pair_features[None], dtype=torch.float32, device=self.device),
            "global_features": torch.tensor(enc.global_features[None], dtype=torch.float32, device=self.device),
            "planet_mask": torch.tensor(enc.planet_mask[None], dtype=torch.bool, device=self.device),
            "own_mask": torch.tensor(enc.own_mask[None], dtype=torch.bool, device=self.device),
        }
        out = self.model(**batch)
        logits = drop_gate_logits(out["source_logits"][0], source_indices, self.no_drop_bias)
        choice = int(torch.argmax(logits).item()) if self.deterministic else int(torch.distributions.Categorical(logits=logits).sample().item())
        self.stats["attempted"] += 1
        if choice <= 0:
            self.stats["kept_actions"] += len(actions)
            return actions
        drop_idx = choice - 1
        kept = [action for idx, action in enumerate(actions) if idx != drop_idx]
        self.stats["kept_actions"] += len(kept)
        self.stats["dropped_actions"] += 1
        return kept


def drop_gate_logits(source_logits: torch.Tensor, source_indices: list[int] | np.ndarray, no_drop_bias: float = 2.0) -> torch.Tensor:
    if isinstance(source_indices, np.ndarray):
        source_indices = [int(x) for x in source_indices.tolist()]
    if not source_indices:
        return torch.tensor([float(no_drop_bias)], dtype=source_logits.dtype, device=source_logits.device)
    idx = torch.tensor(source_indices, dtype=torch.long, device=source_logits.device)
    selected = source_logits[idx]
    keep = torch.logsumexp(selected[..., 1], dim=-1)
    drop = torch.logsumexp(selected[..., 0], dim=-1)
    drop_logits = drop - keep
    no_drop = torch.tensor([float(no_drop_bias)], dtype=source_logits.dtype, device=source_logits.device)
    return torch.cat([no_drop, drop_logits], dim=0)


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int | None]:
    out: list[int | None] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        out.append(None if parsed < 0 else parsed)
    return out
