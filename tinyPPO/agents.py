from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyPPO.features import MAX_PLANETS, encode_obs
from tinyPPO.model import TinyPolicyValueNet

MIN_SHIP_FRACTION = 0.02
ACTION_SLOTS = 3
MAX_ACTIONS_PER_SOURCE_SAFETY = ACTION_SLOTS


def _planet_dict(obs: dict[str, Any]) -> dict[int, list]:
    return {int(p[0]): p for p in obs.get("planets", [])}


def actions_from_decisions(obs: dict[str, Any], player: int, source_slots: np.ndarray, target_slots: np.ndarray, ship_slots: np.ndarray) -> list[list]:
    planets = list(obs.get("planets", []))
    remaining = {i: int(p[5]) for i, p in enumerate(planets)}
    actions: list[list] = []
    for src_i, tgt_i, ship_i in zip(source_slots.tolist(), target_slots.tolist(), ship_slots.tolist(), strict=False):
        if src_i < 0 or src_i >= len(planets) or tgt_i < 0 or tgt_i >= len(planets):
            continue
        src = planets[src_i]
        tgt = planets[tgt_i]
        if int(src[1]) != player or int(src[0]) == int(tgt[0]):
            continue
        src_ships = int(src[5])
        available = remaining.get(src_i, 0)
        if src_ships <= 1 or available <= 0:
            continue
        frac = float(np.clip(float(ship_i), MIN_SHIP_FRACTION, 1.0))
        ships = max(1, min(available, int(src_ships * frac)))
        angle = math.atan2(float(tgt[3]) - float(src[3]), float(tgt[2]) - float(src[2]))
        if math.isfinite(angle) and ships > 0:
            actions.append([int(src[0]), angle, ships])
            remaining[src_i] = available - ships
    return actions


def nearest_planet_agent(obs: dict[str, Any], configuration=None) -> list[list]:
    player = int(obs.get("player", 0))
    planets = list(obs.get("planets", []))
    mine = [p for p in planets if int(p[1]) == player]
    targets = [p for p in planets if int(p[1]) != player]
    actions: list[list] = []
    if not targets:
        return actions
    for src in mine:
        target = min(targets, key=lambda p: math.hypot(float(p[2]) - float(src[2]), float(p[3]) - float(src[3])))
        ships_needed = int(float(target[5])) + 1
        if int(src[5]) >= ships_needed:
            angle = math.atan2(float(target[3]) - float(src[3]), float(target[2]) - float(src[2]))
            actions.append([int(src[0]), angle, ships_needed])
    return actions


def random_policy_agent(obs: dict[str, Any], configuration=None) -> list[list]:
    player = int(obs.get("player", 0))
    actions: list[list] = []
    for p in obs.get("planets", []):
        if int(p[1]) != player or int(p[5]) < 20:
            continue
        if random.random() < 0.35:
            actions.append([int(p[0]), random.uniform(-math.pi, math.pi), max(1, int(float(p[5]) * random.uniform(0.25, 0.75)))])
    return actions


class TinyPPOAgent:
    def __init__(self, checkpoint: str | Path, device: str = "cpu", deterministic: bool = True):
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        cfg = payload.get("model", {})
        self.model = TinyPolicyValueNet(**cfg).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device = torch.device(device)
        self.deterministic = deterministic

    @torch.no_grad()
    def __call__(self, obs: dict[str, Any], configuration=None) -> list[list]:
        player = int(obs.get("player", 0))
        enc = encode_obs(obs, player, players=2)
        batch = {
            "planets": torch.tensor(enc.planets[None], dtype=torch.float32, device=self.device),
            "pair_features": torch.tensor(enc.pair_features[None], dtype=torch.float32, device=self.device),
            "global_features": torch.tensor(enc.global_features[None], dtype=torch.float32, device=self.device),
            "planet_mask": torch.tensor(enc.planet_mask[None], dtype=torch.bool, device=self.device),
            "own_mask": torch.tensor(enc.own_mask[None], dtype=torch.bool, device=self.device),
        }
        out = self.model(**batch)
        source_logits = out["source_logits"][0]
        target_logits = out["target_logits"][0]
        ship_params = out["ship_params"][0]
        own_slots = torch.where(batch["own_mask"][0])[0]
        source_slots: list[int] = []
        target_slots: list[int] = []
        ship_slots: list[float] = []
        for src in own_slots.tolist():
            for slot in range(min(source_logits.size(1), MAX_ACTIONS_PER_SOURCE_SAFETY)):
                if self.deterministic:
                    launch = int(torch.argmax(source_logits[src, slot]).item())
                    tgt = int(torch.argmax(target_logits[src, slot]).item())
                    alpha_beta = ship_params[src, slot, tgt]
                    ship = float((alpha_beta[0] / alpha_beta.sum()).clamp(1e-4, 1.0).item())
                else:
                    launch = int(torch.distributions.Categorical(logits=source_logits[src, slot]).sample().item())
                    tgt = int(torch.distributions.Categorical(logits=target_logits[src, slot]).sample().item())
                    alpha_beta = ship_params[src, slot, tgt]
                    ship = float(torch.distributions.Beta(alpha_beta[0], alpha_beta[1]).sample().clamp(1e-4, 1.0 - 1e-4).item())
                if launch == 1:
                    source_slots.append(src)
                    target_slots.append(tgt)
                    ship_slots.append(ship)
        return actions_from_decisions(obs, player, np.array(source_slots), np.array(target_slots), np.array(ship_slots))
