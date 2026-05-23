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
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0


def apply_ship_fraction_bias(alpha_beta: torch.Tensor, bias: float) -> torch.Tensor:
    if abs(bias) <= 1e-9:
        return alpha_beta
    concentration = alpha_beta.sum(dim=-1, keepdim=True).clamp_min(2.0)
    mean = (alpha_beta[..., :1] / concentration).clamp(1e-4, 1.0 - 1e-4)
    shifted = torch.sigmoid(torch.logit(mean) + float(bias)).clamp(1e-4, 1.0 - 1e-4)
    return torch.cat([shifted * concentration, (1.0 - shifted) * concentration], dim=-1).clamp_min(1e-4)


def _fleet_speed(ships: float, max_speed: float = 6.0) -> float:
    if ships <= 1:
        return 1.0
    ratio = min(max(math.log(max(ships, 1.0)) / math.log(1000.0), 0.0), 1.0)
    return 1.0 + (max_speed - 1.0) * (ratio**1.5)


def _segment_distance_sq(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = min(1.0, max(0.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    return (px - cx) ** 2 + (py - cy) ** 2


def _comet_future_position(obs: dict[str, Any], planet_id: int, eta: int) -> tuple[float, float] | None:
    for group in obs.get("comets", []) or []:
        planet_ids = group.get("planet_ids", []) if isinstance(group, dict) else []
        if planet_id not in planet_ids:
            continue
        try:
            idx = list(planet_ids).index(planet_id)
            path = group.get("paths", [])[idx]
            path_index = int(group.get("path_index", 0))
            future_index = min(max(0, path_index + max(0, eta)), len(path) - 1)
            x, y = path[future_index]
            return float(x), float(y)
        except (IndexError, TypeError, ValueError):
            return None
    return None


def _is_orbital_target(obs: dict[str, Any], planet: list) -> bool:
    if int(planet[0]) in set(obs.get("comet_planet_ids", []) or []):
        return False
    return math.hypot(float(planet[2]) - CENTER, float(planet[3]) - CENTER) + float(planet[4]) < ROTATION_RADIUS_LIMIT


def _future_target_position(obs: dict[str, Any], target: list, eta: int) -> tuple[float, float]:
    comet_pos = _comet_future_position(obs, int(target[0]), eta)
    if comet_pos is not None:
        return comet_pos
    if _is_orbital_target(obs, target):
        angle = math.atan2(float(target[3]) - CENTER, float(target[2]) - CENTER)
        radius = math.hypot(float(target[2]) - CENTER, float(target[3]) - CENTER)
        angle += float(obs.get("angular_velocity", 0.0) or 0.0) * max(0, eta)
        return CENTER + math.cos(angle) * radius, CENTER + math.sin(angle) * radius
    return float(target[2]), float(target[3])


def _aim_angle_and_eta(obs: dict[str, Any], source: list, target: list, ships: int) -> tuple[float, int]:
    speed = _fleet_speed(max(1, ships))
    tx = float(target[2])
    ty = float(target[3])
    eta = 1
    for _ in range(3):
        dist = math.hypot(tx - float(source[2]), ty - float(source[3]))
        eta = max(1, int(math.ceil(dist / max(speed, 1e-6))))
        tx, ty = _future_target_position(obs, target, eta)
    return math.atan2(ty - float(source[3]), tx - float(source[2])), eta


def _path_is_safe(source: list, angle: float, ships: int, ticks: int) -> bool:
    speed = _fleet_speed(max(1, ships))
    x = float(source[2]) + math.cos(angle) * (float(source[4]) + 0.1)
    y = float(source[3]) + math.sin(angle) * (float(source[4]) + 0.1)
    for _ in range(max(1, ticks)):
        nx = x + math.cos(angle) * speed
        ny = y + math.sin(angle) * speed
        if _segment_distance_sq(CENTER, CENTER, x, y, nx, ny) < SUN_RADIUS * SUN_RADIUS:
            return False
        if not (0.0 <= nx <= BOARD_SIZE and 0.0 <= ny <= BOARD_SIZE):
            return False
        x, y = nx, ny
    return True


def _approx_pair_ships(source: list, target: list) -> int:
    src_ships = max(1, int(float(source[5])))
    needed = max(1, int(float(target[5]) + 1.0))
    return min(src_ships, needed)


def safe_target_mask(obs: dict[str, Any], player: int) -> np.ndarray:
    planets = list(obs.get("planets", []))[:MAX_PLANETS]
    mask = np.zeros((MAX_PLANETS, MAX_PLANETS), dtype=np.bool_)
    for src_i, src in enumerate(planets):
        if int(src[1]) != player or int(src[5]) <= 1:
            continue
        for tgt_i, tgt in enumerate(planets):
            if src_i == tgt_i or int(src[0]) == int(tgt[0]):
                continue
            ships = _approx_pair_ships(src, tgt)
            angle, eta = _aim_angle_and_eta(obs, src, tgt, ships)
            mask[src_i, tgt_i] = math.isfinite(angle) and _path_is_safe(src, angle, ships, eta + 3)
    return mask


def apply_target_safety_mask(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    obs: dict[str, Any],
    player: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.tensor(safe_target_mask(obs, player), dtype=torch.bool, device=target_logits.device)
    safe_sources = mask.any(dim=-1)
    masked_targets = target_logits.clone().masked_fill(~mask[:, None, :], -1e9)
    if (~safe_sources).any():
        masked_targets[~safe_sources] = target_logits[~safe_sources]
        source_logits = source_logits.clone()
        source_logits[~safe_sources, :, 1] = -1e9
    return source_logits, masked_targets


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
        angle, eta = _aim_angle_and_eta(obs, src, tgt, ships)
        if math.isfinite(angle) and ships > 0 and _path_is_safe(src, angle, ships, eta + 3):
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
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        deterministic: bool = True,
        launch_bias: float = 0.0,
        ship_bias: float = 0.0,
        launch_temperature: float = 1.0,
    ):
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        cfg = payload.get("model", {})
        self.model = TinyPolicyValueNet(**cfg).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.launch_bias = float(launch_bias)
        self.ship_bias = float(ship_bias)
        self.launch_temperature = max(1e-4, float(launch_temperature))

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
        source_logits = out["source_logits"][0] / self.launch_temperature
        if self.launch_bias:
            source_logits = source_logits.clone()
            source_logits[..., 1] += self.launch_bias
        target_logits = out["target_logits"][0]
        ship_params = apply_ship_fraction_bias(out["ship_params"][0], self.ship_bias)
        source_logits, target_logits = apply_target_safety_mask(source_logits, target_logits, obs, player)
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
