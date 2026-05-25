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
SHIP_BUCKET_MULTIPLIERS = (0.75, 1.0, 1.25, 1.5, 2.0)
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MOVING_AIM_TICKS = 120


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


def _target_moves(obs: dict[str, Any], target: list) -> bool:
    return _comet_future_position(obs, int(target[0]), 1) is not None or _is_orbital_target(obs, target)


def _aim_angle_and_eta(obs: dict[str, Any], source: list, target: list, ships: int) -> tuple[float, int]:
    speed = _fleet_speed(max(1, ships))
    sx = float(source[2])
    sy = float(source[3])
    if _target_moves(obs, target):
        best: tuple[float, float, int] | None = None
        for tick in range(1, MOVING_AIM_TICKS + 1):
            tx, ty = _future_target_position(obs, target, tick)
            dist_to_target = max(0.0, math.hypot(tx - sx, ty - sy) - float(source[4]))
            err = abs(speed * tick - dist_to_target)
            if best is None or err < best[0]:
                best = (err, math.atan2(ty - sy, tx - sx), tick)
            if err <= float(target[4]):
                return math.atan2(ty - sy, tx - sx), tick
        return (float("nan"), 1) if best is None else (float("nan"), best[2])
    dist = max(0.0, math.hypot(float(target[2]) - sx, float(target[3]) - sy) - float(source[4]))
    eta = max(1, int(math.floor(dist / max(speed, 1e-6))))
    return math.atan2(float(target[3]) - sy, float(target[2]) - sx), eta


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


def _incoming_by_planet(obs: dict[str, Any], player: int, horizon: int = 80) -> tuple[dict[int, float], dict[int, float]]:
    planets = list(obs.get("planets", []))
    incoming_friend = {int(p[0]): 0.0 for p in planets}
    incoming_enemy = {int(p[0]): 0.0 for p in planets}
    for fleet in obs.get("fleets", []) or []:
        owner = int(fleet[1])
        fx, fy, angle, ships = float(fleet[2]), float(fleet[3]), float(fleet[4]), float(fleet[6])
        speed = _fleet_speed(ships)
        vx, vy = math.cos(angle), math.sin(angle)
        for planet in planets:
            dx = float(planet[2]) - fx
            dy = float(planet[3]) - fy
            proj = dx * vx + dy * vy
            if proj <= 0.0:
                continue
            eta = proj / max(speed, 1e-6)
            if eta > horizon:
                continue
            perp = abs(dx * vy - dy * vx)
            if perp <= float(planet[4]) + 1.0:
                key = int(planet[0])
                if owner == player:
                    incoming_friend[key] += ships
                elif owner != -1:
                    incoming_enemy[key] += ships
    return incoming_friend, incoming_enemy


def required_ships(
    obs: dict[str, Any],
    player: int,
    source: list,
    target: list,
    incoming: tuple[dict[int, float], dict[int, float]] | None = None,
) -> int:
    incoming_friend, incoming_enemy = incoming if incoming is not None else _incoming_by_planet(obs, player)
    src_ships = max(1, int(float(source[5])))
    base = max(1.0, float(target[5]) + 1.0 + incoming_enemy.get(int(target[0]), 0.0) - incoming_friend.get(int(target[0]), 0.0))
    speed = _fleet_speed(min(src_ships, max(1, int(base))))
    dist = max(0.0, math.hypot(float(target[2]) - float(source[2]), float(target[3]) - float(source[3])) - float(source[4]))
    eta = dist / max(speed, 1e-6)
    if int(target[1]) not in (-1, player):
        base += min(80.0, eta + 2.0) * float(target[6])
    return max(1, int(math.ceil(base)))


def _target_candidate_score(obs: dict[str, Any], player: int, source: list, target: list, required: int) -> float:
    if int(target[0]) == int(source[0]):
        return -1e9
    available = max(0, int(float(source[5])))
    if available < max(1, int(required * SHIP_BUCKET_MULTIPLIERS[0])):
        return -1e9
    dist = math.hypot(float(target[2]) - float(source[2]), float(target[3]) - float(source[3]))
    eta = dist / max(_fleet_speed(max(1, required)), 1e-6)
    owner = int(target[1])
    prod = float(target[6])
    score = 100.0 - dist + 15.0 * prod - 0.7 * required - 2.0 * eta
    if owner not in (-1, player):
        score += 10.0 * prod
    if owner == -1 and float(obs.get("step", 0)) <= 80 and prod >= 3.0 and required <= 20:
        score += 18.0
    if _target_moves(obs, target):
        score -= max(0.0, eta - 24.0) * 2.0
    return score


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


def candidate_target_mask(obs: dict[str, Any], player: int, top_k: int = 6, include_friendly: bool = False) -> np.ndarray:
    planets = list(obs.get("planets", []))[:MAX_PLANETS]
    mask = np.zeros((MAX_PLANETS, MAX_PLANETS), dtype=np.bool_)
    top_k = max(1, int(top_k))
    incoming = _incoming_by_planet(obs, player)
    for src_i, src in enumerate(planets):
        if int(src[1]) != player or int(src[5]) <= 1:
            continue
        scored: list[tuple[float, int]] = []
        for tgt_i, tgt in enumerate(planets):
            if int(tgt[1]) == player and not include_friendly:
                continue
            needed = required_ships(obs, player, src, tgt, incoming=incoming)
            score = _target_candidate_score(obs, player, src, tgt, needed)
            if int(tgt[1]) == player and include_friendly and int(tgt[0]) != int(src[0]):
                friend_in, enemy_in = incoming
                pressure = enemy_in.get(int(tgt[0]), 0.0) - friend_in.get(int(tgt[0]), 0.0)
                dist = math.hypot(float(tgt[2]) - float(src[2]), float(tgt[3]) - float(src[3]))
                score = 35.0 + 2.0 * pressure + float(tgt[6]) - 0.5 * dist
            if score > -1e8:
                scored.append((score, tgt_i))
        scored.sort(reverse=True)
        for _score, tgt_i in scored[: max(top_k * 4, top_k)]:
            tgt = planets[tgt_i]
            ships = min(max(1, int(float(src[5]))), max(1, required_ships(obs, player, src, tgt, incoming=incoming)))
            angle, eta = _aim_angle_and_eta(obs, src, tgt, ships)
            if not (math.isfinite(angle) and _path_is_safe(src, angle, ships, eta + 3)):
                continue
            mask[src_i, tgt_i] = True
            if int(mask[src_i].sum()) >= top_k:
                break
    return mask


def apply_target_safety_mask(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    obs: dict[str, Any],
    player: int,
    target_mask: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.tensor(target_mask if target_mask is not None else safe_target_mask(obs, player), dtype=torch.bool, device=target_logits.device)
    safe_sources = mask.any(dim=-1)
    masked_targets = target_logits.clone().masked_fill(~mask[:, None, :], -1e9)
    if (~safe_sources).any():
        masked_targets[~safe_sources] = target_logits[~safe_sources]
        source_logits = source_logits.clone()
        source_logits[~safe_sources, :, 1] = -1e9
    return source_logits, masked_targets


def actions_from_decisions(
    obs: dict[str, Any],
    player: int,
    source_slots: np.ndarray,
    target_slots: np.ndarray,
    ship_slots: np.ndarray,
    ship_mode: str = "fraction",
) -> list[list]:
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
        if ship_mode == "required_bucket":
            bucket = int(np.clip(int(ship_i), 0, len(SHIP_BUCKET_MULTIPLIERS) - 1))
            needed = required_ships(obs, player, src, tgt)
            ships = max(1, int(math.ceil(needed * SHIP_BUCKET_MULTIPLIERS[bucket])))
            ships = min(available, ships)
        else:
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
        target_top_k: int = 6,
        include_friendly_targets: bool = False,
        target_mask_mode: str = "candidate",
    ):
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        cfg = payload.get("model", {})
        self.model = TinyPolicyValueNet(**cfg).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.ship_mode = "required_bucket" if int(cfg.get("ship_buckets", 0) or 0) > 0 else "fraction"
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.launch_bias = float(launch_bias)
        self.ship_bias = float(ship_bias)
        self.launch_temperature = max(1e-4, float(launch_temperature))
        self.target_top_k = max(1, int(target_top_k))
        self.include_friendly_targets = bool(include_friendly_targets)
        if target_mask_mode not in {"candidate", "safe"}:
            raise ValueError(f"unsupported target_mask_mode: {target_mask_mode!r}")
        self.target_mask_mode = target_mask_mode

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
        if self.ship_mode == "required_bucket" and self.target_mask_mode == "candidate":
            target_mask = candidate_target_mask(obs, player, top_k=self.target_top_k, include_friendly=self.include_friendly_targets)
        else:
            target_mask = safe_target_mask(obs, player)
        source_logits, target_logits = apply_target_safety_mask(source_logits, target_logits, obs, player, target_mask=target_mask)
        ship_params = apply_ship_fraction_bias(out["ship_params"][0], self.ship_bias) if "ship_params" in out else None
        ship_logits = out.get("ship_logits")
        own_slots = torch.where(batch["own_mask"][0])[0]
        source_slots: list[int] = []
        target_slots: list[int] = []
        ship_slots: list[float] = []
        for src in own_slots.tolist():
            for slot in range(min(source_logits.size(1), MAX_ACTIONS_PER_SOURCE_SAFETY)):
                if self.deterministic:
                    launch = int(torch.argmax(source_logits[src, slot]).item())
                    tgt = int(torch.argmax(target_logits[src, slot]).item())
                    if ship_logits is not None:
                        ship = float(torch.argmax(ship_logits[0, src, slot, tgt]).item())
                    else:
                        alpha_beta = ship_params[src, slot, tgt]
                        ship = float((alpha_beta[0] / alpha_beta.sum()).clamp(1e-4, 1.0).item())
                else:
                    launch = int(torch.distributions.Categorical(logits=source_logits[src, slot]).sample().item())
                    tgt = int(torch.distributions.Categorical(logits=target_logits[src, slot]).sample().item())
                    if ship_logits is not None:
                        ship = float(torch.distributions.Categorical(logits=ship_logits[0, src, slot, tgt]).sample().item())
                    else:
                        alpha_beta = ship_params[src, slot, tgt]
                        ship = float(torch.distributions.Beta(alpha_beta[0], alpha_beta[1]).sample().clamp(1e-4, 1.0 - 1e-4).item())
                if launch == 1:
                    source_slots.append(src)
                    target_slots.append(tgt)
                    ship_slots.append(ship)
        return actions_from_decisions(obs, player, np.array(source_slots), np.array(target_slots), np.array(ship_slots), ship_mode=self.ship_mode)
