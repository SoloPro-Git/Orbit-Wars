"""Observation and candidate-action features for rulebase-aligned training."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from training.expert.action_labeling import infer_target_planet_id

BOARD_SIZE = 100.0
MAX_TURNS = 500
MAX_ENTITIES = 64
PLANET_DIM = 21
ACTION_DIM = 32
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0


def _fleet_speed(ships: float, max_speed: float = 6.0) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(max(ships, 1.0)) / math.log(1000.0)
    ratio = min(max(ratio, 0.0), 1.0)
    return 1.0 + (max_speed - 1.0) * (ratio**1.5)


def _score(obs: dict[str, Any], player: int) -> float:
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    return float(
        sum(p[5] for p in planets if int(p[1]) == player)
        + sum(f[6] for f in fleets if int(f[1]) == player)
    )


def _segment_distance_sq(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = min(1.0, max(0.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy


def _sun_collision(src: list, angle: float, ships: float, ticks: int | None = None) -> bool:
    speed = _fleet_speed(ships)
    if ticks is None:
        ticks = int(math.ceil(BOARD_SIZE * 2.0 / max(speed, 1e-6)))
    ax = float(src[2])
    ay = float(src[3])
    bx = ax + math.cos(angle) * speed * max(1, ticks)
    by = ay + math.sin(angle) * speed * max(1, ticks)
    return _segment_distance_sq(CENTER_X, CENTER_Y, ax, ay, bx, by) <= SUN_RADIUS * SUN_RADIUS


def _is_orbital_planet(obs: dict[str, Any], planet: list) -> bool:
    if int(planet[0]) in set(obs.get("comet_planet_ids", [])):
        return False
    return math.hypot(float(planet[2]) - CENTER_X, float(planet[3]) - CENTER_Y) + float(planet[4]) < 50.0


def _comet_remaining(obs: dict[str, Any], planet_id: int) -> int | None:
    for group in obs.get("comets", []) or []:
        planet_ids = group.get("planet_ids", []) if isinstance(group, dict) else []
        if planet_id not in planet_ids:
            continue
        paths = group.get("paths", [])
        path_index = int(group.get("path_index", 0))
        try:
            idx = list(planet_ids).index(planet_id)
            return max(0, len(paths[idx]) - path_index - 1)
        except (ValueError, IndexError, TypeError):
            return None
    return None


def _nearest_enemy_eta(planets: list, player: int, target: list, ships_hint: float) -> float:
    best = 1e9
    for planet in planets:
        if int(planet[1]) in (-1, player):
            continue
        send = max(1.0, min(float(planet[5]), max(1.0, ships_hint)))
        dist = math.hypot(float(target[2]) - float(planet[2]), float(target[3]) - float(planet[3]))
        best = min(best, dist / _fleet_speed(send))
    return best


def _incoming_pressure(obs: dict[str, Any], player: int, target: list, horizon: float) -> tuple[float, float]:
    friendly = 0.0
    enemy = 0.0
    tx = float(target[2])
    ty = float(target[3])
    radius = float(target[4]) + 1.5
    for fleet in obs.get("fleets", []):
        speed = _fleet_speed(float(fleet[6]))
        dx = tx - float(fleet[2])
        dy = ty - float(fleet[3])
        vx = math.cos(float(fleet[4]))
        vy = math.sin(float(fleet[4]))
        proj = dx * vx + dy * vy
        if proj <= 0:
            continue
        eta = proj / max(speed, 1e-6)
        if eta > horizon:
            continue
        perp = abs(dx * vy - dy * vx)
        if perp > radius:
            continue
        if int(fleet[1]) == player:
            friendly += float(fleet[6])
        elif int(fleet[1]) != -1:
            enemy += float(fleet[6])
    return friendly, enemy


def result_value(obs: dict[str, Any], player: int) -> float:
    """Final value target in [-1, 1] from score rank."""
    players = sorted({int(p[1]) for p in obs.get("planets", []) if int(p[1]) >= 0})
    players = sorted(set(players) | {0, 1})
    scores = [(pid, _score(obs, pid)) for pid in players]
    scores.sort(key=lambda row: row[1], reverse=True)
    rank = next((i for i, (pid, _) in enumerate(scores) if pid == player), len(scores) - 1)
    if len(scores) <= 1:
        return 0.0
    return 1.0 - 2.0 * rank / (len(scores) - 1)


@dataclass(frozen=True)
class EncodedPosition:
    planet_features: np.ndarray
    global_features: np.ndarray
    candidate_features: np.ndarray
    candidate_mask: np.ndarray


def encode_planets(
    obs: dict[str, Any],
    player: int,
    max_planets: int | None = None,
    *,
    include_fleets: bool = True,
) -> np.ndarray:
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", []) if include_fleets else []
    n = min(MAX_ENTITIES, len(planets) + len(fleets)) if max_planets is None else max_planets
    out = np.zeros((n, PLANET_DIM), dtype=np.float32)
    comet_ids = set(obs.get("comet_planet_ids", []))
    step = float(obs.get("step", 0))
    initial = {int(p[0]): p for p in obs.get("initial_planets", [])}

    for i, p in enumerate(planets[:n]):
        pid, owner, x, y, radius, ships, production = p
        owner = int(owner)
        if owner == player:
            out[i, 0] = 1.0
        elif owner == -1:
            out[i, 1] = 1.0
        else:
            out[i, 2] = 1.0
        out[i, 3] = (float(x) - 50.0) / 50.0
        out[i, 4] = (float(y) - 50.0) / 50.0
        out[i, 5] = float(radius) / 10.0
        out[i, 6] = math.log1p(float(ships)) / math.log(1000.0)
        out[i, 7] = float(production) / 5.0
        out[i, 8] = float(pid in comet_ids)
        dist_sun = math.hypot(float(x) - 50.0, float(y) - 50.0)
        out[i, 9] = dist_sun / 50.0
        out[i, 10] = float((dist_sun + float(radius)) < 50.0 and pid not in comet_ids)
        out[i, 11] = step / MAX_TURNS
        out[i, 12] = float(obs.get("angular_velocity", 0.0)) * 20.0
        init = initial.get(int(pid))
        if init is not None:
            out[i, 13] = (float(init[2]) - 50.0) / 50.0
            out[i, 14] = (float(init[3]) - 50.0) / 50.0
        out[i, 15] = float(pid) / 64.0
        out[i, 16] = 1.0
        out[i, 17] = 1.0

    offset = min(len(planets), n)
    for j, fleet in enumerate(fleets[: max(0, n - offset)]):
        i = offset + j
        _, owner, x, y, angle, from_planet_id, ships = fleet
        owner = int(owner)
        if owner == player:
            out[i, 0] = 1.0
        elif owner != -1:
            out[i, 2] = 1.0
        out[i, 3] = (float(x) - 50.0) / 50.0
        out[i, 4] = (float(y) - 50.0) / 50.0
        out[i, 5] = 0.1
        out[i, 6] = math.log1p(float(ships)) / math.log(1000.0)
        out[i, 9] = math.hypot(float(x) - 50.0, float(y) - 50.0) / 50.0
        out[i, 11] = step / MAX_TURNS
        out[i, 12] = float(obs.get("angular_velocity", 0.0)) * 20.0
        out[i, 15] = float(from_planet_id) / 64.0
        out[i, 16] = 1.0
        out[i, 18] = 1.0
        out[i, 19] = math.cos(float(angle))
        out[i, 20] = math.sin(float(angle))
    return out


def encode_global(obs: dict[str, Any], player: int) -> np.ndarray:
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    own_prod = sum(float(p[6]) for p in planets if int(p[1]) == player)
    enemy_prod = sum(float(p[6]) for p in planets if int(p[1]) not in (-1, player))
    own_score = _score(obs, player)
    enemy_scores = [_score(obs, pid) for pid in range(4) if pid != player]
    return np.array(
        [
            float(obs.get("step", 0)) / MAX_TURNS,
            float(obs.get("angular_velocity", 0.0)) * 20.0,
            math.log1p(own_score) / math.log(3000.0),
            math.log1p(max(enemy_scores) if enemy_scores else 0.0) / math.log(3000.0),
            own_prod / 40.0,
            enemy_prod / 80.0,
            len([p for p in planets if int(p[1]) == player]) / 40.0,
            len(fleets) / 128.0,
        ],
        dtype=np.float32,
    )


def encode_action(obs: dict[str, Any], player: int, action: list[list]) -> np.ndarray:
    planet_list = obs.get("planets", [])
    planets = {int(p[0]): p for p in planet_list}
    total_ships = sum(float(a[2]) for a in action if len(a) >= 3)
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    out[0] = float(len(action)) / 12.0
    out[1] = math.log1p(total_ships) / math.log(1000.0)
    if not action:
        out[13] = 1.0
        return out

    source_prod = 0.0
    source_ships = 0.0
    target_prod = 0.0
    target_ships = 0.0
    enemy_targets = 0
    neutral_targets = 0
    comet_targets = 0
    eta_sum = 0.0
    labelled = 0
    source_remaining_ratios: list[float] = []
    send_ratios: list[float] = []
    surplus_vals: list[float] = []
    roi_vals: list[float] = []
    target_value_vals: list[float] = []
    enemy_gap_vals: list[float] = []
    own_gap_vals: list[float] = []
    sun_hits = 0
    moving_targets = 0
    comet_life_margins: list[float] = []
    incoming_pressure_vals: list[float] = []
    eta_by_target: dict[int, list[float]] = {}
    comet_ids = set(obs.get("comet_planet_ids", []))
    for from_id, angle, ships in action:
        src = planets.get(int(from_id))
        if src is None:
            continue
        src_ships = max(float(src[5]), 1.0)
        ships_f = max(float(ships), 1.0)
        source_prod += float(src[6])
        source_ships += float(src[5])
        source_remaining_ratios.append(max(0.0, src_ships - ships_f) / src_ships)
        send_ratios.append(min(1.0, ships_f / src_ships))
        target_id = infer_target_planet_id(obs, int(from_id), float(angle), float(ships))
        if target_id is None:
            if _sun_collision(src, float(angle), ships_f):
                sun_hits += 1
            continue
        target = planets.get(int(target_id))
        if target is None:
            continue
        labelled += 1
        dx = float(target[2]) - float(src[2])
        dy = float(target[3]) - float(src[3])
        eta = math.hypot(dx, dy) / _fleet_speed(ships_f)
        eta_sum += eta
        eta_by_target.setdefault(int(target_id), []).append(eta)
        target_prod += float(target[6])
        target_ships += float(target[5])
        if int(target[1]) == -1:
            neutral_targets += 1
        elif int(target[1]) != player:
            enemy_targets += 1
        if target_id in comet_ids:
            comet_targets += 1
            remaining = _comet_remaining(obs, int(target_id))
            if remaining is not None:
                comet_life_margins.append((float(remaining) - eta) / 80.0)
        if _is_orbital_planet(obs, target):
            moving_targets += 1
        if _sun_collision(src, float(angle), ships_f, ticks=int(math.ceil(max(1.0, eta + 3.0)))):
            sun_hits += 1

        required = float(target[5]) + 1.0
        if int(target[1]) not in (-1, player):
            required += float(target[6]) * eta
        friendly, enemy = _incoming_pressure(obs, player, target, eta + 3.0)
        required += max(0.0, enemy - friendly)
        surplus_vals.append((ships_f - required) / 100.0)
        remaining_turns = max(0.0, float(MAX_TURNS - obs.get("step", 0)) - eta)
        economic_value = float(target[6]) * remaining_turns
        target_value_vals.append(economic_value / 500.0)
        roi_vals.append((economic_value - ships_f) / max(ships_f, 1.0) / 10.0)
        enemy_eta = _nearest_enemy_eta(planet_list, player, target, max(1.0, float(target[5]) + 1.0))
        if enemy_eta < 1e8:
            enemy_gap_vals.append((enemy_eta - eta) / 80.0)
        own_etas = []
        for planet in planet_list:
            if int(planet[1]) != player or int(planet[0]) == int(from_id):
                continue
            dist = math.hypot(float(target[2]) - float(planet[2]), float(target[3]) - float(planet[3]))
            own_etas.append(dist / _fleet_speed(max(1.0, min(float(planet[5]), ships_f))))
        if own_etas:
            own_gap_vals.append((min(own_etas) - eta) / 80.0)
        incoming_pressure_vals.append((enemy - friendly) / max(ships_f, 1.0))

    denom = max(len(action), 1)
    labelled_denom = max(labelled, 1)
    out[2] = source_prod / (5.0 * denom)
    out[3] = math.log1p(source_ships) / math.log(2000.0)
    out[4] = target_prod / (5.0 * labelled_denom)
    out[5] = math.log1p(target_ships) / math.log(2000.0)
    out[6] = enemy_targets / denom
    out[7] = neutral_targets / denom
    out[8] = comet_targets / denom
    out[9] = eta_sum / (80.0 * labelled_denom)
    out[10] = total_ships / max(_score(obs, player), 1.0)
    out[11] = float(labelled) / denom
    out[12] = float(len({int(a[0]) for a in action if len(a) >= 3})) / denom
    if source_remaining_ratios:
        out[14] = float(np.mean(source_remaining_ratios))
        out[15] = float(np.min(source_remaining_ratios))
    if send_ratios:
        out[16] = float(np.mean(send_ratios))
        out[17] = float(np.max(send_ratios))
    if surplus_vals:
        out[18] = float(np.mean(surplus_vals))
        out[19] = float(np.min(surplus_vals))
    if roi_vals:
        out[20] = float(np.mean(roi_vals))
    if target_value_vals:
        out[21] = float(np.mean(target_value_vals))
    if enemy_gap_vals:
        out[22] = float(np.mean(enemy_gap_vals))
        out[23] = float(np.min(enemy_gap_vals))
    if own_gap_vals:
        out[24] = float(np.mean(own_gap_vals))
    out[25] = float(sun_hits) / denom
    out[26] = float(max(0, len(action) - labelled)) / denom
    out[27] = float(moving_targets) / denom
    if comet_life_margins:
        out[28] = float(np.mean(comet_life_margins))
    if incoming_pressure_vals:
        out[29] = float(np.mean(incoming_pressure_vals))
    multi_source_targets = sum(1 for etas in eta_by_target.values() if len(etas) > 1)
    out[30] = float(multi_source_targets) / max(len(eta_by_target), 1)
    eta_spreads = [float(np.std(etas)) for etas in eta_by_target.values() if len(etas) > 1]
    if eta_spreads:
        out[31] = float(np.mean(eta_spreads)) / 20.0
    return out


def encode_position(
    obs: dict[str, Any],
    player: int,
    candidates: list[list[list]],
    max_candidates: int = 32,
) -> EncodedPosition:
    cand = np.zeros((max_candidates, ACTION_DIM), dtype=np.float32)
    mask = np.zeros(max_candidates, dtype=np.float32)
    for i, action in enumerate(candidates[:max_candidates]):
        cand[i] = encode_action(obs, player, action)
        mask[i] = 1.0
    return EncodedPosition(
        planet_features=encode_planets(obs, player),
        global_features=encode_global(obs, player),
        candidate_features=cand,
        candidate_mask=mask,
    )
