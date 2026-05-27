from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

BOARD_SIZE = 100.0
CENTER = 50.0
MAX_STEPS = 500.0
MAX_PLANETS = 64
PLANET_FEAT_DIM = 18
PAIR_FEAT_DIM = 24
GLOBAL_FEAT_DIM = 8


@dataclass(frozen=True)
class EncodedObs:
    planets: np.ndarray
    pair_features: np.ndarray
    global_features: np.ndarray
    planet_mask: np.ndarray
    own_mask: np.ndarray
    source_xy: np.ndarray
    planet_ids: list[int]


def score(obs: dict[str, Any], player: int) -> float:
    return float(
        sum(float(p[5]) for p in obs.get("planets", []) if int(p[1]) == player)
        + sum(float(f[6]) for f in obs.get("fleets", []) if int(f[1]) == player)
    )


def final_result(obs: dict[str, Any], player: int, players: int = 2) -> float:
    scores = [score(obs, pid) for pid in range(players)]
    best = max(scores)
    if best <= 0.0:
        return -1.0
    return 1.0 if scores[player] >= best and scores.count(best) == 1 else -1.0


def fleet_speed(ships: float, max_speed: float = 6.0) -> float:
    if ships <= 1:
        return 1.0
    ratio = min(max(math.log(max(ships, 1.0)) / math.log(1000.0), 0.0), 1.0)
    return 1.0 + (max_speed - 1.0) * (ratio**1.5)


def _required_ships_hint(
    target: list,
    player: int,
    arrival: float,
    incoming_friend: float,
    incoming_enemy: float,
) -> float:
    base = max(1.0, float(target[5]) + 1.0 + incoming_enemy - incoming_friend)
    if int(target[1]) not in (-1, player):
        base += min(80.0, arrival + 2.0) * float(target[6])
    return max(1.0, base)


def _public_target_score(source: list, target: list, required: float, arrival: float) -> float:
    dist = math.hypot(float(target[2]) - float(source[2]), float(target[3]) - float(source[3]))
    enemy_produced = arrival * float(target[6]) if int(target[1]) != -1 else 0.0
    enemy_bonus = float(target[6]) if int(target[1]) != -1 else 0.0
    total_ships = required + enemy_produced
    return 100.0 - dist + 15.0 * float(target[6]) + 10.0 * enemy_bonus - 0.7 * total_ships - 2.0 * arrival


def _segment_distance_sq(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy
    t = min(1.0, max(0.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    cx = ax + t * abx
    cy = ay + t * aby
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy


def encode_obs(obs: dict[str, Any], player: int, players: int = 2) -> EncodedObs:
    raw_planets = list(obs.get("planets", []))[:MAX_PLANETS]
    raw_fleets = list(obs.get("fleets", []))
    comet_ids = set(obs.get("comet_planet_ids", []) or [])
    step = float(obs.get("step", 0))
    remaining = max(0.0, MAX_STEPS - step)

    planets = np.zeros((MAX_PLANETS, PLANET_FEAT_DIM), dtype=np.float32)
    pair_features = np.zeros((MAX_PLANETS, MAX_PLANETS, PAIR_FEAT_DIM), dtype=np.float32)
    planet_mask = np.zeros(MAX_PLANETS, dtype=np.bool_)
    own_mask = np.zeros(MAX_PLANETS, dtype=np.bool_)
    source_xy = np.zeros((MAX_PLANETS, 2), dtype=np.float32)
    planet_ids: list[int] = []

    own_score = score(obs, player)
    enemy_scores = [score(obs, pid) for pid in range(players) if pid != player]
    enemy_score = max(enemy_scores) if enemy_scores else 0.0

    incoming_friend = {int(p[0]): 0.0 for p in raw_planets}
    incoming_enemy = {int(p[0]): 0.0 for p in raw_planets}
    for fleet in raw_fleets:
        owner = int(fleet[1])
        fx, fy, angle, ships = float(fleet[2]), float(fleet[3]), float(fleet[4]), float(fleet[6])
        vx, vy = math.cos(angle), math.sin(angle)
        for p in raw_planets:
            dx = float(p[2]) - fx
            dy = float(p[3]) - fy
            proj = dx * vx + dy * vy
            if proj <= 0.0:
                continue
            eta = proj / max(fleet_speed(ships), 1e-6)
            if eta > 80.0:
                continue
            perp = abs(dx * vy - dy * vx)
            if perp <= float(p[4]) + 1.0:
                key = int(p[0])
                if owner == player:
                    incoming_friend[key] += ships
                elif owner != -1:
                    incoming_enemy[key] += ships

    for i, p in enumerate(raw_planets):
        pid, owner, x, y, radius, ships, production = p
        pid = int(pid)
        owner = int(owner)
        planet_ids.append(pid)
        planet_mask[i] = True
        own_mask[i] = owner == player
        source_xy[i] = (float(x), float(y))

        if owner == player:
            planets[i, 0] = 1.0
        elif owner == -1:
            planets[i, 1] = 1.0
        else:
            planets[i, 2] = 1.0
        planets[i, 3] = (float(x) - CENTER) / CENTER
        planets[i, 4] = (float(y) - CENTER) / CENTER
        planets[i, 5] = float(radius) / 10.0
        planets[i, 6] = math.log1p(float(ships)) / math.log(1000.0)
        planets[i, 7] = float(production) / 5.0
        planets[i, 8] = float(pid in comet_ids)
        planets[i, 9] = math.hypot(float(x) - CENTER, float(y) - CENTER) / CENTER
        planets[i, 10] = float(pid) / 128.0
        planets[i, 11] = step / MAX_STEPS
        planets[i, 12] = float(obs.get("angular_velocity", 0.0)) * 20.0
        planets[i, 13] = incoming_friend.get(pid, 0.0) / 500.0
        planets[i, 14] = incoming_enemy.get(pid, 0.0) / 500.0
        planets[i, 15] = (own_score - enemy_score) / 1000.0
        planets[i, 16] = float(len(raw_fleets)) / 128.0
        planets[i, 17] = 1.0

    own_planets = [p for p in raw_planets if int(p[1]) == player]
    enemy_planets = [p for p in raw_planets if int(p[1]) not in (-1, player)]
    own_prod = sum(float(p[6]) for p in own_planets)
    enemy_prod = sum(float(p[6]) for p in enemy_planets)
    comet_life_by_id: dict[int, float] = {}
    for group in obs.get("comets", []) or []:
        planet_ids = group.get("planet_ids", []) if isinstance(group, dict) else []
        paths = group.get("paths", []) if isinstance(group, dict) else []
        path_index = int(group.get("path_index", 0)) if isinstance(group, dict) else 0
        for idx, planet_id in enumerate(planet_ids):
            if idx < len(paths):
                comet_life_by_id[int(planet_id)] = float(max(0, len(paths[idx]) - path_index))
    global_features = np.array(
        [
            step / MAX_STEPS,
            float(obs.get("angular_velocity", 0.0)) * 20.0,
            math.log1p(own_score) / math.log(3000.0),
            math.log1p(enemy_score) / math.log(3000.0),
            len(own_planets) / 40.0,
            len(enemy_planets) / 40.0,
            sum(float(p[6]) for p in own_planets) / 40.0,
            len(raw_fleets) / 128.0,
        ],
        dtype=np.float32,
    )

    for i, src in enumerate(raw_planets):
        sx, sy = float(src[2]), float(src[3])
        src_ships = float(src[5])
        for j, tgt in enumerate(raw_planets):
            tx, ty = float(tgt[2]), float(tgt[3])
            dx = tx - sx
            dy = ty - sy
            dist = math.hypot(dx, dy)
            ships_hint = max(1.0, min(src_ships, max(1.0, float(tgt[5]) + 1.0)))
            speed = fleet_speed(ships_hint)
            arrival = dist / max(speed, 1e-6)
            sun_hit = _segment_distance_sq(CENTER, CENTER, sx, sy, tx, ty) <= 10.0 * 10.0
            target_owner = int(tgt[1])
            friend_in = incoming_friend.get(int(tgt[0]), 0.0)
            enemy_in = incoming_enemy.get(int(tgt[0]), 0.0)
            required = _required_ships_hint(tgt, player, arrival, friend_in, enemy_in)
            public_score = _public_target_score(src, tgt, required, arrival)
            comet_life = comet_life_by_id.get(int(tgt[0]))
            useful_turns = max(0.0, remaining - arrival)
            if comet_life is not None:
                useful_turns = min(useful_turns, max(0.0, comet_life - arrival))
            econ_value = float(tgt[6]) * useful_turns
            owner_bonus = 1.8 if target_owner not in (-1, player) else 1.25 if step < 40 else 1.0
            behind_bonus = 1.15 if own_prod < enemy_prod else 1.0
            contested_penalty = max(0.0, enemy_in - friend_in) * 1.2
            cost = required + arrival * 0.6 + max(0.0, enemy_in - friend_in) + contested_penalty + 1.0
            strategic_score = public_score + 1.5 * econ_value * owner_bonus * behind_bonus / max(cost, 1.0) - 0.2 * contested_penalty
            pair_features[i, j, 0] = dx / BOARD_SIZE
            pair_features[i, j, 1] = dy / BOARD_SIZE
            pair_features[i, j, 2] = dist / (BOARD_SIZE * 1.4143)
            pair_features[i, j, 3] = math.cos(math.atan2(dy, dx)) if dist > 1e-6 else 0.0
            pair_features[i, j, 4] = math.sin(math.atan2(dy, dx)) if dist > 1e-6 else 0.0
            pair_features[i, j, 5] = arrival / 100.0
            pair_features[i, j, 6] = float(src_ships > float(tgt[5]) + 1.0)
            pair_features[i, j, 7] = (src_ships - float(tgt[5])) / 500.0
            pair_features[i, j, 8] = float(tgt[6]) / 5.0
            pair_features[i, j, 9] = float(target_owner == -1)
            pair_features[i, j, 10] = float(target_owner >= 0 and target_owner != player)
            pair_features[i, j, 11] = float(target_owner == player)
            pair_features[i, j, 12] = float(sun_hit)
            pair_features[i, j, 13] = friend_in / 500.0
            pair_features[i, j, 14] = enemy_in / 500.0
            pair_features[i, j, 15] = float(i == j)
            pair_features[i, j, 16] = min(3.0, required / max(1.0, src_ships)) / 3.0
            pair_features[i, j, 17] = math.log1p(required) / math.log(1000.0)
            pair_features[i, j, 18] = max(-2.0, min(2.0, public_score / 100.0))
            pair_features[i, j, 19] = min(3.0, econ_value / 500.0)
            pair_features[i, j, 20] = float(arrival <= max(5.0, remaining - 5.0))
            pair_features[i, j, 21] = 0.0 if comet_life is None else min(2.0, comet_life / 100.0)
            pair_features[i, j, 22] = float(own_prod < enemy_prod)
            pair_features[i, j, 23] = max(-2.0, min(2.0, strategic_score / 100.0))

    return EncodedObs(planets, pair_features, global_features, planet_mask, own_mask, source_xy, planet_ids)
