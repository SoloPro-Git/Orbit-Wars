"""Observation and candidate-action features for rulebase-aligned training."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from training.expert.action_labeling import infer_target_planet_id

BOARD_SIZE = 100.0
MAX_TURNS = 500
PLANET_DIM = 17
ACTION_DIM = 14


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


def encode_planets(obs: dict[str, Any], player: int, max_planets: int | None = None) -> np.ndarray:
    planets = obs.get("planets", [])
    n = len(planets) if max_planets is None else max_planets
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
    planets = {int(p[0]): p for p in obs.get("planets", [])}
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
    comet_ids = set(obs.get("comet_planet_ids", []))
    for from_id, angle, ships in action:
        src = planets.get(int(from_id))
        if src is None:
            continue
        source_prod += float(src[6])
        source_ships += float(src[5])
        target_id = infer_target_planet_id(obs, int(from_id), float(angle), float(ships))
        if target_id is None:
            continue
        target = planets.get(int(target_id))
        if target is None:
            continue
        labelled += 1
        dx = float(target[2]) - float(src[2])
        dy = float(target[3]) - float(src[3])
        eta_sum += math.hypot(dx, dy) / _fleet_speed(float(ships))
        target_prod += float(target[6])
        target_ships += float(target[5])
        if int(target[1]) == -1:
            neutral_targets += 1
        elif int(target[1]) != player:
            enemy_targets += 1
        if target_id in comet_ids:
            comet_targets += 1

    denom = max(len(action), 1)
    out[2] = source_prod / (5.0 * denom)
    out[3] = math.log1p(source_ships) / math.log(2000.0)
    out[4] = target_prod / (5.0 * max(labelled, 1))
    out[5] = math.log1p(target_ships) / math.log(2000.0)
    out[6] = enemy_targets / denom
    out[7] = neutral_targets / denom
    out[8] = comet_targets / denom
    out[9] = eta_sum / (80.0 * max(labelled, 1))
    out[10] = total_ships / max(_score(obs, player), 1.0)
    out[11] = float(labelled) / denom
    out[12] = float(len({int(a[0]) for a in action if len(a) >= 3})) / denom
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

