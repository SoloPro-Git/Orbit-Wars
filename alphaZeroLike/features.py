"""Move-set features for AlphaZero-like candidate search.

Unlike ``training2.features.encode_action``, this encoder keeps every launch in
the candidate action as a separate token.  The network can then reason about
source/target identity and multi-source coordination before pooling the action.
"""

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
GLOBAL_DIM = 8
MOVE_DIM = 20


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
    players = sorted({int(p[1]) for p in obs.get("planets", []) if int(p[1]) >= 0})
    players = sorted(set(players) | {0, 1})
    scores = [(pid, _score(obs, pid)) for pid in players]
    if len(scores) <= 1:
        return 0.0
    score_by_player = dict(scores)
    player_score = score_by_player.get(player, 0.0)
    ranks = [rank for rank, (_, score) in enumerate(sorted(scores, key=lambda row: row[1], reverse=True)) if score == player_score]
    if not ranks:
        return -1.0
    rank = float(sum(ranks) / len(ranks))
    return 1.0 - 2.0 * rank / (len(scores) - 1)


def margin_value(obs: dict[str, Any], player: int, scale: float = 50.0) -> float:
    players = sorted({int(p[1]) for p in obs.get("planets", []) if int(p[1]) >= 0})
    players = sorted(set(players) | {0, 1})
    own = _score(obs, player)
    opponents = [_score(obs, pid) for pid in players if pid != player]
    if not opponents:
        return 0.0
    margin = own - max(opponents)
    return float(math.tanh(margin / max(float(scale), 1e-6)))


@dataclass(frozen=True)
class EncodedSearchPosition:
    planet_features: np.ndarray
    global_features: np.ndarray
    move_features: np.ndarray
    move_source_indices: np.ndarray
    move_target_indices: np.ndarray
    move_mask: np.ndarray
    candidate_mask: np.ndarray


def encode_planets(obs: dict[str, Any], player: int, max_entities: int = MAX_ENTITIES) -> np.ndarray:
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    out = np.zeros((max_entities, PLANET_DIM), dtype=np.float32)
    comet_ids = set(obs.get("comet_planet_ids", []))
    step = float(obs.get("step", 0))
    initial = {int(p[0]): p for p in obs.get("initial_planets", [])}

    for i, p in enumerate(planets[:max_entities]):
        pid, owner, x, y, radius, ships, production = p
        owner = int(owner)
        out[i, 0] = float(owner == player)
        out[i, 1] = float(owner == -1)
        out[i, 2] = float(owner not in (-1, player))
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

    offset = min(len(planets), max_entities)
    for j, fleet in enumerate(fleets[: max(0, max_entities - offset)]):
        i = offset + j
        _, owner, x, y, angle, from_planet_id, ships = fleet
        owner = int(owner)
        out[i, 0] = float(owner == player)
        out[i, 2] = float(owner not in (-1, player))
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


def _planet_index(planets: list[list], planet_id: int | None) -> int:
    if planet_id is None:
        return -1
    for i, planet in enumerate(planets):
        if int(planet[0]) == int(planet_id):
            return i
    return -1


def _encode_move(obs: dict[str, Any], player: int, move: list) -> tuple[np.ndarray, int, int]:
    planets = obs.get("planets", [])
    by_id = {int(p[0]): p for p in planets}
    out = np.zeros(MOVE_DIM, dtype=np.float32)
    if len(move) < 3:
        return out, -1, -1

    src_id = int(move[0])
    angle = float(move[1])
    ships = max(1.0, float(move[2]))
    source = by_id.get(src_id)
    target_id = infer_target_planet_id(obs, src_id, angle, ships)
    target = by_id.get(int(target_id)) if target_id is not None else None
    src_idx = _planet_index(planets, src_id)
    tgt_idx = _planet_index(planets, int(target_id)) if target_id is not None else -1

    out[0] = math.log1p(ships) / math.log(1000.0)
    out[1] = math.cos(angle)
    out[2] = math.sin(angle)
    out[3] = float(src_id) / 64.0
    out[4] = float(target_id if target_id is not None else -1) / 64.0
    out[5] = float(target is None)
    if source is None:
        return out, src_idx, tgt_idx

    source_ships = max(float(source[5]), 1.0)
    out[6] = min(1.0, ships / source_ships)
    out[7] = max(0.0, source_ships - ships) / source_ships
    out[8] = float(source[6]) / 5.0
    if target is None:
        return out, src_idx, tgt_idx

    dx = float(target[2]) - float(source[2])
    dy = float(target[3]) - float(source[3])
    dist = math.hypot(dx, dy)
    eta = dist / _fleet_speed(ships)
    direct = math.atan2(dy, dx)
    offset = math.atan2(math.sin(angle - direct), math.cos(angle - direct))
    out[9] = dist / 140.0
    out[10] = eta / 80.0
    out[11] = math.cos(offset)
    out[12] = math.sin(offset)
    owner = int(target[1])
    out[13] = float(owner == player)
    out[14] = float(owner == -1)
    out[15] = float(owner not in (-1, player))
    out[16] = math.log1p(float(target[5])) / math.log(1000.0)
    out[17] = float(target[6]) / 5.0
    required = float(target[5]) + 1.0
    if owner not in (-1, player):
        required += float(target[6]) * eta
    out[18] = (ships - required) / 100.0
    out[19] = float(target_id in set(obs.get("comet_planet_ids", [])))
    return out, src_idx, tgt_idx


def encode_search_position(
    obs: dict[str, Any],
    player: int,
    candidates: list[list[list]],
    *,
    max_candidates: int = 64,
    max_moves: int = 16,
    max_entities: int = MAX_ENTITIES,
) -> EncodedSearchPosition:
    move_features = np.zeros((max_candidates, max_moves, MOVE_DIM), dtype=np.float32)
    move_sources = np.full((max_candidates, max_moves), -1, dtype=np.int64)
    move_targets = np.full((max_candidates, max_moves), -1, dtype=np.int64)
    move_mask = np.zeros((max_candidates, max_moves), dtype=np.float32)
    candidate_mask = np.zeros(max_candidates, dtype=np.float32)

    for i, action in enumerate(candidates[:max_candidates]):
        candidate_mask[i] = 1.0
        if not action:
            move_mask[i, 0] = 1.0
            continue
        for j, move in enumerate(action[:max_moves]):
            feat, src_idx, tgt_idx = _encode_move(obs, player, move)
            move_features[i, j] = feat
            move_sources[i, j] = src_idx
            move_targets[i, j] = tgt_idx
            move_mask[i, j] = 1.0

    return EncodedSearchPosition(
        planet_features=encode_planets(obs, player, max_entities=max_entities),
        global_features=encode_global(obs, player),
        move_features=move_features,
        move_source_indices=move_sources,
        move_target_indices=move_targets,
        move_mask=move_mask,
        candidate_mask=candidate_mask,
    )
