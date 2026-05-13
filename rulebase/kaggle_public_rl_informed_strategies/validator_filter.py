"""Optional shot-validation wrapper from the public hybrid RL notebook."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np

from .geometry import collides_segment_circle, fleet_speed
from .shot_validator import NumpyShotValidator, VAL_THRESHOLD, encode_shot


def find_target_ray(src_xy: tuple[float, float], angle: float, planets: list) -> int:
    sx, sy = src_xy
    speed = 6.0
    best_id = -1
    best_tick = float("inf")

    for planet in planets:
        pid = int(planet[0])
        px = float(planet[2])
        py = float(planet[3])
        radius = float(planet[4])
        prev_x, prev_y = sx, sy
        for tick in range(1, 80):
            next_x = sx + math.cos(angle) * speed * tick
            next_y = sy + math.sin(angle) * speed * tick
            if collides_segment_circle(prev_x, prev_y, next_x, next_y, px, py, radius + 0.8):
                if tick < best_tick:
                    best_tick = tick
                    best_id = pid
                break
            prev_x, prev_y = next_x, next_y
    return best_id


def load_validator(weights_path: str | Path | None = None) -> NumpyShotValidator | None:
    candidates: list[Path] = []
    if weights_path is not None:
        candidates.append(Path(weights_path))
    candidates.extend(
        [
            Path("/kaggle_simulations/agent/weights.npz"),
            Path.cwd() / "weights.npz",
            Path(__file__).resolve().parent / "weights.npz",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            return NumpyShotValidator(path)
        except Exception:
            return None
    return None


def filter_moves_with_validator(
    obs,
    moves: list[list[float | int]],
    validator: NumpyShotValidator | None,
    threshold: float = VAL_THRESHOLD,
) -> list[list[float | int]]:
    if validator is None or not moves:
        return moves

    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    side = int(obs.get("player", 0) if isinstance(obs, dict) else obs.player)
    owner_by_id = {int(p[0]): int(p[1]) for p in planets}
    xy_by_id = {int(p[0]): (float(p[2]), float(p[3])) for p in planets}

    features = []
    move_indices = []
    for i, move in enumerate(moves):
        try:
            src_id = int(move[0])
            angle = float(move[1])
            ships = int(move[2])
        except Exception:
            continue
        if src_id not in xy_by_id:
            continue
        target_id = find_target_ray(xy_by_id[src_id], angle, planets)
        if target_id < 0 or target_id == src_id:
            continue
        if owner_by_id.get(target_id, -2) == side:
            continue
        encoded = encode_shot(obs, src_id, target_id, ships)
        if encoded is None:
            continue
        features.append(encoded)
        move_indices.append(i)

    if not features:
        return moves

    probs = validator.proba(np.stack(features))
    keep = [True] * len(moves)
    for i, prob in zip(move_indices, probs):
        if float(prob) < threshold:
            keep[i] = False
    return [move for i, move in enumerate(moves) if keep[i]]


def make_validated_agent(
    base_agent: Callable,
    weights_path: str | Path | None = None,
    threshold: float = VAL_THRESHOLD,
) -> Callable:
    validator = load_validator(weights_path)

    def agent(obs, configuration=None):
        moves = base_agent(obs, configuration) if configuration is not None else base_agent(obs)
        return filter_moves_with_validator(obs, moves, validator, threshold=threshold)

    return agent
