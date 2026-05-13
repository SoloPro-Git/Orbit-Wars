"""Numpy shot validator utilities from the public RL hybrid notebook.

The validator is optional: if no weights.npz is present the rule agent simply skips it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_DIM = 24
VAL_THRESHOLD = 0.30


class NumpyShotValidator:
    def __init__(self, weights_path: str | Path):
        weights = np.load(str(weights_path))
        self.w0 = weights["w0"]
        self.b0 = weights["b0"]
        self.w2 = weights["w2"]
        self.b2 = weights["b2"]
        self.w4 = weights["w4"]
        self.b4 = weights["b4"]

    def proba(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.w0.T + self.b0)
        h = np.maximum(0.0, h @ self.w2.T + self.b2)
        z = (h @ self.w4.T + self.b4).reshape(-1)
        return 1.0 / (1.0 + np.exp(-z))


def encode_shot(obs: Any, src_id: int, target_id: int, ships_sent: int) -> np.ndarray | None:
    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    me = int(obs.get("player", 0) if isinstance(obs, dict) else obs.player)
    planet_by_id = {
        int(p[0]): (int(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5]), float(p[6]))
        for p in planets
    }
    if src_id not in planet_by_id or target_id not in planet_by_id:
        return None

    src = planet_by_id[src_id]
    tgt = planet_by_id[target_id]
    src_owner, sx, sy, sr, src_ships, src_prod = src
    tgt_owner, tx, ty, tr, tgt_ships, tgt_prod = tgt

    my_ships_total = sum(int(p[5]) for p in planets if int(p[1]) == me)
    enemy_ships_total = sum(int(p[5]) for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    my_planets = sum(1 for p in planets if int(p[1]) == me)
    enemy_planets = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    dist = max(math.hypot(tx - sx, ty - sy) - sr - tr, 0.0)
    speed = 1.0 + (6.0 - 1.0) * (math.log(max(ships_sent, 1)) / math.log(1000.0)) ** 1.5
    eta = dist / max(speed, 0.5)

    ally_n = ally_s = enemy_n = enemy_s = 0
    for fleet in fleets:
        owner = int(fleet[1])
        ships = int(fleet[6])
        if owner == me:
            ally_n += 1
            ally_s += ships
        else:
            enemy_n += 1
            enemy_s += ships

    turn = int(obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0))
    return np.array(
        [
            src_ships / 100.0,
            src_prod / 5.0,
            sr / 4.0,
            tgt_ships / 100.0,
            tgt_prod / 5.0,
            tr / 4.0,
            1.0 if tgt_owner == me else 0.0,
            1.0 if tgt_owner < 0 else 0.0,
            1.0 if tgt_owner >= 0 and tgt_owner != me else 0.0,
            ships_sent / 100.0,
            ships_sent / max(src_ships, 1),
            dist / 100.0,
            eta / 60.0,
            speed / 6.0,
            ally_n / 10.0,
            ally_s / 100.0,
            enemy_n / 10.0,
            enemy_s / 100.0,
            turn / 500.0,
            my_ships_total / 300.0,
            enemy_ships_total / 300.0,
            my_planets / 20.0,
            enemy_planets / 20.0,
            1.0,
        ],
        dtype=np.float32,
    )
