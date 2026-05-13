"""Rule features borrowed from the local RL pipeline.

These are not neural features. They are small deterministic signals that mirror
what the training pipeline explicitly exposed to the model: remaining economic
value, comet ROI, incoming threat balance, and late-game arrival usefulness.
"""

from __future__ import annotations

import math

from .geometry import distance, fleet_speed
from .state import Fleet, LocalObs, Planet

TOTAL_STEPS = 500


def remaining_steps(local: LocalObs, total_steps: int = TOTAL_STEPS) -> int:
    return max(total_steps - local.step, 0)


def comet_remaining_from_obs(obs, planet_id: int) -> int:
    for group in (obs.get("comets", []) if isinstance(obs, dict) else getattr(obs, "comets", [])):
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = int(group.get("path_index", 0))
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def planet_economic_value(target: Planet, local: LocalObs, arrival_turns: int, comet_life: int | None = None) -> float:
    turns = max(0, remaining_steps(local) - arrival_turns)
    if comet_life is not None:
        turns = min(turns, max(0, comet_life - arrival_turns))
    return target.production * turns


def production_totals(local: LocalObs) -> tuple[float, float]:
    own = sum(p.production for p in local.planets if p.owner == local.player)
    enemy = sum(p.production for p in local.planets if p.owner not in (-1, local.player))
    return own, enemy


def ship_totals(local: LocalObs) -> tuple[int, int]:
    own = sum(p.ships for p in local.planets if p.owner == local.player)
    enemy = sum(p.ships for p in local.planets if p.owner not in (-1, local.player))
    own += sum(f.ships for f in local.fleets if f.owner == local.player)
    enemy += sum(f.ships for f in local.fleets if f.owner not in (-1, local.player))
    return own, enemy


def incoming_balance(target: Planet, fleets: list[Fleet], player: int, horizon: int = 60) -> tuple[int, int, int]:
    friendly = 0
    enemy = 0
    earliest_enemy_eta = horizon + 1
    for fleet in fleets:
        dx = target.x - fleet.x
        dy = target.y - fleet.y
        vx = math.cos(fleet.angle)
        vy = math.sin(fleet.angle)
        proj = dx * vx + dy * vy
        if proj <= 0:
            continue
        perp = abs(dx * vy - dy * vx)
        if perp > target.radius + 1.5:
            continue
        eta = int(proj / fleet_speed(fleet.ships))
        if eta > horizon:
            continue
        if fleet.owner == player:
            friendly += fleet.ships
        else:
            enemy += fleet.ships
            earliest_enemy_eta = min(earliest_enemy_eta, eta)
    return friendly, enemy, earliest_enemy_eta


def strategic_target_score(
    source: Planet,
    target: Planet,
    local: LocalObs,
    base_score: float,
    estimated_ships: int,
    arrival_turns: int,
    comet_life: int | None = None,
) -> float:
    econ_value = planet_economic_value(target, local, arrival_turns, comet_life=comet_life)
    if econ_value <= 0:
        return -1e9

    friendly_in, enemy_in, enemy_eta = incoming_balance(target, local.fleets, local.player)
    contested_penalty = 0.0
    if enemy_in > 0 and enemy_eta <= arrival_turns + 3:
        contested_penalty = enemy_in * 1.2

    own_prod, enemy_prod = production_totals(local)
    own_ships, enemy_ships = ship_totals(local)
    behind_bonus = 1.0
    if own_prod < enemy_prod or own_ships < enemy_ships * 0.85:
        behind_bonus = 1.15

    owner_bonus = 1.0
    if target.owner not in (-1, local.player):
        owner_bonus = 1.8
    elif local.step < 40:
        owner_bonus = 1.25

    cost = estimated_ships + arrival_turns * 0.6 + max(0, enemy_in - friendly_in) + contested_penalty + 1.0
    roi_score = econ_value * owner_bonus * behind_bonus / cost
    proximity = max(0.0, 100.0 - distance(source, target)) * 0.02
    # Keep public scoring as the backbone. RL signals nudge ordering and filter
    # obviously poor late/comet/contested shots without taking over the style.
    return base_score + roi_score * 1.5 + proximity - contested_penalty * 0.2
