"""Target scoring rules imported from public Kaggle rule agents."""

from __future__ import annotations

from .geometry import distance, fleet_speed
from .state import Planet

FORMULA_DIST = 100.0
FORMULA_PROD_MULT = 15.0
FORMULA_ENEMY_BONUS_MULT = 10.0
FORMULA_TOTAL_SHIPS_PERCENT = 0.7


def public_custom_score(source: Planet, target: Planet) -> float:
    dist = distance(source, target)
    min_ships = target.ships + 1
    eta = dist / fleet_speed(min_ships)

    enemy_produced = eta * target.production if target.owner != -1 else 0.0
    enemy_bonus = target.production if target.owner != -1 else 0.0
    total_ships = min_ships + enemy_produced

    return (
        (FORMULA_DIST - dist)
        + (FORMULA_PROD_MULT * target.production)
        + (FORMULA_ENEMY_BONUS_MULT * enemy_bonus)
        - (FORMULA_TOTAL_SHIPS_PERCENT * total_ships)
        - (2.0 * eta)
    )


def closest_planets_to_target(mine: list[Planet], target: Planet) -> list[tuple[Planet, float]]:
    return sorted(((p, distance(p, target)) for p in mine), key=lambda row: row[1])
