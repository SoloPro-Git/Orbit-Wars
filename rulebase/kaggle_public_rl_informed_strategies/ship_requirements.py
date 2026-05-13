"""Multi-source attack ship requirement estimators."""

from __future__ import annotations

from .geometry import distance, fleet_speed, planet_trajectory
from .state import Planet

MIN_SHIPS_MINE_ATTACK = 10


def calculate_required_ships(attacking_planets: list[dict[str, object]], target: Planet, base_ships: int) -> int:
    required = base_ships
    for _ in range(3):
        remainder = required
        max_tick = 0
        for attack in attacking_planets:
            planet = attack["planet"]
            ships = min(int(attack["ships"]), remainder)
            if ships > 0:
                ships = min(int(attack["ships"]), max(ships, MIN_SHIPS_MINE_ATTACK))
            if ships <= 0:
                continue
            max_tick = max(max_tick, int(distance(planet, target) / fleet_speed(ships)))
            remainder -= ships
        new_required = int(base_ships + max_tick * target.production)
        if new_required == required:
            break
        required = new_required
    return required


def calculate_required_ships_moving(
    attacking_planets: list[dict[str, object]],
    target: Planet,
    base_ships: int,
    angular_velocity: float,
) -> int:
    required = base_ships
    target_traj = planet_trajectory(target, angular_velocity)
    for _ in range(3):
        remainder = required
        max_tick = 0
        for attack in attacking_planets:
            planet = attack["planet"]
            ships = min(int(attack["ships"]), remainder)
            if ships > 0:
                ships = min(int(attack["ships"]), max(ships, MIN_SHIPS_MINE_ATTACK))
            if ships <= 0:
                continue
            speed = fleet_speed(ships)
            found_tick = 0
            for tick, (tx, ty) in enumerate(target_traj, start=1):
                turns_to_arrive = int((((planet.x - tx) ** 2 + (planet.y - ty) ** 2) ** 0.5) / speed)
                if abs(turns_to_arrive - tick) <= 1:
                    found_tick = tick
                    break
            max_tick = max(max_tick, found_tick)
            remainder -= ships
        new_required = int(base_ships + max_tick * target.production)
        if new_required == required:
            break
        required = new_required
    return required
