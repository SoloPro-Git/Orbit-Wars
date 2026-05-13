"""Enemy incoming detection and reinforcement planning."""

from __future__ import annotations

import math

from .geometry import collides_segment_circle, fleet_speed, planet_trajectory
from .state import Fleet, Planet

MIN_SHIPS_MINE_ATTACK = 10


def max_enemy_fleet_to_target(
    target: Planet,
    fleets: list[Fleet],
    player: int,
    moving_planets: set[int],
    angular_velocity: float,
) -> int:
    target_traj = planet_trajectory(target, angular_velocity) if target.id in moving_planets else None
    max_enemy = 0
    for fleet in fleets:
        if fleet.owner == player or fleet.ships <= 0:
            continue
        speed = fleet_speed(fleet.ships)
        prev_x, prev_y = fleet.x, fleet.y
        for tick in range(1, 61):
            next_x = fleet.x + math.cos(fleet.angle) * speed * tick
            next_y = fleet.y + math.sin(fleet.angle) * speed * tick
            tx, ty = target_traj[tick - 1] if target_traj is not None else (target.x, target.y)
            if collides_segment_circle(prev_x, prev_y, next_x, next_y, tx, ty, target.radius):
                max_enemy = max(max_enemy, fleet.ships)
                break
            prev_x, prev_y = next_x, next_y
    return max_enemy


def planets_under_attack(
    mine: list[Planet],
    fleets: list[Fleet],
    player: int,
    moving_planets: set[int],
    angular_velocity: float,
) -> dict[int, dict[str, object]]:
    moving_traj = {
        p.id: planet_trajectory(p, angular_velocity)
        for p in mine
        if p.id in moving_planets
    }
    under_attack: dict[int, dict[str, object]] = {}
    seen: set[tuple[int, int]] = set()
    for fleet in fleets:
        if fleet.owner == player or fleet.ships <= 0:
            continue
        speed = fleet_speed(fleet.ships)
        prev_x, prev_y = fleet.x, fleet.y
        for tick in range(1, 61):
            next_x = fleet.x + math.cos(fleet.angle) * speed * tick
            next_y = fleet.y + math.sin(fleet.angle) * speed * tick
            for planet in mine:
                px, py = moving_traj[planet.id][tick - 1] if planet.id in moving_traj else (planet.x, planet.y)
                if not collides_segment_circle(prev_x, prev_y, next_x, next_y, px, py, planet.radius):
                    continue
                key = (planet.id, fleet.id)
                if key in seen:
                    continue
                under_attack.setdefault(planet.id, {"planet": planet, "fleets": []})
                under_attack[planet.id]["fleets"].append({"fleet": fleet, "arrive_tick": tick})
                seen.add(key)
            prev_x, prev_y = next_x, next_y
    return under_attack


def reinforcement_plans(
    mine: list[Planet],
    under_attack: dict[int, dict[str, object]],
    reinforcement_trajectories: list[dict[str, object]],
) -> dict[int, dict[str, int]]:
    plans: dict[int, dict[str, int]] = {}
    for planet in mine:
        if planet.id not in under_attack:
            continue
        attacking = sorted(under_attack[planet.id]["fleets"], key=lambda row: row["arrive_tick"])
        incoming = sorted(
            [row for row in reinforcement_trajectories if row["target"].id == planet.id],
            key=lambda row: row["arrive_tick"],
        )

        available = planet.ships
        previous_tick = 0
        reinf_idx = 0
        for attack in attacking:
            arrive_tick = int(attack["arrive_tick"])
            available += int((arrive_tick - previous_tick) * planet.production)
            while reinf_idx < len(incoming) and incoming[reinf_idx]["arrive_tick"] <= arrive_tick:
                available += int(incoming[reinf_idx]["total_ships"])
                reinf_idx += 1
            available -= int(attack["fleet"].ships)
            previous_tick = arrive_tick
            if available < 0:
                plans[planet.id] = {
                    "ships_needed": int(max(MIN_SHIPS_MINE_ATTACK, abs(available))),
                    "needed_by_tick": arrive_tick,
                }
                break
    return plans
