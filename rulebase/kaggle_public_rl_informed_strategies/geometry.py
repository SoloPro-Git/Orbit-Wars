"""Geometry, speed, and collision helpers for Orbit Wars rules."""

from __future__ import annotations

import math

from .state import Planet

CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0


def distance(a: Planet, b: Planet) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def fleet_speed(ships: int | float, max_speed: float = MAX_SPEED) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(max(1.0, float(ships))) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (max_speed - 1.0) * (ratio**1.5)


def travel_ticks(src: Planet, target: Planet, ships: int | float) -> int:
    return max(1, int(math.floor(distance(src, target) / fleet_speed(ships))))


def angle_to(src: Planet, target: Planet) -> float:
    return math.atan2(target.y - src.y, target.x - src.x)


def collides_segment_circle(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cx: float,
    cy: float,
    radius: float,
) -> bool:
    vx = x2 - x1
    vy = y2 - y1
    wx = cx - x1
    wy = cy - y1
    length_sq = vx * vx + vy * vy
    if length_sq == 0:
        return (x1 - cx) ** 2 + (y1 - cy) ** 2 <= radius * radius

    t = (wx * vx + wy * vy) / length_sq
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * vx
    closest_y = y1 + t * vy
    return (closest_x - cx) ** 2 + (closest_y - cy) ** 2 <= radius * radius


def sun_collision(src: Planet, ships: int | float, angle: float, ticks: int = 61) -> bool:
    speed = fleet_speed(ships)
    prev_x, prev_y = src.x, src.y
    for tick in range(1, ticks):
        x = src.x + math.cos(angle) * speed * tick
        y = src.y + math.sin(angle) * speed * tick
        if collides_segment_circle(prev_x, prev_y, x, y, CENTER_X, CENTER_Y, SUN_RADIUS):
            return True
        prev_x, prev_y = x, y
    return False


def planet_trajectory(planet: Planet, angular_velocity: float, ticks: int = 60) -> list[tuple[float, float]]:
    angle = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    radius = math.hypot(planet.x - CENTER_X, planet.y - CENTER_Y)
    return [
        (
            CENTER_X + radius * math.cos(angle + angular_velocity * tick),
            CENTER_Y + radius * math.sin(angle + angular_velocity * tick),
        )
        for tick in range(1, ticks + 1)
    ]


def find_angle_to_moving_planet(
    src: Planet,
    target: Planet,
    ships: int,
    angular_velocity: float,
    ticks: int = 60,
) -> tuple[float | None, int | None]:
    speed = fleet_speed(ships)
    for tick, (tx, ty) in enumerate(planet_trajectory(target, angular_velocity, ticks), start=1):
        dist_to_target = math.hypot(tx - src.x, ty - src.y) - src.radius
        if abs(speed * tick - dist_to_target) > target.radius:
            continue
        angle = math.atan2(ty - src.y, tx - src.x)
        if sun_collision(src, ships, angle):
            return None, None
        return angle, tick
    return None, None
