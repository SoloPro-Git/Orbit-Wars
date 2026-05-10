"""Expert action labeling helpers.

Map expert actions ``[from_planet_id, angle, num_ships]`` to a target planet id
using a physics-aware approximation aligned with Orbit Wars rules.
"""
from __future__ import annotations

import math
from typing import Any


_BOARD_SIZE = 100.0
_SUN_X = 50.0
_SUN_Y = 50.0
_SUN_RADIUS = 10.0
_MAX_SHIP_SPEED = 6.0


def _fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(max(ships, 1.0)) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (_MAX_SHIP_SPEED - 1.0) * (ratio ** 1.5)


def _segment_hits_circle(
    x1: float, y1: float, x2: float, y2: float, cx: float, cy: float, r: float
) -> bool:
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy
    a = dx * dx + dy * dy
    if a < 1e-12:
        return (fx * fx + fy * fy) <= (r * r)
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return False
    s = math.sqrt(disc)
    t1 = (-b - s) / (2.0 * a)
    t2 = (-b + s) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def _planet_position_at_step(
    planet: list[float],
    step_offset: int,
    angular_velocity: float,
    initial_by_id: dict[int, tuple[float, float, float]],
    comet_pos_by_step: dict[int, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    pid = int(planet[0])
    radius = float(planet[4])
    if comet_pos_by_step and pid in comet_pos_by_step:
        return comet_pos_by_step[pid]

    x = float(planet[2])
    y = float(planet[3])

    dist_to_center = math.hypot(x - _SUN_X, y - _SUN_Y)
    is_orbiting = (dist_to_center + radius) < 50.0
    if not is_orbiting:
        return x, y

    init = initial_by_id.get(pid)
    if init is None:
        return x, y
    init_x, init_y, _ = init
    orbital_r = math.hypot(init_x - _SUN_X, init_y - _SUN_Y)
    cur_angle = math.atan2(y - _SUN_Y, x - _SUN_X)
    future_angle = cur_angle + angular_velocity * step_offset
    return (
        _SUN_X + orbital_r * math.cos(future_angle),
        _SUN_Y + orbital_r * math.sin(future_angle),
    )


def infer_target_planet_id(
    observation: dict[str, Any],
    from_planet_id: int,
    angle: float,
    num_ships: float,
    max_steps: int = 120,
) -> int | None:
    """Infer which planet an expert launch is aiming to hit first.

    Simulates one fleet moving in a straight line, and detects first collision
    with planets whose positions may move (orbit/comet) per step.
    """
    planets = observation.get("planets", [])
    if not planets:
        return None

    source = None
    for p in planets:
        if int(p[0]) == int(from_planet_id):
            source = p
            break
    if source is None:
        return None

    sx = float(source[2]) + float(source[4]) * math.cos(angle)
    sy = float(source[3]) + float(source[4]) * math.sin(angle)

    v = _fleet_speed(float(num_ships))
    vx = math.cos(angle) * v
    vy = math.sin(angle) * v

    initial_by_id: dict[int, tuple[float, float, float]] = {}
    for ip in observation.get("initial_planets", []):
        initial_by_id[int(ip[0])] = (float(ip[2]), float(ip[3]), float(ip[4]))

    # comet current/future positions from paths
    comet_positions_per_step: dict[int, dict[int, tuple[float, float]]] = {}
    for g in observation.get("comets", []):
        pids = g.get("planet_ids", [])
        paths = g.get("paths", [])
        path_index = int(g.get("path_index", 0))
        for i, pid in enumerate(pids):
            if i >= len(paths):
                continue
            path = paths[i]
            series: dict[int, tuple[float, float]] = {}
            for s in range(max_steps + 1):
                idx = path_index + s
                if 0 <= idx < len(path):
                    series[s] = (float(path[idx][0]), float(path[idx][1]))
            comet_positions_per_step[int(pid)] = series

    x, y = sx, sy
    for step in range(1, max_steps + 1):
        nx = x + vx
        ny = y + vy

        # out-of-board / sun collision => fleet removed
        if nx < 0.0 or nx > _BOARD_SIZE or ny < 0.0 or ny > _BOARD_SIZE:
            return None
        if _segment_hits_circle(x, y, nx, ny, _SUN_X, _SUN_Y, _SUN_RADIUS):
            return None

        hit_candidates: list[tuple[float, int]] = []
        for p in planets:
            pid = int(p[0])
            if pid == int(from_planet_id):
                continue
            r = float(p[4])
            comet_series = comet_positions_per_step.get(pid, {})
            comet_pos = comet_series.get(step)
            px, py = _planet_position_at_step(
                p,
                step_offset=step,
                angular_velocity=float(observation.get("angular_velocity", 0.0)),
                initial_by_id=initial_by_id,
                comet_pos_by_step=({pid: comet_pos} if comet_pos else None),
            )
            if _segment_hits_circle(x, y, nx, ny, px, py, r):
                d = math.hypot(px - x, py - y)
                hit_candidates.append((d, pid))

        if hit_candidates:
            hit_candidates.sort(key=lambda t: t[0])
            return hit_candidates[0][1]

        x, y = nx, ny

    return None
