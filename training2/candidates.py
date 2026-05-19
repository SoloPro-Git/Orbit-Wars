"""Candidate action generation."""
from __future__ import annotations

from collections.abc import Callable
import math
import random

BOARD_SIZE = 100.0
MAX_TURNS = 500
SUN_RADIUS = 10.0


def _canonical(action: list[list]) -> tuple:
    return tuple((int(a[0]), round(float(a[1]), 6), int(a[2])) for a in action if len(a) >= 3)


def _fleet_speed(ships: float, max_speed: float = 6.0) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(max(ships, 1.0)) / math.log(1000.0)
    ratio = min(max(ratio, 0.0), 1.0)
    return 1.0 + (max_speed - 1.0) * (ratio**1.5)


def _segment_distance_sq(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = min(1.0, max(0.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return (px - cx) ** 2 + (py - cy) ** 2


def _sun_collision(src: list, angle: float, ships: int, ticks: int) -> bool:
    speed = _fleet_speed(ships)
    ax = float(src[2])
    ay = float(src[3])
    bx = ax + math.cos(angle) * speed * max(1, ticks)
    by = ay + math.sin(angle) * speed * max(1, ticks)
    return _segment_distance_sq(50.0, 50.0, ax, ay, bx, by) <= SUN_RADIUS * SUN_RADIUS


def _comet_future_position(obs: dict, planet_id: int, eta: int) -> tuple[float, float] | None:
    for group in obs.get("comets", []) or []:
        planet_ids = group.get("planet_ids", []) if isinstance(group, dict) else []
        if planet_id not in planet_ids:
            continue
        try:
            idx = list(planet_ids).index(planet_id)
            path = group.get("paths", [])[idx]
            path_index = int(group.get("path_index", 0))
            future_index = min(max(0, path_index + max(0, eta)), len(path) - 1)
            x, y = path[future_index]
            return float(x), float(y)
        except (IndexError, TypeError, ValueError):
            return None
    return None


def _is_orbital_target(obs: dict, planet: list) -> bool:
    if int(planet[0]) in set(obs.get("comet_planet_ids", [])):
        return False
    return math.hypot(float(planet[2]) - 50.0, float(planet[3]) - 50.0) + float(planet[4]) < 50.0


def _future_target_position(obs: dict, target: list, eta: int) -> tuple[float, float]:
    comet_pos = _comet_future_position(obs, int(target[0]), eta)
    if comet_pos is not None:
        return comet_pos
    if _is_orbital_target(obs, target):
        angle = math.atan2(float(target[3]) - 50.0, float(target[2]) - 50.0)
        radius = math.hypot(float(target[2]) - 50.0, float(target[3]) - 50.0)
        angle += float(obs.get("angular_velocity", 0.0)) * max(0, eta)
        return 50.0 + math.cos(angle) * radius, 50.0 + math.sin(angle) * radius
    return float(target[2]), float(target[3])


def _aim_angle_and_eta(obs: dict, source: list, target: list, ships: int) -> tuple[float, int]:
    speed = _fleet_speed(max(1, ships))
    tx = float(target[2])
    ty = float(target[3])
    eta = 1
    for _ in range(3):
        dist = math.hypot(tx - float(source[2]), ty - float(source[3]))
        eta = max(1, int(math.ceil(dist / max(speed, 1e-6))))
        tx, ty = _future_target_position(obs, target, eta)
    return math.atan2(ty - float(source[3]), tx - float(source[2])), eta


def _enemy_sendable(enemy: list) -> int:
    return max(0, int(float(enemy[5]) * 0.85 + float(enemy[6]) * 6.0))


def _source_safe_after_send(source: list, ships: int, planets: list, player: int) -> bool:
    remaining = max(0, int(source[5]) - ships)
    for enemy in planets:
        if int(enemy[1]) in (-1, player):
            continue
        dist = math.hypot(float(source[2]) - float(enemy[2]), float(source[3]) - float(enemy[3]))
        if dist > 42.0:
            continue
        sendable = _enemy_sendable(enemy)
        if sendable <= 0:
            continue
        eta = dist / _fleet_speed(max(1, sendable))
        projected = remaining + int(float(source[6]) * eta)
        if projected < sendable + 5:
            return False
    return True


def _heuristic_candidates(obs: dict, player: int, limit: int = 6) -> list[list[list]]:
    planets = obs.get("planets", [])
    mine = [p for p in planets if int(p[1]) == player and float(p[5]) >= 6.0]
    targets = [p for p in planets if int(p[1]) != player and float(p[6]) > 0.0]
    step = int(obs.get("step", 0))
    rows: list[tuple[float, list[list]]] = []
    for source in mine:
        source_ships = int(source[5])
        for target in targets:
            required = int(float(target[5]) + 1)
            angle, eta = _aim_angle_and_eta(obs, source, target, max(1, required))
            if int(target[1]) not in (-1, player):
                required += int(float(target[6]) * eta)
            # Small hold margin mirrors the promoted post-capture hold lesson.
            if int(target[1]) != -1 or float(target[6]) >= 3.0:
                required += 4
            ships = min(source_ships, max(1, required))
            if ships <= 0 or ships > source_ships:
                continue
            if source_ships - ships < max(4, int(float(source[6]))):
                continue
            if not _source_safe_after_send(source, ships, planets, player):
                continue
            if _sun_collision(source, angle, ships, eta + 3):
                continue
            remaining_value = float(target[6]) * max(0, MAX_TURNS - step - eta)
            if int(target[1]) == -1 and remaining_value < ships * 1.05:
                continue
            score = remaining_value - ships * 1.2 - eta * 0.5
            if int(target[1]) != -1:
                score += float(target[6]) * 30.0
            if int(target[0]) in set(obs.get("comet_planet_ids", [])):
                score -= max(0, eta - 12) * 8.0
            rows.append((score, [[int(source[0]), float(angle), int(ships)]]))
    rows.sort(key=lambda row: row[0], reverse=True)
    return [candidate for score, candidate in rows[:limit] if score > 0.0]


def build_candidates(
    obs: dict,
    rulebase_agent: Callable[[dict], list[list]],
    max_candidates: int = 32,
    include_noop: bool = True,
    extra_candidates: list[list[list]] | None = None,
    include_heuristics: bool = True,
) -> tuple[list[list[list]], int]:
    oracle = rulebase_agent(obs) or []
    candidates: list[list[list]] = []
    seen: set[tuple] = set()

    def add(action: list[list]) -> None:
        key = _canonical(action)
        if key in seen:
            return
        seen.add(key)
        candidates.append([[int(a[0]), float(a[1]), int(a[2])] for a in action if len(a) >= 3])

    add(oracle)
    oracle_index = 0
    if include_noop:
        add([])
    for move in oracle:
        add([move])
    for i in range(len(oracle)):
        add([m for j, m in enumerate(oracle) if j != i])
    if include_heuristics:
        for candidate in _heuristic_candidates(obs, int(obs.get("player", 0))):
            add(candidate)
    for candidate in extra_candidates or []:
        add(candidate)
    return candidates[:max_candidates], oracle_index


def shuffle_candidates(
    candidates: list[list[list]],
    target_index: int,
    rng: random.Random,
) -> tuple[list[list[list]], int]:
    """Shuffle candidates and return the target candidate's new index."""
    if not candidates:
        return candidates, target_index
    indexed = list(enumerate(candidates))
    rng.shuffle(indexed)
    shuffled = [candidate for _, candidate in indexed]
    new_target = next((i for i, (old_idx, _) in enumerate(indexed) if old_idx == target_index), 0)
    return shuffled, new_target
