"""Observation parsing helpers shared by public rule strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: float


@dataclass(slots=True)
class Fleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass(slots=True)
class LocalObs:
    planets: list[Planet]
    mine: list[Planet]
    targets: list[Planet]
    fleets: list[Fleet]
    player: int
    step: int
    angular_velocity: float
    initial_planets: list[Planet]
    comet_planet_ids: set[int]


def obs_get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def parse_planet(raw: Any) -> Planet:
    return Planet(
        id=int(raw[0]),
        owner=int(raw[1]),
        x=float(raw[2]),
        y=float(raw[3]),
        radius=float(raw[4]),
        ships=int(raw[5]),
        production=float(raw[6]),
    )


def parse_fleet(raw: Any, fallback_id: int = 0) -> Fleet:
    # Orbit Wars fleets are [id, owner, x, y, angle, from_planet_id, ships].
    return Fleet(
        id=int(raw[0]) if len(raw) > 0 else fallback_id,
        owner=int(raw[1]),
        x=float(raw[2]),
        y=float(raw[3]),
        angle=float(raw[4]),
        from_planet_id=int(raw[5]),
        ships=int(raw[6]),
    )


def comet_ids_from_obs(obs: Any) -> set[int]:
    explicit = obs_get(obs, "comet_planet_ids", None)
    if explicit:
        return {int(pid) for pid in explicit}

    comet_ids: set[int] = set()
    for comet in obs_get(obs, "comets", []) or []:
        for pid in comet.get("planet_ids", []):
            comet_ids.add(int(pid))
    return comet_ids


def parse_observation(obs: Any) -> LocalObs:
    player = int(obs_get(obs, "player", -2))
    planets = [parse_planet(p) for p in obs_get(obs, "planets", [])]
    fleets = [parse_fleet(f, i) for i, f in enumerate(obs_get(obs, "fleets", []))]
    initial_raw = obs_get(obs, "initial_planets", []) or []
    initial_planets = [parse_planet(p) for p in initial_raw]

    return LocalObs(
        planets=planets,
        mine=[p for p in planets if p.owner == player],
        targets=[p for p in planets if p.owner != player],
        fleets=fleets,
        player=player,
        step=int(obs_get(obs, "step", 0)),
        angular_velocity=float(obs_get(obs, "angular_velocity", 0.0) or 0.0),
        initial_planets=initial_planets,
        comet_planet_ids=comet_ids_from_obs(obs),
    )
