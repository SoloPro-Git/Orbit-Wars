"""Fast, Kaggle-like Orbit Wars simulator.

This module intentionally does not patch or import Kaggle's official
``orbit_wars.py`` implementation.  It mirrors the public rules and exposes the
small API shape used by this repository: ``reset(players)``, ``step(actions)``,
``run(agents)``, and ``steps[-1][pid]["observation"]``.
"""

from __future__ import annotations

import math
import random
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a project dependency.
    np = None

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration.
    njit = None


Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]
HAS_NUMBA = njit is not None and np is not None


class Struct(dict):
    """Tiny attr-access dict compatible with the agents in this repo."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


@dataclass
class FastConfig:
    episodeSteps: int = 500
    actTimeout: float = 1.0
    shipSpeed: float = 6.0
    sunRadius: float = 10.0
    boardSize: float = 100.0
    cometSpeed: float = 4.0
    seed: int | None = None


def _config_from(configuration: dict[str, Any] | None) -> FastConfig:
    cfg = FastConfig()
    for key, value in (configuration or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def distance(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_segment_distance(p, v, w) -> float:
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v)
    t = max(
        0,
        min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2),
    )
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)


def _point_to_segment_distance_sq(px, py, vx, vy, wx, wy) -> float:
    dx = wx - vx
    dy = wy - vy
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        qx = px - vx
        qy = py - vy
        return qx * qx + qy * qy
    t = ((px - vx) * dx + (py - vy) * dy) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    proj_x = vx + t * dx
    proj_y = vy + t * dy
    qx = px - proj_x
    qy = py - proj_y
    return qx * qx + qy * qy


if HAS_NUMBA:

    @njit(cache=True)
    def _segment_distance_sq_numba(px, py, vx, vy, wx, wy):
        dx = wx - vx
        dy = wy - vy
        l2 = dx * dx + dy * dy
        if l2 == 0.0:
            qx = px - vx
            qy = py - vy
            return qx * qx + qy * qy
        t = ((px - vx) * dx + (py - vy) * dy) / l2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        proj_x = vx + t * dx
        proj_y = vy + t * dy
        qx = px - proj_x
        qy = py - proj_y
        return qx * qx + qy * qy


    @njit(cache=True)
    def _fleet_planet_hits_numba(old_new, planet_xyr, ship_speed):
        n_fleets = old_new.shape[0]
        n_planets = planet_xyr.shape[0]
        hit_planet = np.full(n_fleets, -1, np.int64)
        remove = np.zeros(n_fleets, np.bool_)
        log_1000 = math.log(1000.0)
        for i in range(n_fleets):
            old_x = old_new[i, 0]
            old_y = old_new[i, 1]
            angle = old_new[i, 2]
            ships = old_new[i, 3]
            speed = 1.0 + (ship_speed - 1.0) * (math.log(ships) / log_1000) ** 1.5
            if speed > ship_speed:
                speed = ship_speed
            new_x = old_x + math.cos(angle) * speed
            new_y = old_y + math.sin(angle) * speed
            old_new[i, 4] = new_x
            old_new[i, 5] = new_y
            for j in range(n_planets):
                radius = planet_xyr[j, 2]
                if _segment_distance_sq_numba(planet_xyr[j, 0], planet_xyr[j, 1], old_x, old_y, new_x, new_y) < radius * radius:
                    hit_planet[i] = j
                    remove[i] = True
                    break
            if remove[i]:
                continue
            if not (0.0 <= new_x <= BOARD_SIZE and 0.0 <= new_y <= BOARD_SIZE):
                remove[i] = True
                continue
            if _segment_distance_sq_numba(CENTER, CENTER, old_x, old_y, new_x, new_y) < SUN_RADIUS * SUN_RADIUS:
                remove[i] = True
        return hit_planet, remove


    @njit(cache=True)
    def _sweep_hits_numba(fleet_xy, initial_remove, segments):
        n_fleets = fleet_xy.shape[0]
        n_segments = segments.shape[0]
        remove = initial_remove.copy()
        hit_segment = np.full(n_fleets, -1, np.int64)
        for s in range(n_segments):
            old_x = segments[s, 0]
            old_y = segments[s, 1]
            new_x = segments[s, 2]
            new_y = segments[s, 3]
            radius = segments[s, 4]
            radius_sq = radius * radius
            for i in range(n_fleets):
                if remove[i]:
                    continue
                if _segment_distance_sq_numba(fleet_xy[i, 0], fleet_xy[i, 1], old_x, old_y, new_x, new_y) < radius_sq:
                    hit_segment[i] = s
                    remove[i] = True
        return hit_segment, remove



def generate_planets(rng=None):
    if rng is None:
        rng = random
    planets = []
    num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    id_counter = 0
    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        angle = rng.uniform(0, math.pi / 2)
        min_orbital = ROTATION_RADIUS_LIMIT - r
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
        orbital_r = rng.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)
        if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
            continue
        if (BOARD_SIZE - x) - r < 0 or (BOARD_SIZE - y) - r < 0:
            continue
        if (x - CENTER) < r + 5 or (y - CENTER) < r + 5:
            continue
        ships = min(rng.randint(5, 99), rng.randint(5, 99))
        temp_planets = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]
        valid = True
        for tp in temp_planets:
            for p in planets:
                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            planets.extend(temp_planets)
            id_counter += 4
            static_groups += 1

    attempts = 0
    max_attempts = 5000
    has_orbiting = False
    while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < max_attempts):
        attempts += 1
        if attempts >= max_attempts:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        x = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        y = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        orbital_radius = distance((x, y), (CENTER, CENTER))
        if orbital_radius < SUN_RADIUS + r + 10:
            continue
        if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
            if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                continue
        valid = True
        ships = rng.randint(5, 30)
        temp_planets = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]
        for tp in temp_planets:
            tp_orbital = distance((tp[2], tp[3]), (CENTER, CENTER))
            tp_is_rotating = tp_orbital + tp[4] < ROTATION_RADIUS_LIMIT
            for p in planets:
                p_orbital = distance((p[2], p[3]), (CENTER, CENTER))
                p_is_rotating = p_orbital + p[4] < ROTATION_RADIUS_LIMIT
                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
                if tp_is_rotating != p_is_rotating:
                    if abs(tp_orbital - p_orbital) < tp[4] + p[4] + PLANET_CLEARANCE:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            if orbital_radius + r < ROTATION_RADIUS_LIMIT:
                has_orbiting = True
            planets.extend(temp_planets)
            id_counter += 4
    return planets


def generate_comet_paths(
    initial_planets,
    angular_velocity,
    spawn_step,
    comet_planet_ids=None,
    comet_speed=4.0,
    rng=None,
):
    if rng is None:
        rng = random
    comet_planet_ids = set(comet_planet_ids or [])
    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue
        b = a * math.sqrt(1 - e**2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)
        dense = []
        num = 5000
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            dense.append((CENTER + ex * cos_phi - ey * sin_phi, CENTER + ex * sin_phi + ey * cos_phi))
        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            dx = dense[i][0] - dense[i - 1][0]
            dy = dense[i][1] - dense[i - 1][1]
            cum += math.sqrt(dx * dx + dy * dy)
            if cum >= target:
                path.append(dense[i])
                target += comet_speed
        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i
        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue
        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]
        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if planet[0] in comet_planet_ids:
                continue
            dx = planet[2] - CENTER
            dy = planet[3] - CENTER
            pr = math.sqrt(dx * dx + dy * dy)
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)
        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            dx = cx - CENTER
            dy = cy - CENTER
            if math.sqrt(dx * dx + dy * dy) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break
            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    dx = sp[0] - planet[2]
                    dy = sp[1] - planet[3]
                    if math.sqrt(dx * dx + dy * dy) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx**2 + dy**2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    dx = sp[0] - px
                    dy = sp[1] - py
                    if math.sqrt(dx * dx + dy * dy) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return paths
    return None


def _copy_planets(planets: list[list]) -> list[list]:
    return [p.copy() for p in planets]


def _copy_fleets(fleets: list[list]) -> list[list]:
    return [f.copy() for f in fleets]


def _copy_comets(comets: list[dict]) -> list[dict]:
    return [
        {
            "planet_ids": list(group["planet_ids"]),
            "paths": [[pt.copy() for pt in path] for path in group["paths"]],
            "path_index": group["path_index"],
        }
        for group in comets
    ]


class FastOrbitWarsEnv:
    """A lightweight environment with the subset of Kaggle API used locally."""

    def __init__(
        self,
        configuration: dict[str, Any] | None = None,
        debug: bool = False,
        keep_history: bool = True,
        copy_observations: bool = True,
        use_numba: bool = False,
    ):
        self.configuration = _config_from(configuration)
        self.debug = debug
        self.keep_history = keep_history
        self.copy_observations = copy_observations
        self.use_numba = use_numba and HAS_NUMBA
        self.steps: list[list[Struct]] = []
        self.done = False
        self.info: dict[str, Any] = {}
        self.num_agents = 0
        self._state_planets: list[list] = []
        self._initial_planets: list[list] = []
        self._fleets: list[list] = []
        self._comets: list[dict] = []
        self._comet_planet_ids: list[int] = []
        self._next_fleet_id = 0
        self._angular_velocity = 0.0
        self._step = 0
        self._initial_geom: dict[int, tuple[float, float, float, bool]] = {}

    def reset(self, num_agents: int = 2):
        self.num_agents = num_agents
        self.done = False
        seed = self.configuration.seed
        if seed is None:
            seed = random.randrange(2**31)
        self.info["seed"] = seed
        init_rng = random.Random(seed)
        self._angular_velocity = init_rng.uniform(0.025, 0.05)
        self._state_planets = generate_planets(init_rng)
        self._initial_planets = _copy_planets(self._state_planets)
        self._initial_geom = self._build_initial_geom(self._initial_planets)
        self._fleets = []
        self._next_fleet_id = 0
        self._comets = []
        self._comet_planet_ids = []
        self._step = 0

        num_groups = len(self._state_planets) // 4
        if num_groups > 0:
            home_group = init_rng.randint(0, num_groups - 1)
            base = home_group * 4
            if num_agents == 2:
                self._state_planets[base][1] = 0
                self._state_planets[base][5] = 10
                self._state_planets[base + 3][1] = 1
                self._state_planets[base + 3][5] = 10
            elif num_agents == 4:
                for j in range(4):
                    self._state_planets[base + j][1] = j
                    self._state_planets[base + j][5] = 10
        self.steps = [self._make_step([[] for _ in range(num_agents)], [0] * num_agents, ["ACTIVE"] * num_agents)]
        return self.steps[-1]

    def step(self, actions: list[list]):
        if self.done:
            raise RuntimeError("Environment done")
        if not self.steps:
            self.reset(2)
        actions = list(actions)
        if len(actions) < self.num_agents:
            actions.extend([[] for _ in range(self.num_agents - len(actions))])
        actions = actions[: self.num_agents]
        rewards = [0] * self.num_agents
        statuses = ["ACTIVE"] * self.num_agents
        self._advance(actions, rewards, statuses)
        frame = self._make_step(actions, rewards, statuses)
        if self.keep_history:
            self.steps.append(frame)
        else:
            self.steps[:] = [frame]
        return frame

    def run(self, agents: list[Callable | str]):
        self.reset(len(agents))
        while not self.done:
            actions = []
            for pid, agent in enumerate(agents):
                if isinstance(agent, str):
                    agent_fn = random_agent if agent == "random" else starter_agent
                else:
                    agent_fn = agent
                obs = self.steps[-1][pid]["observation"]
                try:
                    action = agent_fn(obs, self.configuration)
                except TypeError:
                    action = agent_fn(obs)
                actions.append(action)
            self.step(actions)
        return self.steps

    def _observation(self, player: int, snapshot: tuple | None = None) -> Struct:
        if snapshot is None:
            snapshot = self._observation_snapshot()
        planets, initial_planets, fleets, comets, comet_planet_ids = snapshot
        obs = Struct(
            player=player,
            angular_velocity=self._angular_velocity,
            planets=planets,
            initial_planets=initial_planets,
            fleets=fleets,
            next_fleet_id=self._next_fleet_id,
            comets=comets,
            comet_planet_ids=comet_planet_ids,
            remainingOverageTime=60,
        )
        if player == 0:
            obs.step = self._step
        return obs

    def _observation_snapshot(self) -> tuple:
        if not self.copy_observations:
            return (
                self._state_planets,
                self._initial_planets,
                self._fleets,
                self._comets,
                self._comet_planet_ids,
            )
        return (
            _copy_planets(self._state_planets),
            _copy_planets(self._initial_planets),
            _copy_fleets(self._fleets),
            _copy_comets(self._comets),
            list(self._comet_planet_ids),
        )

    def _make_step(self, actions: list[list], rewards: list[int], statuses: list[str]) -> list[Struct]:
        snapshot = self._observation_snapshot()
        return [
            Struct(
                action=actions[pid] if pid < len(actions) else [],
                reward=rewards[pid],
                info={},
                observation=self._observation(pid, snapshot),
                status=statuses[pid],
            )
            for pid in range(self.num_agents)
        ]

    @staticmethod
    def _build_initial_geom(planets: list[list]) -> dict[int, tuple[float, float, float, bool]]:
        geom = {}
        for planet in planets:
            dx = planet[2] - CENTER
            dy = planet[3] - CENTER
            orbital_r = math.sqrt(dx**2 + dy**2)
            initial_angle = math.atan2(dy, dx)
            is_rotating = orbital_r + planet[4] < ROTATION_RADIUS_LIMIT
            geom[planet[0]] = (orbital_r, initial_angle, planet[4], is_rotating)
        return geom

    def _expire_comets(self) -> None:
        expired = []
        for group in self._comets:
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                if idx >= len(group["paths"][i]):
                    expired.append(pid)
        if expired:
            expired_set = set(expired)
            self._state_planets = [p for p in self._state_planets if p[0] not in expired_set]
            self._initial_planets = [p for p in self._initial_planets if p[0] not in expired_set]
            self._comet_planet_ids = [pid for pid in self._comet_planet_ids if pid not in expired_set]
            for group in self._comets:
                group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
            self._comets = [g for g in self._comets if g["planet_ids"]]

    def _advance(self, actions: list[list], rewards: list[int], statuses: list[str]) -> None:
        self._expire_comets()
        point_segment_dist_sq = _point_to_segment_distance_sq
        step = self._step
        if (step + 1) in COMET_SPAWN_STEPS:
            episode_seed = self.info.get("seed", 0) or 0
            comet_rng = random.Random(f"orbit_wars-comet-{episode_seed}-{step + 1}")
            comet_paths = generate_comet_paths(
                self._initial_planets,
                self._angular_velocity,
                step + 1,
                self._comet_planet_ids,
                self.configuration.cometSpeed,
                rng=comet_rng,
            )
            if comet_paths:
                next_id = max(p[0] for p in self._state_planets) + 1
                comet_ships = min(
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                    comet_rng.randint(1, 99),
                )
                group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
                for i in range(4):
                    pid = next_id + i
                    group["planet_ids"].append(pid)
                    self._comet_planet_ids.append(pid)
                    planet = [pid, -1, -99, -99, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
                    self._state_planets.append(planet)
                    self._initial_planets.append(planet[:])
                    self._initial_geom[pid] = (
                        math.sqrt((-99 - CENTER) ** 2 + (-99 - CENTER) ** 2),
                        math.atan2(-99 - CENTER, -99 - CENTER),
                        COMET_RADIUS,
                        False,
                    )
                self._comets.append(group)

        planets_by_id = {p[0]: p for p in self._state_planets}
        for player_id, action in enumerate(actions):
            if not action or not isinstance(action, list):
                continue
            for move in action:
                if len(move) != 3:
                    continue
                from_id, angle, ships = move
                ships = int(ships)
                from_planet = planets_by_id.get(from_id)
                if from_planet is not None and from_planet[1] == player_id:
                    if from_planet[5] >= ships and ships > 0:
                        from_planet[5] -= ships
                        start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
                        start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
                        self._fleets.append([
                            self._next_fleet_id,
                            player_id,
                            start_x,
                            start_y,
                            angle,
                            from_id,
                            ships,
                        ])
                        self._next_fleet_id += 1

        for planet in self._state_planets:
            if planet[1] != -1:
                planet[5] += planet[6]

        fleets_to_remove: set[int] = set()
        combat_lists = {p[0]: [] for p in self._state_planets}
        max_speed = self.configuration.shipSpeed
        state_planets = self._state_planets
        fleets = self._fleets
        if self.use_numba and fleets and state_planets:
            old_new = np.empty((len(fleets), 6), dtype=np.float64)
            for i, fleet in enumerate(fleets):
                old_new[i, 0] = fleet[2]
                old_new[i, 1] = fleet[3]
                old_new[i, 2] = fleet[4]
                old_new[i, 3] = fleet[6]
                old_new[i, 4] = fleet[2]
                old_new[i, 5] = fleet[3]
            planet_xyr = np.empty((len(state_planets), 3), dtype=np.float64)
            for i, planet in enumerate(state_planets):
                planet_xyr[i, 0] = planet[2]
                planet_xyr[i, 1] = planet[3]
                planet_xyr[i, 2] = planet[4]
            hit_planets, remove_flags = _fleet_planet_hits_numba(old_new, planet_xyr, max_speed)
            for i, fleet in enumerate(fleets):
                fleet[2] = float(old_new[i, 4])
                fleet[3] = float(old_new[i, 5])
                hit_idx = int(hit_planets[i])
                if hit_idx >= 0:
                    combat_lists[state_planets[hit_idx][0]].append(fleet)
                if bool(remove_flags[i]):
                    fleets_to_remove.add(fleet[0])
        else:
            log_1000 = math.log(1000)
            for fleet in fleets:
                angle = fleet[4]
                ships = fleet[6]
                speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / log_1000) ** 1.5
                speed = min(speed, max_speed)
                old_x = fleet[2]
                old_y = fleet[3]
                fleet[2] += math.cos(angle) * speed
                fleet[3] += math.sin(angle) * speed
                new_x = fleet[2]
                new_y = fleet[3]
                hit_planet = False
                for planet in state_planets:
                    radius = planet[4]
                    if point_segment_dist_sq(planet[2], planet[3], old_x, old_y, new_x, new_y) < radius**2:
                        combat_lists[planet[0]].append(fleet)
                        fleets_to_remove.add(fleet[0])
                        hit_planet = True
                        break
                if hit_planet:
                    continue
                if not (0 <= new_x <= BOARD_SIZE and 0 <= new_y <= BOARD_SIZE):
                    fleets_to_remove.add(fleet[0])
                    continue
                if point_segment_dist_sq(CENTER, CENTER, old_x, old_y, new_x, new_y) < SUN_RADIUS**2:
                    fleets_to_remove.add(fleet[0])
                    continue

        comet_pid_set = set(self._comet_planet_ids)
        sweep_candidates = self._fleets
        def sweep_fleets(planet, old_pos, new_pos):
            if old_pos == new_pos:
                return
            old_x, old_y = old_pos
            new_x, new_y = new_pos
            radius = planet[4]
            radius_sq = radius**2
            for fleet in sweep_candidates:
                if fleet[0] in fleets_to_remove:
                    continue
                if point_segment_dist_sq(fleet[2], fleet[3], old_x, old_y, new_x, new_y) < radius_sq:
                    combat_lists[planet[0]].append(fleet)
                    fleets_to_remove.add(fleet[0])

        move_step = self._step
        regular_sweep_segments = []
        regular_sweep_planet_ids = []
        for planet in self._state_planets:
            if planet[0] in comet_pid_set:
                continue
            initial = self._initial_geom.get(planet[0])
            if not initial:
                continue
            orbital_r, initial_angle, _radius, is_rotating = initial
            old_pos = (planet[2], planet[3])
            if is_rotating:
                current_angle = initial_angle + self._angular_velocity * move_step
                planet[2] = CENTER + orbital_r * math.cos(current_angle)
                planet[3] = CENTER + orbital_r * math.sin(current_angle)
            new_pos = (planet[2], planet[3])
            if self.use_numba and sweep_candidates and old_pos != new_pos:
                regular_sweep_segments.append([old_pos[0], old_pos[1], new_pos[0], new_pos[1], planet[4]])
                regular_sweep_planet_ids.append(planet[0])
            else:
                sweep_fleets(planet, old_pos, new_pos)

        if self.use_numba and regular_sweep_segments:
            fleet_xy = np.empty((len(sweep_candidates), 2), dtype=np.float64)
            initial_remove = np.zeros(len(sweep_candidates), dtype=np.bool_)
            for i, fleet in enumerate(sweep_candidates):
                fleet_xy[i, 0] = fleet[2]
                fleet_xy[i, 1] = fleet[3]
                initial_remove[i] = fleet[0] in fleets_to_remove
            segments = np.asarray(regular_sweep_segments, dtype=np.float64)
            hit_segments, remove_flags = _sweep_hits_numba(fleet_xy, initial_remove, segments)
            for i, fleet in enumerate(sweep_candidates):
                hit_idx = int(hit_segments[i])
                if hit_idx >= 0:
                    combat_lists[regular_sweep_planet_ids[hit_idx]].append(fleet)
                if bool(remove_flags[i]) and not bool(initial_remove[i]):
                    fleets_to_remove.add(fleet[0])

        expired_comet_pids = []
        comet_sweep_segments = []
        comet_sweep_planet_ids = []
        for group in self._comets:
            group["path_index"] += 1
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                planet = planets_by_id.get(pid)
                if planet is None:
                    continue
                p_path = group["paths"][i]
                if idx >= len(p_path):
                    expired_comet_pids.append(pid)
                else:
                    old_pos = (planet[2], planet[3])
                    planet[2] = p_path[idx][0]
                    planet[3] = p_path[idx][1]
                    if old_pos[0] >= 0:
                        new_pos = (planet[2], planet[3])
                        if self.use_numba and sweep_candidates and old_pos != new_pos:
                            comet_sweep_segments.append([old_pos[0], old_pos[1], new_pos[0], new_pos[1], planet[4]])
                            comet_sweep_planet_ids.append(planet[0])
                        else:
                            sweep_fleets(planet, old_pos, new_pos)

        if self.use_numba and comet_sweep_segments:
            fleet_xy = np.empty((len(sweep_candidates), 2), dtype=np.float64)
            initial_remove = np.zeros(len(sweep_candidates), dtype=np.bool_)
            for i, fleet in enumerate(sweep_candidates):
                fleet_xy[i, 0] = fleet[2]
                fleet_xy[i, 1] = fleet[3]
                initial_remove[i] = fleet[0] in fleets_to_remove
            segments = np.asarray(comet_sweep_segments, dtype=np.float64)
            hit_segments, remove_flags = _sweep_hits_numba(fleet_xy, initial_remove, segments)
            for i, fleet in enumerate(sweep_candidates):
                hit_idx = int(hit_segments[i])
                if hit_idx >= 0:
                    combat_lists[comet_sweep_planet_ids[hit_idx]].append(fleet)
                if bool(remove_flags[i]) and not bool(initial_remove[i]):
                    fleets_to_remove.add(fleet[0])

        if expired_comet_pids:
            expired_set = set(expired_comet_pids)
            self._state_planets = [p for p in self._state_planets if p[0] not in expired_set]
            self._initial_planets = [p for p in self._initial_planets if p[0] not in expired_set]
            self._comet_planet_ids = [pid for pid in self._comet_planet_ids if pid not in expired_set]
            for group in self._comets:
                group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in expired_set]
            self._comets = [g for g in self._comets if g["planet_ids"]]

        if fleets_to_remove:
            self._fleets = [f for f in self._fleets if f[0] not in fleets_to_remove]

        planets_by_id = {p[0]: p for p in self._state_planets}
        for pid, planet_fleets in combat_lists.items():
            planet = planets_by_id.get(pid)
            if not planet or not planet_fleets:
                continue
            player_ships = {}
            for fleet in planet_fleets:
                owner = fleet[1]
                player_ships[owner] = player_ships.get(owner, 0) + fleet[6]
            sorted_players = sorted(player_ships.items(), key=lambda item: item[1], reverse=True)
            top_player, top_ships = sorted_players[0]
            if len(sorted_players) > 1:
                survivor_ships = top_ships - sorted_players[1][1]
                if sorted_players[0][1] == sorted_players[1][1]:
                    survivor_ships = 0
                survivor_owner = top_player if survivor_ships > 0 else -1
            else:
                survivor_owner = top_player
                survivor_ships = top_ships
            if survivor_ships > 0:
                if planet[1] == survivor_owner:
                    planet[5] += survivor_ships
                else:
                    planet[5] -= survivor_ships
                    if planet[5] < 0:
                        planet[1] = survivor_owner
                        planet[5] = abs(planet[5])

        terminated = False
        if self._step >= self.configuration.episodeSteps - 2:
            terminated = True
        alive_players = set()
        for p in self._state_planets:
            if p[1] != -1:
                alive_players.add(p[1])
        for f in self._fleets:
            alive_players.add(f[1])
        if len(alive_players) <= 1:
            terminated = True
        if terminated:
            self.done = True
            for i in range(self.num_agents):
                statuses[i] = "DONE"
            scores = [0] * self.num_agents
            for p in self._state_planets:
                if p[1] != -1:
                    scores[p[1]] += p[5]
            for f in self._fleets:
                scores[f[1]] += f[6]
            max_score = max(scores)
            for i in range(self.num_agents):
                rewards[i] = 1 if scores[i] == max_score and max_score > 0 else -1
        self._step += 1


def make_fast_orbit_wars(
    configuration: dict[str, Any] | None = None,
    debug: bool = False,
    keep_history: bool = True,
    copy_observations: bool = True,
    use_numba: bool = False,
) -> FastOrbitWarsEnv:
    return FastOrbitWarsEnv(
        configuration=configuration,
        debug=debug,
        keep_history=keep_history,
        copy_observations=copy_observations,
        use_numba=use_numba,
    )


def random_agent(obs):
    moves = []
    player = obs.get("player", 0)
    planets = [Planet(*p) for p in obs.get("planets", [])]
    for p in planets:
        if p.owner == player and p.ships > 0:
            angle = random.uniform(0, 2 * math.pi)
            ships = p.ships // 2
            if ships >= 20:
                moves.append([p.id, angle, ships])
    return moves


def starter_agent(obs):
    moves = []
    player = obs.get("player", 0)
    planets = [Planet(*p) for p in obs.get("planets", [])]
    static_targets = []
    for p in planets:
        orbital_r = math.sqrt((p.x - CENTER) ** 2 + (p.y - CENTER) ** 2)
        if orbital_r + p.radius >= ROTATION_RADIUS_LIMIT and p.owner != player:
            static_targets.append(p)
    my_planets = [p for p in planets if p.owner == player]
    for mp in my_planets:
        if mp.ships <= 0:
            continue
        closest = None
        min_dist = float("inf")
        for t in static_targets:
            dist = math.sqrt((mp.x - t.x) ** 2 + (mp.y - t.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest = t
        if closest:
            angle = math.atan2(closest.y - mp.y, closest.x - mp.x)
            ships = mp.ships // 2
            if ships >= 20:
                moves.append([mp.id, angle, ships])
    return moves
