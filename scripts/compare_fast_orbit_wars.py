"""Compare the fast Orbit Wars simulator against Kaggle's official env."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kaggle_environments import make

from training2.fast_orbit_wars import make_fast_orbit_wars


def raw_obs(env: Any, pid: int) -> Any:
    return env.steps[-1][pid]["observation"]


def getv(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def deterministic_policy(obs: Any, rng: random.Random) -> list[list[float]]:
    player = getv(obs, "player", 0)
    planets = getv(obs, "planets", [])
    moves = []
    targets = [p for p in planets if int(p[1]) != player]
    if not targets:
        return moves
    for p in planets:
        if int(p[1]) != player or float(p[5]) < 8:
            continue
        if rng.random() > 0.45:
            continue
        target = min(
            targets,
            key=lambda t: math.hypot(float(p[2]) - float(t[2]), float(p[3]) - float(t[3])) + rng.random() * 3.0,
        )
        angle = math.atan2(float(target[3]) - float(p[3]), float(target[2]) - float(p[2]))
        ships = max(1, int(float(p[5]) * rng.uniform(0.25, 0.75)))
        moves.append([p[0], angle, ships])
    return moves


def normalize_obs(obs: Any) -> dict[str, Any]:
    return {
        "planets": getv(obs, "planets", []),
        "initial_planets": getv(obs, "initial_planets", []),
        "fleets": getv(obs, "fleets", []),
        "comets": getv(obs, "comets", []),
        "comet_planet_ids": getv(obs, "comet_planet_ids", []),
        "next_fleet_id": getv(obs, "next_fleet_id", None),
        "angular_velocity": getv(obs, "angular_velocity", None),
        "step": getv(obs, "step", None),
    }


def close(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    return a == b


def assert_same(official: Any, fast: Any, players: int, step: int, tol: float) -> None:
    for pid in range(players):
        os = official.steps[-1][pid]
        fs = fast.steps[-1][pid]
        if os["status"] != fs["status"] or os["reward"] != fs["reward"]:
            raise AssertionError(
                f"status/reward mismatch step={step} pid={pid}: "
                f"{os['status']}/{os['reward']} vs {fs['status']}/{fs['reward']}"
            )
        oo = normalize_obs(os["observation"])
        fo = normalize_obs(fs["observation"])
        if not close(oo, fo, tol):
            raise AssertionError(
                "observation mismatch "
                + json.dumps({"step": step, "pid": pid, "official": oo, "fast": fo}, default=str)[:4000]
            )


def compare(seed: int, players: int, episode_steps: int, tol: float, use_numba: bool) -> int:
    official = make("orbit_wars", configuration={"episodeSteps": episode_steps, "seed": seed}, debug=True)
    fast = make_fast_orbit_wars({"episodeSteps": episode_steps, "seed": seed}, use_numba=use_numba)
    official.reset(players)
    fast.reset(players)
    assert_same(official, fast, players, 0, tol)
    rng = random.Random(seed + 17)
    steps = 0
    for step in range(episode_steps):
        actions = [deterministic_policy(raw_obs(official, pid), rng) for pid in range(players)]
        official.step(actions)
        fast.step(actions)
        steps = step + 1
        assert_same(official, fast, players, steps, tol)
        if all(s["status"] != "ACTIVE" for s in official.steps[-1]):
            break
    return steps


def bench(seed: int, players: int, episode_steps: int, games: int, use_numba: bool) -> dict[str, float]:
    action_traces = []
    for game in range(games):
        env = make("orbit_wars", configuration={"episodeSteps": episode_steps, "seed": seed + game}, debug=True)
        env.reset(players)
        rng = random.Random(seed + game + 17)
        game_actions = []
        for _ in range(episode_steps):
            actions = [deterministic_policy(raw_obs(env, pid), rng) for pid in range(players)]
            game_actions.append(actions)
            env.step(actions)
            if all(s["status"] != "ACTIVE" for s in env.steps[-1]):
                break
        action_traces.append(game_actions)

    t0 = time.perf_counter()
    for game, actions_list in enumerate(action_traces):
        env = make("orbit_wars", configuration={"episodeSteps": episode_steps, "seed": seed + game}, debug=True)
        env.reset(players)
        for actions in actions_list:
            env.step(actions)
            if all(s["status"] != "ACTIVE" for s in env.steps[-1]):
                break
    official_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for game, actions_list in enumerate(action_traces):
        env = make_fast_orbit_wars(
            {"episodeSteps": episode_steps, "seed": seed + game},
            keep_history=False,
            use_numba=use_numba,
        )
        env.reset(players)
        for actions in actions_list:
            env.step(actions)
            if all(s["status"] != "ACTIVE" for s in env.steps[-1]):
                break
    fast_s = time.perf_counter() - t0
    return {"official_s": official_s, "fast_s": fast_s, "speedup": official_s / max(fast_s, 1e-9)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=180)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--bench-games", type=int, default=3)
    parser.add_argument("--numba", action="store_true")
    args = parser.parse_args()

    total_steps = 0
    for seed in range(args.seeds):
        total_steps += compare(seed, args.players, args.episode_steps, args.tol, args.numba)
    metrics = bench(10_000, args.players, args.episode_steps, args.bench_games, args.numba)
    print(json.dumps({"matched_seeds": args.seeds, "matched_steps": total_steps, "numba": args.numba, **metrics}, indent=2))


if __name__ == "__main__":
    main()
