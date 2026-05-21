"""Balanced 4P league for historical regular strategy variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path("/data2/solo/Orbit-Wars")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulebase.kaggle_public_rl_informed_strategies.rl_informed_agent import RLInformedPublicRuleAgent
from rulebase.kaggle_public_rl_informed_strategies.strategy_config import (
    CHAMPION_OPPONENT_VARIANTS,
    HISTORICAL_BEST_VARIANTS,
)
from training2.fast_orbit_wars import make_fast_orbit_wars

OUT_DIR = ROOT / "rulebase/kaggle_public_rl_informed_strategies/experiments"


def _params_key(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _preferred_name(name: str) -> tuple[int, str]:
    if name == "regular":
        return (0, name)
    if name == "regular_config":
        return (9, name)
    if name.startswith("unread"):
        return (1, name)
    if name.startswith("tail_"):
        return (2, name)
    if name.startswith("mp_"):
        return (3, name)
    return (4, name)


def variant_pool(pool: str, include_public: bool, names: list[str] | None) -> dict[str, dict]:
    raw = CHAMPION_OPPONENT_VARIANTS if pool == "champion" else HISTORICAL_BEST_VARIANTS
    selected = dict(raw)
    if names:
        selected = {name: raw[name] for name in names}
    if not include_public:
        selected = {name: params for name, params in selected.items() if params is not None}

    by_hash: dict[str, tuple[str, dict]] = {}
    for name, params in selected.items():
        if params is None:
            continue
        key = _params_key(params)
        current = by_hash.get(key)
        if current is None or _preferred_name(name) < _preferred_name(current[0]):
            by_hash[key] = (name, dict(params))
    return dict(sorted((name, params) for name, params in by_hash.values()))


def make_agent(params: dict):
    instance = RLInformedPublicRuleAgent(**params)

    def agent(obs, configuration=None):
        return instance.act(obs)

    return agent


def build_tasks(
    variants: dict[str, dict],
    games_per_seat: int,
    base_seed: int,
    episode_steps: int,
    use_numba: bool,
) -> list[dict]:
    names = list(variants)
    tasks = []
    for variant_index, name in enumerate(names):
        for seat in range(4):
            for i in range(games_per_seat):
                rng = random.Random(base_seed * 1000003 + variant_index * 10007 + seat * 101 + i)
                opponents = [other for other in names if other != name]
                chosen = rng.sample(opponents, 3)
                agents = chosen[:]
                agents.insert(seat, name)
                tasks.append(
                    {
                        "seed": base_seed + i,
                        "variant": name,
                        "seat": seat,
                        "agents": agents,
                        "episode_steps": episode_steps,
                        "use_numba": use_numba,
                    }
                )
    return tasks


def run_task(task: dict, variants: dict[str, dict]) -> dict:
    agents = [make_agent(variants[name]) for name in task["agents"]]
    env = make_fast_orbit_wars(
        {"seed": int(task["seed"]), "episodeSteps": int(task["episode_steps"])},
        keep_history=False,
        use_numba=bool(task["use_numba"]),
    )
    env.run(agents)
    rewards = [env.steps[-1][idx]["reward"] for idx in range(4)]
    seat = int(task["seat"])
    reward = rewards[seat]
    rank = 1 + sum(other > reward for other in rewards)
    win = reward == max(rewards) and reward > 0
    return {
        "variant": task["variant"],
        "seed": task["seed"],
        "seat": f"p{seat}",
        "agents": ",".join(task["agents"]),
        "opponents": ",".join(name for idx, name in enumerate(task["agents"]) if idx != seat),
        "variant_reward": reward,
        "rewards": ";".join(str(value) for value in rewards),
        "rank": rank,
        "win": bool(win),
    }


def summarize(rows: list[dict]) -> list[dict]:
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    summaries = []
    for name, group in by_variant.items():
        games = len(group)
        wins = sum(str(row["win"]).lower() == "true" for row in group)
        avg_rank = sum(float(row["rank"]) for row in group) / games
        avg_reward = sum(float(row["variant_reward"]) for row in group) / games
        by_seat = {}
        for seat in ("p0", "p1", "p2", "p3"):
            seat_rows = [row for row in group if row["seat"] == seat]
            seat_wins = sum(str(row["win"]).lower() == "true" for row in seat_rows)
            by_seat[seat] = {
                "games": len(seat_rows),
                "wins": seat_wins,
                "win_rate": seat_wins / len(seat_rows) if seat_rows else 0.0,
            }
        summaries.append(
            {
                "variant": name,
                "games": games,
                "wins": wins,
                "win_rate": wins / games,
                "avg_rank": avg_rank,
                "avg_reward": avg_reward,
                "seat": by_seat,
            }
        )
    return sorted(summaries, key=lambda row: (row["win_rate"], -row["avg_rank"], row["avg_reward"]), reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=["champion", "historical"], default="champion")
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--include-public", action="store_true")
    parser.add_argument("--games-per-seat", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=12000)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(64, max(1, os.cpu_count() or 1)))
    parser.add_argument("--no-numba", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = variant_pool(args.pool, args.include_public, args.names)
    tasks = build_tasks(
        variants=variants,
        games_per_seat=args.games_per_seat,
        base_seed=args.base_seed,
        episode_steps=args.episode_steps,
        use_numba=not args.no_numba,
    )
    print(
        f"Running historical regular league pool={args.pool} variants={len(variants)} "
        f"games={len(tasks)} games_per_seat={args.games_per_seat} workers={args.workers} "
        f"numba={not args.no_numba}",
        flush=True,
    )
    print("variants:")
    for name in variants:
        print(f"  {name}")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, task, variants) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if idx <= 10 or idx % 50 == 0 or idx == len(tasks):
                print(
                    f"[{idx}/{len(tasks)}] {row['variant']} {row['seat']} "
                    f"seed={row['seed']} reward={row['variant_reward']} rank={row['rank']}",
                    flush=True,
                )

    summaries = summarize(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tag = hashlib.sha1(",".join(variants).encode("utf-8")).hexdigest()[:8]
    json_path = OUT_DIR / f"historical_regular_league_{args.pool}_{stamp}_{tag}.json"
    csv_path = OUT_DIR / f"historical_regular_league_{args.pool}_{stamp}_{tag}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "pool": args.pool,
                "variants": list(variants),
                "games_per_seat": args.games_per_seat,
                "base_seed": args.base_seed,
                "episode_steps": args.episode_steps,
                "use_numba": not args.no_numba,
                "summaries": summaries,
                "games": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n==== HISTORICAL REGULAR LEAGUE ====")
    for row in summaries:
        print(
            f"{row['variant']}: wins={row['wins']}/{row['games']} "
            f"wr={row['win_rate']:.3f} avg_rank={row['avg_rank']:.3f} "
            f"avg_reward={row['avg_reward']:.3f}"
        )
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")


if __name__ == "__main__":
    main()
