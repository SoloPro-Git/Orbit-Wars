"""Parallel 2-player ablation runner backed by the fast Orbit Wars simulator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path("/data2/solo/Orbit-Wars")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulebase.kaggle_public_rl_informed_strategies.rl_informed_agent import RLInformedPublicRuleAgent
from rulebase.kaggle_public_rl_informed_strategies.strategy_config import (
    ABLATION_SUITES,
    CHAMPION_OPPONENT_VARIANTS,
    REGULAR_CONFIG,
)
from rulebase.kaggle_public_strategies.public_rule_agent import PublicRuleAgent
from training2.fast_orbit_wars import make_fast_orbit_wars

OUT_DIR = ROOT / "rulebase/kaggle_public_rl_informed_strategies/experiments"


def make_variant_agent(params: dict):
    instance = RLInformedPublicRuleAgent(**params)

    def agent(obs, configuration=None):
        return instance.act(obs)

    return agent


def make_public_agent():
    instance = PublicRuleAgent()

    def agent(obs, configuration=None):
        return instance.act(obs)

    return agent


def make_config_agent(params: dict):
    instance = RLInformedPublicRuleAgent(**params)

    def agent(obs, configuration=None):
        return instance.act(obs)

    return agent


def make_regular_agent():
    return make_config_agent(REGULAR_CONFIG.to_agent_kwargs())


def make_opponent_agent(opponent: str):
    if opponent == "public_original":
        return make_public_agent()
    if opponent == "regular":
        return make_regular_agent()
    if opponent in CHAMPION_OPPONENT_VARIANTS:
        params = CHAMPION_OPPONENT_VARIANTS[opponent]
        if params is None:
            return make_public_agent()
        return make_config_agent(dict(params))
    raise ValueError(f"Unknown opponent: {opponent}")


def run_one(seed: int, variant_as_p0: bool, params: dict, opponent: str, use_numba: bool) -> dict:
    variant = make_variant_agent(params)
    opponent_agent = make_opponent_agent(opponent)
    env = make_fast_orbit_wars({"seed": seed}, keep_history=False, use_numba=use_numba)
    if variant_as_p0:
        env.run([variant, opponent_agent])
        variant_reward = env.steps[-1][0]["reward"]
        opponent_reward = env.steps[-1][1]["reward"]
        seat = "variant_p0"
    else:
        env.run([opponent_agent, variant])
        opponent_reward = env.steps[-1][0]["reward"]
        variant_reward = env.steps[-1][1]["reward"]
        seat = "variant_p1"

    if variant_reward > opponent_reward:
        outcome = "win"
    elif variant_reward < opponent_reward:
        outcome = "loss"
    else:
        outcome = "draw"

    return {
        "seed": seed,
        "seat": seat,
        "variant_reward": variant_reward,
        "opponent": opponent,
        "opponent_reward": opponent_reward,
        "outcome": outcome,
        "steps": env.steps[-1][0]["observation"].get("step", None),
    }


def run_task(task: dict) -> dict:
    row = run_one(
        seed=int(task["seed"]),
        variant_as_p0=bool(task["variant_as_p0"]),
        params=dict(task["params"]),
        opponent=str(task["opponent"]),
        use_numba=bool(task["use_numba"]),
    )
    return {"variant": task["variant"], **row}


def summarize_variant(name: str, params: dict, rows: list[dict]) -> dict:
    wins = sum(1 for row in rows if row["outcome"] == "win")
    losses = sum(1 for row in rows if row["outcome"] == "loss")
    draws = sum(1 for row in rows if row["outcome"] == "draw")
    return {
        "variant": name,
        "params": params,
        "games": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(rows),
        "non_loss_rate": (wins + draws) / len(rows),
    }


def build_tasks(variants: dict[str, dict], games_per_seat: int, opponent: str, use_numba: bool) -> list[dict]:
    tasks = []
    for name, params in variants.items():
        for i in range(games_per_seat):
            tasks.append(
                {
                    "variant": name,
                    "params": params,
                    "seed": 42 + i,
                    "variant_as_p0": True,
                    "opponent": opponent,
                    "use_numba": use_numba,
                }
            )
        for i in range(games_per_seat):
            tasks.append(
                {
                    "variant": name,
                    "params": params,
                    "seed": 1042 + i,
                    "variant_as_p0": False,
                    "opponent": opponent,
                    "use_numba": use_numba,
                }
            )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(ABLATION_SUITES), default="additive")
    parser.add_argument("--opponent", choices=sorted(CHAMPION_OPPONENT_VARIANTS), default="regular")
    parser.add_argument("--games-per-seat", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--no-numba", action="store_true")
    return parser.parse_args()


def main(suite: str, games_per_seat: int, workers: int, opponent: str, use_numba: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = ABLATION_SUITES[suite]
    tasks = build_tasks(variants, games_per_seat, opponent, use_numba)
    all_rows = []

    print(
        f"Running FAST suite={suite}, opponent={opponent}: {len(tasks)} games, {len(variants)} variants, "
        f"workers={workers}, games_per_seat={games_per_seat}, numba={use_numba}",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            all_rows.append(row)
            print(
                f"[{idx}/{len(tasks)}] {row['variant']} {row['seat']} "
                f"seed={row['seed']} -> {row['outcome']}",
                flush=True,
            )

    rows_by_variant = {name: [] for name in variants}
    for row in all_rows:
        rows_by_variant[row["variant"]].append(row)

    summaries = [
        summarize_variant(name, params, rows_by_variant[name])
        for name, params in variants.items()
    ]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"{suite}_fast_ablation_{stamp}.json"
    csv_path = OUT_DIR / f"{suite}_fast_ablation_{stamp}.csv"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "environment": "fast_orbit_wars",
        "use_numba": use_numba,
        "suite": suite,
        "opponent": opponent,
        "games_per_seat": games_per_seat,
        "workers": workers,
        "summaries": summaries,
        "games": all_rows,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n==== FAST SUMMARY TABLE ====")
    for row in sorted(summaries, key=lambda r: r["win_rate"], reverse=True):
        print(
            f"{row['variant']}: {row['wins']}-{row['losses']}-{row['draws']} "
            f"win_rate={row['win_rate']:.2f} non_loss={row['non_loss_rate']:.2f} "
            f"params={row['params']}"
        )
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        suite=args.suite,
        games_per_seat=args.games_per_seat,
        workers=args.workers,
        opponent=args.opponent,
        use_numba=not args.no_numba,
    )
