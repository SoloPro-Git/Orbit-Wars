"""Parallel 4-player evaluator for strategy variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KAGGLE_ENVS_LOG_LEVEL", "ERROR")

from kaggle_environments import make

ROOT = Path("/data2/solo/Orbit-Wars")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulebase.kaggle_public_rl_informed_strategies.rl_informed_agent import RLInformedPublicRuleAgent
from rulebase.kaggle_public_rl_informed_strategies.strategy_config import (
    ABLATION_SUITES,
    CHAMPION_OPPONENT_VARIANTS,
)
from rulebase.kaggle_public_strategies.public_rule_agent import PublicRuleAgent

OUT_DIR = ROOT / "rulebase/kaggle_public_rl_informed_strategies/experiments"


def make_config_agent(params: dict):
    instance = RLInformedPublicRuleAgent(**params)

    def agent(obs, configuration=None):
        return instance.act(obs)

    return agent


def make_public_agent():
    instance = PublicRuleAgent()

    def agent(obs, configuration=None):
        return instance.act(obs)

    return agent


def make_named_agent(name: str):
    params = CHAMPION_OPPONENT_VARIANTS[name]
    if params is None:
        return make_public_agent()
    return make_config_agent(dict(params))


def run_one(seed: int, variant_seat: int, variant_name: str, params: dict, opponents: list[str]) -> dict:
    agents = [make_named_agent(opponents[i % len(opponents)]) for i in range(3)]
    agents.insert(variant_seat, make_config_agent(dict(params)))
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run(agents)
    rewards = [env.steps[-1][idx].reward for idx in range(4)]
    variant_reward = rewards[variant_seat]
    rank = 1 + sum(reward > variant_reward for reward in rewards)
    tied_best = variant_reward == max(rewards) and variant_reward > 0
    eliminated = variant_reward < max(rewards)
    return {
        "variant": variant_name,
        "seed": seed,
        "seat": f"p{variant_seat}",
        "opponents": ",".join(opponents),
        "variant_reward": variant_reward,
        "rewards": ";".join(str(r) for r in rewards),
        "rank": rank,
        "win": bool(tied_best),
        "loss": bool(eliminated),
        "steps": len(env.steps),
    }


def run_task(task: dict) -> dict:
    return run_one(
        seed=int(task["seed"]),
        variant_seat=int(task["variant_seat"]),
        variant_name=str(task["variant"]),
        params=dict(task["params"]),
        opponents=list(task["opponents"]),
    )


def build_tasks(variants: dict[str, dict], games_per_seat: int, opponents: list[str]) -> list[dict]:
    tasks = []
    for name, params in variants.items():
        for seat in range(4):
            for i in range(games_per_seat):
                tasks.append(
                    {
                        "variant": name,
                        "params": params,
                        "seed": 9000 + i,
                        "variant_seat": seat,
                        "opponents": opponents,
                    }
                )
    return tasks


def summarize_variant(name: str, params: dict, rows: list[dict]) -> dict:
    wins = sum(1 for row in rows if row["win"])
    losses = sum(1 for row in rows if row["loss"])
    draws_or_ties = len(rows) - wins - losses
    avg_rank = sum(float(row["rank"]) for row in rows) / len(rows)
    return {
        "variant": name,
        "params": params,
        "games": len(rows),
        "wins": wins,
        "losses": losses,
        "draws_or_ties": draws_or_ties,
        "win_rate": wins / len(rows),
        "non_loss_rate": (wins + draws_or_ties) / len(rows),
        "avg_rank": avg_rank,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(ABLATION_SUITES), default="multiplayer_diplomacy")
    parser.add_argument("--opponents", nargs=3, choices=sorted(CHAMPION_OPPONENT_VARIANTS), default=["regular", "recapture_s45_e180_w50_b40_regular", "regular"])
    parser.add_argument("--games-per-seat", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    return parser.parse_args()


def main(suite: str, games_per_seat: int, workers: int, opponents: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = ABLATION_SUITES[suite]
    tasks = build_tasks(variants, games_per_seat, opponents)
    all_rows = []
    print(
        f"Running 4P suite={suite}, opponents={opponents}: {len(tasks)} games, "
        f"{len(variants)} variants, workers={workers}, games_per_seat={games_per_seat}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            all_rows.append(row)
            print(
                f"[{idx}/{len(tasks)}] {row['variant']} {row['seat']} seed={row['seed']} "
                f"reward={row['variant_reward']} rank={row['rank']}",
                flush=True,
            )

    rows_by_variant = {name: [] for name in variants}
    for row in all_rows:
        rows_by_variant[row["variant"]].append(row)
    summaries = [summarize_variant(name, params, rows_by_variant[name]) for name, params in variants.items()]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    opponent_tag = hashlib.sha1(",".join(opponents).encode("utf-8")).hexdigest()[:8]
    json_path = OUT_DIR / f"{suite}_4p_ablation_{stamp}_{opponent_tag}.json"
    csv_path = OUT_DIR / f"{suite}_4p_ablation_{stamp}_{opponent_tag}.csv"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suite": suite,
        "opponents": opponents,
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

    print("\n==== 4P SUMMARY TABLE ====")
    for row in sorted(summaries, key=lambda r: (r["win_rate"], -r["avg_rank"]), reverse=True):
        print(
            f"{row['variant']}: {row['wins']}-{row['losses']}-{row['draws_or_ties']} "
            f"win_rate={row['win_rate']:.2f} non_loss={row['non_loss_rate']:.2f} avg_rank={row['avg_rank']:.2f} "
            f"params={row['params']}"
        )
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    main(suite=args.suite, games_per_seat=args.games_per_seat, workers=args.workers, opponents=args.opponents)
