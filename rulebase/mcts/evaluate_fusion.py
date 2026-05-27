"""Small fast-simulator probes for rulebase.mcts fusion selectors."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulebase.kaggle_public_rl_informed_strategies.fast_ablation_eval import make_regular_agent
from rulebase.mcts.fusion import RuleFusionConfig, RuleFusionSelector
from training2 import make_fast_orbit_wars


def make_fusion_agent(config: RuleFusionConfig):
    instance = RuleFusionSelector(config)

    def agent(obs, configuration=None):
        return instance.act(obs, configuration)

    return agent


def run_one(task: dict) -> dict:
    mode = str(task["mode"])
    seat = int(task["seat"])
    seed = int(task["seed"])
    num_players = 2 if mode == "2p" else 4
    config = RuleFusionConfig(
        two_player_config=str(task["two_player_config"]),
        four_player_config=str(task["four_player_config"]),
    )
    env = make_fast_orbit_wars(
        {"seed": seed, "episodeSteps": int(task["episode_steps"])},
        keep_history=False,
        use_numba=bool(task["use_numba"]),
    )
    agents = [make_regular_agent() for _ in range(num_players)]
    agents[seat] = make_fusion_agent(config)
    env.run(agents)
    rewards = [env.steps[-1][idx]["reward"] for idx in range(num_players)]
    best_other = max(rewards[:seat] + rewards[seat + 1 :])
    variant_reward = rewards[seat]
    return {
        "mode": mode,
        "seed": seed,
        "seat": seat,
        "variant_reward": variant_reward,
        "best_other": best_other,
        "rewards": rewards,
        "win": variant_reward > best_other,
        "loss": variant_reward < best_other,
        "rank": 1 + sum(reward > variant_reward for reward in rewards),
    }


def build_tasks(args: argparse.Namespace) -> list[dict]:
    tasks = []
    if args.mode in ("2p", "both"):
        for i in range(args.games):
            for seat in (0, 1):
                tasks.append(
                    {
                        "mode": "2p",
                        "seed": args.seed_base_2p + i + seat * 1000,
                        "seat": seat,
                        "episode_steps": args.episode_steps,
                        "use_numba": args.use_numba,
                        "two_player_config": args.two_player_config,
                        "four_player_config": args.four_player_config,
                    }
                )
    if args.mode in ("4p", "both"):
        for i in range(args.games):
            for seat in range(4):
                tasks.append(
                    {
                        "mode": "4p",
                        "seed": args.seed_base_4p + i + seat * 100,
                        "seat": seat,
                        "episode_steps": args.episode_steps,
                        "use_numba": args.use_numba,
                        "two_player_config": args.two_player_config,
                        "four_player_config": args.four_player_config,
                    }
                )
    return tasks


def summarize(rows: list[dict]) -> None:
    for mode in ("2p", "4p"):
        subset = [row for row in rows if row["mode"] == mode]
        if not subset:
            continue
        wins = sum(1 for row in subset if row["win"])
        losses = sum(1 for row in subset if row["loss"])
        draws = len(subset) - wins - losses
        avg_rank = sum(float(row["rank"]) for row in subset) / len(subset)
        top = sum(1 for row in subset if row["variant_reward"] == 1)
        print(
            f"{mode}: strict={wins} loss={losses} draw={draws} "
            f"top={top}/{len(subset)} avg_rank={avg_rank:.3f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("2p", "4p", "both"), default="both")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=140)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base-2p", type=int, default=8100)
    parser.add_argument("--seed-base-4p", type=int, default=9900)
    parser.add_argument("--two-player-config", default="p2move_regular")
    parser.add_argument("--four-player-config", default="regular")
    parser.add_argument("--use-numba", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['mode']} seed={row['seed']} seat={row['seat']} "
                f"reward={row['variant_reward']} best_other={row['best_other']} "
                f"rewards={row['rewards']}",
                flush=True,
            )
    print("\nSUMMARY")
    summarize(rows)


if __name__ == "__main__":
    main()
