from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ray

from tinyPPO.agents import TinyPPOAgent
from tinyPPO.bridge_agents import parse_float_list
from tinyPPO.eval import run_matchups
from training2.rulebase_bridge import make_rulebase_agent


@ray.remote
def _eval_part(config: dict[str, Any], games: int, seed: int) -> dict[str, float]:
    ckpt = Path(config["checkpoint"])
    result = run_matchups(
        lambda: TinyPPOAgent(
            ckpt,
            device=config["device"],
            deterministic=not config["stochastic"],
            launch_bias=config["launch_bias"],
            ship_bias=config["ship_bias"],
            launch_temperature=config["launch_temperature"],
            target_top_k=config["target_top_k"],
            include_friendly_targets=config["include_friendly_targets"],
            target_mask_mode=config["target_mask_mode"],
            target_pair_weight=config["target_pair_weight"],
        ),
        lambda: make_rulebase_agent("regular"),
        games=games,
        seed=seed,
        episode_steps=config["episode_steps"],
        use_numba=config["use_numba"],
        progress=False,
    )
    result.update(
        {
            "checkpoint": str(ckpt),
            "launch_bias": config["launch_bias"],
            "ship_bias": config["ship_bias"],
            "launch_temperature": config["launch_temperature"],
            "target_pair_weight": config["target_pair_weight"],
            "target_mask_mode": config["target_mask_mode"],
            "target_top_k": config["target_top_k"],
            "include_friendly_targets": config["include_friendly_targets"],
            "stochastic": config["stochastic"],
        }
    )
    return result


def _merge(parts: list[dict[str, float]]) -> dict[str, float]:
    games = sum(float(part["games"]) for part in parts)
    wins = sum(float(part["wins"]) for part in parts)
    losses = sum(float(part["losses"]) for part in parts)
    draws = sum(float(part["draws"]) for part in parts)
    base = dict(parts[0]) if parts else {}
    base.update(
        {
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winrate": wins / max(1.0, games),
            "nonloss": (wins + draws) / max(1.0, games),
            "mean_reward": (wins - losses) / max(1.0, games),
        }
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--games-per-task", type=int, default=2)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--gpus-per-worker", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=940000)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--launch-biases", default="-0.5,-0.25,0.0,0.25")
    parser.add_argument("--ship-biases", default="0.0")
    parser.add_argument("--launch-temperatures", default="1.0")
    parser.add_argument("--target-pair-weights", default="1.0")
    parser.add_argument("--target-mask-modes", default="candidate")
    parser.add_argument("--target-top-k", type=int, default=6)
    parser.add_argument("--include-friendly-targets", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--variant-seed-stride", type=int, default=0)
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    ray.init(address=None if args.ray_address == "local" else args.ray_address, ignore_reinit_error=True)
    base: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "episode_steps": args.episode_steps,
        "use_numba": not args.no_numba,
        "target_top_k": args.target_top_k,
        "include_friendly_targets": bool(args.include_friendly_targets),
        "stochastic": bool(args.stochastic),
    }
    configs: list[dict[str, Any]] = []
    for mask_mode in [item.strip() for item in args.target_mask_modes.split(",") if item.strip()]:
        for pair_weight in parse_float_list(args.target_pair_weights):
            for temperature in parse_float_list(args.launch_temperatures):
                for ship_bias in parse_float_list(args.ship_biases):
                    for launch_bias in parse_float_list(args.launch_biases):
                        configs.append(
                            {
                                **base,
                                "target_mask_mode": mask_mode,
                                "target_pair_weight": pair_weight,
                                "launch_temperature": temperature,
                                "ship_bias": ship_bias,
                                "launch_bias": launch_bias,
                            }
                        )

    all_rows: list[dict[str, float]] = []
    for variant_index, config in enumerate(configs):
        refs = []
        remaining = args.games
        task_idx = 0
        while remaining > 0:
            part_games = min(args.games_per_task, remaining)
            remaining -= part_games
            refs.append(
                _eval_part.options(num_cpus=args.cpus_per_worker, num_gpus=args.gpus_per_worker).remote(
                    config,
                    part_games,
                    args.seed + variant_index * args.variant_seed_stride + task_idx * args.games_per_task,
                )
            )
            task_idx += 1
        parts: list[dict[str, float]] = []
        while refs:
            ready, refs = ray.wait(refs, num_returns=1)
            part = ray.get(ready[0])
            parts.append(part)
            print(json.dumps({"event": "part_done", **part}, sort_keys=True), flush=True)
        merged = _merge(parts)
        all_rows.append(merged)
        print(json.dumps({"event": "variant_done", **merged}, sort_keys=True), flush=True)

    all_rows.sort(key=lambda row: (float(row["nonloss"]), float(row["winrate"]), float(row["mean_reward"])), reverse=True)
    print("\nTop variants:")
    for row in all_rows:
        print(
            f"bias={row['launch_bias']} temp={row['launch_temperature']} ship={row['ship_bias']} "
            f"pair={row['target_pair_weight']} mask={row['target_mask_mode']} "
            f"W/L/D={int(row['wins'])}/{int(row['losses'])}/{int(row['draws'])} "
            f"wr={float(row['winrate']):.3f} nonloss={float(row['nonloss']):.3f}"
        )


if __name__ == "__main__":
    main()
