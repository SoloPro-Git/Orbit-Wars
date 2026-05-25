from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ray

from tinyPPO.bridge_agents import parse_float_list, parse_int_list
from tinyPPO.eval_bridge import _run_agent_pair
from tinyPPO.bridge_agents import RegularSourceBridgeAgent
from training2.rulebase_bridge import make_rulebase_agent


@ray.remote
def _eval_part(config: dict[str, Any], games: int, seed: int) -> dict[str, float]:
    ckpt = Path(config["checkpoint"])
    if config["variant"] == "regular_anchor":
        agent_factory = lambda: make_rulebase_agent("regular")
    else:
        agent_factory = lambda: RegularSourceBridgeAgent(
            ckpt,
            device=config["device"],
            threshold=config["threshold"],
            apply_prob=config["apply_prob"],
            max_source_drops=config["max_source_drops"],
            max_drop_frac=config["max_drop_frac"],
            min_keep_actions=config["min_keep_actions"],
            min_anchor_actions_to_filter=config["min_anchor_actions_to_filter"],
            launch_bias=config["launch_bias"],
            launch_temperature=config["launch_temperature"],
            reduce=config["reduce"],
        )
    result = _run_agent_pair(
        agent_factory,
        lambda: make_rulebase_agent("regular"),
        games=games,
        seed=seed,
        episode_steps=config["episode_steps"],
        use_numba=config["use_numba"],
        progress=False,
        desc=str(config["variant"]),
    )
    result.update(
        {
            "variant": config["variant"],
            "threshold": config.get("threshold"),
            "apply_prob": config.get("apply_prob", 0.0),
            "max_source_drops": config.get("max_source_drops"),
            "max_drop_frac": config.get("max_drop_frac"),
            "min_anchor_actions_to_filter": config.get("min_anchor_actions_to_filter"),
        }
    )
    return result


def _merge(parts: list[dict[str, float]]) -> dict[str, float]:
    games = sum(float(part["games"]) for part in parts)
    wins = sum(float(part["wins"]) for part in parts)
    losses = sum(float(part["losses"]) for part in parts)
    draws = sum(float(part["draws"]) for part in parts)
    out: dict[str, float] = {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": wins / max(1.0, games),
        "nonloss": (wins + draws) / max(1.0, games),
        "mean_reward": (wins - losses) / max(1.0, games),
    }
    for key in (
        "bridge_calls",
        "bridge_attempted",
        "bridge_anchor_actions",
        "bridge_kept_actions",
        "bridge_dropped_actions",
    ):
        if any(key in part for part in parts):
            out[key] = sum(float(part.get(key, 0.0)) for part in parts)
    anchor = max(1.0, out.get("bridge_anchor_actions", 0.0))
    calls = max(1.0, out.get("bridge_calls", 0.0))
    if "bridge_anchor_actions" in out:
        out["bridge_kept_frac"] = out.get("bridge_kept_actions", 0.0) / anchor
        out["bridge_dropped_frac"] = out.get("bridge_dropped_actions", 0.0) / anchor
        out["bridge_attempt_frac"] = out.get("bridge_attempted", 0.0) / calls
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--games-per-task", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--gpus-per-worker", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=940000)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--thresholds", default="0.5,0.7,0.85")
    parser.add_argument("--apply-probs", default="1.0")
    parser.add_argument("--max-source-drops-list", default="1")
    parser.add_argument("--max-drop-frac", type=float, default=0.15)
    parser.add_argument("--min-keep-actions", type=int, default=1)
    parser.add_argument("--min-anchor-actions-to-filter", type=int, default=4)
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--launch-temperature", type=float, default=1.0)
    parser.add_argument("--reduce", choices=["noisy_or", "max", "mean"], default="max")
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    ray.init(address=None if args.ray_address == "local" else args.ray_address, ignore_reinit_error=True)
    base: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "episode_steps": args.episode_steps,
        "use_numba": not args.no_numba,
        "max_drop_frac": args.max_drop_frac,
        "min_keep_actions": args.min_keep_actions,
        "min_anchor_actions_to_filter": args.min_anchor_actions_to_filter,
        "launch_bias": args.launch_bias,
        "launch_temperature": args.launch_temperature,
        "reduce": args.reduce,
    }
    configs: list[dict[str, Any]] = [{**base, "variant": "regular_anchor"}]
    for apply_prob in parse_float_list(args.apply_probs):
        for threshold in parse_float_list(args.thresholds):
            for max_drops in parse_int_list(args.max_source_drops_list):
                configs.append(
                    {
                        **base,
                        "variant": "regular_source_bridge",
                        "threshold": threshold,
                        "apply_prob": apply_prob,
                        "max_source_drops": max_drops,
                    }
                )

    all_rows: list[dict[str, float | str | int | None]] = []
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
                    args.seed + variant_index * 100000 + task_idx * args.games_per_task,
                )
            )
            task_idx += 1
        parts: list[dict[str, float]] = []
        while refs:
            ready, refs = ray.wait(refs, num_returns=min(args.workers, len(refs)))
            for ref in ready:
                part = ray.get(ref)
                parts.append(part)
                print(json.dumps({"event": "part_done", **part}, sort_keys=True), flush=True)
        merged = _merge(parts)
        merged.update(
            {
                "variant": config["variant"],
                "threshold": config.get("threshold"),
                "apply_prob": config.get("apply_prob", 0.0),
                "max_source_drops": config.get("max_source_drops"),
                "max_drop_frac": config.get("max_drop_frac"),
                "min_anchor_actions_to_filter": config.get("min_anchor_actions_to_filter"),
            }
        )
        all_rows.append(merged)
        print(json.dumps({"event": "variant_done", **merged}, sort_keys=True), flush=True)

    all_rows.sort(key=lambda row: (float(row["nonloss"]), float(row["winrate"]), float(row["mean_reward"])), reverse=True)
    print("\nTop variants:")
    for row in all_rows:
        print(
            f"{row['variant']} p={row['apply_prob']} th={row['threshold']} drops={row['max_source_drops']} "
            f"W/L/D={int(row['wins'])}/{int(row['losses'])}/{int(row['draws'])} "
            f"wr={float(row['winrate']):.3f} nonloss={float(row['nonloss']):.3f} "
            f"kept={float(row.get('bridge_kept_frac', 1.0)):.3f} attempted={float(row.get('bridge_attempt_frac', 0.0)):.3f}"
        )


if __name__ == "__main__":
    main()
