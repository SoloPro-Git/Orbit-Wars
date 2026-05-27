from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tinyPPO.eval_regular_imitation import evaluate


DEFAULT_BIAS_GRID = "-0.35,-0.25,-0.15,-0.05,0.0,0.05,0.15"


def _float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _score_metrics(metrics: dict[str, float]) -> float:
    regular_density = max(1e-6, float(metrics.get("regular_actions_per_state", 0.0)))
    model_density = max(1e-6, float(metrics.get("model_actions_per_state", 0.0)))
    density_ratio = model_density / regular_density
    density_penalty = min(2.0, abs(math.log(density_ratio)))
    return (
        0.38 * float(metrics.get("action_f1", 0.0))
        + 0.22 * float(metrics.get("source_target_f1", 0.0))
        + 0.18 * float(metrics.get("source_f1", 0.0))
        + 0.12 * float(metrics.get("micro_action_f1", 0.0))
        - 0.10 * density_penalty
        - 0.03 * float(metrics.get("action_count_mae", 0.0))
    )


def _density_ok(metrics: dict[str, float], min_ratio: float, max_ratio: float) -> bool:
    regular_density = max(1e-6, float(metrics.get("regular_actions_per_state", 0.0)))
    model_density = float(metrics.get("model_actions_per_state", 0.0))
    ratio = model_density / regular_density
    return min_ratio <= ratio <= max_ratio


def _best_density_matched(sweep: list[dict[str, float]], min_ratio: float, max_ratio: float) -> dict[str, float]:
    matched = [item for item in sweep if _density_ok(item, min_ratio, max_ratio)]
    candidates = matched if matched else sweep
    return max(candidates, key=_score_metrics)


def _make_eval_args(args: argparse.Namespace, checkpoint: str, seed: int, launch_bias_grid: str) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=checkpoint,
        players_list=args.players_list,
        games_per_players=args.games_per_players,
        seed=seed,
        episode_steps=args.episode_steps,
        rows_per_game=args.rows_per_game,
        max_rows=args.max_rows,
        keep_noop_prob=args.keep_noop_prob,
        device=args.device,
        aggression=0.0,
        launch_bias=0.0,
        launch_bias_grid=launch_bias_grid,
        ship_bias=args.ship_bias,
        launch_temperature=args.launch_temperature,
        target_top_k=args.target_top_k,
        include_friendly_targets=args.include_friendly_targets,
        target_mask_mode=args.target_mask_mode,
        target_pair_weight=args.target_pair_weight,
        stochastic=args.stochastic,
        diagnose_policy=args.diagnose_policy,
        no_numba=args.no_numba,
        progress=args.progress,
    )


def _eval_checkpoint(args: argparse.Namespace, checkpoint: str, seed: int, launch_bias_grid: str) -> dict[str, Any]:
    result = evaluate(_make_eval_args(args, checkpoint, seed, launch_bias_grid))
    sweep = result.get("sweep")
    if not isinstance(sweep, list):
        raise RuntimeError("phase1 gate expects --launch-bias-grid evaluation output")
    best = _best_density_matched(sweep, args.min_density_ratio, args.max_density_ratio)
    return {
        "checkpoint": checkpoint,
        "seed": seed,
        "states": result.get("states"),
        "best_density_matched": best,
        "selected_score": _score_metrics(best),
        "sweep": sweep,
    }


def _aggregate(seed_results: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "source_f1",
        "source_target_f1",
        "action_f1",
        "micro_action_f1",
        "action_count_mae",
        "model_actions_per_state",
        "regular_actions_per_state",
        "model_errors",
    ]
    aggregate: dict[str, float] = {}
    for key in keys:
        values = [float(item["best_density_matched"].get(key, 0.0)) for item in seed_results]
        aggregate[key] = sum(values) / max(1, len(values))
    aggregate["selected_score"] = sum(float(item["selected_score"]) for item in seed_results) / max(1, len(seed_results))
    aggregate["density_ratio"] = aggregate["model_actions_per_state"] / max(1e-6, aggregate["regular_actions_per_state"])
    return aggregate


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    candidate_results = [
        _eval_checkpoint(args, args.checkpoint, seed, args.launch_bias_grid)
        for seed in seeds
    ]
    candidate_summary = _aggregate(candidate_results)

    baseline_results: list[dict[str, Any]] = []
    baseline_summary: dict[str, float] | None = None
    if args.baseline_checkpoint:
        baseline_grid = args.baseline_launch_bias_grid or args.launch_bias_grid
        baseline_results = [
            _eval_checkpoint(args, args.baseline_checkpoint, seed, baseline_grid)
            for seed in seeds
        ]
        baseline_summary = _aggregate(baseline_results)

    reasons: list[str] = []
    pass_phase1 = True
    density_ratio = candidate_summary["density_ratio"]
    if not (args.min_density_ratio <= density_ratio <= args.max_density_ratio):
        pass_phase1 = False
        reasons.append(f"density_ratio {density_ratio:.3f} outside [{args.min_density_ratio:.3f}, {args.max_density_ratio:.3f}]")
    for metric, threshold in [
        ("action_f1", args.min_action_f1),
        ("source_target_f1", args.min_source_target_f1),
        ("source_f1", args.min_source_f1),
    ]:
        value = float(candidate_summary.get(metric, 0.0))
        if value < threshold:
            pass_phase1 = False
            reasons.append(f"{metric} {value:.3f} below {threshold:.3f}")
    if baseline_summary is not None:
        delta = candidate_summary["selected_score"] - baseline_summary["selected_score"]
        if delta < args.min_score_delta_vs_baseline:
            pass_phase1 = False
            reasons.append(f"score_delta_vs_baseline {delta:.4f} below {args.min_score_delta_vs_baseline:.4f}")
    if not reasons:
        reasons.append("candidate passes configured same-state imitation gate")

    return {
        "pass_phase1": pass_phase1,
        "reasons": reasons,
        "candidate": {
            "checkpoint": args.checkpoint,
            "summary": candidate_summary,
            "seeds": candidate_results,
        },
        "baseline": {
            "checkpoint": args.baseline_checkpoint,
            "summary": baseline_summary,
            "seeds": baseline_results,
        },
        "config": {
            "players_list": args.players_list,
            "seeds": seeds,
            "games_per_players": args.games_per_players,
            "rows_per_game": args.rows_per_game,
            "episode_steps": args.episode_steps,
            "target_mask_mode": args.target_mask_mode,
            "launch_bias_grid": _float_list(args.launch_bias_grid),
            "min_density_ratio": args.min_density_ratio,
            "max_density_ratio": args.max_density_ratio,
            "min_action_f1": args.min_action_f1,
            "min_source_target_f1": args.min_source_target_f1,
            "min_source_f1": args.min_source_f1,
            "min_score_delta_vs_baseline": args.min_score_delta_vs_baseline,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate TinyPPO Phase 1 by same-state regular imitation, not winrate.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--players-list", default="2")
    parser.add_argument("--seeds", default="992000,993000,994000")
    parser.add_argument("--games-per-players", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=180)
    parser.add_argument("--rows-per-game", type=int, default=12)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--keep-noop-prob", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ship-bias", type=float, default=0.0)
    parser.add_argument("--launch-temperature", type=float, default=1.0)
    parser.add_argument("--launch-bias-grid", default=DEFAULT_BIAS_GRID)
    parser.add_argument("--baseline-launch-bias-grid", default="-0.25")
    parser.add_argument("--target-top-k", type=int, default=6)
    parser.add_argument("--include-friendly-targets", action="store_true")
    parser.add_argument("--target-mask-mode", choices=["candidate", "safe", "all_planets"], default="all_planets")
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--diagnose-policy", action="store_true")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--min-density-ratio", type=float, default=0.70)
    parser.add_argument("--max-density-ratio", type=float, default=1.35)
    parser.add_argument("--min-action-f1", type=float, default=0.30)
    parser.add_argument("--min-source-target-f1", type=float, default=0.30)
    parser.add_argument("--min-source-f1", type=float, default=0.38)
    parser.add_argument("--min-score-delta-vs-baseline", type=float, default=0.02)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    result = run_gate(args)
    encoded = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
