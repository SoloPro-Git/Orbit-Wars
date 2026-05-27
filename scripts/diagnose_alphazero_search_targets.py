"""Diagnose whether AlphaZero-like search targets contain useful signal."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphaZeroLike.candidates import CandidateConfig, CandidateGenerator
from alphaZeroLike.mcts import MCTSConfig, ShallowPUCTSearch, _raw_obs, evaluate_candidates
from alphaZeroLike.model import AlphaZeroLikeNet
from alphaZeroLike.proposal import ProposalConfig, proposals_from_model
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_model(checkpoint: Path, device: str) -> AlphaZeroLikeNet:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = AlphaZeroLikeNet(
        d_model=int(model_cfg.get("d_model", args.get("d_model", 192))),
        nhead=int(model_cfg.get("nhead", args.get("nhead", 6))),
        layers=int(model_cfg.get("layers", args.get("layers", 4))),
        dropout=float(model_cfg.get("dropout", args.get("dropout", 0.10))),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def _entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0.0]
    return float(-(probs * np.log(probs)).sum()) if probs.size else 0.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "candidate_count",
        "value_range",
        "unique_value_count",
        "positive_candidate_rate",
        "nonnegative_candidate_rate",
        "prior_entropy",
        "policy_entropy",
        "prior_policy_l1",
        "prior_value_corr",
        "visited_value_range",
        "visited_count",
        "best_value",
        "oracle_value",
    ]
    out: dict[str, Any] = {"positions": len(rows)}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if key in row]
        out[f"{key}_mean"] = _mean(vals)
        out[f"{key}_p50"] = _percentile(vals, 50)
        out[f"{key}_p90"] = _percentile(vals, 90)
    out["all_values_equal_rate"] = _mean([float(row["all_values_equal"]) for row in rows])
    out["best_is_oracle_rate"] = _mean([float(row["best_is_oracle"]) for row in rows])
    out["selected_is_oracle_rate"] = _mean([float(row["selected_is_oracle"]) for row in rows])
    out["policy_top_is_prior_top_rate"] = _mean([float(row["policy_top_is_prior_top"]) for row in rows])
    out["policy_top_is_value_best_rate"] = _mean([float(row["policy_top_is_value_best"]) for row in rows])
    return out


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint = _resolve(args.checkpoint)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    model = _load_model(checkpoint, device)

    candidate_cfg = CandidateConfig(
        max_candidates=args.max_candidates,
        include_noop=not args.no_noop,
        include_heuristics=not args.no_heuristics,
        oracle=args.oracle,
        use_rulebase=not args.no_rulebase_candidates,
    )
    proposal_cfg = ProposalConfig(
        enabled=not args.no_proposals,
        num_candidates=args.proposal_num_candidates,
        num_full_actions=args.proposal_num_full_actions,
        num_sampled_actions=args.proposal_num_sampled_actions,
        raw_sampled_actions=args.proposal_raw_sampled_actions,
        send_threshold=args.proposal_send_threshold,
        max_sources=args.proposal_max_sources,
        top_targets_per_source=args.proposal_top_targets_per_source,
    )
    mcts_cfg = MCTSConfig(
        simulations=args.simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        dirichlet_frac=0.0,
        rollout_depth=args.rollout_depth,
        max_candidates=args.max_candidates,
        max_moves=args.max_moves,
        value_mode=args.value_mode,
        margin_scale=args.margin_scale,
    )

    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    rng = random.Random(args.seed)

    for ep in range(args.episodes):
        env = make_fast_orbit_wars(
            {"episodeSteps": args.episode_steps, "seed": args.seed + ep},
            keep_history=False,
            use_numba=not args.no_numba,
        )
        env.reset(args.players)
        agents = {pid: make_rulebase_agent(args.oracle) for pid in range(args.players)}
        search = ShallowPUCTSearch(
            model,
            CandidateGenerator(candidate_cfg),
            mcts_cfg,
            device=device,
            opponent_factory=lambda: make_rulebase_agent(args.oracle),
            proposal_cfg=proposal_cfg,
        )
        for step in range(args.episode_steps):
            if step >= args.warmup_steps and step % args.sample_stride == 0:
                for player in range(args.players):
                    if len(rows) >= args.positions:
                        break
                    obs = _raw_obs(env, player)
                    proposal_seed = args.seed * 1_000_003 + ep * 10_007 + step * 101 + player
                    torch.manual_seed(proposal_seed)
                    extra = proposals_from_model(obs, player, model, device, proposal_cfg)
                    candidates, oracle_index = CandidateGenerator(candidate_cfg)(obs, extra_candidates=extra)
                    if not candidates:
                        continue
                    priors, root_value = evaluate_candidates(
                        model,
                        obs,
                        player,
                        candidates,
                        device=device,
                        max_candidates=args.max_candidates,
                        max_moves=args.max_moves,
                    )
                    priors = priors[: len(candidates)]
                    priors = priors / max(float(priors.sum()), 1e-12)
                    opponents = {pid: make_rulebase_agent(args.oracle) for pid in range(args.players) if pid != player}
                    values = np.asarray(
                        [search._rollout_value(env, player, candidate, opponents) for candidate in candidates],
                        dtype=np.float64,
                    )
                    torch.manual_seed(proposal_seed)
                    result = search.search(env, player, add_noise=False, rng=rng)
                    policy = result.policy_target[: len(candidates)]
                    visits = result.visits[: len(candidates)]
                    visited = visits > 0.0
                    value_range = float(values.max() - values.min()) if values.size else 0.0
                    visited_values = values[visited] if values.size == visits.size else np.asarray([], dtype=np.float64)
                    prior_top = int(np.argmax(priors)) if priors.size else 0
                    policy_top = int(np.argmax(policy)) if policy.size else 0
                    value_best = int(np.argmax(values)) if values.size else 0
                    row = {
                        "episode": ep,
                        "step": step,
                        "player": player,
                        "candidate_count": len(candidates),
                        "root_value": float(root_value),
                        "value_range": value_range,
                        "unique_value_count": int(len({round(float(v), 6) for v in values.tolist()})),
                        "all_values_equal": bool(value_range <= args.equal_eps),
                        "positive_candidate_rate": float((values > 0.0).mean()) if values.size else 0.0,
                        "nonnegative_candidate_rate": float((values >= 0.0).mean()) if values.size else 0.0,
                        "best_value": float(values[value_best]) if values.size else 0.0,
                        "oracle_value": float(values[oracle_index]) if oracle_index < len(values) else 0.0,
                        "best_is_oracle": bool(value_best == oracle_index),
                        "selected_is_oracle": bool(int(result.selected_index) == oracle_index),
                        "prior_entropy": _entropy(priors),
                        "policy_entropy": _entropy(policy),
                        "prior_policy_l1": float(np.abs(priors - policy).sum()) if priors.size == policy.size else 0.0,
                        "prior_value_corr": _corr(priors, values),
                        "visited_count": int(visited.sum()),
                        "visited_value_range": float(visited_values.max() - visited_values.min()) if visited_values.size else 0.0,
                        "policy_top_is_prior_top": bool(policy_top == prior_top),
                        "policy_top_is_value_best": bool(policy_top == value_best),
                    }
                    rows.append(row)
                    if len(examples) < args.examples:
                        top = sorted(
                            range(len(candidates)),
                            key=lambda i: (float(values[i]), float(priors[i])),
                            reverse=True,
                        )[: min(5, len(candidates))]
                        examples.append(
                            {
                                **row,
                                "top_candidates": [
                                    {
                                        "index": int(i),
                                        "value": float(values[i]),
                                        "prior": float(priors[i]),
                                        "policy": float(policy[i]) if i < len(policy) else 0.0,
                                        "visits": float(visits[i]) if i < len(visits) else 0.0,
                                        "is_oracle": bool(i == oracle_index),
                                        "action": candidates[i],
                                    }
                                    for i in top
                                ],
                            }
                        )
                if len(rows) >= args.positions:
                    break

            actions = []
            for player in range(args.players):
                obs = _raw_obs(env, player)
                actions.append(agents[player](obs) or [])
            env.step(actions)
            if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                break
        if len(rows) >= args.positions:
            break

    return {
        "checkpoint": str(checkpoint.resolve()),
        "config": vars(args),
        "summary": _summarize(rows),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="alphaZeroLike/checkpoints/latest.pt")
    parser.add_argument("--out", default="alphaZeroLike/logs/search_target_diagnostics_latest.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=180)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=12)
    parser.add_argument("--oracle", default="rl_informed_regular")
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--rollout-depth", type=int, default=1)
    parser.add_argument("--value-mode", choices=["rank", "margin_tanh"], default="rank")
    parser.add_argument("--margin-scale", type=float, default=50.0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--max-moves", type=int, default=8)
    parser.add_argument("--proposal-num-candidates", type=int, default=16)
    parser.add_argument("--proposal-num-full-actions", type=int, default=8)
    parser.add_argument("--proposal-num-sampled-actions", type=int, default=8)
    parser.add_argument("--proposal-raw-sampled-actions", type=int, default=64)
    parser.add_argument("--proposal-send-threshold", type=float, default=0.30)
    parser.add_argument("--proposal-max-sources", type=int, default=4)
    parser.add_argument("--proposal-top-targets-per-source", type=int, default=2)
    parser.add_argument("--equal-eps", type=float, default=1e-9)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--no-proposals", action="store_true")
    parser.add_argument("--no-rulebase-candidates", action="store_true")
    parser.add_argument("--no-heuristics", action="store_true")
    parser.add_argument("--no-noop", action="store_true")
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    report = diagnose(args)
    out = _resolve(args.out)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
