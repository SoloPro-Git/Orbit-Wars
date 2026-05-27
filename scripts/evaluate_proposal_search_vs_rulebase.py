"""Evaluate proposal-driven shallow search against the local rulebase."""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphaZeroLike.candidates import CandidateConfig, CandidateGenerator
from alphaZeroLike.mcts import MCTSConfig, ShallowPUCTSearch, _raw_obs
from alphaZeroLike.model import AlphaZeroLikeNet
from alphaZeroLike.proposal import ProposalConfig
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_model(checkpoint: Path, device: str) -> AlphaZeroLikeNet:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    model = AlphaZeroLikeNet(
        d_model=int(model_cfg.get("d_model", args.get("d_model", 192))),
        nhead=int(model_cfg.get("nhead", args.get("nhead", 6))),
        layers=int(model_cfg.get("layers", args.get("layers", 4))),
        dropout=float(model_cfg.get("dropout", args.get("dropout", 0.10))),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def _run_one(payload: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
    seed, model_pid, raw_args = payload
    args = argparse.Namespace(**raw_args)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model = _load_model(_resolve(args.checkpoint), device)  # type: ignore[arg-type]
    rng = random.Random(seed)
    env = make_fast_orbit_wars(
        {"episodeSteps": args.episode_steps, "seed": seed},
        keep_history=False,
        use_numba=not args.no_numba,
    )
    env.reset(args.players)
    opponents = {pid: make_rulebase_agent(args.oracle) for pid in range(args.players) if pid != model_pid}
    search = ShallowPUCTSearch(
        model,
        CandidateGenerator(
            CandidateConfig(
                max_candidates=args.max_candidates,
                include_noop=not args.no_noop,
                include_heuristics=not args.no_heuristics,
                oracle=args.oracle,
                use_rulebase=not args.no_rulebase_candidates,
            )
        ),
        MCTSConfig(
            simulations=args.simulations,
            c_puct=args.c_puct,
            temperature=args.temperature,
            dirichlet_frac=args.dirichlet_frac,
            rollout_depth=args.rollout_depth,
            max_candidates=args.max_candidates,
            max_moves=args.max_moves,
            value_mode=args.value_mode,
            margin_scale=args.margin_scale,
            root_eval_mode=args.root_eval_mode,
            max_root_evals=args.max_root_evals,
        ),
        device=device,
        opponent_factory=lambda: make_rulebase_agent(args.oracle),
        proposal_cfg=ProposalConfig(
            enabled=not args.no_proposals,
            num_candidates=args.proposal_num_candidates,
            num_full_actions=args.proposal_num_full_actions,
            num_sampled_actions=args.proposal_num_sampled_actions,
            raw_sampled_actions=args.proposal_raw_sampled_actions,
            send_threshold=args.proposal_send_threshold,
            max_sources=args.proposal_max_sources,
            top_targets_per_source=args.proposal_top_targets_per_source,
            source_temperature=args.proposal_source_temperature,
            target_temperature=args.proposal_target_temperature,
            ship_noise_std=args.proposal_ship_noise_std,
            sample_source_prob=args.proposal_sample_source_prob,
        ),
    )

    search_calls = 0
    policy_calls = 0
    selected_oracle = 0
    for model_turn in range(args.episode_steps):
        actions = []
        for pid in range(args.players):
            obs = _raw_obs(env, pid)
            if pid == model_pid:
                if model_turn % max(1, args.search_stride) == 0:
                    result = search.search(env, pid, add_noise=args.add_noise, rng=rng)
                    search_calls += 1
                else:
                    result = search.policy_action(env, pid, rng=rng)
                    policy_calls += 1
                selected_oracle += int(result.selected_index == 0)
                actions.append(result.actions[result.selected_index] if result.actions else [])
            else:
                actions.append(opponents[pid](obs) or [])
        env.step(actions)
        if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
            break

    rewards = [int(env.steps[-1][pid]["reward"]) for pid in range(args.players)]
    model_reward = rewards[model_pid]
    opp_best = max(reward for pid, reward in enumerate(rewards) if pid != model_pid)
    outcome = "win" if model_reward > opp_best else "loss" if model_reward < opp_best else "draw"
    total_calls = search_calls + policy_calls
    return {
        "seed": seed,
        "model_pid": model_pid,
        "rewards": rewards,
        "model_reward": model_reward,
        "opp_best_reward": opp_best,
        "outcome": outcome,
        "search_calls": search_calls,
        "policy_calls": policy_calls,
        "selected_oracle_rate": selected_oracle / max(total_calls, 1),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    wins = sum(row["outcome"] == "win" for row in rows)
    losses = sum(row["outcome"] == "loss" for row in rows)
    draws = sum(row["outcome"] == "draw" for row in rows)
    return {
        "games": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / max(total, 1),
        "nonloss_rate": (wins + draws) / max(total, 1),
        "mean_reward": sum(float(row["model_reward"]) for row in rows) / max(total, 1),
        "selected_oracle_rate": sum(float(row["selected_oracle_rate"]) for row in rows) / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="alphaZeroLike/checkpoints/latest.pt")
    parser.add_argument("--out", default="alphaZeroLike/logs/eval_proposal_search_vs_rulebase_latest.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--oracle", default="rl_informed_regular")
    parser.add_argument("--simulations", type=int, default=2)
    parser.add_argument("--rollout-depth", type=int, default=4)
    parser.add_argument("--search-stride", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--dirichlet-frac", type=float, default=0.0)
    parser.add_argument("--value-mode", choices=["rank", "margin_tanh"], default="rank")
    parser.add_argument("--margin-scale", type=float, default=50.0)
    parser.add_argument("--root-eval-mode", choices=["puct", "exhaustive"], default="puct")
    parser.add_argument("--max-root-evals", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--max-moves", type=int, default=8)
    parser.add_argument("--proposal-num-candidates", type=int, default=16)
    parser.add_argument("--proposal-num-full-actions", type=int, default=8)
    parser.add_argument("--proposal-num-sampled-actions", type=int, default=8)
    parser.add_argument("--proposal-raw-sampled-actions", type=int, default=64)
    parser.add_argument("--proposal-send-threshold", type=float, default=0.40)
    parser.add_argument("--proposal-max-sources", type=int, default=4)
    parser.add_argument("--proposal-top-targets-per-source", type=int, default=2)
    parser.add_argument("--proposal-source-temperature", type=float, default=1.0)
    parser.add_argument("--proposal-target-temperature", type=float, default=1.0)
    parser.add_argument("--proposal-ship-noise-std", type=float, default=0.10)
    parser.add_argument("--proposal-sample-source-prob", type=float, default=0.65)
    parser.add_argument("--add-noise", action="store_true")
    parser.add_argument("--no-proposals", action="store_true")
    parser.add_argument("--no-rulebase-candidates", action="store_true")
    parser.add_argument("--no-heuristics", action="store_true")
    parser.add_argument("--no-noop", action="store_true")
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    raw_args = vars(args)
    tasks = [(args.seed + game // args.players, game % args.players, raw_args) for game in range(args.games)]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_run_one, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row["seed"], row["model_pid"]))

    report = {"config": raw_args, "summary": _summary(rows), "games": rows}
    out = _resolve(args.out)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
