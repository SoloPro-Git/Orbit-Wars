"""Evaluate an AlphaZero-like agent against the local rulebase in fast env."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphaZeroLike.agent import AlphaZeroLikeAgent
from alphaZeroLike.candidates import CandidateConfig
from alphaZeroLike.proposal import ProposalConfig
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent


def _resolve(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def _make_az_agent(args: argparse.Namespace, az_player: int, seed: int):
    agent = AlphaZeroLikeAgent(
        checkpoint=_resolve(args.checkpoint),
        init_from_training2=_resolve(args.init_from_training2),
        device=args.device,
        candidate_config=CandidateConfig(
            max_candidates=args.max_candidates,
            include_noop=not args.no_noop,
            include_heuristics=not args.no_heuristics,
            oracle=args.oracle,
            use_rulebase=not args.no_rulebase_candidates,
        ),
        use_model_proposals=not args.no_model_proposals,
        proposal_config=ProposalConfig(
            enabled=not args.no_model_proposals,
            num_candidates=args.proposal_num_candidates,
            num_full_actions=args.proposal_num_full_actions,
            num_sampled_actions=args.proposal_num_sampled_actions,
            raw_sampled_actions=args.proposal_raw_sampled_actions,
            send_threshold=args.proposal_send_threshold,
            max_sources=args.proposal_max_sources,
        ),
        rulebase_anchor=args.rulebase_anchor,
        anchor_min_prob=args.anchor_min_prob,
        anchor_min_margin=args.anchor_min_margin,
        proposal_explore_prob=args.proposal_explore_prob,
        proposal_explore_top_k=args.proposal_explore_top_k,
        proposal_explore_only_model=not args.proposal_explore_any_candidate,
        proposal_explore_max_extra_moves=args.proposal_explore_max_extra_moves,
        proposal_explore_max_ship_ratio=args.proposal_explore_max_ship_ratio,
        proposal_explore_max_extra_ships=args.proposal_explore_max_extra_ships,
        proposal_explore_min_anchor_source_jaccard=args.proposal_explore_min_anchor_source_jaccard,
        proposal_explore_project_anchor_sources=args.proposal_explore_project_anchor_sources,
        proposal_explore_project_anchor_source_probs=args.proposal_explore_project_anchor_source_probs,
        proposal_explore_max_anchor_source_drops=args.proposal_explore_max_anchor_source_drops,
        proposal_explore_min_anchor_moves=args.proposal_explore_min_anchor_moves,
        seed=seed * 10 + az_player,
    )
    stats = {
        "turns": 0,
        "explore_attempts": 0,
        "explore_used": 0,
        "explore_blocked": 0,
        "explore_anchor_moves": 0,
        "explore_selected_moves": 0,
        "explore_dropped_moves": 0,
        "selected_anchor": 0,
        "anchor_prob_sum": 0.0,
    }

    def act(obs: dict) -> list[list]:
        action = agent.act(obs)
        stats["turns"] += 1
        stats["explore_attempts"] += int(agent.last_explore_attempted)
        stats["explore_used"] += int(agent.last_explored)
        stats["explore_blocked"] += int(agent.last_explore_blocked)
        if agent.last_explored:
            stats["explore_anchor_moves"] += int(agent.last_explore_anchor_moves)
            stats["explore_selected_moves"] += int(agent.last_explore_selected_moves)
            stats["explore_dropped_moves"] += max(
                0,
                int(agent.last_explore_anchor_moves) - int(agent.last_explore_selected_moves),
            )
        stats["selected_anchor"] += int(agent.last_selected_idx == 0)
        stats["anchor_prob_sum"] += float(agent.last_anchor_prob)
        return action

    return act, stats


def _run_one(payload: tuple[int, int, dict[str, Any]]) -> dict[str, Any]:
    seed, az_player, raw_args = payload
    args = argparse.Namespace(**raw_args)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    try:
        import torch

        torch.set_num_threads(max(1, int(args.torch_threads)))
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    env = make_fast_orbit_wars(
        {"episodeSteps": args.episode_steps, "seed": seed},
        keep_history=False,
        use_numba=not args.no_numba,
    )
    agents = [make_rulebase_agent(args.oracle) for _ in range(args.players)]
    az_act, az_stats = _make_az_agent(args, az_player, seed)
    agents[az_player] = az_act
    env.run(agents)
    rewards = [int(env.steps[-1][pid]["reward"]) for pid in range(args.players)]
    az_reward = rewards[az_player]
    opp_best = max(reward for pid, reward in enumerate(rewards) if pid != az_player)
    outcome = "win" if az_reward > opp_best else "loss" if az_reward < opp_best else "draw"
    return {
        "seed": seed,
        "az_player": az_player,
        "rewards": rewards,
        "az_reward": az_reward,
        "opp_best_reward": opp_best,
        "outcome": outcome,
        "az_stats": az_stats,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    wins = sum(row["outcome"] == "win" for row in rows)
    losses = sum(row["outcome"] == "loss" for row in rows)
    draws = sum(row["outcome"] == "draw" for row in rows)
    by_seat: dict[int, dict[str, Any]] = {}
    for row in rows:
        seat = int(row["az_player"])
        bucket = by_seat.setdefault(seat, {"games": 0, "wins": 0, "losses": 0, "draws": 0})
        bucket["games"] += 1
        bucket["wins"] += int(row["outcome"] == "win")
        bucket["losses"] += int(row["outcome"] == "loss")
        bucket["draws"] += int(row["outcome"] == "draw")
    stats_total = {
        "turns": 0,
        "explore_attempts": 0,
        "explore_used": 0,
        "explore_blocked": 0,
        "explore_anchor_moves": 0,
        "explore_selected_moves": 0,
        "explore_dropped_moves": 0,
        "selected_anchor": 0,
        "anchor_prob_sum": 0.0,
    }
    for row in rows:
        for key in stats_total:
            value = row.get("az_stats", {}).get(key, 0.0)
            stats_total[key] += float(value) if key == "anchor_prob_sum" else int(value)
    for bucket in by_seat.values():
        games = max(int(bucket["games"]), 1)
        bucket["win_rate"] = bucket["wins"] / games
        bucket["nonloss_rate"] = (bucket["wins"] + bucket["draws"]) / games
    return {
        "games": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / max(total, 1),
        "nonloss_rate": (wins + draws) / max(total, 1),
        "mean_reward": sum(float(row["az_reward"]) for row in rows) / max(total, 1),
        "explore_attempt_rate": stats_total["explore_attempts"] / max(stats_total["turns"], 1),
        "explore_used_rate": stats_total["explore_used"] / max(stats_total["turns"], 1),
        "explore_blocked_rate": stats_total["explore_blocked"] / max(stats_total["turns"], 1),
        "explore_mean_anchor_moves": stats_total["explore_anchor_moves"] / max(stats_total["explore_used"], 1),
        "explore_mean_selected_moves": stats_total["explore_selected_moves"] / max(stats_total["explore_used"], 1),
            "explore_mean_dropped_moves": stats_total["explore_dropped_moves"] / max(stats_total["explore_used"], 1),
        "selected_anchor_rate": stats_total["selected_anchor"] / max(stats_total["turns"], 1),
        "mean_anchor_prob": stats_total["anchor_prob_sum"] / max(stats_total["turns"], 1),
        "explore_stats": stats_total,
        "by_seat": by_seat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="alphaZeroLike/checkpoints/latest.pt")
    parser.add_argument("--init-from-training2")
    parser.add_argument("--out", default="alphaZeroLike/logs/eval_alphazero_vs_rulebase_latest.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--oracle", default="rl_informed_regular")
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--rulebase-anchor", choices=["disabled", "confidence", "always"], default="confidence")
    parser.add_argument("--anchor-min-prob", type=float, default=0.55)
    parser.add_argument("--anchor-min-margin", type=float, default=0.10)
    parser.add_argument("--proposal-explore-prob", type=float, default=0.0)
    parser.add_argument("--proposal-explore-top-k", type=int, default=8)
    parser.add_argument("--proposal-explore-any-candidate", action="store_true")
    parser.add_argument("--proposal-explore-max-extra-moves", type=int, default=1)
    parser.add_argument("--proposal-explore-max-ship-ratio", type=float, default=1.25)
    parser.add_argument("--proposal-explore-max-extra-ships", type=int, default=20)
    parser.add_argument("--proposal-explore-min-anchor-source-jaccard", type=float, default=0.0)
    parser.add_argument("--proposal-explore-project-anchor-sources", action="store_true")
    parser.add_argument("--proposal-explore-project-anchor-source-probs", action="store_true")
    parser.add_argument("--proposal-explore-max-anchor-source-drops", type=int)
    parser.add_argument("--proposal-explore-min-anchor-moves", type=int, default=1)
    parser.add_argument("--proposal-num-candidates", type=int, default=16)
    parser.add_argument("--proposal-num-full-actions", type=int, default=8)
    parser.add_argument("--proposal-num-sampled-actions", type=int, default=8)
    parser.add_argument("--proposal-raw-sampled-actions", type=int, default=64)
    parser.add_argument("--proposal-send-threshold", type=float, default=0.30)
    parser.add_argument("--proposal-max-sources", type=int, default=4)
    parser.add_argument("--no-model-proposals", action="store_true")
    parser.add_argument("--no-rulebase-candidates", action="store_true")
    parser.add_argument("--no-heuristics", action="store_true")
    parser.add_argument("--no-noop", action="store_true")
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    try:
        import torch

        torch.set_num_threads(max(1, int(args.torch_threads)))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    raw_args = vars(args)
    tasks = [
        (args.seed + game // args.players, game % args.players, raw_args)
        for game in range(args.games)
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_run_one, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: (row["seed"], row["az_player"]))
    report = {
        "config": raw_args,
        "summary": _summarize(rows),
        "games": rows,
    }
    out = Path(_resolve(args.out) or args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
