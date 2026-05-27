"""Generate policy rows by distilling source-only bridge or rulebase decisions.

The source bridge is not a submission target. It is a teacher that converts the
proposal source head into a candidate action label while keeping rulebase
targets and ship counts. The resulting rows train the AlphaZeroLike policy head
to choose the bridge action from a candidate set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphaZeroLike.agent import AlphaZeroLikeAgent
from alphaZeroLike.candidates import CandidateConfig, CandidateGenerator
from alphaZeroLike.features import result_value
from alphaZeroLike.proposal import ProposalConfig, proposal_labels
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent


def _raw_obs(env, player: int) -> dict:
    state = env.steps[-1][player]
    obs = dict(state["observation"])
    obs["player"] = player
    return obs


def _canonical(action: list[list]) -> tuple:
    return tuple((int(move[0]), round(float(move[1]), 6), int(move[2])) for move in action if len(move) >= 3)


def _add_candidate(candidates: list[list[list]], action: list[list]) -> int:
    key = _canonical(action)
    for idx, candidate in enumerate(candidates):
        if _canonical(candidate) == key:
            return idx
    candidates.append([[int(move[0]), float(move[1]), int(move[2])] for move in action if len(move) >= 3])
    return len(candidates) - 1


def generate_game(args: argparse.Namespace, seed: int, model_pid: int) -> tuple[list[dict], dict]:
    env = make_fast_orbit_wars(
        {"episodeSteps": args.episode_steps, "seed": seed},
        keep_history=False,
        use_numba=not args.no_numba,
    )
    env.reset(args.players)
    teacher = None
    if args.teacher_mode == "source_bridge":
        teacher = AlphaZeroLikeAgent(
            checkpoint=args.checkpoint,
            device=args.device,
            candidate_config=CandidateConfig(
                max_candidates=args.max_candidates,
                include_noop=not args.no_noop,
                include_heuristics=not args.no_heuristics,
                oracle=args.oracle,
                use_rulebase=True,
            ),
            use_model_proposals=True,
            proposal_config=ProposalConfig(
                enabled=True,
                send_threshold=args.proposal_send_threshold,
                num_candidates=args.proposal_num_candidates,
                num_full_actions=args.proposal_num_full_actions,
                max_sources=args.proposal_max_sources,
            ),
            rulebase_anchor="always",
            proposal_explore_prob=1.0,
            proposal_explore_project_anchor_source_probs=True,
            proposal_explore_min_anchor_moves=args.proposal_explore_min_anchor_moves,
            seed=seed * 10 + model_pid,
        )
    student = None
    if args.student_checkpoint:
        student = AlphaZeroLikeAgent(
            checkpoint=args.student_checkpoint,
            device=args.device,
            candidate_config=CandidateConfig(
                max_candidates=args.max_candidates,
                include_noop=not args.no_noop,
                include_heuristics=not args.no_heuristics,
                oracle=args.oracle,
                use_rulebase=True,
            ),
            use_model_proposals=not args.student_no_model_proposals,
            proposal_config=ProposalConfig(
                enabled=not args.student_no_model_proposals,
                send_threshold=args.proposal_send_threshold,
                num_candidates=args.proposal_num_candidates,
                num_full_actions=args.proposal_num_full_actions,
                max_sources=args.proposal_max_sources,
            ),
            rulebase_anchor="disabled",
            seed=seed * 100 + model_pid,
        )
    generator = CandidateGenerator(
        CandidateConfig(
            max_candidates=args.max_candidates,
            include_noop=not args.no_noop,
            include_heuristics=not args.no_heuristics,
            oracle=args.oracle,
            use_rulebase=True,
        )
    )
    rule_agents = [make_rulebase_agent(args.oracle) for _ in range(args.players)]
    rows: list[dict] = []
    stats = {"turns": 0, "changed": 0, "blocked": 0, "rows": 0, "student_turns": 0}

    for _ in range(args.episode_steps):
        actions = []
        for pid in range(args.players):
            obs = _raw_obs(env, pid)
            if pid == model_pid:
                candidates, _ = generator(obs)
                if args.teacher_mode == "rulebase":
                    teacher_action = candidates[0] if candidates else []
                    teacher_changed = False
                    teacher_blocked = False
                else:
                    assert teacher is not None
                    teacher_action = teacher.act(obs) or []
                    teacher_changed = bool(teacher.last_explored)
                    teacher_blocked = bool(teacher.last_explore_blocked)
                selected_idx = _add_candidate(candidates, teacher_action)
                policy_target = [0.0] * len(candidates)
                policy_target[selected_idx] = 1.0
                rows.append(
                    {
                        "obs": obs,
                        "player": pid,
                        "candidates": candidates,
                        "policy_target": policy_target,
                        "max_moves": args.max_moves,
                        "teacher_mode": args.teacher_mode,
                        "teacher_changed": teacher_changed,
                        "student_driven": bool(student is not None),
                        **proposal_labels(obs, pid, teacher_action),
                    }
                )
                stats["turns"] += 1
                stats["changed"] += int(teacher_changed)
                stats["blocked"] += int(teacher_blocked)
                if student is not None:
                    action = student.act(obs) or []
                    stats["student_turns"] += 1
                else:
                    action = teacher_action
                actions.append(action)
            else:
                actions.append(rule_agents[pid](obs) or [])
        env.step(actions)
        if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
            break

    final_obs = _raw_obs(env, model_pid)
    value = result_value(final_obs, model_pid)
    for row in rows:
        row["value"] = value
    stats["rows"] = len(rows)
    stats["value"] = float(value)
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="alphaZeroLike/checkpoints/latest.pt")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--teacher-mode", choices=["source_bridge", "rulebase"], default="source_bridge")
    parser.add_argument("--out", default="data/alphaZeroLike/source_bridge_policy_20260522.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--oracle", default="rl_informed_regular")
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--max-moves", type=int, default=16)
    parser.add_argument("--proposal-send-threshold", type=float, default=0.30)
    parser.add_argument("--proposal-num-candidates", type=int, default=16)
    parser.add_argument("--proposal-num-full-actions", type=int, default=8)
    parser.add_argument("--proposal-max-sources", type=int, default=4)
    parser.add_argument("--proposal-explore-min-anchor-moves", type=int, default=1)
    parser.add_argument("--student-no-model-proposals", action="store_true")
    parser.add_argument("--no-heuristics", action="store_true")
    parser.add_argument("--no-noop", action="store_true")
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    totals = {"turns": 0, "changed": 0, "blocked": 0, "rows": 0, "value_sum": 0.0}
    with out.open("w") as f:
        tasks = [
            (game, args.seed + game // args.players, game % args.players)
            for game in range(args.games)
        ]
        if args.workers <= 1:
            results = []
            for game, seed, model_pid in tasks:
                rows, stats = generate_game(args, seed, model_pid)
                results.append((game, seed, model_pid, rows, stats))
        else:
            results = []
            with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
                futures = {
                    pool.submit(generate_game, args, seed, model_pid): (game, seed, model_pid)
                    for game, seed, model_pid in tasks
                }
                for future in as_completed(futures):
                    game, seed, model_pid = futures[future]
                    rows, stats = future.result()
                    results.append((game, seed, model_pid, rows, stats))
                    print({"game": game, "seed": seed, "model_pid": model_pid, **stats}, flush=True)
        for game, seed, model_pid, rows, stats in sorted(results, key=lambda row: row[0]):
            for row in rows:
                f.write(json.dumps(row) + "\n")
            totals["turns"] += stats["turns"]
            totals["changed"] += stats["changed"]
            totals["blocked"] += stats["blocked"]
            totals["rows"] += stats["rows"]
            totals["value_sum"] += stats["value"]
            if args.workers <= 1:
                print({"game": game, "seed": seed, "model_pid": model_pid, **stats}, flush=True)
    totals["changed_rate"] = totals["changed"] / max(totals["turns"], 1)
    totals["blocked_rate"] = totals["blocked"] / max(totals["turns"], 1)
    totals["mean_value"] = totals["value_sum"] / max(args.games, 1)
    print({"out": str(out), **totals}, flush=True)


if __name__ == "__main__":
    main()
