"""Compare model proposal candidates with REGULAR_CONFIG actions in fast env."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from alphaZeroLike.model import AlphaZeroLikeNet
from alphaZeroLike.proposal import ProposalConfig, proposal_labels, proposals_from_model
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent


def _raw_obs(env: Any, player: int) -> dict:
    return env.steps[-1][player]["observation"]


def _load_model(checkpoint: Path, device: str) -> AlphaZeroLikeNet:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = AlphaZeroLikeNet(
        d_model=int(ckpt_args.get("d_model", 192)),
        nhead=int(ckpt_args.get("nhead", 6)),
        layers=int(ckpt_args.get("layers", 4)),
        dropout=float(ckpt_args.get("dropout", 0.10)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def _active_pairs(labels: dict) -> set[tuple[int, int]]:
    return {
        (i, int(labels["proposal_target"][i]))
        for i, (send, valid) in enumerate(zip(labels.get("proposal_send", []), labels.get("proposal_valid", [])))
        if float(send) > 0.5 and float(valid) > 0.5
    }


def _active_sources(labels: dict) -> set[int]:
    return {src for src, _ in _active_pairs(labels)}


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def _legal_action(obs: dict, player: int, action: list[list]) -> bool:
    by_id = {int(p[0]): p for p in obs.get("planets", [])}
    used: dict[int, int] = {}
    for move in action:
        if len(move) < 3:
            return False
        src = int(move[0])
        ships = int(move[2])
        planet = by_id.get(src)
        if planet is None or int(planet[1]) != player or ships < 1:
            return False
        used[src] = used.get(src, 0) + ships
        if used[src] > int(float(planet[5])):
            return False
    return True


def _compare(rule_labels: dict, cand_labels: dict) -> dict[str, float]:
    rule_sources = _active_sources(rule_labels)
    cand_sources = _active_sources(cand_labels)
    rule_pairs = _active_pairs(rule_labels)
    cand_pairs = _active_pairs(cand_labels)
    if rule_pairs:
        pair_recall = len(rule_pairs & cand_pairs) / len(rule_pairs)
    else:
        pair_recall = 1.0 if not cand_pairs else 0.0
    return {
        "source_jaccard": _jaccard(rule_sources, cand_sources),
        "pair_jaccard": _jaccard(rule_pairs, cand_pairs),
        "pair_recall": float(pair_recall),
        "source_exact": float(rule_sources == cand_sources),
        "pair_exact": float(rule_pairs == cand_pairs),
    }


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="alphaZeroLike/checkpoints/latest.pt")
    parser.add_argument("--out", default="alphaZeroLike/logs/proposal_rollout_eval_latest.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--episode-steps", type=int, default=180)
    parser.add_argument("--four-player-prob", type=float, default=0.3)
    parser.add_argument("--oracle", default="rl_informed_regular")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--send-threshold", type=float, default=0.30)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--num-full-actions", type=int, default=8)
    parser.add_argument("--num-sampled-actions", type=int, default=8)
    parser.add_argument("--max-sources", type=int, default=4)
    parser.add_argument("--top-targets-per-source", type=int, default=2)
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    model = _load_model(Path(args.checkpoint), device)
    cfg = ProposalConfig(
        num_candidates=args.num_candidates,
        num_full_actions=args.num_full_actions,
        num_sampled_actions=args.num_sampled_actions,
        send_threshold=args.send_threshold,
        max_sources=args.max_sources,
        top_targets_per_source=args.top_targets_per_source,
    )

    top1_source_j: list[float] = []
    top1_pair_j: list[float] = []
    best_source_j: list[float] = []
    best_pair_j: list[float] = []
    best_pair_recall: list[float] = []
    active_top1_source_j: list[float] = []
    active_top1_pair_j: list[float] = []
    active_best_source_j: list[float] = []
    active_best_pair_j: list[float] = []
    active_best_pair_recall: list[float] = []
    legal_rates: list[float] = []
    num_props: list[int] = []
    rule_sizes: list[int] = []
    top1_sizes: list[int] = []
    examples: list[dict] = []
    rows = 0
    active_rows = 0
    games_2p = 0
    games_4p = 0

    progress = tqdm(total=args.episodes, desc="[ProposalRolloutEval]", unit="game")
    for ep in range(args.episodes):
        rng = random.Random(args.seed + ep)
        players = 4 if rng.random() < args.four_player_prob else 2
        games_4p += int(players == 4)
        games_2p += int(players == 2)
        env = make_fast_orbit_wars(
            {"episodeSteps": args.episode_steps, "seed": args.seed + ep},
            keep_history=False,
            use_numba=not args.no_numba,
        )
        env.reset(players)
        agents = {pid: make_rulebase_agent(args.oracle) for pid in range(players)}
        for _ in range(args.episode_steps):
            actions: list[list[list]] = []
            for pid in range(players):
                obs = _raw_obs(env, pid)
                rule_action = agents[pid](obs) or []
                rule_labels = proposal_labels(obs, pid, rule_action)
                props = proposals_from_model(obs, pid, model, device, cfg)
                rows += 1
                rule_pair_count = len(_active_pairs(rule_labels))
                active_rows += int(rule_pair_count > 0)
                rule_sizes.append(rule_pair_count)
                num_props.append(len(props))
                legal_rates.append(float(all(_legal_action(obs, pid, prop) for prop in props)))
                if props:
                    top_labels = proposal_labels(obs, pid, props[0])
                    top_cmp = _compare(rule_labels, top_labels)
                    top1_source_j.append(top_cmp["source_jaccard"])
                    top1_pair_j.append(top_cmp["pair_jaccard"])
                    top1_sizes.append(len(_active_pairs(top_labels)))
                    best_cmp = max((_compare(rule_labels, proposal_labels(obs, pid, p)) for p in props), key=lambda x: x["pair_jaccard"])
                    best_source_j.append(best_cmp["source_jaccard"])
                    best_pair_j.append(best_cmp["pair_jaccard"])
                    best_pair_recall.append(best_cmp["pair_recall"])
                else:
                    top1_source_j.append(0.0 if rule_pair_count else 1.0)
                    top1_pair_j.append(0.0 if rule_pair_count else 1.0)
                    top1_sizes.append(0)
                    best_source_j.append(0.0 if rule_pair_count else 1.0)
                    best_pair_j.append(0.0 if rule_pair_count else 1.0)
                    best_pair_recall.append(0.0 if rule_pair_count else 1.0)
                if rule_pair_count > 0:
                    active_top1_source_j.append(top1_source_j[-1])
                    active_top1_pair_j.append(top1_pair_j[-1])
                    active_best_source_j.append(best_source_j[-1])
                    active_best_pair_j.append(best_pair_j[-1])
                    active_best_pair_recall.append(best_pair_recall[-1])
                if len(examples) < 8 and rule_pair_count > 0:
                    examples.append(
                        {
                            "episode": ep,
                            "player": pid,
                            "step": int(obs.get("step", 0)),
                            "rule_pairs": sorted([list(x) for x in _active_pairs(rule_labels)]),
                            "top1_pairs": sorted([list(x) for x in _active_pairs(proposal_labels(obs, pid, props[0]))]) if props else [],
                            "num_proposals": len(props),
                            "top1_pair_jaccard": round(top1_pair_j[-1], 3),
                            "best_pair_jaccard": round(best_pair_j[-1], 3),
                        }
                    )
                actions.append(rule_action)
            env.step(actions)
            if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                break
        progress.update(1)
        progress.set_postfix(rows=rows, active=active_rows)
    progress.close()

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "oracle": args.oracle,
        "episodes": args.episodes,
        "games_2p": games_2p,
        "games_4p": games_4p,
        "rows": rows,
        "active_rows": active_rows,
        "proposal_config": cfg.__dict__,
        "metrics": {
            "legal_candidate_set_rate": _mean(legal_rates),
            "num_proposals_mean": _mean(num_props),
            "rule_action_size_mean": _mean(rule_sizes),
            "top1_action_size_mean": _mean(top1_sizes),
            "top1_source_jaccard": _mean(top1_source_j),
            "top1_pair_jaccard": _mean(top1_pair_j),
            "best_source_jaccard": _mean(best_source_j),
            "best_pair_jaccard": _mean(best_pair_j),
            "best_pair_recall": _mean(best_pair_recall),
            "active_top1_source_jaccard": _mean(active_top1_source_j),
            "active_top1_pair_jaccard": _mean(active_top1_pair_j),
            "active_best_source_jaccard": _mean(active_best_source_j),
            "active_best_pair_jaccard": _mean(active_best_pair_j),
            "active_best_pair_recall": _mean(active_best_pair_recall),
        },
        "examples": examples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
