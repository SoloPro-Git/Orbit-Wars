from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch

from training.expert.action_labeling import infer_target_planet_id
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent

from tinyPPO.agents import (
    ACTION_SLOTS,
    TinyPPOAgent,
    all_planets_target_mask,
    apply_target_safety_mask,
    candidate_target_mask,
    safe_target_mask,
)
from tinyPPO.features import MAX_PLANETS, encode_obs, score


def _int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _turn_bucket(turn: int, width: int) -> str:
    lo = (max(0, int(turn)) // width) * width
    return f"{lo:03d}-{lo + width - 1:03d}"


def _outcome(model_score: float, opponent_score: float) -> str:
    if model_score > opponent_score:
        return "win"
    if model_score < opponent_score:
        return "loss"
    return "draw"


def _owned_planets(obs: dict[str, Any], player: int) -> int:
    return sum(1 for planet in obs.get("planets", []) if int(planet[1]) == int(player))


def _owned_ships(obs: dict[str, Any], player: int) -> float:
    return float(sum(float(planet[5]) for planet in obs.get("planets", []) if int(planet[1]) == int(player)))


def _empty_stats() -> dict[str, float]:
    return {
        "samples": 0.0,
        "model_actions": 0.0,
        "regular_actions": 0.0,
        "action_gap": 0.0,
        "abs_action_gap": 0.0,
        "model_zero_regular_nonzero": 0.0,
        "regular_zero_model_nonzero": 0.0,
        "both_zero": 0.0,
        "owned_planets": 0.0,
        "owned_ships": 0.0,
        "score_gap": 0.0,
        "source_tp": 0.0,
        "source_pred": 0.0,
        "source_true": 0.0,
        "pair_tp": 0.0,
        "pair_pred": 0.0,
        "pair_true": 0.0,
        "ship_matched": 0.0,
        "ship_abs_error": 0.0,
        "ship_rel_error": 0.0,
        "target_rank_actions": 0.0,
        "target_rank_covered": 0.0,
        "target_rank_sum": 0.0,
        "target_rank_mrr": 0.0,
        "target_rank_top1": 0.0,
        "target_rank_top3": 0.0,
        "target_rank_top5": 0.0,
        "target_rank_uncovered": 0.0,
        "true_owner_own": 0.0,
        "true_owner_neutral": 0.0,
        "true_owner_enemy": 0.0,
        "pred_owner_own": 0.0,
        "pred_owner_neutral": 0.0,
        "pred_owner_enemy": 0.0,
        "matched_owner_own": 0.0,
        "matched_owner_neutral": 0.0,
        "matched_owner_enemy": 0.0,
        "missed_owner_own": 0.0,
        "missed_owner_neutral": 0.0,
        "missed_owner_enemy": 0.0,
        "extra_owner_own": 0.0,
        "extra_owner_neutral": 0.0,
        "extra_owner_enemy": 0.0,
    }


def _owner_bucket(obs: dict[str, Any], target_idx: int, player: int) -> str:
    planets = list(obs.get("planets", []))
    if target_idx < 0 or target_idx >= len(planets):
        return "unknown"
    owner = int(planets[target_idx][1])
    if owner == int(player):
        return "own"
    if owner == -1:
        return "neutral"
    return "enemy"


def _action_items(obs: dict[str, Any], player: int, actions: list[list]) -> list[tuple[int, int, int]]:
    planets = list(obs.get("planets", []))[:MAX_PLANETS]
    id_to_idx = {int(p[0]): idx for idx, p in enumerate(planets)}
    items: list[tuple[int, int, int]] = []
    for action in actions or []:
        if not isinstance(action, list) or len(action) < 3:
            continue
        src_idx = id_to_idx.get(int(action[0]))
        if src_idx is None or src_idx >= len(planets):
            continue
        ships = max(1, int(float(action[2])))
        target_id = infer_target_planet_id(obs, int(action[0]), float(action[1]), ships)
        tgt_idx = id_to_idx.get(int(target_id)) if target_id is not None else None
        if tgt_idx is None or tgt_idx == src_idx or tgt_idx >= len(planets):
            continue
        items.append((int(src_idx), int(tgt_idx), int(ships)))
    return items


def _counter_from_items(items: list[tuple[int, int, int]], pair: bool) -> Counter:
    if pair:
        return Counter((src_idx, tgt_idx) for src_idx, tgt_idx, _ships in items)
    return Counter(src_idx for src_idx, _tgt_idx, _ships in items)


def _add_owner_counts(stats: dict[str, float], obs: dict[str, Any], player: int, prefix: str, pairs: Counter) -> None:
    for (_src_idx, tgt_idx), count in pairs.items():
        owner = _owner_bucket(obs, int(tgt_idx), player)
        key = f"{prefix}_owner_{owner}"
        if key in stats:
            stats[key] += float(count)


def _add_ship_errors(
    stats: dict[str, float],
    model_items: list[tuple[int, int, int]],
    regular_items: list[tuple[int, int, int]],
) -> None:
    model_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
    regular_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
    for src_idx, tgt_idx, ships in model_items:
        model_by_pair[(src_idx, tgt_idx)].append(int(ships))
    for src_idx, tgt_idx, ships in regular_items:
        regular_by_pair[(src_idx, tgt_idx)].append(int(ships))
    for key in set(model_by_pair) & set(regular_by_pair):
        pred = sorted(model_by_pair[key])
        true = sorted(regular_by_pair[key])
        for pred_ships, true_ships in zip(pred, true, strict=False):
            abs_err = abs(float(pred_ships) - float(true_ships))
            stats["ship_matched"] += 1.0
            stats["ship_abs_error"] += abs_err
            stats["ship_rel_error"] += abs_err / max(1.0, float(true_ships))


def _add_action_overlap(
    stats: dict[str, float],
    obs: dict[str, Any],
    player: int,
    model_action: list[list],
    regular_label: list[list],
) -> None:
    model_items = _action_items(obs, player, model_action)
    regular_items = _action_items(obs, player, regular_label)
    model_sources = _counter_from_items(model_items, pair=False)
    regular_sources = _counter_from_items(regular_items, pair=False)
    model_pairs = _counter_from_items(model_items, pair=True)
    regular_pairs = _counter_from_items(regular_items, pair=True)
    matched_sources = model_sources & regular_sources
    matched_pairs = model_pairs & regular_pairs
    missed_pairs = regular_pairs - model_pairs
    extra_pairs = model_pairs - regular_pairs

    stats["source_tp"] += float(sum(matched_sources.values()))
    stats["source_pred"] += float(sum(model_sources.values()))
    stats["source_true"] += float(sum(regular_sources.values()))
    stats["pair_tp"] += float(sum(matched_pairs.values()))
    stats["pair_pred"] += float(sum(model_pairs.values()))
    stats["pair_true"] += float(sum(regular_pairs.values()))
    _add_ship_errors(stats, model_items, regular_items)
    _add_owner_counts(stats, obs, player, "true", regular_pairs)
    _add_owner_counts(stats, obs, player, "pred", model_pairs)
    _add_owner_counts(stats, obs, player, "matched", matched_pairs)
    _add_owner_counts(stats, obs, player, "missed", missed_pairs)
    _add_owner_counts(stats, obs, player, "extra", extra_pairs)


@torch.no_grad()
def _add_regular_target_ranks(
    stats: dict[str, float],
    model_agent: TinyPPOAgent,
    obs: dict[str, Any],
    player: int,
    regular_label: list[list],
) -> None:
    regular_items = _action_items(obs, player, regular_label)
    if not regular_items:
        return
    enc = encode_obs(obs, player, players=2)
    device = model_agent.device
    batch = {
        "planets": torch.tensor(enc.planets[None], dtype=torch.float32, device=device),
        "pair_features": torch.tensor(enc.pair_features[None], dtype=torch.float32, device=device),
        "global_features": torch.tensor(enc.global_features[None], dtype=torch.float32, device=device),
        "planet_mask": torch.tensor(enc.planet_mask[None], dtype=torch.bool, device=device),
        "own_mask": torch.tensor(enc.own_mask[None], dtype=torch.bool, device=device),
    }
    out = model_agent.model(**batch)
    source_logits = out["source_logits"][0] / model_agent.launch_temperature
    if model_agent.launch_bias:
        source_logits = source_logits.clone()
        source_logits[..., 1] += model_agent.launch_bias
    target_logits = out["target_logits"][0]
    if "target_pair_logits" in out and abs(model_agent.target_pair_weight) > 1e-9:
        target_logits = target_logits + model_agent.target_pair_weight * out["target_pair_logits"][0][:, None, :]
    if model_agent.ship_mode == "required_bucket" and model_agent.target_mask_mode == "candidate":
        target_mask = candidate_target_mask(
            obs,
            player,
            top_k=model_agent.target_top_k,
            include_friendly=model_agent.include_friendly_targets,
        )
    elif model_agent.target_mask_mode == "all_planets":
        target_mask = all_planets_target_mask(obs, player)
    else:
        target_mask = safe_target_mask(obs, player)
    _source_logits, target_logits = apply_target_safety_mask(source_logits, target_logits, obs, player, target_mask=target_mask)
    slot_by_source: Counter[int] = Counter()
    for source_idx, target_idx, _ships in regular_items:
        slot = int(slot_by_source[source_idx])
        slot_by_source[source_idx] += 1
        if slot >= min(ACTION_SLOTS, target_logits.shape[1]):
            continue
        stats["target_rank_actions"] += 1.0
        if not (0 <= target_idx < MAX_PLANETS and bool(target_mask[source_idx, target_idx])):
            stats["target_rank_uncovered"] += 1.0
            continue
        valid_mask = torch.tensor(target_mask[source_idx], dtype=torch.bool, device=device)
        logits = target_logits[source_idx, slot]
        label_score = logits[target_idx]
        rank = int((logits[valid_mask] > label_score).sum().item()) + 1
        stats["target_rank_covered"] += 1.0
        stats["target_rank_sum"] += float(rank)
        stats["target_rank_mrr"] += 1.0 / float(rank)
        stats["target_rank_top1"] += float(rank <= 1)
        stats["target_rank_top3"] += float(rank <= 3)
        stats["target_rank_top5"] += float(rank <= 5)


def _add_step(
    stats: dict[str, float],
    obs: dict[str, Any],
    player: int,
    model_action: list[list],
    regular_label: list[list],
    model_agent: TinyPPOAgent | None,
) -> None:
    model_actions = len(model_action)
    regular_actions = len(regular_label)
    gap = float(regular_actions - model_actions)
    stats["samples"] += 1.0
    stats["model_actions"] += float(model_actions)
    stats["regular_actions"] += float(regular_actions)
    stats["action_gap"] += gap
    stats["abs_action_gap"] += abs(gap)
    stats["model_zero_regular_nonzero"] += float(model_actions == 0 and regular_actions > 0)
    stats["regular_zero_model_nonzero"] += float(regular_actions == 0 and model_actions > 0)
    stats["both_zero"] += float(regular_actions == 0 and model_actions == 0)
    stats["owned_planets"] += float(_owned_planets(obs, player))
    stats["owned_ships"] += float(_owned_ships(obs, player))
    stats["score_gap"] += float(score(obs, player) - score(obs, 1 - player))
    _add_action_overlap(stats, obs, player, model_action, regular_label)
    if model_agent is not None:
        _add_regular_target_ranks(stats, model_agent, obs, player, regular_label)


def _prf(tp: float, pred: float, true: float) -> tuple[float, float, float]:
    precision = tp / pred if pred > 0 else (1.0 if true <= 0 else 0.0)
    recall = tp / true if true > 0 else (1.0 if pred <= 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _owner_dict(stats: dict[str, float], prefix: str, denom: float) -> dict[str, float]:
    return {
        owner: float(stats.get(f"{prefix}_owner_{owner}", 0.0)) / max(1.0, denom)
        for owner in ("own", "neutral", "enemy")
    }


def _finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    samples = max(1.0, float(stats.get("samples", 0.0)))
    source_p, source_r, source_f1 = _prf(
        float(stats.get("source_tp", 0.0)),
        float(stats.get("source_pred", 0.0)),
        float(stats.get("source_true", 0.0)),
    )
    pair_p, pair_r, pair_f1 = _prf(
        float(stats.get("pair_tp", 0.0)),
        float(stats.get("pair_pred", 0.0)),
        float(stats.get("pair_true", 0.0)),
    )
    ship_matched = max(1.0, float(stats.get("ship_matched", 0.0)))
    rank_covered = max(1.0, float(stats.get("target_rank_covered", 0.0)))
    rank_actions = max(1.0, float(stats.get("target_rank_actions", 0.0)))
    true_pairs = float(stats.get("pair_true", 0.0))
    pred_pairs = float(stats.get("pair_pred", 0.0))
    return {
        "samples": float(stats.get("samples", 0.0)),
        "model_actions_mean": float(stats.get("model_actions", 0.0)) / samples,
        "regular_actions_mean": float(stats.get("regular_actions", 0.0)) / samples,
        "action_gap_mean": float(stats.get("action_gap", 0.0)) / samples,
        "abs_action_gap_mean": float(stats.get("abs_action_gap", 0.0)) / samples,
        "model_zero_regular_nonzero_rate": float(stats.get("model_zero_regular_nonzero", 0.0)) / samples,
        "regular_zero_model_nonzero_rate": float(stats.get("regular_zero_model_nonzero", 0.0)) / samples,
        "both_zero_rate": float(stats.get("both_zero", 0.0)) / samples,
        "owned_planets_mean": float(stats.get("owned_planets", 0.0)) / samples,
        "owned_ships_mean": float(stats.get("owned_ships", 0.0)) / samples,
        "score_gap_mean": float(stats.get("score_gap", 0.0)) / samples,
        "source_precision": source_p,
        "source_recall": source_r,
        "source_f1": source_f1,
        "source_target_precision": pair_p,
        "source_target_recall": pair_r,
        "source_target_f1": pair_f1,
        "ship_matched_actions": float(stats.get("ship_matched", 0.0)),
        "ship_abs_error_mean": float(stats.get("ship_abs_error", 0.0)) / ship_matched,
        "ship_rel_error_mean": float(stats.get("ship_rel_error", 0.0)) / ship_matched,
        "target_rank_actions": float(stats.get("target_rank_actions", 0.0)),
        "target_rank_covered_rate": float(stats.get("target_rank_covered", 0.0)) / rank_actions,
        "target_rank_mean": float(stats.get("target_rank_sum", 0.0)) / rank_covered,
        "target_rank_mrr": float(stats.get("target_rank_mrr", 0.0)) / rank_covered,
        "target_rank_top1": float(stats.get("target_rank_top1", 0.0)) / rank_covered,
        "target_rank_top3": float(stats.get("target_rank_top3", 0.0)) / rank_covered,
        "target_rank_top5": float(stats.get("target_rank_top5", 0.0)) / rank_covered,
        "target_rank_uncovered_rate": float(stats.get("target_rank_uncovered", 0.0)) / rank_actions,
        "true_owner_rate": _owner_dict(stats, "true", true_pairs),
        "pred_owner_rate": _owner_dict(stats, "pred", pred_pairs),
        "missed_owner_rate_per_true_pair": _owner_dict(stats, "missed", true_pairs),
        "extra_owner_rate_per_pred_pair": _owner_dict(stats, "extra", pred_pairs),
    }


def _merge_stats(parts: list[dict[str, float]]) -> dict[str, float]:
    merged = _empty_stats()
    for part in parts:
        for key, value in part.items():
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged


@ray.remote
class RolloutDiagnoseActor:
    def __init__(
        self,
        checkpoint: str,
        device: str,
        deterministic: bool,
        launch_bias: float,
        ship_bias: float,
        launch_temperature: float,
        target_mask_mode: str,
        target_pair_weight: float,
    ):
        if device.startswith("cuda:"):
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
            device = "cuda:0"
        self.model_agent = TinyPPOAgent(
            checkpoint,
            device=device,
            deterministic=deterministic,
            launch_bias=launch_bias,
            ship_bias=ship_bias,
            launch_temperature=launch_temperature,
            target_mask_mode=target_mask_mode,
            target_pair_weight=target_pair_weight,
        )
    def run_games(
        self,
        jobs: list[tuple[int, int]],
        episode_steps: int,
        turn_bucket: int,
        use_numba: bool,
    ) -> dict[str, Any]:
        overall = _empty_stats()
        by_turn: dict[str, dict[str, float]] = defaultdict(_empty_stats)
        by_outcome: dict[str, dict[str, float]] = defaultdict(_empty_stats)
        outcome_counts: Counter[str] = Counter()
        final_score_gap_sum = 0.0
        game_lengths: list[int] = []

        for seed, model_seat in jobs:
            step_records: list[tuple[dict[str, Any], int, int, list[list], list[list]]] = []
            label_agent = make_rulebase_agent("regular")
            regular_agents = [make_rulebase_agent("regular") for _ in range(2)]
            agents = []
            for pid in range(2):
                if pid == model_seat:

                    def model_logged(obs: dict[str, Any], configuration=None, pid: int = pid) -> list[list]:
                        del configuration
                        model_action = self.model_agent(obs) or []
                        regular_label = label_agent(obs) or []
                        step_records.append(
                            (
                                obs,
                                int(pid),
                                int(obs.get("step", len(step_records))),
                                model_action,
                                regular_label,
                            )
                        )
                        return model_action

                    agents.append(model_logged)
                else:

                    regular_agent = regular_agents[pid]

                    def regular(obs: dict[str, Any], configuration=None, regular_agent=regular_agent) -> list[list]:
                        del configuration
                        return regular_agent(obs) or []

                    agents.append(regular)

            env = make_fast_orbit_wars(
                {"episodeSteps": episode_steps, "seed": seed},
                keep_history=False,
                use_numba=use_numba,
            )
            env.run(agents)
            final_frame = env.steps[-1] if env.steps else []
            final_obs = final_frame[model_seat].get("observation", {}) if final_frame else {}
            model_score = float(score(final_obs, model_seat)) if final_obs else 0.0
            other_score = float(score(final_obs, 1 - model_seat)) if final_obs else 0.0
            game_outcome = _outcome(model_score, other_score)
            outcome_counts[game_outcome] += 1
            final_score_gap_sum += model_score - other_score
            game_lengths.append(int(getattr(env, "_step", 0)))

            for obs, player, turn, model_action, regular_label in step_records:
                _add_step(overall, obs, player, model_action, regular_label, self.model_agent)
                _add_step(by_turn[_turn_bucket(turn, turn_bucket)], obs, player, model_action, regular_label, self.model_agent)
                _add_step(by_outcome[game_outcome], obs, player, model_action, regular_label, self.model_agent)

        games = max(1, len(jobs))
        return {
            "games": float(len(jobs)),
            "wins": float(outcome_counts.get("win", 0)),
            "losses": float(outcome_counts.get("loss", 0)),
            "draws": float(outcome_counts.get("draw", 0)),
            "final_score_gap_mean": final_score_gap_sum / float(games),
            "game_length_mean": float(np.mean(game_lengths)) if game_lengths else 0.0,
            "overall_raw": overall,
            "by_turn_raw": dict(by_turn),
            "by_outcome_raw": dict(by_outcome),
        }


def _merge_reports(parts: list[dict[str, Any]]) -> dict[str, Any]:
    games = sum(float(part["games"]) for part in parts)
    wins = sum(float(part["wins"]) for part in parts)
    losses = sum(float(part["losses"]) for part in parts)
    draws = sum(float(part["draws"]) for part in parts)
    weighted_gap = sum(float(part["final_score_gap_mean"]) * float(part["games"]) for part in parts)
    weighted_length = sum(float(part["game_length_mean"]) * float(part["games"]) for part in parts)
    overall = _merge_stats([part["overall_raw"] for part in parts])

    turn_accum: dict[str, list[dict[str, float]]] = defaultdict(list)
    outcome_accum: dict[str, list[dict[str, float]]] = defaultdict(list)
    for part in parts:
        for key, value in part["by_turn_raw"].items():
            turn_accum[key].append(value)
        for key, value in part["by_outcome_raw"].items():
            outcome_accum[key].append(value)

    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": wins / max(1.0, games),
        "nonloss": (wins + draws) / max(1.0, games),
        "final_score_gap_mean": weighted_gap / max(1.0, games),
        "game_length_mean": weighted_length / max(1.0, games),
        "overall": _finalize_stats(overall),
        "by_turn": {key: _finalize_stats(_merge_stats(values)) for key, values in sorted(turn_accum.items())},
        "by_outcome": {key: _finalize_stats(_merge_stats(values)) for key, values in sorted(outcome_accum.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="path|name")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=260527)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--actors", type=int, default=32)
    parser.add_argument("--cpus-per-actor", type=float, default=1.0)
    parser.add_argument("--gpus-per-actor", type=float, default=0.0)
    parser.add_argument("--gpu-ids-manual", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--ship-bias", type=float, default=0.0)
    parser.add_argument("--launch-temperature", type=float, default=1.0)
    parser.add_argument("--target-mask-mode", choices=["candidate", "safe", "all_planets"], default="candidate")
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--turn-bucket", type=int, default=50)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-temp-dir", default="", help="Optional Ray temp/session directory, useful when /tmp is low on space.")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    ray_init_kwargs = {
        "address": args.ray_address,
        "ignore_reinit_error": True,
        "runtime_env": {
            "excludes": [
                "swanlog/**",
                "wandb/**",
                "tinyPPO/data/*.pkl",
                "tinyPPO/runs/**/*.pt",
                "tinyPPO/runs/**/*.pkl",
                "tinyPPO/runs/**/train.log",
            ]
        },
    }
    if args.ray_address == "local":
        ray_init_kwargs["address"] = None
    if args.ray_temp_dir:
        ray_init_kwargs["_temp_dir"] = args.ray_temp_dir
    ray.init(**ray_init_kwargs)
    gpu_ids = _int_list(args.gpu_ids_manual)
    jobs = [(int(args.seed) + i, i % 2) for i in range(int(args.games))]
    reports: dict[str, Any] = {}
    actor_count = max(1, min(int(args.actors), len(jobs)))
    shards = [jobs[i::actor_count] for i in range(actor_count)]

    for raw_checkpoint in args.checkpoint:
        parts = raw_checkpoint.split("|", 1)
        path = Path(parts[0]).expanduser().resolve()
        name = parts[1] if len(parts) == 2 else path.stem
        if not path.exists():
            raise FileNotFoundError(path)
        actors = [
            RolloutDiagnoseActor.options(
                num_cpus=float(args.cpus_per_actor),
                num_gpus=0.0 if gpu_ids else float(args.gpus_per_actor),
            ).remote(
                str(path),
                f"cuda:{gpu_ids[i % len(gpu_ids)]}" if gpu_ids else str(args.device),
                not bool(args.stochastic),
                float(args.launch_bias),
                float(args.ship_bias),
                float(args.launch_temperature),
                str(args.target_mask_mode),
                float(args.target_pair_weight),
            )
            for i in range(actor_count)
        ]
        refs = [
            actor.run_games.remote(shard, int(args.episode_steps), int(args.turn_bucket), not bool(args.no_numba))
            for actor, shard in zip(actors, shards, strict=True)
            if shard
        ]
        parts_out: list[dict[str, Any]] = []
        pending = list(refs)
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            parts_out.extend(ray.get(done))
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        reports[name] = _merge_reports(parts_out)
        print(json.dumps({"event": "diagnosed", "name": name, "report": reports[name]}, ensure_ascii=True), flush=True)

    result = {"args": vars(args), "reports": reports}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
