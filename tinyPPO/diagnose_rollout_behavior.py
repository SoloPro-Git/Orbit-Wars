from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import ray

from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent

from tinyPPO.agents import TinyPPOAgent
from tinyPPO.features import score


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
    }


def _add_step(stats: dict[str, float], obs: dict[str, Any], player: int, model_actions: int, regular_actions: int) -> None:
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


def _finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    samples = max(1.0, float(stats.get("samples", 0.0)))
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
        )
        self.regular_agent = make_rulebase_agent("regular")

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
            step_records: list[tuple[dict[str, Any], int, int, int, int]] = []
            agents = []
            for pid in range(2):
                if pid == model_seat:

                    def model_logged(obs: dict[str, Any], configuration=None, pid: int = pid) -> list[list]:
                        del configuration
                        model_action = self.model_agent(obs) or []
                        regular_label = self.regular_agent(obs) or []
                        step_records.append(
                            (
                                obs,
                                int(pid),
                                int(obs.get("step", len(step_records))),
                                len(model_action),
                                len(regular_label),
                            )
                        )
                        return model_action

                    agents.append(model_logged)
                else:

                    def regular(obs: dict[str, Any], configuration=None) -> list[list]:
                        del configuration
                        return self.regular_agent(obs) or []

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

            for obs, player, turn, model_count, regular_count in step_records:
                _add_step(overall, obs, player, model_count, regular_count)
                _add_step(by_turn[_turn_bucket(turn, turn_bucket)], obs, player, model_count, regular_count)
                _add_step(by_outcome[game_outcome], obs, player, model_count, regular_count)

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
    parser.add_argument("--turn-bucket", type=int, default=50)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    ray.init(
        address=args.ray_address,
        ignore_reinit_error=True,
        runtime_env={
            "excludes": [
                "swanlog/**",
                "wandb/**",
                "tinyPPO/data/*.pkl",
                "tinyPPO/runs/**/*.pt",
                "tinyPPO/runs/**/*.pkl",
                "tinyPPO/runs/**/train.log",
            ]
        },
    )
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
