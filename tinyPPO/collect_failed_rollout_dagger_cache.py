from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import ray

from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent

from tinyPPO.agents import TinyPPOAgent
from tinyPPO.features import score
from tinyPPO.imitation_regular import row_from_regular_action
from tinyPPO.imitation_regular_ray import _int_list

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _turn_bucket(turn: int) -> str:
    lo = (max(0, turn) // 50) * 50
    return f"{lo}-{lo + 49}"


def _reward_bucket(reward: float) -> str:
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def _outcome_bucket(model_score: float, opponent_score: float) -> str:
    if model_score > opponent_score:
        return "win"
    if model_score < opponent_score:
        return "loss"
    return "draw"


def _row_action_count(row: Any) -> int:
    labelled = getattr(row, "labelled", None)
    if labelled is not None:
        return int(labelled)
    return int(np.asarray(getattr(row, "launch_mask")).sum())


def _owned_planets(obs: dict[str, Any], player: int) -> int:
    return sum(1 for planet in obs.get("planets", []) if int(planet[1]) == int(player))


def _owned_ships(obs: dict[str, Any], player: int) -> float:
    return float(sum(float(planet[5]) for planet in obs.get("planets", []) if int(planet[1]) == int(player)))


def _score_gap(obs: dict[str, Any], player: int, players: int) -> float:
    own_score = float(score(obs, player))
    opponents = [pid for pid in range(players) if pid != player]
    if not opponents:
        return 0.0
    return own_score - max(float(score(obs, pid)) for pid in opponents)


@ray.remote
class FailedRolloutDaggerActor:
    def __init__(
        self,
        checkpoint: str,
        device: str,
        deterministic: bool,
        launch_bias: float,
        ship_bias: float,
        launch_temperature: float,
        target_top_k: int,
        target_mask_mode: str,
        target_pair_weight: float,
    ):
        if device.startswith("cuda:"):
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
            device = "cuda:0"
        self.checkpoint = checkpoint
        self.model_agent = TinyPPOAgent(
            checkpoint,
            device=device,
            deterministic=deterministic,
            launch_bias=launch_bias,
            ship_bias=ship_bias,
            launch_temperature=launch_temperature,
            target_top_k=target_top_k,
            target_mask_mode=target_mask_mode,
            target_pair_weight=target_pair_weight,
        )

    def collect_games(
        self,
        jobs: list[tuple[int, int, int]],
        episode_steps: int,
        keep_noop_prob: float,
        sample_stride: int,
        rows_per_game: int,
        min_turn: int,
        max_turn: int,
        max_final_reward: float,
        keep_outcomes: list[str],
        min_abs_action_gap: int,
        min_action_gap: int,
        max_action_gap: int,
        min_owned_planets: int,
        min_owned_ships: float,
        min_score_gap: float,
        use_numba: bool,
        target_mask_mode: str,
    ) -> tuple[list[Any], dict[str, float], dict[str, dict[str, float]]]:
        rows: list[Any] = []
        metrics: dict[str, float] = {
            "games": 0.0,
            "kept_games": 0.0,
            "samples": 0.0,
            "labelled_actions": 0.0,
            "skipped_actions": 0.0,
            "regular_label_actions": 0.0,
            "model_actions": 0.0,
        }
        buckets: dict[str, Counter] = {
            "turn": Counter(),
            "reward": Counter(),
            "action_gap": Counter(),
            "game_outcome": Counter(),
        }
        for seed, players, model_seat in jobs:
            random.seed(seed)
            np.random.seed(seed)
            raw_rows: list[dict[str, Any]] = []
            label_agent = make_rulebase_agent("regular")
            regular_agents = [make_rulebase_agent("regular") for _ in range(players)]
            agents = []
            for pid in range(players):
                if pid == model_seat:

                    def model_logged(obs: dict[str, Any], configuration=None, pid: int = pid) -> list[list]:
                        del configuration
                        model_action = self.model_agent(obs) or []
                        row_obs = copy.deepcopy(obs)
                        regular_label = label_agent(copy.deepcopy(obs)) or []
                        if regular_label or random.random() < keep_noop_prob:
                            owned_planets = _owned_planets(obs, pid)
                            owned_ships = _owned_ships(obs, pid)
                            score_gap = _score_gap(obs, pid, players)
                            raw_rows.append(
                                {
                                    "obs": row_obs,
                                    "player": pid,
                                    "label_action": regular_label,
                                    "turn": int(obs.get("step", len(raw_rows))),
                                    "model_action_count": len(model_action),
                                    "regular_action_count": len(regular_label),
                                    "owned_planets": owned_planets,
                                    "owned_ships": owned_ships,
                                    "score_gap": score_gap,
                                    "raw_index": len(raw_rows),
                                }
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
            metrics["games"] += 1.0
            final_frame = env.steps[-1] if env.steps else []
            final_reward = float(final_frame[model_seat].get("reward", 0.0)) if final_frame else 0.0
            final_status = str(final_frame[model_seat].get("status", "")) if final_frame else ""
            final_obs = final_frame[model_seat].get("observation", {}) if final_frame else {}
            opponent_seat = 1 - model_seat if players == 2 else max((pid for pid in range(players) if pid != model_seat), key=lambda pid: score(final_obs, pid), default=model_seat)
            model_score = float(score(final_obs, model_seat)) if final_obs else 0.0
            opponent_score = float(score(final_obs, opponent_seat)) if final_obs else 0.0
            outcome = _outcome_bucket(model_score, opponent_score)
            game_length = int(getattr(env, "_step", 0))
            buckets["game_outcome"][outcome] += 1
            if keep_outcomes and outcome not in keep_outcomes:
                continue
            if not keep_outcomes and final_reward > max_final_reward:
                continue
            filtered: list[dict[str, Any]] = []
            for raw in raw_rows[:: max(1, sample_stride)]:
                turn = int(raw["turn"])
                if min_turn >= 0 and turn < min_turn:
                    continue
                if max_turn >= 0 and turn > max_turn:
                    continue
                action_gap = int(raw["regular_action_count"]) - int(raw["model_action_count"])
                if min_abs_action_gap >= 0 and abs(action_gap) < min_abs_action_gap:
                    continue
                if min_action_gap >= 0 and action_gap < min_action_gap:
                    continue
                if max_action_gap >= 0 and action_gap > max_action_gap:
                    continue
                if min_owned_planets >= 0 and int(raw["owned_planets"]) < min_owned_planets:
                    continue
                if min_owned_ships >= 0.0 and float(raw["owned_ships"]) < min_owned_ships:
                    continue
                if min_score_gap > -1.0e30 and float(raw["score_gap"]) < min_score_gap:
                    continue
                filtered.append(raw)
            if rows_per_game > 0 and len(filtered) > rows_per_game:
                filtered = random.sample(filtered, rows_per_game)
            if not filtered:
                continue
            metrics["kept_games"] += 1.0
            for sampled_index, raw in enumerate(filtered):
                row = row_from_regular_action(raw["obs"], int(raw["player"]), raw["label_action"], players=players, target_mask_mode=target_mask_mode)
                if row is None:
                    continue
                action_gap = int(raw["regular_action_count"]) - int(raw["model_action_count"])
                row.dagger_checkpoint = self.checkpoint  # type: ignore[attr-defined]
                row.dagger_seed = int(seed)  # type: ignore[attr-defined]
                row.dagger_players = int(players)  # type: ignore[attr-defined]
                row.dagger_model_seat = int(model_seat)  # type: ignore[attr-defined]
                row.dagger_player = int(raw["player"])  # type: ignore[attr-defined]
                row.dagger_raw_index = int(raw["raw_index"])  # type: ignore[attr-defined]
                row.dagger_sampled_index = int(sampled_index)  # type: ignore[attr-defined]
                row.dagger_turn_index = int(raw["turn"])  # type: ignore[attr-defined]
                row.dagger_obs_step = int(raw["turn"])  # type: ignore[attr-defined]
                row.dagger_model_action_count = int(raw["model_action_count"])  # type: ignore[attr-defined]
                row.dagger_regular_label_action_count = int(raw["regular_action_count"])  # type: ignore[attr-defined]
                row.dagger_action_gap = int(action_gap)  # type: ignore[attr-defined]
                row.dagger_owned_planets = int(raw["owned_planets"])  # type: ignore[attr-defined]
                row.dagger_owned_ships = float(raw["owned_ships"])  # type: ignore[attr-defined]
                row.dagger_score_gap = float(raw["score_gap"])  # type: ignore[attr-defined]
                row.dagger_game_length = int(game_length)  # type: ignore[attr-defined]
                row.dagger_model_final_reward = float(final_reward)  # type: ignore[attr-defined]
                row.dagger_model_final_status = final_status  # type: ignore[attr-defined]
                row.dagger_model_score = float(model_score)  # type: ignore[attr-defined]
                row.dagger_opponent_score = float(opponent_score)  # type: ignore[attr-defined]
                row.dagger_model_outcome = outcome  # type: ignore[attr-defined]
                rows.append(row)
                metrics["labelled_actions"] += float(_row_action_count(row))
                metrics["skipped_actions"] += float(getattr(row, "skipped", 0))
                metrics["regular_label_actions"] += float(raw["regular_action_count"])
                metrics["model_actions"] += float(raw["model_action_count"])
                buckets["turn"][_turn_bucket(int(raw["turn"]))] += 1
                buckets["reward"][_reward_bucket(final_reward)] += 1
                buckets.setdefault("outcome", Counter())[outcome] += 1
                buckets.setdefault("owned_planets", Counter())[str(min(40, int(raw["owned_planets"])))] += 1
                buckets.setdefault("score_gap", Counter())[str(int(float(raw["score_gap"]) // 500 * 500))] += 1
                gap_bucket = str(max(-20, min(20, action_gap)))
                buckets["action_gap"][gap_bucket] += 1
            metrics["samples"] = float(len(rows))
        bucket_metrics = {name: {key: float(value) for key, value in counter.items()} for name, counter in buckets.items()}
        return rows, metrics, bucket_metrics


def _build_jobs(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    players = int(args.players)
    if args.eval_aligned_jobs:
        games_per_task = max(1, int(args.eval_games_per_task))
        return [
            (int(args.seed) + game, players, (game % games_per_task) % players)
            for game in range(args.games)
        ]
    if args.seed_base_count > 0:
        jobs: list[tuple[int, int, int]] = []
        for base_idx in range(args.seed_base_count):
            base = int(args.seed_base_start) + base_idx * int(args.seed_base_stride)
            for local_game in range(max(1, int(args.games_per_seed_base))):
                global_game = len(jobs)
                jobs.append((base + local_game, players, global_game % players))
        return jobs
    return [
        (args.seed + 30_000_000 + game, players, game % players)
        for game in range(args.games)
    ]


def collect(args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)
    jobs = _build_jobs(args)
    actor_count = max(1, min(args.actors, len(jobs)))
    shards = [jobs[i::actor_count] for i in range(actor_count)]
    gpu_ids = _int_list(args.gpu_ids_manual)
    actors = [
        FailedRolloutDaggerActor.options(
            num_cpus=args.cpus_per_actor,
            num_gpus=0.0 if gpu_ids else args.gpus_per_actor,
        ).remote(
            args.checkpoint,
            f"cuda:{gpu_ids[i % len(gpu_ids)]}" if gpu_ids else args.device,
            not args.stochastic,
            args.launch_bias,
            args.ship_bias,
            args.launch_temperature,
            args.target_top_k,
            args.model_target_mask_mode,
            args.target_pair_weight,
        )
        for i in range(actor_count)
    ]
    refs = [
        actor.collect_games.remote(
            shard,
            args.episode_steps,
            args.keep_noop_prob,
            args.sample_stride,
            args.rows_per_game,
            args.min_turn,
            args.max_turn,
            args.max_final_reward,
            [item.strip() for item in args.keep_outcomes.split(",") if item.strip()],
            args.min_abs_action_gap,
            args.min_action_gap,
            args.max_action_gap,
            args.min_owned_planets,
            args.min_owned_ships,
            args.min_score_gap,
            not args.no_numba,
            args.row_target_mask_mode,
        )
        for actor, shard in zip(actors, shards, strict=True)
        if shard
    ]
    rows: list[Any] = []
    metrics: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "games": 0.0,
        "kept_games": 0.0,
        "samples": 0.0,
        "labelled_actions": 0.0,
        "skipped_actions": 0.0,
        "regular_label_actions": 0.0,
        "model_actions": 0.0,
        "actors": float(actor_count),
        "players": float(args.players),
        "max_final_reward": float(args.max_final_reward),
        "keep_outcomes": args.keep_outcomes,
        "min_turn": float(args.min_turn),
        "max_turn": float(args.max_turn),
        "min_abs_action_gap": float(args.min_abs_action_gap),
        "min_action_gap": float(args.min_action_gap),
        "max_action_gap": float(args.max_action_gap),
        "min_owned_planets": float(args.min_owned_planets),
        "min_owned_ships": float(args.min_owned_ships),
        "min_score_gap": float(args.min_score_gap),
        "row_target_mask_mode": args.row_target_mask_mode,
        "model_target_mask_mode": args.model_target_mask_mode,
        "target_top_k": float(args.target_top_k),
        "target_pair_weight": float(args.target_pair_weight),
        "eval_aligned_jobs": float(bool(args.eval_aligned_jobs)),
        "eval_games_per_task": float(args.eval_games_per_task),
    }
    bucket_totals: dict[str, Counter] = {"turn": Counter(), "reward": Counter(), "action_gap": Counter()}
    progress = tqdm(total=len(refs), desc="collect failed rollout DAgger", dynamic_ncols=True) if tqdm is not None else None
    pending = list(refs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        for ref in done:
            part_rows, part_metrics, part_buckets = ray.get(ref)
            rows.extend(part_rows)
            for key in [
                "games",
                "kept_games",
                "labelled_actions",
                "skipped_actions",
                "regular_label_actions",
                "model_actions",
            ]:
                metrics[key] = float(metrics.get(key, 0.0)) + float(part_metrics.get(key, 0.0))
            metrics["samples"] = float(len(rows))
            for name, bucket in part_buckets.items():
                bucket_totals.setdefault(name, Counter()).update({key: int(value) for key, value in bucket.items()})
        if progress is not None:
            progress.update(len(done))
            progress.set_postfix(samples=len(rows), kept_games=int(metrics["kept_games"]))
    if progress is not None:
        progress.close()
    for actor in actors:
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass
    metrics["buckets"] = {name: dict(counter) for name, counter in bucket_totals.items()}
    metrics["mean_regular_actions_per_row"] = metrics["regular_label_actions"] / max(1.0, metrics["samples"])
    metrics["mean_model_actions_per_row"] = metrics["model_actions"] / max(1.0, metrics["samples"])
    metrics["mean_action_gap_per_row"] = (metrics["regular_label_actions"] - metrics["model_actions"]) / max(1.0, metrics["samples"])
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--ray-temp-dir", default="", help="Optional Ray temp/session directory, useful when /tmp is low on space.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed-base-start", type=int, default=0, help="If --seed-base-count > 0, collect seeds seed_base_start + i*seed_base_stride + local_game.")
    parser.add_argument("--seed-base-count", type=int, default=0)
    parser.add_argument("--seed-base-stride", type=int, default=100)
    parser.add_argument("--games-per-seed-base", type=int, default=4)
    parser.add_argument("--eval-aligned-jobs", action="store_true", help="Use the same seed/model-seat expansion as eval_policy_sweep_ray for one variant.")
    parser.add_argument("--eval-games-per-task", type=int, default=4, help="games-per-task used by eval_policy_sweep_ray; only affects --eval-aligned-jobs seat assignment.")
    parser.add_argument("--actors", type=int, default=16)
    parser.add_argument("--cpus-per-actor", type=float, default=1.0)
    parser.add_argument("--gpus-per-actor", type=float, default=0.0)
    parser.add_argument("--gpu-ids-manual", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--rows-per-game", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--keep-noop-prob", type=float, default=0.05)
    parser.add_argument(
        "--row-target-mask-mode",
        choices=["candidate", "safe", "all_planets"],
        default="candidate",
        help="Target mask stored in newly collected failed-rollout DAgger rows.",
    )
    parser.add_argument("--max-final-reward", type=float, default=0.0)
    parser.add_argument("--keep-outcomes", default="", help="Comma list from loss,draw,win using the same score comparison as online eval. If set, overrides --max-final-reward filtering.")
    parser.add_argument("--min-turn", type=int, default=-1)
    parser.add_argument("--max-turn", type=int, default=-1)
    parser.add_argument("--min-abs-action-gap", type=int, default=-1)
    parser.add_argument("--min-action-gap", type=int, default=-1, help="Minimum signed gap regular_action_count - model_action_count.")
    parser.add_argument("--max-action-gap", type=int, default=-1, help="Maximum signed gap regular_action_count - model_action_count.")
    parser.add_argument("--min-owned-planets", type=int, default=-1)
    parser.add_argument("--min-owned-ships", type=float, default=-1.0)
    parser.add_argument("--min-score-gap", type=float, default=-1.0e30, help="Minimum score(player)-best_opponent_score for the sampled model-state.")
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--ship-bias", type=float, default=0.0)
    parser.add_argument("--launch-temperature", type=float, default=1.0)
    parser.add_argument("--target-top-k", type=int, default=6)
    parser.add_argument("--model-target-mask-mode", choices=["candidate", "safe", "all_planets"], default="candidate")
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260527)
    parser.add_argument("--no-numba", action="store_true")
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
    if args.ray_temp_dir:
        ray_init_kwargs["_temp_dir"] = args.ray_temp_dir
    ray.init(**ray_init_kwargs)
    rows, metrics = collect(args)
    metrics["cache_path"] = args.out
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"rows": rows, "metrics": metrics, "args": vars(args)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    print(json.dumps({"event": "failed_rollout_dagger_cache_saved", "metrics": metrics}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
