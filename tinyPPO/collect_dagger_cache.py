from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import ray

from tinyPPO.imitation_regular_ray import DaggerCollectActor, _int_list, _players_list

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _checkpoints(args: argparse.Namespace) -> list[str]:
    paths = [item.strip() for item in args.dagger_checkpoints.split(",") if item.strip()]
    if args.dagger_checkpoint:
        paths.insert(0, args.dagger_checkpoint)
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("at least one checkpoint is required")
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing DAgger checkpoints: {missing}")
    return paths


def collect(args: argparse.Namespace) -> tuple[list[Any], dict[str, float]]:
    checkpoints = _checkpoints(args)
    players_values = _players_list(args.players_list)
    jobs = [
        (args.seed + 20_000_000 + players * 1_000_000 + game, players, game % players)
        for players in players_values
        for game in range(args.dagger_games_per_players)
    ]
    if not jobs:
        return [], {"games": 0.0, "samples": 0.0}

    actor_count = max(1, min(args.dagger_actors, len(jobs)))
    shards = [jobs[i::actor_count] for i in range(actor_count)]
    gpu_ids = _int_list(args.dagger_gpu_ids_manual)
    actors = [
        DaggerCollectActor.options(
            num_cpus=args.dagger_cpus_per_actor,
            num_gpus=0.0 if gpu_ids else args.dagger_gpus_per_actor,
        ).remote(
            checkpoints[i % len(checkpoints)],
            f"cuda:{gpu_ids[i % len(gpu_ids)]}" if gpu_ids else args.dagger_device,
            not args.dagger_stochastic,
            args.launch_bias,
            args.ship_bias,
            args.launch_temperature,
            args.dagger_model_seat_only,
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
            not args.no_numba,
        )
        for actor, shard in zip(actors, shards, strict=True)
        if shard
    ]

    rows: list[Any] = []
    metrics = {
        "games": 0.0,
        "samples": 0.0,
        "labelled_actions": 0.0,
        "skipped_actions": 0.0,
        "regular_label_actions": 0.0,
        "model_seat_label_actions": 0.0,
        "actors": float(actor_count),
        "checkpoints": float(len(checkpoints)),
        "model_seat_only": float(bool(args.dagger_model_seat_only)),
    }
    progress = tqdm(total=len(refs), desc="collect DAgger cache", dynamic_ncols=True) if tqdm is not None else None
    pending = list(refs)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        for ref in done:
            part_rows, part_metrics = ray.get(ref)
            rows.extend(part_rows)
            for key in [
                "games",
                "labelled_actions",
                "skipped_actions",
                "regular_label_actions",
                "model_seat_label_actions",
            ]:
                metrics[key] += float(part_metrics.get(key, 0.0))
            metrics["samples"] = float(len(rows))
        if progress is not None:
            progress.update(len(done))
            progress.set_postfix(samples=len(rows), games=int(metrics["games"]))
    if progress is not None:
        progress.close()

    for actor in actors:
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--out", required=True)
    parser.add_argument("--players-list", default="2")
    parser.add_argument("--dagger-checkpoint", default="")
    parser.add_argument("--dagger-checkpoints", default="")
    parser.add_argument("--dagger-games-per-players", type=int, default=0)
    parser.add_argument("--dagger-actors", type=int, default=8)
    parser.add_argument("--dagger-cpus-per-actor", type=float, default=1.0)
    parser.add_argument("--dagger-gpus-per-actor", type=float, default=0.0)
    parser.add_argument("--dagger-gpu-ids-manual", default="")
    parser.add_argument("--dagger-device", default="cpu")
    parser.add_argument("--dagger-stochastic", action="store_true")
    parser.add_argument("--dagger-model-seat-only", action="store_true")
    parser.add_argument("--rows-per-game", type=int, default=16)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--keep-noop-prob", type=float, default=0.15)
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--ship-bias", type=float, default=0.0)
    parser.add_argument("--launch-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260526)
    parser.add_argument("--no-numba", action="store_true")
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
    rows, metrics = collect(args)
    metrics["cache_path"] = args.out
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"rows": rows, "metrics": metrics, "args": vars(args)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    print(json.dumps({"event": "dagger_cache_saved", "metrics": metrics}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
