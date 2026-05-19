"""Ray offline dataset generation for stage1 regular-rulebase alignment."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KAGGLE_ENGINES_LOG_LEVEL"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import ray
import yaml
from tqdm import tqdm

from training2.candidates import build_candidates, shuffle_candidates
from training2.envs import make_orbit_wars_env
from training2.features import encode_position, result_value
from training2.proposal import proposal_labels
from training2.rulebase_bridge import make_rulebase_agent


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        p = PROJECT_ROOT / path
    with p.open() as f:
        return yaml.safe_load(f) or {}


def _raw_obs(env, player: int) -> dict:
    return env.steps[-1][player]["observation"]


def _prepare_output(path: Path, output_shards: int) -> list[Path]:
    if output_shards <= 1:
        path.parent.mkdir(parents=True, exist_ok=True)
        return [path]
    out_dir = path if path.suffix != ".jsonl" else path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_shard in out_dir.glob("part-*.jsonl"):
        old_shard.unlink()
    return [out_dir / f"part-{i:05d}.jsonl" for i in range(output_shards)]


@ray.remote
class DatasetWorker:
    def __init__(
        self,
        worker_id: int,
        oracle: str,
        max_candidates: int,
        four_player_prob: float,
        shuffle_training_candidates: bool,
        env_backend: str,
        env_use_numba: bool,
    ) -> None:
        self.worker_id = worker_id
        self.oracle = oracle
        self.max_candidates = max_candidates
        self.four_player_prob = four_player_prob
        self.shuffle_training_candidates = shuffle_training_candidates
        self.env_backend = env_backend
        self.env_use_numba = env_use_numba

    def generate(self, episodes: int, seed_offset: int) -> dict:
        rows: list[dict] = []
        games_2p = 0
        games_4p = 0
        lengths: list[int] = []
        target_counts: dict[int, int] = {}
        candidate_counts: dict[int, int] = {}
        proposal_label_rows = 0
        for ep in range(episodes):
            seed = seed_offset + self.worker_id * 1_000_000 + ep
            rng = random.Random(seed)
            players = 4 if rng.random() < self.four_player_prob else 2
            games_4p += int(players == 4)
            games_2p += int(players == 2)
            env = make_orbit_wars_env(
                {"episodeSteps": 500, "seed": seed},
                backend=self.env_backend,
                debug=True,
                use_numba=self.env_use_numba,
            )
            env.reset(players)
            agents = {pid: make_rulebase_agent(self.oracle) for pid in range(players)}
            pending: list[dict] = []
            steps = 0
            for steps in range(500):
                actions = []
                for pid in range(players):
                    obs = _raw_obs(env, pid)
                    candidates, oracle_idx = build_candidates(obs, agents[pid], max_candidates=self.max_candidates)
                    oracle_action = candidates[oracle_idx] if candidates else []
                    if self.shuffle_training_candidates:
                        candidates, oracle_idx = shuffle_candidates(candidates, oracle_idx, rng)
                    enc = encode_position(obs, pid, candidates, max_candidates=self.max_candidates)
                    target_counts[int(oracle_idx)] = target_counts.get(int(oracle_idx), 0) + 1
                    candidate_counts[len(candidates)] = candidate_counts.get(len(candidates), 0) + 1
                    labels = proposal_labels(obs, pid, oracle_action)
                    proposal_label_rows += int(bool(labels.get("proposal_valid")))
                    pending.append(
                        {
                            "player": pid,
                            "planets": enc.planet_features.tolist(),
                            "global": enc.global_features.tolist(),
                            "candidates": enc.candidate_features.tolist(),
                            "candidate_mask": enc.candidate_mask.tolist(),
                            "target": int(oracle_idx),
                            **labels,
                        }
                    )
                    actions.append(oracle_action)
                env.step(actions)
                if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                    break
            finals = [_raw_obs(env, pid) for pid in range(players)]
            for row in pending:
                row["value"] = result_value(finals[row["player"]], row["player"])
            rows.extend(pending)
            lengths.append(steps + 1)
        return {
            "rows": rows,
            "episodes": episodes,
            "games_2p": games_2p,
            "games_4p": games_4p,
            "samples": len(rows),
            "avg_game_length": float(np.mean(lengths)) if lengths else 0.0,
            "target_counts": target_counts,
            "candidate_counts": candidate_counts,
            "proposal_label_rows": proposal_label_rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training2/config/default.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--out")
    parser.add_argument("--output-shards", type=int)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    model_cfg = cfg.get("model", {})
    gen_cfg = cfg.get("offline_data", {})
    stage = cfg.get("stage1", {})
    ray_cfg = cfg.get("ray", {})

    episodes = args.episodes or int(gen_cfg.get("episodes", 10000))
    workers = args.workers or int(gen_cfg.get("num_workers", ray_cfg.get("num_data_workers", 8)))
    episodes_per_task = args.episodes_per_task or int(gen_cfg.get("episodes_per_task", 4))
    out_path = Path(args.out or gen_cfg.get("output_path", "data/training2/stage1_regular.jsonl"))
    output_shards = args.output_shards or int(gen_cfg.get("output_shards", 1))
    shard_paths = _prepare_output(out_path, output_shards)

    ray_temp = Path(ray_cfg.get("temp_dir", "training2/.ray_temp")).resolve()
    ray_temp.mkdir(parents=True, exist_ok=True)
    ray_address = ray_cfg.get("address")
    if ray_address:
        ray.init(address=str(ray_address), ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=str(ray_temp))

    oracle = str(stage.get("oracle", "rl_informed_regular"))
    max_candidates = int(model_cfg.get("max_candidates", 32))
    four_player_prob = float(gen_cfg.get("four_player_prob", stage.get("data_four_player_prob", 1.0)))
    shuffle_training_candidates = bool(gen_cfg.get("shuffle_candidates", True))
    env_cfg = cfg.get("env", {})
    env_backend = str(stage.get("env_backend", env_cfg.get("backend", "kaggle")))
    env_use_numba = bool(stage.get("env_use_numba", env_cfg.get("use_numba", False)))
    actors = [
        DatasetWorker.options(num_gpus=0).remote(
            i,
            oracle,
            max_candidates,
            four_player_prob,
            shuffle_training_candidates,
            env_backend,
            env_use_numba,
        )
        for i in range(workers)
    ]
    print(
        json.dumps(
            {
                "stage": "generate_stage1_dataset",
                "episodes": episodes,
                "workers": workers,
                "episodes_per_task": episodes_per_task,
                "four_player_prob": four_player_prob,
                "shuffle_candidates": shuffle_training_candidates,
                "env_backend": env_backend,
                "env_use_numba": env_use_numba,
                "output_shards": len(shard_paths),
                "out": str(out_path),
                "shard_example": str(shard_paths[0]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    remaining = episodes
    futures = []
    future_shards = {}
    task_id = 0
    while remaining > 0:
        n = min(episodes_per_task, remaining)
        actor = actors[task_id % workers]
        ref = actor.generate.remote(n, seed_offset=task_id * 10_000_000)
        futures.append(ref)
        future_shards[ref] = task_id % len(shard_paths)
        remaining -= n
        task_id += 1

    stats = {
        "episodes": 0,
        "samples": 0,
        "games_2p": 0,
        "games_4p": 0,
        "proposal_label_rows": 0,
    }
    target_counts: dict[int, int] = {}
    candidate_counts: dict[int, int] = {}
    pending = list(futures)
    writers = [path.open("w") for path in shard_paths]
    try:
        progress = tqdm(total=len(futures), desc="[GenerateStage1]", unit="task")
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            ref = ready[0]
            chunk = ray.get(ref)
            f = writers[future_shards[ref]]
            for row in chunk["rows"]:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush()
            stats["episodes"] += chunk["episodes"]
            stats["samples"] += chunk["samples"]
            stats["games_2p"] += chunk["games_2p"]
            stats["games_4p"] += chunk["games_4p"]
            stats["proposal_label_rows"] += chunk.get("proposal_label_rows", 0)
            for key, value in chunk.get("target_counts", {}).items():
                target_counts[int(key)] = target_counts.get(int(key), 0) + int(value)
            for key, value in chunk.get("candidate_counts", {}).items():
                candidate_counts[int(key)] = candidate_counts.get(int(key), 0) + int(value)
            print(
                json.dumps(
                    {
                        **stats,
                        "target_top": sorted(target_counts.items(), key=lambda item: -item[1])[:8],
                        "candidate_count_top": sorted(candidate_counts.items(), key=lambda item: -item[1])[:8],
                        "out": str(out_path),
                        "output_shards": len(shard_paths),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            progress.update(1)
            progress.set_postfix(
                episodes=stats["episodes"],
                samples=stats["samples"],
                size_mb=f"{sum(path.stat().st_size for path in shard_paths if path.exists()) / 1024 / 1024:.1f}",
            )
        progress.close()
    finally:
        for writer in writers:
            writer.close()

    ray.shutdown()
    print(
        json.dumps(
            {
                **stats,
                "target_counts": dict(sorted(target_counts.items())),
                "candidate_counts": dict(sorted(candidate_counts.items())),
                "out": str(out_path),
                "shards": [str(path) for path in shard_paths],
                "done": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
