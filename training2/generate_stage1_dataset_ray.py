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
from kaggle_environments import make
from tqdm import tqdm

from training2.candidates import build_candidates
from training2.features import encode_position, result_value
from training2.rulebase_bridge import make_rulebase_agent


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        p = PROJECT_ROOT / path
    with p.open() as f:
        return yaml.safe_load(f) or {}


def _raw_obs(env, player: int) -> dict:
    return env.steps[-1][player]["observation"]


@ray.remote
class DatasetWorker:
    def __init__(self, worker_id: int, oracle: str, max_candidates: int, four_player_prob: float) -> None:
        self.worker_id = worker_id
        self.oracle = oracle
        self.max_candidates = max_candidates
        self.four_player_prob = four_player_prob

    def generate(self, episodes: int, seed_offset: int) -> dict:
        rows: list[dict] = []
        games_2p = 0
        games_4p = 0
        lengths: list[int] = []
        for ep in range(episodes):
            seed = seed_offset + self.worker_id * 1_000_000 + ep
            rng = random.Random(seed)
            players = 4 if rng.random() < self.four_player_prob else 2
            games_4p += int(players == 4)
            games_2p += int(players == 2)
            env = make("orbit_wars", configuration={"episodeSteps": 500, "seed": seed}, debug=True)
            env.reset(players)
            agents = {pid: make_rulebase_agent(self.oracle) for pid in range(players)}
            pending: list[dict] = []
            steps = 0
            for steps in range(500):
                actions = []
                for pid in range(players):
                    obs = _raw_obs(env, pid)
                    candidates, oracle_idx = build_candidates(obs, agents[pid], max_candidates=self.max_candidates)
                    enc = encode_position(obs, pid, candidates, max_candidates=self.max_candidates)
                    pending.append(
                        {
                            "player": pid,
                            "planets": enc.planet_features.tolist(),
                            "global": enc.global_features.tolist(),
                            "candidates": enc.candidate_features.tolist(),
                            "candidate_mask": enc.candidate_mask.tolist(),
                            "target": int(oracle_idx),
                        }
                    )
                    actions.append(candidates[oracle_idx] if candidates else [])
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
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training2/config/default.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--out")
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
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ray_temp = Path(ray_cfg.get("temp_dir", "training2/.ray_temp")).resolve()
    ray_temp.mkdir(parents=True, exist_ok=True)
    ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=str(ray_temp))

    oracle = str(stage.get("oracle", "rl_informed_regular"))
    max_candidates = int(model_cfg.get("max_candidates", 32))
    four_player_prob = float(gen_cfg.get("four_player_prob", stage.get("data_four_player_prob", 1.0)))
    actors = [
        DatasetWorker.options(num_gpus=0).remote(i, oracle, max_candidates, four_player_prob)
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
                "out": str(out_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    remaining = episodes
    futures = []
    task_id = 0
    while remaining > 0:
        n = min(episodes_per_task, remaining)
        actor = actors[task_id % workers]
        futures.append(actor.generate.remote(n, seed_offset=task_id * 10_000_000))
        remaining -= n
        task_id += 1

    stats = {"episodes": 0, "samples": 0, "games_2p": 0, "games_4p": 0}
    pending = list(futures)
    with out_path.open("w") as f:
        progress = tqdm(total=len(futures), desc="[GenerateStage1]", unit="task")
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            chunk = ray.get(ready[0])
            for row in chunk["rows"]:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush()
            stats["episodes"] += chunk["episodes"]
            stats["samples"] += chunk["samples"]
            stats["games_2p"] += chunk["games_2p"]
            stats["games_4p"] += chunk["games_4p"]
            print(json.dumps({**stats, "out": str(out_path)}, ensure_ascii=False), flush=True)
            progress.update(1)
            progress.set_postfix(
                episodes=stats["episodes"],
                samples=stats["samples"],
                size_mb=f"{out_path.stat().st_size / 1024 / 1024:.1f}",
            )
        progress.close()

    ray.shutdown()
    print(json.dumps({**stats, "out": str(out_path), "done": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
