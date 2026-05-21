"""Generate AlphaZeroLike proposal-head data from the strongest regular rulebase."""

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
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import ray
from tqdm import tqdm

from alphaZeroLike.features import encode_global, encode_planets
from alphaZeroLike.proposal import proposal_labels
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent


def _raw_obs(env: Any, player: int) -> dict:
    return env.steps[-1][player]["observation"]


def _has_active_launch(labels: dict) -> bool:
    return any(float(send) > 0.5 and float(valid) > 0.5 for send, valid in zip(labels.get("proposal_send", []), labels.get("proposal_valid", [])))


def _prepare_output(path: Path, output_shards: int) -> list[Path]:
    if output_shards <= 1:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        return [path]
    out_dir = path if path.suffix != ".jsonl" else path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.jsonl"):
        old.unlink()
    return [out_dir / f"part-{i:05d}.jsonl" for i in range(output_shards)]


@ray.remote
class ProposalDataWorker:
    def __init__(
        self,
        worker_id: int,
        oracle: str,
        four_player_prob: float,
        episode_steps: int,
        keep_noop_prob: float,
        use_numba: bool,
    ) -> None:
        self.worker_id = worker_id
        self.oracle = oracle
        self.four_player_prob = four_player_prob
        self.episode_steps = episode_steps
        self.keep_noop_prob = keep_noop_prob
        self.use_numba = use_numba

    def generate(self, episodes: int, seed_offset: int) -> dict:
        rows: list[dict] = []
        active_rows = 0
        skipped_noop = 0
        games_2p = 0
        games_4p = 0
        lengths: list[int] = []
        for ep in range(episodes):
            seed = seed_offset + self.worker_id * 1_000_000 + ep
            rng = random.Random(seed)
            players = 4 if rng.random() < self.four_player_prob else 2
            games_4p += int(players == 4)
            games_2p += int(players == 2)
            env = make_fast_orbit_wars(
                {"episodeSteps": self.episode_steps, "seed": seed},
                keep_history=False,
                use_numba=self.use_numba,
            )
            env.reset(players)
            agents = {pid: make_rulebase_agent(self.oracle) for pid in range(players)}
            steps = 0
            for steps in range(self.episode_steps):
                actions: list[list[list]] = []
                obs_by_player: list[dict] = []
                for pid in range(players):
                    obs = _raw_obs(env, pid)
                    action = agents[pid](obs) or []
                    labels = proposal_labels(obs, pid, action)
                    active = _has_active_launch(labels)
                    active_rows += int(active)
                    if active or rng.random() < self.keep_noop_prob:
                        rows.append(
                            {
                                "player": pid,
                                "planets": encode_planets(obs, pid).tolist(),
                                "global": encode_global(obs, pid).tolist(),
                                **labels,
                                "source": "regular_rulebase_proposal",
                            }
                        )
                    else:
                        skipped_noop += 1
                    obs_by_player.append(obs)
                    actions.append(action)
                env.step(actions)
                if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                    break
            lengths.append(steps + 1)
        return {
            "rows": rows,
            "episodes": episodes,
            "samples": len(rows),
            "active_rows": active_rows,
            "skipped_noop": skipped_noop,
            "games_2p": games_2p,
            "games_4p": games_4p,
            "avg_game_length": float(np.mean(lengths)) if lengths else 0.0,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/alphaZeroLike/proposal_regular_20260521")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--output-shards", type=int, default=128)
    parser.add_argument("--oracle", default="rl_informed_regular")
    parser.add_argument("--four-player-prob", type=float, default=0.3)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--keep-noop-prob", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026052100)
    parser.add_argument("--ray-address")
    parser.add_argument("--ray-temp-dir", default="/tmp/azpdata")
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    shard_paths = _prepare_output(out_path, args.output_shards)
    if args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
    else:
        temp_dir = Path(args.ray_temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=str(temp_dir))

    actors = [
        ProposalDataWorker.options(num_cpus=1, num_gpus=0).remote(
            i,
            args.oracle,
            args.four_player_prob,
            args.episode_steps,
            args.keep_noop_prob,
            not args.no_numba,
        )
        for i in range(args.workers)
    ]
    print(
        json.dumps(
            {
                "stage": "generate_proposal_stage1_data",
                "out": str(out_path),
                "episodes": args.episodes,
                "workers": args.workers,
                "episodes_per_task": args.episodes_per_task,
                "output_shards": len(shard_paths),
                "oracle": args.oracle,
                "four_player_prob": args.four_player_prob,
                "keep_noop_prob": args.keep_noop_prob,
                "use_numba": not args.no_numba,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    remaining = args.episodes
    futures = []
    future_shards = {}
    task_id = 0
    while remaining > 0:
        n = min(args.episodes_per_task, remaining)
        actor = actors[task_id % len(actors)]
        ref = actor.generate.remote(n, args.seed + task_id * 10_000)
        futures.append(ref)
        future_shards[ref] = task_id % len(shard_paths)
        remaining -= n
        task_id += 1

    stats = {"episodes": 0, "samples": 0, "active_rows": 0, "skipped_noop": 0, "games_2p": 0, "games_4p": 0}
    pending = list(futures)
    writers = [path.open("w") for path in shard_paths]
    try:
        progress = tqdm(total=len(futures), desc="[AZProposalData]", unit="task")
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            ref = ready[0]
            chunk = ray.get(ref)
            writer = writers[future_shards[ref]]
            for row in chunk["rows"]:
                writer.write(json.dumps(row, separators=(",", ":")) + "\n")
            writer.flush()
            for key in stats:
                stats[key] += int(chunk.get(key, 0))
            progress.update(1)
            progress.set_postfix(rows=stats["samples"], active=stats["active_rows"])
            print(json.dumps({"event": "chunk", **{f"data/{k}": v for k, v in stats.items()}}, ensure_ascii=False), flush=True)
    finally:
        for writer in writers:
            writer.close()

    print(json.dumps({"event": "done", **{f"data/{k}": v for k, v in stats.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
