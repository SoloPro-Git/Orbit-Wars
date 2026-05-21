"""Ray training loop for the AlphaZero-like Orbit Wars prototype."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KAGGLE_ENGINES_LOG_LEVEL"] = "0"
os.environ["SWANLAB_NO_INTERACTIVE"] = "1"
os.environ["SWANLAB_DISABLE_INTERACTIVE"] = "1"
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import ray
import torch
import yaml
from tqdm import tqdm

from alphaZeroLike.candidates import CandidateConfig
from alphaZeroLike.compat import load_training2_stage1_backbone
from alphaZeroLike.mcts import MCTSConfig
from alphaZeroLike.model import AlphaZeroLikeNet
from alphaZeroLike.proposal import ProposalConfig
from alphaZeroLike.self_play import generate_game
from alphaZeroLike.train import train_batch


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open() as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _resolve(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


def _make_model(model_cfg: dict[str, Any], device: str) -> AlphaZeroLikeNet:
    return AlphaZeroLikeNet(
        d_model=int(model_cfg.get("d_model", 192)),
        nhead=int(model_cfg.get("nhead", 6)),
        layers=int(model_cfg.get("layers", 4)),
        dropout=float(model_cfg.get("dropout", 0.10)),
    ).to(device)


def _dataclass_kwargs(cls, cfg: dict[str, Any]) -> dict[str, Any]:
    fields = set(getattr(cls, "__dataclass_fields__", {}).keys())
    out = {key: value for key, value in (cfg or {}).items() if key in fields}
    if "ship_ratio_choices" in out and isinstance(out["ship_ratio_choices"], list):
        out["ship_ratio_choices"] = tuple(float(x) for x in out["ship_ratio_choices"])
    return out


def _resolve_worker_count(value: Any, resources: dict[str, float], ray_cfg: dict[str, Any], kind: str) -> int:
    if isinstance(value, (int, float)):
        return max(1, int(value))
    raw = str(value).strip().lower()
    if raw in {"auto_gpu_x3", "auto-gpu-x3", "auto_gpu_workers"}:
        workers_per_gpu = int(ray_cfg.get("max_gpu_workers_per_gpu", 3))
        return max(1, int(float(resources.get("GPU", 0.0)) * workers_per_gpu))
    if raw in {"auto_rollout_cpu", "auto-rollout-cpu"}:
        resource_name = str(ray_cfg.get("rollout_resource") or "")
        resource = float(resources.get(resource_name, 0.0)) if resource_name else 0.0
        if resource <= 0.0:
            resource = float(resources.get("CPU", 1.0))
        per_worker = float(ray_cfg.get("rollout_resource_per_worker", ray_cfg.get("cpus_per_rollout", 1.0)))
        return max(1, int(resource // max(per_worker, 1e-9)))
    if raw == "auto_cpu":
        per_worker = float(ray_cfg.get(f"cpus_per_{kind}", 1.0))
        return max(1, int(float(resources.get("CPU", 1.0)) // max(per_worker, 1e-9)))
    return max(1, int(float(raw)))


def _init_swanlab(config: dict[str, Any], args) -> Any | None:
    if args.no_swanlab:
        return None
    key = os.environ.get("SWANLAB_API_KEY")
    key_path = PROJECT_ROOT / "training/config/swanlab_key.txt"
    if not key and key_path.exists():
        key = key_path.read_text().strip()
    if key:
        os.environ["SWANLAB_API_KEY"] = key
    import swanlab

    swan_cfg = config.get("swanlab", {})
    return swanlab.init(
        project=str(swan_cfg.get("project", "orbit-wars")),
        experiment_name=str(swan_cfg.get("experiment", "alphaZeroLike")),
        mode=str(swan_cfg.get("mode", "cloud")),
        config={"alphaZeroLike": config, "args": vars(args)},
    )


@ray.remote
class AZRolloutActor:
    def __init__(
        self,
        worker_id: int,
        model_cfg: dict[str, Any],
        env_cfg: dict[str, Any],
        mcts_cfg: dict[str, Any],
        candidate_cfg: dict[str, Any],
        proposal_cfg: dict[str, Any],
        opponent_cfg: dict[str, Any],
        device: str,
        torch_threads: int | None = None,
    ) -> None:
        if torch_threads is not None and torch_threads > 0:
            torch.set_num_threads(int(torch_threads))
            torch.set_num_interop_threads(int(torch_threads))
        self.worker_id = worker_id
        self.device = device
        self.model = _make_model(model_cfg, device)
        self.env_cfg = env_cfg
        self.mcts_cfg = MCTSConfig(**_dataclass_kwargs(MCTSConfig, mcts_cfg))
        self.candidate_cfg = CandidateConfig(**_dataclass_kwargs(CandidateConfig, candidate_cfg))
        self.proposal_cfg = ProposalConfig(**_dataclass_kwargs(ProposalConfig, proposal_cfg))
        self.opponent_cfg = opponent_cfg or {}

    def rollout(self, state_dict: dict[str, torch.Tensor], games: int, seed_offset: int) -> dict[str, Any]:
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        rows: list[dict] = []
        lengths: list[int] = []
        values: list[float] = []
        t0 = time.time()
        for game in range(games):
            game_rows = generate_game(
                self.model,
                seed=seed_offset + self.worker_id * 1_000_000 + game,
                players=int(self.env_cfg.get("players", 2)),
                episode_steps=int(self.env_cfg.get("episode_steps", 500)),
                use_numba=bool(self.env_cfg.get("use_numba", True)),
                opponent_oracle=str(self.env_cfg.get("opponent_oracle", "rl_informed_regular")),
                device=self.device,
                mcts_cfg=self.mcts_cfg,
                candidate_cfg=self.candidate_cfg,
                proposal_cfg=self.proposal_cfg,
                opponent_checkpoints=[
                    _resolve(str(path)) or str(path)
                    for path in self.opponent_cfg.get("checkpoints", [])
                ],
                opponent_rulebase_weight=float(self.opponent_cfg.get("rulebase_weight", 1.0)),
                opponent_checkpoint_weight=float(self.opponent_cfg.get("checkpoint_weight", 0.0)),
                opponent_device=str(self.opponent_cfg.get("device", "cpu")),
            )
            rows.extend(game_rows)
            lengths.append(len(game_rows))
            if game_rows:
                values.append(float(game_rows[-1].get("value", 0.0)))
        return {
            "rows": rows,
            "games": games,
            "samples": len(rows),
            "avg_game_rows": float(np.mean(lengths)) if lengths else 0.0,
            "avg_value": float(np.mean(values)) if values else 0.0,
            "sec": time.time() - t0,
        }


@ray.remote
class AZTrainerActor:
    def __init__(
        self,
        model_cfg: dict[str, Any],
        train_cfg: dict[str, Any],
        device: str,
        resume: str | None,
        init_from_training2: str | None,
    ) -> None:
        self.device = device
        self.model = _make_model(model_cfg, device)
        if resume and Path(resume).exists():
            ckpt = torch.load(resume, map_location=device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.init_report = {"resume": resume}
        elif init_from_training2:
            self.init_report = load_training2_stage1_backbone(self.model, init_from_training2, map_location=device)
        else:
            self.init_report = {"scratch": True}
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 2e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        self.replay: list[dict] = []

    def add_rows(self, rows: list[dict], replay_capacity: int) -> dict[str, float]:
        self.replay.extend(rows)
        self.replay = self.replay[-int(replay_capacity):]
        return {"train/replay_size": float(len(self.replay))}

    def update(
        self,
        updates: int,
        batch_size: int,
        value_weight: float,
        entropy_weight: float,
    ) -> dict[str, float]:
        if not self.replay:
            return {"train/replay_size": 0.0}
        self.model.train()
        accum: dict[str, float] = {}
        for _ in range(int(updates)):
            batch = random.sample(self.replay, k=min(int(batch_size), len(self.replay)))
            metrics = train_batch(
                self.model,
                self.opt,
                batch,
                device=self.device,
                value_weight=value_weight,
                entropy_weight=entropy_weight,
            )
            for key, value in metrics.items():
                accum[f"train/{key}"] = accum.get(f"train/{key}", 0.0) + float(value)
        out = {key: value / max(int(updates), 1) for key, value in accum.items()}
        out["train/replay_size"] = float(len(self.replay))
        out["train/lr"] = float(self.opt.param_groups[0]["lr"])
        if self.device.startswith("cuda"):
            out["gpu/memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            out["gpu/max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        return out

    def state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.model.state_dict().items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict, strict=False)

    def init_info(self) -> dict[str, Any]:
        return self.init_report

    def save(self, path: str, iteration: int, config: dict[str, Any]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "iteration": iteration,
                "config": config,
            },
            path,
        )


def _average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(states) == 1:
        return states[0]
    avg: dict[str, torch.Tensor] = {}
    for key in states[0]:
        value = states[0][key]
        if torch.is_floating_point(value):
            avg[key] = torch.stack([state[key].float() for state in states], dim=0).mean(dim=0).to(value.dtype)
        else:
            avg[key] = value
    return avg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="alphaZeroLike/config/default.yaml")
    parser.add_argument("--stage", help="Override run.stage from YAML.")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--allow-no-swanlab", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    stage_name = args.stage or cfg.get("run", {}).get("stage", "mixed")
    stage_override = cfg.get("stages", {}).get(stage_name, {})
    cfg = _deep_update(cfg, stage_override)
    cfg.setdefault("run", {})["stage"] = stage_name

    train_cfg = cfg.get("training", {})
    ray_cfg = cfg.get("ray", {})
    model_cfg = cfg.get("model", {})
    output_dir = Path(train_cfg.get("output_dir", "alphaZeroLike/checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = _resolve(str(train_cfg.get("resume_from") or ""))
    init_from_training2 = _resolve(str(train_cfg.get("init_from_training2") or ""))
    if resume and not Path(resume).exists():
        resume = None

    swan = None
    try:
        swan = _init_swanlab(cfg, args)
    except Exception as exc:
        if args.allow_no_swanlab:
            print(f"[SwanLab] init failed, continuing: {exc}", flush=True)
        else:
            raise

    runtime_env: dict[str, Any] = {}
    runtime_env_vars = {str(k): str(v) for k, v in dict(ray_cfg.get("runtime_env_vars", {}) or {}).items()}
    if runtime_env_vars:
        runtime_env["env_vars"] = runtime_env_vars
    ray_address = ray_cfg.get("address")
    if ray_address:
        ray.init(address=str(ray_address), ignore_reinit_error=True, runtime_env=runtime_env or None)
    else:
        ray_temp = Path(ray_cfg.get("temp_dir", "alphaZeroLike/.ray_temp")).resolve()
        ray_temp.mkdir(parents=True, exist_ok=True)
        ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=str(ray_temp), runtime_env=runtime_env or None)

    resources = ray.cluster_resources()
    trainer_workers = _resolve_worker_count(ray_cfg.get("num_trainer_workers", 1), resources, ray_cfg, "trainer")
    rollout_workers = _resolve_worker_count(ray_cfg.get("num_rollout_workers", 1), resources, ray_cfg, "rollout")
    rollout_options: dict[str, Any] = {
        "num_cpus": float(ray_cfg.get("cpus_per_rollout", 1.0)),
        "num_gpus": float(ray_cfg.get("gpus_per_rollout", 0.0)),
    }
    if ray_cfg.get("rollout_resource"):
        rollout_options["resources"] = {
            str(ray_cfg["rollout_resource"]): float(
                ray_cfg.get("rollout_resource_per_worker", ray_cfg.get("cpus_per_rollout", 1.0))
            )
        }

    trainers = [
        AZTrainerActor.options(
            num_cpus=float(ray_cfg.get("cpus_per_trainer", 0.5)),
            num_gpus=float(ray_cfg.get("gpus_per_trainer", 0.0)),
        ).remote(
            model_cfg,
            train_cfg,
            str(train_cfg.get("device", "cpu")),
            resume,
            init_from_training2,
        )
        for _ in range(trainer_workers)
    ]
    rollouts = [
        AZRolloutActor.options(**rollout_options).remote(
            i,
            model_cfg,
            cfg.get("env", {}),
            cfg.get("mcts", {}),
            cfg.get("candidate", {}),
            cfg.get("proposal", {}),
            cfg.get("opponents", {}),
            str(train_cfg.get("rollout_device", "cpu")),
        )
        for i in range(rollout_workers)
    ]
    init_infos = ray.get([trainer.init_info.remote() for trainer in trainers])
    print(
        json.dumps(
            {
                "stage": stage_name,
                "resume": resume,
                "init_from_training2": init_from_training2,
                "train_workers": trainer_workers,
                "rollout_workers": rollout_workers,
                "ray_resources": resources,
                "init": init_infos[0],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    iterations = int(train_cfg.get("max_iterations", 100))
    games_per_iteration = int(train_cfg.get("games_per_iteration", 64))
    games_per_task = int(train_cfg.get("games_per_rollout_task", 1))
    replay_capacity = int(train_cfg.get("replay_capacity", 200000))
    updates = int(train_cfg.get("updates_per_iteration", 50))
    batch_size = int(train_cfg.get("batch_size", 128))
    value_weight = float(train_cfg.get("value_loss_weight", 1.0))
    entropy_weight = float(train_cfg.get("entropy_weight", 0.01))
    seed_base = int(train_cfg.get("seed", 70_000_000))
    save_interval = max(1, int(train_cfg.get("save_interval", 1)))

    progress = tqdm(range(iterations), desc=f"[AZLike:{stage_name}]", unit="iter")
    for iteration in progress:
        t0 = time.time()
        states = ray.get([trainer.state_dict_cpu.remote() for trainer in trainers])
        model_state = _average_state_dicts(states)

        remaining = games_per_iteration
        parts: list[dict[str, Any]] = []
        in_flight: dict[Any, int] = {}
        task_id = 0

        def submit(actor_idx: int) -> None:
            nonlocal remaining, task_id
            games = min(games_per_task, remaining)
            if games <= 0:
                return
            remaining -= games
            task_id += 1
            ref = rollouts[actor_idx].rollout.remote(
                model_state,
                games,
                seed_base + iteration * 1_000_000 + task_id * 10_000,
            )
            in_flight[ref] = actor_idx

        for actor_idx in range(min(rollout_workers, games_per_iteration)):
            submit(actor_idx)
        rollout_t0 = time.time()
        while in_flight:
            done, _ = ray.wait(list(in_flight), num_returns=1)
            for ref in done:
                actor_idx = in_flight.pop(ref)
                parts.append(ray.get(ref))
                submit(actor_idx)
        rollout_sec = time.time() - rollout_t0

        rows = [row for part in parts for row in part["rows"]]
        split = [rows[i::trainer_workers] for i in range(trainer_workers)]
        ray.get([trainer.add_rows.remote(split[i], replay_capacity) for i, trainer in enumerate(trainers)])
        train_t0 = time.time()
        train_parts = ray.get(
            [trainer.update.remote(updates, batch_size, value_weight, entropy_weight) for trainer in trainers]
        )
        train_sec = time.time() - train_t0
        states = ray.get([trainer.state_dict_cpu.remote() for trainer in trainers])
        model_state = _average_state_dicts(states)
        ray.get([trainer.load_state_dict.remote(model_state) for trainer in trainers])

        log: dict[str, float | int | str] = {
            "iteration": iteration,
            "stage": stage_name,
            "time/iteration_sec": time.time() - t0,
            "time/rollout_sec": rollout_sec,
            "time/train_sec": train_sec,
            "rollout/games": sum(int(p["games"]) for p in parts),
            "rollout/samples": len(rows),
            "rollout/avg_game_rows": float(np.mean([p["avg_game_rows"] for p in parts])) if parts else 0.0,
            "rollout/avg_value": float(np.mean([p["avg_value"] for p in parts])) if parts else 0.0,
            "rollout/workers": rollout_workers,
            "train/workers": trainer_workers,
        }
        for part in train_parts:
            for key, value in part.items():
                log[key] = float(log.get(key, 0.0)) + float(value) / max(len(train_parts), 1)

        print(json.dumps(log, ensure_ascii=False), flush=True)
        if swan is not None:
            swan.log(log, step=iteration)
        progress.set_postfix(loss=log.get("train/loss", 0.0), rows=log["rollout/samples"])

        if iteration % save_interval == 0 or iteration == iterations - 1:
            ray.get([trainers[0].save.remote(str(output_dir / "latest.pt"), iteration, cfg)])

    ray.get([trainers[0].save.remote(str(output_dir / "latest.pt"), iterations - 1, cfg)])


if __name__ == "__main__":
    main()
