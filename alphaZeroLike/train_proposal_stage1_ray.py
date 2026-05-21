"""Ray proposal-head pretraining from AlphaZeroLike proposal JSONL data."""

from __future__ import annotations

import argparse
import json
import os
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

import ray
import torch

from alphaZeroLike.compat import load_training2_stage1_backbone
from alphaZeroLike.model import AlphaZeroLikeNet
from alphaZeroLike.train_proposal_stage1 import (
    _freeze_except_proposal,
    _iter_jsonl_paths,
    _iter_rows,
    _next_balanced_batch,
    proposal_pretrain_batch,
)


def _resolve(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


def _make_model(device: str) -> AlphaZeroLikeNet:
    return AlphaZeroLikeNet().to(device)


def _init_swanlab(args: argparse.Namespace) -> Any | None:
    if args.no_swanlab:
        return None
    key = os.environ.get("SWANLAB_API_KEY")
    key_path = PROJECT_ROOT / "training/config/swanlab_key.txt"
    if not key and key_path.exists():
        key = key_path.read_text().strip()
    if key:
        os.environ["SWANLAB_API_KEY"] = key

    import swanlab

    return swanlab.init(
        project=str(args.swanlab_project),
        experiment_name=str(args.swanlab_experiment),
        mode=str(args.swanlab_mode),
        config={"alphaZeroLike_proposal_ray": vars(args)},
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


@ray.remote
class ProposalTrainerActor:
    def __init__(
        self,
        actor_id: int,
        data_path: str,
        device: str,
        resume: str | None,
        init_from_training2: str | None,
        lr: float,
        weight_decay: float,
        seed: int,
        train_backbone: bool,
    ) -> None:
        self.actor_id = actor_id
        self.device = device
        self.model = _make_model(device)
        if resume and Path(resume).exists():
            ckpt = torch.load(resume, map_location=device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.init_info = {"resume": resume}
        elif init_from_training2:
            report = load_training2_stage1_backbone(self.model, init_from_training2, map_location=device)
            self.init_info = {"init_from_training2": init_from_training2, "loaded_tensors": len(report["loaded"])}
        else:
            self.init_info = {"scratch": True}
        if not train_backbone:
            _freeze_except_proposal(self.model)
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        paths = _iter_jsonl_paths(data_path)
        if not paths:
            raise FileNotFoundError(data_path)
        self.row_iter = _iter_rows(paths, shuffle_files=True, seed=seed + actor_id * 1009)

    def train_steps(self, steps: int, batch_size: int, active_row_frac: float) -> dict[str, float]:
        self.model.train()
        accum: dict[str, float] = {}
        t0 = time.time()
        for _ in range(int(steps)):
            rows = _next_balanced_batch(self.row_iter, int(batch_size), active_row_frac=active_row_frac)
            metrics = proposal_pretrain_batch(self.model, self.opt, rows, device=self.device)
            for key, value in metrics.items():
                accum[f"proposal/{key}"] = accum.get(f"proposal/{key}", 0.0) + float(value)
        out = {key: value / max(int(steps), 1) for key, value in accum.items()}
        out["proposal/sec"] = time.time() - t0
        out["proposal/lr"] = float(self.opt.param_groups[0]["lr"])
        if self.device.startswith("cuda"):
            out["gpu/memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            out["gpu/max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        return out

    def state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.model.state_dict().items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict, strict=False)

    def init_report(self) -> dict[str, Any]:
        return self.init_info

    def save(self, path: str, steps: int, args: dict[str, Any]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict(), "proposal_steps": steps, "args": args}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/alphaZeroLike/proposal_regular_20260521")
    parser.add_argument("--out", default="alphaZeroLike/checkpoints/latest.pt")
    parser.add_argument("--resume", default="alphaZeroLike/checkpoints/latest.pt")
    parser.add_argument("--init-from-training2", default="training2/checkpoints/stage1_tactical_entities_20260519/latest.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--gpus-per-worker", type=float, default=1.0 / 3.0)
    parser.add_argument("--cpus-per-worker", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--sync-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--active-row-frac", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--ray-address")
    parser.add_argument("--ray-temp-dir", default="/tmp/azpray")
    parser.add_argument("--swanlab-project", default="orbit-wars")
    parser.add_argument("--swanlab-experiment", default="alphaZeroLike-proposal-ray")
    parser.add_argument("--swanlab-mode", default="cloud")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--allow-no-swanlab", action="store_true")
    args = parser.parse_args()

    resume = _resolve(args.resume)
    init_from_training2 = _resolve(args.init_from_training2)
    if resume and not Path(resume).exists():
        resume = None
    data_path = _resolve(args.data)
    out_path = _resolve(args.out)

    swan = None
    try:
        swan = _init_swanlab(args)
    except Exception as exc:
        if args.allow_no_swanlab:
            print(f"[SwanLab] init failed, continuing: {exc}", flush=True)
        else:
            raise

    if args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
    else:
        temp_dir = Path(args.ray_temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=str(temp_dir))

    actors = [
        ProposalTrainerActor.options(num_cpus=args.cpus_per_worker, num_gpus=args.gpus_per_worker).remote(
            i,
            data_path,
            args.device,
            resume,
            init_from_training2,
            args.lr,
            args.weight_decay,
            args.seed,
            args.train_backbone,
        )
        for i in range(args.workers)
    ]
    init = ray.get(actors[0].init_report.remote())
    print(json.dumps({"stage": "proposal_pretrain_ray", "workers": args.workers, "init": init}, ensure_ascii=False), flush=True)

    completed = 0
    while completed < args.steps:
        chunk = min(args.sync_interval, args.steps - completed)
        parts = ray.get([actor.train_steps.remote(chunk, args.batch_size, args.active_row_frac) for actor in actors])
        states = ray.get([actor.state_dict_cpu.remote() for actor in actors])
        avg_state = _average_state_dicts(states)
        ray.get([actor.load_state_dict.remote(avg_state) for actor in actors])
        completed += chunk
        log: dict[str, float | int] = {"step": completed, "workers": args.workers}
        for part in parts:
            for key, value in part.items():
                log[key] = float(log.get(key, 0.0)) + float(value) / max(len(parts), 1)
        print(json.dumps(log, ensure_ascii=False), flush=True)
        if swan is not None:
            swan.log(log, step=completed)

    ray.get([actors[0].save.remote(out_path, args.steps, vars(args))])
    print(json.dumps({"saved": out_path, "steps": args.steps}, ensure_ascii=False), flush=True)
    if swan is not None:
        swan.finish()


if __name__ == "__main__":
    main()
