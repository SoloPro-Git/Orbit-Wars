"""Ray stage-1 training: align candidate model to REGULAR_CONFIG rulebase.

Stage 1 is behavior cloning plus a real-game gate:
- Ray data workers generate demonstrations from the strongest regular rulebase.
- The trainer learns to rank the regular rulebase candidate first.
- Ray eval workers periodically run 100 two-player games:
  player0 = model-reranked candidate agent, player1 = regular rulebase.
- The stage passes when model win rate reaches the configured threshold.
"""
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

try:
    import ray
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Ray 未安装，请先安装 ray") from exc

from kaggle_environments import make

from training2.batching import pad_planets
from training2.candidates import build_candidates
from training2.features import encode_position, result_value
from training2.model import CandidatePolicyValueNet
from training2.rulebase_bridge import make_rulebase_agent


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        p = PROJECT_ROOT / path
    with p.open() as f:
        return yaml.safe_load(f) or {}


def _iter_jsonl_paths(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"offline dataset not found: {path}")
        if path.is_dir():
            out.extend(sorted(path.glob("*.jsonl")))
        else:
            out.append(path)
    return out


def _load_offline_rows(paths: list[str], max_samples: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for path in _iter_jsonl_paths(paths):
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if max_samples is not None and len(rows) >= max_samples:
                    return rows
    return rows


def _canonical_action(action: list[list]) -> tuple:
    return tuple((int(a[0]), round(float(a[1]), 6), int(a[2])) for a in action if len(a) >= 3)


def _normalize_action(action: Any) -> list[list]:
    if not isinstance(action, list):
        return []
    out: list[list] = []
    for item in action:
        if not isinstance(item, list | tuple) or len(item) < 3:
            continue
        out.append([int(item[0]), float(item[1]), int(item[2])])
    return out


def _historical_candidates(
    obs: dict,
    action: list[list],
    rulebase_agent,
    max_candidates: int,
    include_noop: bool = True,
) -> list[list[list]]:
    candidates: list[list[list]] = []
    seen: set[tuple] = set()

    def add(candidate: list[list]) -> None:
        key = _canonical_action(candidate)
        if key in seen:
            return
        seen.add(key)
        candidates.append(_normalize_action(candidate))

    add(action)
    if include_noop:
        add([])
    for move in action:
        add([move])
    for i in range(len(action)):
        add([m for j, m in enumerate(action) if j != i])
    regular = rulebase_agent(obs) or []
    add(regular)
    for move in regular:
        add([move])
    return candidates[:max_candidates]


def _load_historical_replay_rows(
    paths: list[str],
    oracle: str,
    max_candidates: int,
    max_samples: int | None = None,
    keep_noop_prob: float = 0.2,
) -> list[dict]:
    rows: list[dict] = []
    rng = random.Random(20260515)
    rulebase_agent = make_rulebase_agent(oracle)
    for path in _iter_jsonl_paths(paths):
        raw_rows: list[dict] = []
        final_reward_by_player: dict[int, float] = {}
        final_obs_by_player: dict[int, dict] = {}
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                raw_rows.append(raw)
                if raw.get("done"):
                    pid = int(raw.get("player_id", raw.get("observation", {}).get("player", 0)))
                    final_reward_by_player[pid] = float(raw.get("reward", 0.0))
                    if isinstance(raw.get("observation"), dict):
                        final_obs_by_player[pid] = raw["observation"]

        for raw in raw_rows:
            obs = raw.get("observation")
            if not isinstance(obs, dict) or not obs.get("planets"):
                continue
            player = int(raw.get("player_id", obs.get("player", 0)))
            obs = dict(obs)
            obs["player"] = player
            action = _normalize_action(raw.get("actions", []))
            if not action and rng.random() > keep_noop_prob:
                continue
            candidates = _historical_candidates(obs, action, rulebase_agent, max_candidates=max_candidates)
            enc = encode_position(obs, player, candidates, max_candidates=max_candidates)
            value = final_reward_by_player.get(player)
            if value is None:
                final_obs = final_obs_by_player.get(player)
                value = result_value(final_obs, player) if final_obs else float(raw.get("reward", 0.0))
            rows.append(_row_from_encoded(enc, player, target=0, value=float(value)))
            rows[-1]["source"] = "historical_replay"
            if max_samples is not None and len(rows) >= max_samples:
                return rows
    return rows


def _raw_obs(env, player: int) -> dict:
    return env.steps[-1][player]["observation"]


def _score(obs: dict, player: int) -> float:
    return float(
        sum(p[5] for p in obs.get("planets", []) if int(p[1]) == player)
        + sum(f[6] for f in obs.get("fleets", []) if int(f[1]) == player)
    )


def _row_from_encoded(enc, player: int, target: int, value: float | None = None) -> dict:
    row = {
        "player": player,
        "planets": enc.planet_features.tolist(),
        "global": enc.global_features.tolist(),
        "candidates": enc.candidate_features.tolist(),
        "candidate_mask": enc.candidate_mask.tolist(),
        "target": int(target),
    }
    if value is not None:
        row["value"] = float(value)
    return row


@ray.remote
class Stage1DataActor:
    def __init__(
        self,
        worker_id: int,
        oracle: str,
        max_candidates: int,
        four_player_prob: float,
    ) -> None:
        self.worker_id = worker_id
        self.oracle = oracle
        self.max_candidates = max_candidates
        self.four_player_prob = four_player_prob

    def generate(self, episodes: int, seed_offset: int) -> dict:
        rows: list[dict] = []
        game_lengths: list[int] = []
        games_2p = 0
        games_4p = 0
        for ep in range(episodes):
            seed = seed_offset + self.worker_id * 100000 + ep
            rng = random.Random(seed)
            players = 4 if rng.random() < self.four_player_prob else 2
            if players == 4:
                games_4p += 1
            else:
                games_2p += 1
            env = make("orbit_wars", configuration={"episodeSteps": 500, "seed": seed}, debug=True)
            env.reset(players)
            agents = {pid: make_rulebase_agent(self.oracle) for pid in range(players)}
            pending: list[dict] = []
            steps = 0
            for steps in range(500):
                actions = []
                for pid in range(players):
                    obs = _raw_obs(env, pid)
                    candidates, oracle_idx = build_candidates(
                        obs,
                        agents[pid],
                        max_candidates=self.max_candidates,
                    )
                    enc = encode_position(obs, pid, candidates, max_candidates=self.max_candidates)
                    pending.append(_row_from_encoded(enc, pid, oracle_idx))
                    actions.append(candidates[oracle_idx] if candidates else [])
                env.step(actions)
                if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                    break

            finals = [_raw_obs(env, pid) for pid in range(players)]
            for row in pending:
                row["value"] = result_value(finals[row["player"]], row["player"])
            rows.extend(pending)
            game_lengths.append(steps + 1)
        return {
            "rows": rows,
            "samples": len(rows),
            "episodes": episodes,
            "games_2p": games_2p,
            "games_4p": games_4p,
            "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
        }


@ray.remote
class Stage1EvalActor:
    def __init__(
        self,
        worker_id: int,
        oracle: str,
        max_candidates: int,
        four_player_prob: float,
    ) -> None:
        self.worker_id = worker_id
        self.oracle = oracle
        self.max_candidates = max_candidates
        self.four_player_prob = four_player_prob

    def evaluate(self, state_dict: dict, games: int, seed_offset: int, device: str = "cpu") -> dict:
        model = CandidatePolicyValueNet().to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        wins = losses = draws = 0
        margins: list[float] = []
        games_2p = 0
        games_4p = 0
        for game in range(games):
            seed = seed_offset + self.worker_id * 100000 + game
            rng = random.Random(seed)
            players = 4 if rng.random() < self.four_player_prob else 2
            if players == 4:
                games_4p += 1
            else:
                games_2p += 1
            env = make("orbit_wars", configuration={"episodeSteps": 500, "seed": seed}, debug=True)
            env.reset(players)
            model_pid = game % players
            agents = {pid: make_rulebase_agent(self.oracle) for pid in range(players)}

            for _ in range(500):
                actions = []
                for pid in range(players):
                    obs = _raw_obs(env, pid)
                    if pid == model_pid:
                        candidates, _ = build_candidates(
                            obs,
                            agents[pid],
                            max_candidates=self.max_candidates,
                        )
                        enc = encode_position(obs, pid, candidates, max_candidates=self.max_candidates)
                        with torch.no_grad():
                            logits, _ = model(
                                torch.tensor(enc.planet_features, dtype=torch.float32, device=device).unsqueeze(0),
                                torch.tensor(enc.global_features, dtype=torch.float32, device=device).unsqueeze(0),
                                torch.tensor(enc.candidate_features, dtype=torch.float32, device=device).unsqueeze(0),
                                torch.tensor(enc.candidate_mask, dtype=torch.float32, device=device).unsqueeze(0),
                            )
                        idx = int(logits.argmax(-1).item())
                        actions.append(candidates[idx] if idx < len(candidates) else candidates[0])
                    else:
                        actions.append(agents[pid](obs) or [])
                env.step(actions)
                if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                    break

            final_obs = _raw_obs(env, model_pid)
            my_score = _score(final_obs, model_pid)
            opp_best = max(_score(final_obs, pid) for pid in range(players) if pid != model_pid)
            margin = my_score - opp_best
            margins.append(margin)
            if margin > 0:
                wins += 1
            elif margin < 0:
                losses += 1
            else:
                draws += 1
        return {
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / max(games, 1),
            "draw_rate": draws / max(games, 1),
            "avg_margin": float(np.mean(margins)) if margins else 0.0,
            "games_2p": games_2p,
            "games_4p": games_4p,
        }


def _train_batch(model, opt, rows: list[dict], device: str) -> dict:
    planets = pad_planets([r["planets"] for r in rows]).to(device)
    glob = torch.tensor([r["global"] for r in rows], dtype=torch.float32, device=device)
    candidates = torch.tensor([r["candidates"] for r in rows], dtype=torch.float32, device=device)
    mask = torch.tensor([r["candidate_mask"] for r in rows], dtype=torch.float32, device=device)
    target = torch.tensor([r["target"] for r in rows], dtype=torch.long, device=device)
    value_target = torch.tensor([r["value"] for r in rows], dtype=torch.float32, device=device)

    logits, value = model(planets, glob, candidates, mask)
    policy_loss = F.cross_entropy(logits, target)
    value_loss = F.mse_loss(value, value_target)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    return {
        "train/loss": float(loss.item()),
        "train/policy_loss": float(policy_loss.item()),
        "train/value_loss": float(value_loss.item()),
        "train/entropy": float(entropy.item()),
        "train/oracle_top1": float((logits.argmax(dim=-1) == target).float().mean().item()),
        "align/train_oracle_top1": float((logits.argmax(dim=-1) == target).float().mean().item()),
        "train/value_mean": float(value.mean().item()),
    }


@ray.remote
class Stage1TrainerActor:
    def __init__(self, model_cfg: dict, train_cfg: dict, device: str, resume: str | None = None) -> None:
        self.device = device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("TrainerActor requested CUDA but torch.cuda.is_available() is False")
        self.model = CandidatePolicyValueNet(
            d_model=int(model_cfg.get("d_model", 192)),
            nhead=int(model_cfg.get("nhead", 6)),
            layers=int(model_cfg.get("layers", 4)),
            dropout=float(model_cfg.get("dropout", 0.10)),
        ).to(self.device)
        if resume:
            ckpt = torch.load(resume, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 2e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

    def update(self, rows: list[dict], updates: int, batch_size: int, align_limit: int) -> dict:
        self.model.train()
        metrics_accum: dict[str, float] = {}
        for _ in range(updates):
            batch = random.sample(rows, k=min(batch_size, len(rows)))
            metrics = _train_batch(self.model, self.opt, batch, self.device)
            for key, value in metrics.items():
                metrics_accum[key] = metrics_accum.get(key, 0.0) + value
        metrics_accum = {key: value / max(updates, 1) for key, value in metrics_accum.items()}
        align_metrics = _alignment_metrics(
            self.model,
            rows[-align_limit:] if align_limit > 0 else rows,
            self.device,
            batch_size=batch_size,
        )
        log = {
            "train/lr": self.opt.param_groups[0]["lr"],
            **metrics_accum,
            **align_metrics,
        }
        if self.device.startswith("cuda"):
            log.update(
                {
                    "gpu/memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "gpu/memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                    "gpu/max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
                }
            )
        return log

    def state_dict_cpu(self) -> dict:
        return {key: value.detach().cpu() for key, value in self.model.state_dict().items()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict, strict=False)

    def save(self, path: str, iteration: int, config: dict, best_win_rate: float) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "iteration": iteration,
                "config": config,
                "best_win_rate": best_win_rate,
            },
            path,
        )


def _average_state_dicts(states: list[dict]) -> dict:
    if len(states) == 1:
        return states[0]
    avg = {}
    for key in states[0]:
        value = states[0][key]
        if torch.is_floating_point(value):
            avg[key] = torch.stack([state[key].float() for state in states], dim=0).mean(dim=0).to(value.dtype)
        else:
            avg[key] = value
    return avg


def _alignment_metrics(model, rows: list[dict], device: str, batch_size: int) -> dict:
    """Measure exact policy agreement with the regular rulebase labels."""
    if not rows:
        return {}
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    total_entropy = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            planets = pad_planets([r["planets"] for r in batch]).to(device)
            glob = torch.tensor([r["global"] for r in batch], dtype=torch.float32, device=device)
            candidates = torch.tensor([r["candidates"] for r in batch], dtype=torch.float32, device=device)
            mask = torch.tensor([r["candidate_mask"] for r in batch], dtype=torch.float32, device=device)
            target = torch.tensor([r["target"] for r in batch], dtype=torch.long, device=device)
            logits, _ = model(planets, glob, candidates, mask)
            pred = logits.argmax(dim=-1)
            n = target.numel()
            total += n
            correct += int((pred == target).sum().item())
            total_loss += float(F.cross_entropy(logits, target, reduction="sum").item())
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
            total_entropy += float(entropy.sum().item())
    model.train()
    return {
        "align/new_oracle_top1": correct / max(total, 1),
        "align/new_action_exact_match": correct / max(total, 1),
        "align/new_policy_loss": total_loss / max(total, 1),
        "align/new_entropy": total_entropy / max(total, 1),
        "align/new_samples": total,
    }


def _init_swanlab(config: dict, args) -> Any | None:
    key = os.environ.get("SWANLAB_API_KEY")
    if not key:
        key_file = PROJECT_ROOT / "training" / "config" / "swanlab_key.txt"
        if key_file.exists():
            os.environ["SWANLAB_API_KEY"] = key_file.read_text().strip()
    import swanlab

    swanlab.init(
            project=config.get("swanlab_project", "orbit-wars"),
            experiment_name=config.get("swanlab_experiment", "training2-stage1-regular-align"),
            config={"training2_stage1": config, "args": vars(args)},
            mode=config.get("swanlab_mode", "cloud"),
            public=False,
            launcher=False,
        )
    swanlab.log({"stage1/swanlab_initialized": 1}, step=0)
    return swanlab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training2/config/default.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--episodes-per-worker", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--updates-per-iter", type=int)
    parser.add_argument("--eval-games", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--eval-gpus-per-worker", type=float)
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--allow-no-swanlab", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    run_cfg = cfg.get("run", {})
    selected_stage = str(run_cfg.get("stage", "stage1")).lower()
    if selected_stage not in {"stage1", "regular_align"}:
        raise ValueError(
            f"training2.train_stage1_ray only runs stage1/regular_align, got run.stage={selected_stage!r}. "
            "Set run.stage: stage1 in training2/config/default.yaml, or use the dedicated self-play entrypoint later."
        )
    model_cfg = cfg.get("model", {})
    stage = cfg.get("stage1", {})
    train_cfg = cfg.get("training", {})
    ray_cfg = cfg.get("ray", {})

    workers = args.workers or int(ray_cfg.get("num_data_workers", 8))
    eval_workers = args.eval_workers or int(ray_cfg.get("num_eval_workers", 8))
    trainer_workers = int(ray_cfg.get("num_trainer_workers", 1))
    gpus_per_trainer = float(ray_cfg.get("gpus_per_trainer", ray_cfg.get("trainer_num_gpus", 1.0)))
    episodes_per_worker = args.episodes_per_worker or int(ray_cfg.get("episodes_per_worker", 2))
    iterations = args.iterations or int(stage.get("max_iterations", 500))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 256))
    updates_per_iter = args.updates_per_iter or int(train_cfg.get("updates_per_iteration", 100))
    device = args.device or train_cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Stage1 配置要求使用 CUDA，但当前 torch.cuda.is_available() 为 False")
    eval_device = args.eval_device or str(stage.get("eval_device", "cpu"))
    if eval_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("eval_device 配置为 CUDA，但当前没有可用 GPU")

    max_candidates = int(model_cfg.get("max_candidates", 32))
    oracle = str(stage.get("oracle", "rl_informed_regular"))
    data_four_player_prob = float(stage.get("data_four_player_prob", 1.0))
    eval_four_player_prob = float(stage.get("eval_four_player_prob", 0.0))
    eval_games = args.eval_games or int(stage.get("eval_games", 100))
    eval_interval = int(stage.get("eval_interval", 10))
    threshold = float(stage.get("win_rate_threshold", 0.50))
    replay_capacity = int(stage.get("replay_capacity", 200000))
    data_source = str(stage.get("data_source", "online")).lower()
    offline_paths = [str(p) for p in stage.get("offline_data_paths", [])]
    offline_max_samples_raw = stage.get("offline_max_samples")
    offline_max_samples = int(offline_max_samples_raw) if offline_max_samples_raw else None
    historical_cfg = cfg.get("historical_replay", {})
    historical_enabled = bool(historical_cfg.get("enabled", False))
    historical_paths = [str(p) for p in historical_cfg.get("paths", [])]
    historical_max_samples_raw = historical_cfg.get("max_samples")
    historical_max_samples = int(historical_max_samples_raw) if historical_max_samples_raw else None
    historical_keep_noop_prob = float(historical_cfg.get("keep_noop_prob", 0.2))
    output_dir = Path(stage.get("output_dir", "training2/checkpoints/stage1_regular"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_swanlab:
        swan = None
    else:
        try:
            swan = _init_swanlab(
                {**train_cfg, **stage, **ray_cfg, "historical_replay": historical_cfg, "device": device},
                args,
            )
        except Exception as exc:
            if args.allow_no_swanlab:
                print(f"[SwanLab] 初始化失败，按 --allow-no-swanlab 继续: {exc}")
                swan = None
            else:
                raise RuntimeError(
                    "SwanLab 初始化失败。请确认 SWANLAB_API_KEY 或 training/config/swanlab_key.txt 可用；"
                    "调试时可显式加 --allow-no-swanlab。"
                ) from exc

    ray_temp = Path(ray_cfg.get("temp_dir", "training2/.ray_temp")).resolve()
    ray_temp.mkdir(parents=True, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=str(ray_temp),
        _memory=1_000_000_000,
        num_gpus=torch.cuda.device_count() if torch.cuda.is_available() else 0,
    )

    trainer_actors = [
        Stage1TrainerActor.options(num_gpus=gpus_per_trainer).remote(
            model_cfg,
            train_cfg,
            device,
            args.resume,
        )
        for _ in range(trainer_workers)
    ]

    replay: list[dict] = []
    data_actors = []
    offline_rows: list[dict] = []
    historical_rows: list[dict] = []
    if historical_enabled and historical_paths:
        historical_rows = _load_historical_replay_rows(
            historical_paths,
            oracle=oracle,
            max_candidates=max_candidates,
            max_samples=historical_max_samples,
            keep_noop_prob=historical_keep_noop_prob,
        )
        replay.extend(historical_rows[-replay_capacity:])
    if data_source == "online":
        data_actors = [
            Stage1DataActor.options(num_gpus=0).remote(i, oracle, max_candidates, data_four_player_prob)
            for i in range(workers)
        ]
    elif data_source == "offline":
        offline_rows = _load_offline_rows(offline_paths, max_samples=offline_max_samples) if offline_paths else []
        if not offline_rows and not historical_rows:
            raise ValueError("stage1.data_source=offline but no offline or historical rows were loaded")
        replay.extend(offline_rows[-replay_capacity:])
        replay = replay[-replay_capacity:]
    else:
        raise ValueError(f"unknown stage1.data_source={data_source!r}; expected online/offline")
    eval_gpus_per_worker = (
        args.eval_gpus_per_worker
        if args.eval_gpus_per_worker is not None
        else float(ray_cfg.get("eval_gpus_per_worker", 0.0 if eval_device == "cpu" else 0.25))
    )
    eval_actors = [
        Stage1EvalActor.options(num_gpus=eval_gpus_per_worker).remote(i, oracle, max_candidates, eval_four_player_prob)
        for i in range(eval_workers)
    ]

    best_win_rate = -1.0
    print(
        json.dumps(
            {
                "stage": "stage1_regular_align",
                "workers": workers,
                "eval_workers": eval_workers,
                "episodes_per_worker": episodes_per_worker,
                "eval_games": eval_games,
                "threshold": threshold,
                "device": device,
                "eval_device": eval_device,
                "eval_gpus_per_worker": eval_gpus_per_worker,
                "trainer_workers": trainer_workers,
                "gpus_per_trainer": gpus_per_trainer,
                "data_four_player_prob": data_four_player_prob,
                "eval_four_player_prob": eval_four_player_prob,
                "data_source": data_source,
                "offline_samples": len(offline_rows),
                "historical_replay_enabled": historical_enabled,
                "historical_replay_samples": len(historical_rows),
                "historical_keep_noop_prob": historical_keep_noop_prob,
                "oracle": oracle,
            },
            ensure_ascii=False,
        )
    )

    try:
        progress = tqdm(range(iterations), desc="[Stage1]", unit="iter")
        for iteration in progress:
            t0 = time.time()
            if data_source == "online":
                refs = [
                    actor.generate.remote(episodes_per_worker, seed_offset=iteration * 10_000_000)
                    for actor in data_actors
                ]
                generated = ray.get(refs)
                new_rows = [row for chunk in generated for row in chunk["rows"]]
                replay.extend(new_rows)
                replay = replay[-replay_capacity:]
                data_log = {
                    "data/new_samples": len(new_rows),
                    "data/replay_size": len(replay),
                    "data/historical_replay_samples": len(historical_rows),
                    "data/episodes": sum(c["episodes"] for c in generated),
                    "data/games_2p": sum(c["games_2p"] for c in generated),
                    "data/games_4p": sum(c["games_4p"] for c in generated),
                    "data/four_player_ratio_actual": (
                        sum(c["games_4p"] for c in generated) / max(sum(c["episodes"] for c in generated), 1)
                    ),
                    "data/avg_game_length": float(np.mean([c["avg_game_length"] for c in generated])),
                }
            else:
                new_rows = random.sample(replay, k=min(int(stage.get("alignment_eval_samples", 4096)), len(replay)))
                data_log = {
                    "data/new_samples": 0,
                    "data/replay_size": len(replay),
                    "data/offline_samples": len(offline_rows),
                    "data/historical_replay_samples": len(historical_rows),
                    "data/episodes": 0,
                    "data/games_2p": 0,
                    "data/games_4p": 0,
                    "data/four_player_ratio_actual": data_four_player_prob,
                    "data/avg_game_length": 0.0,
                }

            shard_refs = [
                actor.update.remote(
                    replay,
                    updates_per_iter,
                    batch_size,
                    int(stage.get("alignment_eval_samples", 4096)),
                )
                for actor in trainer_actors
            ]
            shard_logs = ray.get(shard_refs)
            states = ray.get([actor.state_dict_cpu.remote() for actor in trainer_actors])
            averaged_state = _average_state_dicts(states)
            ray.get([actor.load_state_dict.remote(averaged_state) for actor in trainer_actors])
            train_log: dict[str, float] = {}
            for shard_log in shard_logs:
                for key, value in shard_log.items():
                    train_log[key] = train_log.get(key, 0.0) + float(value)
            train_log = {key: value / max(len(shard_logs), 1) for key, value in train_log.items()}
            train_log["train/trainer_workers"] = trainer_workers
            train_log["train/gpus_per_trainer"] = gpus_per_trainer

            log = {
                "iteration": iteration,
                "stage": 1,
                **data_log,
                "time/iteration_sec": time.time() - t0,
                **train_log,
            }

            ckpt_path = output_dir / "latest.pt"
            ray.get(trainer_actors[0].save.remote(str(ckpt_path), iteration, cfg, best_win_rate))

            should_eval = (iteration + 1) % eval_interval == 0
            if should_eval:
                model_cpu_state = ray.get(trainer_actors[0].state_dict_cpu.remote())
                games_per_actor = [eval_games // eval_workers] * eval_workers
                for i in range(eval_games % eval_workers):
                    games_per_actor[i] += 1
                eval_refs = [
                    actor.evaluate.remote(
                        model_cpu_state,
                        games_per_actor[i],
                        seed_offset=90_000_000 + iteration * 1_000_000,
                        device=eval_device,
                    )
                    for i, actor in enumerate(eval_actors)
                    if games_per_actor[i] > 0
                ]
                eval_parts = ray.get(eval_refs)
                games = sum(p["games"] for p in eval_parts)
                wins = sum(p["wins"] for p in eval_parts)
                losses = sum(p["losses"] for p in eval_parts)
                draws = sum(p["draws"] for p in eval_parts)
                eval_games_2p = sum(p["games_2p"] for p in eval_parts)
                eval_games_4p = sum(p["games_4p"] for p in eval_parts)
                avg_margin = float(np.mean([p["avg_margin"] for p in eval_parts]))
                win_rate = wins / max(games, 1)
                log.update(
                    {
                        "eval/games": games,
                        "eval/wins": wins,
                        "eval/losses": losses,
                        "eval/draws": draws,
                        "eval/games_2p": eval_games_2p,
                        "eval/games_4p": eval_games_4p,
                        "eval/four_player_ratio_actual": eval_games_4p / max(games, 1),
                        "eval/win_rate_vs_regular": win_rate,
                        "eval/draw_rate_vs_regular": draws / max(games, 1),
                        "eval/avg_margin_vs_regular": avg_margin,
                        "eval/pass_stage1": float(win_rate >= threshold),
                    }
                )
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    ray.get(trainer_actors[0].save.remote(str(output_dir / "best.pt"), iteration, cfg, best_win_rate))

            if swan:
                swan.log(log, step=iteration)
            print(json.dumps(log, ensure_ascii=False), flush=True)
            progress.set_postfix(
                loss=f"{log.get('train/loss', 0.0):.3f}",
                align=f"{log.get('align/new_oracle_top1', 0.0):.3f}",
                win=f"{log.get('eval/win_rate_vs_regular', -1.0):.3f}",
            )

            if log.get("eval/win_rate_vs_regular", -1.0) >= threshold:
                print(f"[Stage1] PASSED: win_rate={log['eval/win_rate_vs_regular']:.3f} >= {threshold:.3f}")
                break
    finally:
        if swan:
            swan.finish()
        ray.shutdown()


if __name__ == "__main__":
    main()
