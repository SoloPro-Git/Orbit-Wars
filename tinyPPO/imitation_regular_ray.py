from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from training2.rulebase_bridge import make_rulebase_agent

from tinyPPO.agents import ACTION_SLOTS, SHIP_BUCKET_MULTIPLIERS, TinyPPOAgent
from tinyPPO.eval import run_matchups
from tinyPPO.imitation_regular import _collect_one_game, bc_loss, stack_rows, unpack
from tinyPPO.model import TinyPolicyValueNet


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _players_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--players-list must not be empty")
    return values


@ray.remote(num_cpus=1)
def collect_game_task(
    seed: int,
    players: int,
    episode_steps: int,
    keep_noop_prob: float,
    sample_stride: int,
    rows_per_game: int,
    use_numba: bool,
) -> tuple[list[Any], dict[str, float]]:
    return _collect_one_game(seed, players, episode_steps, keep_noop_prob, sample_stride, rows_per_game, use_numba)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _average_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in states[0]:
        acc = None
        for state in states:
            value = state[key].float()
            acc = value.clone() if acc is None else acc.add_(value)
        out[key] = (acc / float(len(states))).to(states[0][key].dtype)
    return out


@ray.remote
class BCTrainEvalActor:
    def __init__(
        self,
        rows: list[Any],
        model_cfg: dict[str, int],
        batch_size: int,
        val_frac: float,
        lr: float,
        weight_decay: float,
        max_grad_norm: float,
        launch_pos_weight: float,
        target_loss_weight: float,
        ship_loss_weight: float,
        critical_action_weight: float,
        seed: int,
    ):
        torch.set_num_threads(1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cfg = dict(model_cfg)
        self.model = TinyPolicyValueNet(**self.model_cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.launch_pos_weight = float(launch_pos_weight)
        self.target_loss_weight = float(target_loss_weight)
        self.ship_loss_weight = float(ship_loss_weight)
        self.critical_action_weight = float(critical_action_weight)
        dataset = stack_rows(rows)
        generator = torch.Generator().manual_seed(seed)
        val_size = max(1, int(len(dataset) * val_frac))
        train_size = max(1, len(dataset) - val_size)
        train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)
        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
        self.samples = len(dataset)

    def train_epoch(self, state: dict[str, torch.Tensor], epoch: int) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        del epoch
        self.model.load_state_dict({key: value.to(self.device) for key, value in state.items()})
        self.model.train()
        sums: dict[str, float] = {}
        count = 0
        for batch in self.train_loader:
            self.opt.zero_grad(set_to_none=True)
            loss, metrics = bc_loss(
                self.model,
                unpack(batch, self.device),
                self.launch_pos_weight,
                self.target_loss_weight,
                self.ship_loss_weight,
                self.critical_action_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()
            n = int(batch[0].shape[0])
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value) * n
            count += n
        metrics = {f"train_{key}": value / max(1, count) for key, value in sums.items()}
        metrics["samples"] = float(self.samples)
        metrics.update({f"val_{key}": value for key, value in self._eval_loader().items()})
        return _cpu_state_dict(self.model), metrics

    @torch.no_grad()
    def _eval_loader(self) -> dict[str, float]:
        self.model.eval()
        sums: dict[str, float] = {}
        count = 0
        for batch in self.val_loader:
            _, metrics = bc_loss(
                self.model,
                unpack(batch, self.device),
                self.launch_pos_weight,
                self.target_loss_weight,
                self.ship_loss_weight,
                self.critical_action_weight,
            )
            n = int(batch[0].shape[0])
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value) * n
            count += n
        return {key: value / max(1, count) for key, value in sums.items()}

    def eval_vs_regular(
        self,
        state: dict[str, torch.Tensor],
        games: int,
        seed: int,
        stochastic: bool,
        launch_bias: float,
        ship_bias: float,
        launch_temperature: float,
    ) -> dict[str, float]:
        self.model.load_state_dict({key: value.to(self.device) for key, value in state.items()})
        fd, path = tempfile.mkstemp(prefix="regular_bc_ray_eval_", suffix=".pt")
        os.close(fd)
        torch.save({"state_dict": _cpu_state_dict(self.model), "model": self.model_cfg, "update": 0, "metrics": {}}, path)
        try:
            return run_matchups(
                lambda: TinyPPOAgent(
                    path,
                    device=str(self.device),
                    deterministic=not stochastic,
                    launch_bias=launch_bias,
                    ship_bias=ship_bias,
                    launch_temperature=launch_temperature,
                ),
                lambda: make_rulebase_agent("regular"),
                games=games,
                seed=seed,
                episode_steps=500,
                use_numba=True,
                progress=False,
            )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


@ray.remote
class BCEvalActor:
    def __init__(self, model_cfg: dict[str, int]):
        torch.set_num_threads(1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cfg = dict(model_cfg)
        self.model = TinyPolicyValueNet(**self.model_cfg).to(self.device)

    def eval_vs_regular(
        self,
        state: dict[str, torch.Tensor],
        games: int,
        seed: int,
        stochastic: bool,
        launch_bias: float,
        ship_bias: float,
        launch_temperature: float,
    ) -> dict[str, float]:
        self.model.load_state_dict({key: value.to(self.device) for key, value in state.items()})
        fd, path = tempfile.mkstemp(prefix="regular_bc_ray_eval_", suffix=".pt")
        os.close(fd)
        torch.save({"state_dict": _cpu_state_dict(self.model), "model": self.model_cfg, "update": 0, "metrics": {}}, path)
        try:
            return run_matchups(
                lambda: TinyPPOAgent(
                    path,
                    device=str(self.device),
                    deterministic=not stochastic,
                    launch_bias=launch_bias,
                    ship_bias=ship_bias,
                    launch_temperature=launch_temperature,
                ),
                lambda: make_rulebase_agent("regular"),
                games=games,
                seed=seed,
                episode_steps=500,
                use_numba=True,
                progress=True,
                desc="regular-bc eval",
            )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


def _weighted_mean(parts: list[dict[str, float]]) -> dict[str, float]:
    total = sum(float(p.get("samples", 1.0)) for p in parts)
    out: dict[str, float] = {}
    for part in parts:
        weight = float(part.get("samples", 1.0)) / max(1.0, total)
        for key, value in part.items():
            if key == "samples":
                continue
            out[key] = out.get(key, 0.0) + float(value) * weight
    out["samples"] = total
    return out


def _merge_eval(parts: list[dict[str, float]]) -> dict[str, float]:
    games = sum(float(p["games"]) for p in parts)
    wins = sum(float(p["wins"]) for p in parts)
    losses = sum(float(p["losses"]) for p in parts)
    draws = sum(float(p["draws"]) for p in parts)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": wins / max(1.0, games),
        "nonloss": (wins + draws) / max(1.0, games),
        "mean_reward": (wins - losses) / max(1.0, games),
    }


def save_checkpoint(path: Path, state: dict[str, torch.Tensor], model_cfg: dict[str, int], epoch: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "model": model_cfg, "update": epoch, "metrics": metrics}, path)


def _flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_metrics(value, name))
        elif isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
            out[name] = float(value)
    return out


def _init_swanlab(args: argparse.Namespace) -> Any | None:
    if args.no_swanlab:
        return None
    key = os.environ.get("SWANLAB_API_KEY")
    if not key:
        key_file = Path("training/config/swanlab_key.txt")
        if key_file.exists():
            os.environ["SWANLAB_API_KEY"] = key_file.read_text().strip()
    os.environ.setdefault("SWANLAB_NO_INTERACTIVE", "1")
    os.environ.setdefault("SWANLAB_DISABLE_INTERACTIVE", "1")
    try:
        import swanlab

        run = swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_experiment,
            config={"tinyPPO_regular_bc_ray": vars(args)},
            mode=args.swanlab_mode,
            public=False,
            launcher=False,
        )
        swanlab.log({"bc/swanlab_initialized": 1}, step=0)
        return swanlab if run is None else run
    except Exception as exc:
        if not args.allow_no_swanlab:
            raise
        print(json.dumps({"event": "swanlab_init_failed", "error": str(exc)}, ensure_ascii=True), flush=True)
        return None


def _log_swanlab(swan: Any | None, payload: dict[str, Any], step: int) -> None:
    if swan is None:
        return
    metrics = _flatten_metrics(payload)
    if not metrics:
        return
    try:
        swan.log(metrics, step=step)
    except Exception as exc:
        print(json.dumps({"event": "swanlab_log_failed", "step": step, "error": str(exc)}, ensure_ascii=True), flush=True)


def collect_dataset(args: argparse.Namespace) -> tuple[list[Any], dict[str, float]]:
    cache_path = Path(args.dataset_cache) if args.dataset_cache else None
    if cache_path is not None and cache_path.exists() and not args.refresh_dataset:
        with cache_path.open("rb") as fh:
            payload = pickle.load(fh)
        rows = payload["rows"]
        metrics = dict(payload.get("metrics", {}))
        metrics["cache_loaded"] = 1.0
        metrics["cache_path"] = str(cache_path)
        return rows, metrics

    players_values = _players_list(args.players_list)
    jobs = [
        (args.seed + players * 1_000_000 + game, players)
        for players in players_values
        for game in range(args.games_per_players)
    ]
    futures = [
        collect_game_task.remote(
            seed,
            players,
            args.episode_steps,
            args.keep_noop_prob,
            args.sample_stride,
            args.rows_per_game,
            not args.no_numba,
        )
        for seed, players in jobs
    ]
    rows: list[Any] = []
    labelled = 0.0
    skipped = 0.0
    completed = 0
    progress = tqdm(total=len(futures), desc="ray collect regular BC", dynamic_ncols=True) if tqdm is not None else None
    pending = list(futures)
    while pending:
        done, pending = ray.wait(pending, num_returns=min(64, len(pending)))
        for ref in done:
            game_rows, metrics = ray.get(ref)
            rows.extend(game_rows)
            labelled += float(metrics.get("labelled_actions", 0.0))
            skipped += float(metrics.get("skipped_actions", 0.0))
            completed += 1
        if progress is not None:
            progress.update(len(done))
            progress.set_postfix(samples=len(rows), labelled=int(labelled))
    if progress is not None:
        progress.close()
    metrics = {
        "games": float(completed),
        "samples": float(len(rows)),
        "labelled_actions": labelled,
        "skipped_actions": skipped,
        "players_modes": float(len(players_values)),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump({"rows": rows, "metrics": metrics, "args": vars(args)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
        metrics["cache_saved"] = 1.0
        metrics["cache_path"] = str(cache_path)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--out", default="tinyPPO/regular_bc_ray.pt")
    parser.add_argument("--best-out", default="tinyPPO/regular_bc.pt")
    parser.add_argument("--resume", default="", help="Resume model weights from a regular BC checkpoint. Epoch numbering continues from checkpoint update.")
    parser.add_argument("--players-list", default="2")
    parser.add_argument("--games-per-players", type=int, default=2000)
    parser.add_argument("--dataset-cache", default="", help="Pickle cache for collected BC rows. Existing cache is reused unless --refresh-dataset is set.")
    parser.add_argument("--refresh-dataset", action="store_true")
    parser.add_argument("--rows-per-game", type=int, default=12)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--keep-noop-prob", type=float, default=0.15)
    parser.add_argument("--trainers", type=int, default=8)
    parser.add_argument("--cpus-per-trainer", type=float, default=2.0)
    parser.add_argument("--gpus-per-trainer", type=float, default=1.0)
    parser.add_argument("--eval-actors", type=int, default=1)
    parser.add_argument("--eval-cpus-per-actor", type=float, default=4.0)
    parser.add_argument("--gpus-per-eval-actor", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--launch-pos-weight", type=float, default=4.0)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--ship-loss-weight", type=float, default=0.5)
    parser.add_argument("--critical-action-weight", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.08)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=40)
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--max-pending-evals", type=int, default=2)
    parser.add_argument("--eval-min-launch-recall", type=float, default=0.0, help="Skip slow online eval until validation launch recall reaches this value.")
    parser.add_argument("--target-nonloss", type=float, default=0.50)
    parser.add_argument("--target-winrate", type=float, default=0.20)
    parser.add_argument("--eval-stochastic", action="store_true")
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--ship-bias", type=float, default=0.0)
    parser.add_argument("--launch-temperature", type=float, default=1.0)
    parser.add_argument("--swanlab-project", default="orbit-wars")
    parser.add_argument("--swanlab-experiment", default="tinyppo-regular-bc-ray")
    parser.add_argument("--swanlab-mode", default="cloud")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--allow-no-swanlab", action="store_true")
    parser.add_argument("--seed", type=int, default=260525)
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    swan = _init_swanlab(args)
    rows, collect_metrics = collect_dataset(args)
    if not rows:
        raise RuntimeError("no rows collected")
    print(json.dumps({"collect": collect_metrics}, ensure_ascii=True), flush=True)
    _log_swanlab(swan, {"collect": collect_metrics}, 0)

    shards = [rows[i :: args.trainers] for i in range(args.trainers)]
    resume_state: dict[str, torch.Tensor] | None = None
    resume_update = 0
    resume_model_cfg: dict[str, int] | None = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_state = {key: value.detach().cpu() for key, value in resume_payload["state_dict"].items()}
        resume_update = int(resume_payload.get("update", 0))
        if isinstance(resume_payload.get("model"), dict):
            resume_model_cfg = {key: int(value) for key, value in resume_payload["model"].items()}
        print(
            json.dumps(
                {"event": "resume_loaded", "path": args.resume, "resume_update": resume_update, "model": resume_model_cfg},
                ensure_ascii=True,
            ),
            flush=True,
        )

    model_cfg = resume_model_cfg or {
        "hidden": args.hidden,
        "heads": args.heads,
        "layers": args.layers,
        "ship_buckets": len(SHIP_BUCKET_MULTIPLIERS),
        "action_slots": ACTION_SLOTS,
    }
    init_model = TinyPolicyValueNet(**model_cfg)
    state = resume_state or _cpu_state_dict(init_model)
    actors = [
        BCTrainEvalActor.options(num_cpus=args.cpus_per_trainer, num_gpus=args.gpus_per_trainer).remote(
            shard,
            model_cfg,
            args.batch_size,
            args.val_frac,
            args.lr,
            args.weight_decay,
            args.max_grad_norm,
            args.launch_pos_weight,
            args.target_loss_weight,
            args.ship_loss_weight,
            args.critical_action_weight,
            args.seed + i,
        )
        for i, shard in enumerate(shards)
        if shard
    ]
    eval_actors = [
        BCEvalActor.options(num_cpus=args.eval_cpus_per_actor, num_gpus=args.gpus_per_eval_actor).remote(model_cfg)
        for _ in range(args.eval_actors)
    ]
    best_nonloss = -1.0
    best_metrics: dict[str, Any] = {}
    pending_eval_refs: dict[Any, int] = {}
    pending_eval_states: dict[int, dict[str, torch.Tensor]] = {}
    eval_parts_by_epoch: dict[int, list[dict[str, float]]] = {}
    eval_expected_parts: dict[int, int] = {}
    eval_completion_count = 0
    stop_after_epoch: int | None = None
    eval_actor_cursor = 0

    def maybe_finish_eval(eval_epoch: int) -> None:
        nonlocal best_nonloss, best_metrics, eval_completion_count, stop_after_epoch
        expected = eval_expected_parts.get(eval_epoch, 1)
        if len(eval_parts_by_epoch.get(eval_epoch, [])) < expected:
            return
        eval_metrics = _merge_eval(eval_parts_by_epoch.pop(eval_epoch))
        eval_expected_parts.pop(eval_epoch, None)
        eval_state = pending_eval_states.pop(eval_epoch, None)
        eval_completion_count += 1
        summary = {
            "event": "async_eval_done",
            "epoch": eval_epoch,
            "eval_completion": eval_completion_count,
            "eval_vs_regular": eval_metrics,
        }
        print(json.dumps(summary, ensure_ascii=True), flush=True)
        _log_swanlab(swan, summary, eval_epoch)
        if eval_state is not None:
            save_checkpoint(Path(args.out), eval_state, model_cfg, eval_epoch, {"collect": collect_metrics, **summary})
            if eval_metrics["nonloss"] > best_nonloss:
                best_nonloss = eval_metrics["nonloss"]
                best_metrics = dict(summary)
                save_checkpoint(Path(args.best_out), eval_state, model_cfg, eval_epoch, {"collect": collect_metrics, **summary, "best": True})
                save_checkpoint(
                    Path(args.out).with_name(f"regular_bc_ray_best_e{eval_epoch:04d}.pt"),
                    eval_state,
                    model_cfg,
                    eval_epoch,
                    {"collect": collect_metrics, **summary, "best": True},
                )
        if eval_metrics["nonloss"] >= args.target_nonloss and eval_metrics["winrate"] >= args.target_winrate:
            stop_after_epoch = eval_epoch
            print(json.dumps({"event": "target_reached", "epoch": eval_epoch, "metrics": eval_metrics}, ensure_ascii=True), flush=True)

    def drain_ready_evals() -> None:
        nonlocal best_nonloss, best_metrics, eval_completion_count, stop_after_epoch
        if not pending_eval_refs:
            return
        ready, _not_ready = ray.wait(list(pending_eval_refs), num_returns=len(pending_eval_refs), timeout=0.0)
        for ref in ready:
            eval_epoch = pending_eval_refs.pop(ref)
            part = ray.get(ref)
            eval_parts_by_epoch.setdefault(eval_epoch, []).append(part)
            maybe_finish_eval(eval_epoch)

    eval_actor_cursor = 0
    start_epoch = resume_update + 1 if resume_update > 0 else 1
    for epoch in range(start_epoch, args.epochs + 1):
        drain_ready_evals()
        if stop_after_epoch is not None:
            break
        state_refs = [actor.train_epoch.remote(state, epoch) for actor in actors]
        results = ray.get(state_refs)
        state = _average_states([item[0] for item in results])
        train_metrics = _weighted_mean([item[1] for item in results])
        summary: dict[str, Any] = {"epoch": epoch, "train": train_metrics}

        should_eval = (
            epoch % args.eval_interval == 0
            and float(train_metrics.get("val_launch_recall", 0.0)) >= args.eval_min_launch_recall
            and len(eval_expected_parts) < args.max_pending_evals
        )
        if should_eval:
            actors_for_eval = min(len(eval_actors), max(1, args.eval_games))
            selected_eval_actors = [eval_actors[(eval_actor_cursor + i) % len(eval_actors)] for i in range(actors_for_eval)]
            eval_actor_cursor = (eval_actor_cursor + actors_for_eval) % len(eval_actors)
            games_parts = [args.eval_games // actors_for_eval] * actors_for_eval
            for i in range(args.eval_games % actors_for_eval):
                games_parts[i] += 1
            eval_state = {key: value.detach().cpu() for key, value in state.items()}
            eval_refs = []
            for i, (actor, games) in enumerate(zip(selected_eval_actors, games_parts, strict=True)):
                if games <= 0:
                    continue
                eval_refs.append(actor.eval_vs_regular.remote(
                    state,
                    games,
                    args.seed + 10_000 + epoch * 1_000 + i * 100,
                    args.eval_stochastic,
                    args.launch_bias,
                    args.ship_bias,
                    args.launch_temperature,
                ))
            pending_eval_states[epoch] = eval_state
            for ref in eval_refs:
                pending_eval_refs[ref] = epoch
            eval_expected_parts[epoch] = len(eval_refs)
            summary["async_eval_submitted"] = {
                "parts": float(len(eval_refs)),
                "games": float(args.eval_games),
                "pending_evals": float(len(eval_expected_parts)),
            }
            print(json.dumps(summary, ensure_ascii=True), flush=True)
            _log_swanlab(swan, summary, epoch)
        else:
            if epoch % args.eval_interval == 0:
                if float(train_metrics.get("val_launch_recall", 0.0)) < args.eval_min_launch_recall:
                    summary["eval_skipped"] = {
                        "reason": "val_launch_recall_below_threshold",
                        "val_launch_recall": float(train_metrics.get("val_launch_recall", 0.0)),
                        "threshold": args.eval_min_launch_recall,
                    }
                else:
                    summary["eval_skipped"] = {
                        "reason": "max_pending_evals",
                        "pending_evals": float(len(eval_expected_parts)),
                        "max_pending_evals": float(args.max_pending_evals),
                    }
            print(json.dumps(summary, ensure_ascii=True), flush=True)
            _log_swanlab(swan, summary, epoch)
        drain_ready_evals()

    while pending_eval_refs and stop_after_epoch is None:
        done, _not_ready = ray.wait(list(pending_eval_refs), num_returns=1, timeout=30.0)
        if not done:
            print(
                json.dumps(
                    {"event": "async_eval_wait", "pending_parts": len(pending_eval_refs), "pending_evals": len(eval_expected_parts)},
                    ensure_ascii=True,
                ),
                flush=True,
            )
            continue
        for ref in done:
            eval_epoch = pending_eval_refs.pop(ref)
            part = ray.get(ref)
            eval_parts_by_epoch.setdefault(eval_epoch, []).append(part)
            maybe_finish_eval(eval_epoch)
    print(json.dumps({"event": "done", "best_nonloss": best_nonloss, "best": best_metrics}, ensure_ascii=True), flush=True)
    if swan is not None:
        swan.finish()


if __name__ == "__main__":
    main()
