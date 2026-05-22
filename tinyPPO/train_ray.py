from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

from tinyPPO.agents import ACTION_SLOTS, MAX_ACTIONS_PER_SOURCE_SAFETY, nearest_planet_agent, random_policy_agent
from tinyPPO.features import score
from tinyPPO.model import TinyPolicyValueNet
from tinyPPO.ppo import PPOConfig, PPOUpdater, RolloutBuffer
from tinyPPO.train import (
    add_weighted_rows,
    collect_episode,
    init_swanlab_or_none,
    log_swanlab,
    replay_rows_for_update,
    save_checkpoint,
    sample_policy_action,
    set_seed,
)
from training2 import make_fast_orbit_wars


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for headless runs.
    tqdm = None


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _merge_eval_parts(parts: list[dict]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for key in ("eval_vs_random", "eval_vs_nearest"):
        games = wins = losses = draws = reward_sum = 0.0
        for part in parts:
            data = part[key]
            g = float(data.get("games", 0.0))
            games += g
            wins += float(data.get("wins", 0.0))
            losses += float(data.get("losses", 0.0))
            draws += float(data.get("draws", 0.0))
            reward_sum += float(data.get("mean_reward", 0.0)) * g
        merged[key] = {
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winrate": wins / max(1.0, games),
            "nonloss": (wins + draws) / max(1.0, games),
            "mean_reward": reward_sum / max(1.0, games),
        }
    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="tinyPPO/runs/ray_random_2p_v1")
    parser.add_argument("--ray-address", default=None, help="Ray address. Defaults to RAY_ADDRESS env; use 'auto' for an existing local cluster.")
    parser.add_argument("--ray-runtime-env-id", default="", help="Optional env marker to force fresh Ray runtime env on workers.")
    parser.add_argument("--ray-working-dir", default="", help="Optional Ray working_dir, matching training2 style for multi-node clusters.")
    parser.add_argument("--gpu-ids", default="", help="Comma list to expose before ray.init, e.g. local '1,2,3,4,5,6,7' or remote '0,1,2,3,4,5,6,7,8'.")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0, help="0 means len(gpu_ids) * workers_per_gpu, or Ray's visible GPU count * workers_per_gpu.")
    parser.add_argument("--cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--gpus-per-worker", type=float, default=1.0 / 3.0)
    parser.add_argument("--learner-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes-per-worker", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--opponent-mode", choices=["random", "self", "mix"], default="random")
    parser.add_argument("--curriculum", action="store_true", help="Train vs random first, then switch training opponent to latest checkpoint.")
    parser.add_argument("--random-winrate-threshold", type=float, default=0.90)
    parser.add_argument("--selfplay-entropy-coef", type=float, default=0.04)
    parser.add_argument("--latest-opponent-stochastic", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-first", action="store_true")
    parser.add_argument("--sync-eval", action="store_true", help="Block training during eval instead of running Ray eval actors asynchronously.")
    parser.add_argument("--eval-workers", type=int, default=3)
    parser.add_argument("--gpus-per-eval-worker", type=float, default=1.0 / 3.0)
    parser.add_argument("--eval-gpu-ids", default="", help="Comma list forced inside eval actors, e.g. '0'. Use with --gpus-per-eval-worker 0 to keep eval on a GPU excluded from rollout Ray resources.")
    parser.add_argument("--eval-node-ip", default="", help="Pin async eval actors to this Ray node IP. Defaults to the driver node.")
    parser.add_argument("--cpus-per-eval-worker", type=float, default=1.0)
    parser.add_argument("--stop-eval-confirmations", type=int, default=2)
    parser.add_argument("--stop-winrate", type=float, default=0.55)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--action-slots", type=int, default=ACTION_SLOTS)
    parser.add_argument("--max-actions-per-source-safety", type=int, default=MAX_ACTIONS_PER_SOURCE_SAFETY)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--replay-updates", type=int, default=2, help="Keep this many previous update batches for age-decayed PPO replay. 0 disables replay.")
    parser.add_argument("--replay-ratio", type=float, default=0.25, help="Replay samples as a fraction of fresh rollout samples.")
    parser.add_argument("--replay-age-decay", type=float, default=0.50, help="Per-update replay loss weight decay.")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--swanlab-project", default="orbit-wars")
    parser.add_argument("--swanlab-experiment", default="tinyPPO-ray")
    parser.add_argument("--swanlab-mode", default="cloud")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--allow-no-swanlab", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    import ray

    runtime_env: dict = {}
    runtime_env_vars = {
        "PYTHONPATH": ".",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "KAGGLE_ENGINES_LOG_LEVEL": "0",
    }
    if args.ray_runtime_env_id:
        runtime_env_vars["TINYPPO_RUNTIME_ENV_ID"] = str(args.ray_runtime_env_id)
    runtime_env["env_vars"] = runtime_env_vars
    if args.ray_working_dir:
        runtime_env["working_dir"] = str(Path(args.ray_working_dir).resolve())
    ray_address = args.ray_address or os.environ.get("RAY_ADDRESS")
    ray.init(address=ray_address, ignore_reinit_error=True, runtime_env=runtime_env)
    eval_node_ip = args.eval_node_ip
    if not eval_node_ip:
        eval_node_ip = ray.util.get_node_ip_address()

    visible_gpu_count = len([x for x in args.gpu_ids.split(",") if x.strip()]) if args.gpu_ids else int(ray.cluster_resources().get("GPU", 0))
    eval_gpu_reserve = 0.0 if args.sync_eval else max(0, args.eval_workers) * max(0.0, args.gpus_per_eval_worker)
    rollout_gpu_budget = max(args.gpus_per_worker, float(visible_gpu_count) - eval_gpu_reserve)
    num_workers = args.num_workers or max(1, int(rollout_gpu_budget / max(args.gpus_per_worker, 1e-6)))

    @ray.remote(num_cpus=args.cpus_per_worker, num_gpus=args.gpus_per_worker)
    class RolloutWorker:
        def __init__(self, model_cfg: dict, episode_steps: int, opponent_mode: str, use_numba: bool):
            torch.set_num_threads(1)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = TinyPolicyValueNet(**model_cfg).to(self.device)
            self.opponent_model = TinyPolicyValueNet(**model_cfg).to(self.device)
            self.episode_steps = episode_steps
            self.opponent_mode = opponent_mode
            self.use_numba = use_numba

        def set_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
            self.model.load_state_dict(state_dict)
            self.model.eval()

        def set_opponent_weights(self, state_dict: dict[str, torch.Tensor] | None) -> TinyPolicyValueNet | None:
            if state_dict is None:
                return None
            self.opponent_model.load_state_dict(state_dict)
            self.opponent_model.eval()
            return self.opponent_model

        def collect(
            self,
            state_dict: dict[str, torch.Tensor],
            opponent_state_dict: dict[str, torch.Tensor] | None,
            train_mode: str,
            latest_opponent_deterministic: bool,
            max_actions_per_source: int,
            seed_base: int,
            episodes: int,
        ) -> tuple[list[dict], dict[str, float]]:
            self.set_weights(state_dict)
            opponent_model = self.set_opponent_weights(opponent_state_dict)
            rows: list[dict] = []
            metrics: list[dict[str, float]] = []
            for i in range(episodes):
                ep_rows, ep_metrics = collect_episode(
                    self.model,
                    seed_base + i,
                    self.device,
                    self.episode_steps,
                    train_mode,
                    self.use_numba,
                    opponent_model=opponent_model if train_mode == "latest" else None,
                    opponent_deterministic=latest_opponent_deterministic,
                    max_actions_per_source=max_actions_per_source,
                )
                rows.extend(ep_rows)
                metrics.append(ep_metrics)
            return rows, {
                "episodes": float(episodes),
                "samples": float(len(rows)),
                "rollout_result_p0": float(np.mean([m["result_p0"] for m in metrics])) if metrics else 0.0,
                "mean_launches": float(np.mean([m["mean_launches"] for m in metrics])) if metrics else 0.0,
                "self_opponent_frac": float(np.mean([m["opponent_self"] for m in metrics])) if metrics else 0.0,
                "opponent_latest": float(np.mean([m["opponent_latest"] for m in metrics])) if metrics else 0.0,
            }

    @ray.remote(num_cpus=args.cpus_per_eval_worker, num_gpus=args.gpus_per_eval_worker)
    class EvalWorker:
        def __init__(self, model_cfg: dict, episode_steps: int, use_numba: bool, max_actions_per_source: int, eval_gpu_ids: str = ""):
            torch.set_num_threads(1)
            if eval_gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu_ids)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = TinyPolicyValueNet(**model_cfg).to(self.device)
            self.episode_steps = episode_steps
            self.use_numba = use_numba
            self.max_actions_per_source = max_actions_per_source

        def set_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
            self.model.load_state_dict(state_dict)
            self.model.eval()

        def model_agent(self, obs: dict, configuration=None) -> list[list]:
            actions, _row = sample_policy_action(
                self.model,
                obs,
                self.device,
                deterministic=True,
                max_actions_per_source=self.max_actions_per_source,
            )
            return actions

        def _run_matchups(self, opponent_name: str, games: int, seed: int) -> dict[str, float]:
            opponent = random_policy_agent if opponent_name == "random" else nearest_planet_agent
            wins = losses = draws = 0
            reward_sum = 0.0
            for i in range(games):
                model_seat = i % 2
                agents = [opponent, opponent]
                agents[model_seat] = self.model_agent
                env = make_fast_orbit_wars(
                    {"episodeSteps": self.episode_steps, "seed": seed + i},
                    keep_history=False,
                    use_numba=self.use_numba,
                )
                env.run(agents)
                obs = env.steps[-1][model_seat]["observation"]
                model_score = score(obs, model_seat)
                other_score = score(obs, 1 - model_seat)
                if model_score > other_score:
                    wins += 1
                    reward_sum += 1.0
                elif model_score < other_score:
                    losses += 1
                    reward_sum -= 1.0
                else:
                    draws += 1
            return {
                "games": float(games),
                "wins": float(wins),
                "losses": float(losses),
                "draws": float(draws),
                "winrate": wins / max(1, games),
                "nonloss": (wins + draws) / max(1, games),
                "mean_reward": reward_sum / max(1, games),
            }

        def evaluate(self, state_dict: dict[str, torch.Tensor], update: int, games: int, seed: int) -> dict:
            self.set_weights(state_dict)
            return {
                "update": update,
                "eval_vs_random": self._run_matchups("random", games, seed),
                "eval_vs_nearest": self._run_matchups("nearest", games, seed + 100_000),
            }

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    swan = init_swanlab_or_none(args, "ray")
    model_cfg = {"hidden": args.hidden, "heads": args.heads, "layers": args.layers, "ship_buckets": 0, "action_slots": args.action_slots}
    learner_device = torch.device(args.learner_device)
    model = TinyPolicyValueNet(**model_cfg).to(learner_device)
    updater = PPOUpdater(model, PPOConfig(learning_rate=args.lr, entropy_coef=args.entropy_coef), device=str(learner_device))
    workers = [RolloutWorker.remote(model_cfg, args.episode_steps, args.opponent_mode, not args.no_numba) for _ in range(num_workers)]
    eval_actor_options = {"resources": {f"node:{eval_node_ip}": 0.001}}
    eval_workers = [] if args.sync_eval else [
        EvalWorker.options(**eval_actor_options).remote(model_cfg, args.episode_steps, not args.no_numba, args.max_actions_per_source_safety)
        if not args.eval_gpu_ids
        else EvalWorker.options(**eval_actor_options).remote(model_cfg, args.episode_steps, not args.no_numba, args.max_actions_per_source_safety, args.eval_gpu_ids)
        for _ in range(max(1, args.eval_workers))
    ]
    print(
        json.dumps(
            {
                "ray_workers": num_workers,
                "eval_workers": len(eval_workers),
                "visible_gpus": visible_gpu_count,
                "gpus_per_worker": args.gpus_per_worker,
                "gpus_per_eval_worker": args.gpus_per_eval_worker,
                "eval_node_ip": eval_node_ip,
            },
            ensure_ascii=False,
        )
    )

    best_winrate = -1.0
    phase = "random" if args.curriculum else args.opponent_mode
    latest_opponent_state = cpu_state_dict(model)
    replay_batches: deque[tuple[int, list[dict]]] = deque(maxlen=max(0, args.replay_updates))
    pending_eval_refs: dict[object, int] = {}
    eval_parts: dict[int, list[dict]] = {}
    eval_expected_parts: dict[int, int] = {}
    nearest_stop_streak = 0
    stop_after_update: int | None = None
    for update in range(1, args.updates + 1):
        ready_refs = []
        if pending_eval_refs:
            ready_refs, _not_ready = ray.wait(list(pending_eval_refs), num_returns=len(pending_eval_refs), timeout=0.0)
        for ref in ready_refs:
            eval_update = pending_eval_refs.pop(ref)
            part = ray.get(ref)
            eval_parts.setdefault(eval_update, []).append(part)
            if len(eval_parts[eval_update]) >= eval_expected_parts.get(eval_update, 1):
                merged_eval = _merge_eval_parts(eval_parts.pop(eval_update))
                eval_expected_parts.pop(eval_update, None)
                eval_summary = {"update": eval_update, **merged_eval, "phase": phase, "event": "async_eval_done"}
                print(json.dumps(eval_summary, ensure_ascii=False), flush=True)
                log_swanlab(swan, eval_summary, eval_update)
                if merged_eval["eval_vs_nearest"]["winrate"] > best_winrate:
                    best_winrate = merged_eval["eval_vs_nearest"]["winrate"]
                    save_checkpoint(out_dir / "best.pt", model, args, update, {"async_eval_update": eval_update, **merged_eval})
                if args.curriculum and phase == "random" and merged_eval["eval_vs_random"]["winrate"] >= args.random_winrate_threshold:
                    phase = "latest"
                    updater.cfg.entropy_coef = max(updater.cfg.entropy_coef, args.selfplay_entropy_coef)
                    latest_opponent_state = cpu_state_dict(model)
                    print(
                        json.dumps(
                            {
                                "event": "phase_transition",
                                "update": update,
                                "eval_update": eval_update,
                                "to": phase,
                                "reason": f"eval_vs_random winrate {merged_eval['eval_vs_random']['winrate']:.3f} >= {args.random_winrate_threshold:.3f}",
                                "entropy_coef": updater.cfg.entropy_coef,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if merged_eval["eval_vs_nearest"]["winrate"] >= args.stop_winrate:
                    nearest_stop_streak += 1
                else:
                    nearest_stop_streak = 0
                if nearest_stop_streak >= args.stop_eval_confirmations:
                    stop_after_update = update

        if stop_after_update is not None:
            print(json.dumps({"event": "stop_confirmed", "update": update, "confirmations": nearest_stop_streak}, ensure_ascii=False), flush=True)
            break

        print(json.dumps({"event": "collect_start", "update": update, "phase": phase, "workers": num_workers}, ensure_ascii=False), flush=True)
        weights_ref = ray.put(cpu_state_dict(model))
        opponent_ref = ray.put(latest_opponent_state) if phase == "latest" else ray.put(None)
        futures = [
            worker.collect.remote(
                weights_ref,
                opponent_ref,
                phase,
                not args.latest_opponent_stochastic,
                args.max_actions_per_source_safety,
                args.seed + update * 1_000_000 + wid * 10_000,
                args.episodes_per_worker,
            )
            for wid, worker in enumerate(workers)
        ]
        results = []
        pending = list(futures)
        pbar = tqdm(total=len(pending), desc=f"collect u{update} {phase}", dynamic_ncols=True) if tqdm is not None else None
        while pending:
            done, pending = ray.wait(pending, num_returns=1, timeout=10.0)
            if not done:
                print(
                    json.dumps(
                        {"event": "collect_heartbeat", "update": update, "done": len(results), "pending": len(pending)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            rows, metrics = ray.get(done[0])
            results.append((rows, metrics))
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(samples=sum(len(r) for r, _m in results), refresh=False)
            else:
                print(
                    json.dumps(
                        {"event": "worker_done", "update": update, "done": len(results), "total": len(futures), "samples": len(rows)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if pbar is not None:
            pbar.close()
        print(json.dumps({"event": "collect_done", "update": update, "worker_results": len(results)}, ensure_ascii=False), flush=True)

        buffer = RolloutBuffer()
        worker_metrics = []
        fresh_rows_for_replay: list[dict] = []
        for rows, metrics in results:
            worker_metrics.append(metrics)
            add_weighted_rows(buffer, rows, 1.0)
            fresh_rows_for_replay.extend(dict(row) for row in rows)

        replay_rows, replay_metrics = replay_rows_for_update(
            replay_batches,
            update,
            len(buffer),
            args.replay_ratio,
            args.replay_age_decay,
        )
        for row in replay_rows:
            buffer.add(**row)
        if args.replay_updates > 0:
            replay_batches.append((update, fresh_rows_for_replay))

        train_metrics = updater.update(buffer)
        print(json.dumps({"event": "update_done", "update": update, "samples": len(buffer), "train": train_metrics}, ensure_ascii=False), flush=True)
        summary = {
            "update": update,
            "workers": num_workers,
            "episodes": num_workers * args.episodes_per_worker,
            "samples": len(buffer),
            "train": train_metrics,
            "rollout_result_p0": float(np.mean([m["rollout_result_p0"] for m in worker_metrics])) if worker_metrics else 0.0,
            "mean_launches": float(np.mean([m["mean_launches"] for m in worker_metrics])) if worker_metrics else 0.0,
            "self_opponent_frac": float(np.mean([m["self_opponent_frac"] for m in worker_metrics])) if worker_metrics else 0.0,
            "latest_opponent_frac": float(np.mean([m["opponent_latest"] for m in worker_metrics])) if worker_metrics else 0.0,
            "phase": phase,
            "replay": replay_metrics,
        }

        should_eval = update % args.eval_interval == 0 or (args.eval_first and update == 1)
        if should_eval:
            ckpt_path = out_dir / "latest.pt"
            save_checkpoint(ckpt_path, model, args, update, summary)
            eval_state_ref = ray.put(cpu_state_dict(model))
            if args.sync_eval:
                print(json.dumps({"event": "sync_eval_start", "update": update, "games_each": args.eval_games}, ensure_ascii=False), flush=True)
                temp_worker = EvalWorker.options(
                    num_gpus=args.gpus_per_eval_worker,
                    num_cpus=args.cpus_per_eval_worker,
                    resources={f"node:{eval_node_ip}": 0.001},
                ).remote(
                    model_cfg,
                    args.episode_steps,
                    not args.no_numba,
                    args.max_actions_per_source_safety,
                )
                part = ray.get(temp_worker.evaluate.remote(eval_state_ref, update, args.eval_games, args.seed + 300_000 + update * 1_000))
                merged_eval = _merge_eval_parts([part])
                summary.update(merged_eval)
            else:
                games_left = args.eval_games
                parts = []
                for i, worker in enumerate(eval_workers):
                    shard_games = games_left // (len(eval_workers) - i)
                    games_left -= shard_games
                    if shard_games <= 0:
                        continue
                    parts.append(
                        worker.evaluate.remote(
                            eval_state_ref,
                            update,
                            shard_games,
                            args.seed + 300_000 + update * 10_000 + i * 1_000,
                        )
                    )
                for ref in parts:
                    pending_eval_refs[ref] = update
                eval_expected_parts[update] = len(parts)
                print(
                    json.dumps(
                        {"event": "async_eval_submitted", "update": update, "parts": len(parts), "games_each_total": args.eval_games},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        save_checkpoint(out_dir / "latest.pt", model, args, update, summary)
        if phase == "latest":
            latest_opponent_state = cpu_state_dict(model)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        log_swanlab(swan, summary, update)
        print(json.dumps(summary, ensure_ascii=False))

    if swan is not None:
        swan.finish()


if __name__ == "__main__":
    main()
