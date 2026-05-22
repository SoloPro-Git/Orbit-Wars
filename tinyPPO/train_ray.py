from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from tinyPPO.agents import MAX_ACTIONS_PER_SOURCE_SAFETY, SHIP_FRACTIONS, TinyPPOAgent, nearest_planet_agent, random_policy_agent
from tinyPPO.eval import run_matchups
from tinyPPO.model import TinyPolicyValueNet
from tinyPPO.ppo import PPOConfig, PPOUpdater, RolloutBuffer
from tinyPPO.train import collect_episode, save_checkpoint, set_seed


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for headless runs.
    tqdm = None


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="tinyPPO/runs/ray_random_2p_v1")
    parser.add_argument("--ray-address", default=None, help="Use 'auto' for an existing Ray cluster.")
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
    parser.add_argument("--stop-winrate", type=float, default=0.55)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--max-actions-per-source-safety", type=int, default=MAX_ACTIONS_PER_SOURCE_SAFETY)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--no-numba", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    import ray

    ray.init(address=args.ray_address, ignore_reinit_error=True, runtime_env={})

    visible_gpu_count = len([x for x in args.gpu_ids.split(",") if x.strip()]) if args.gpu_ids else int(ray.cluster_resources().get("GPU", 0))
    num_workers = args.num_workers or max(1, visible_gpu_count * args.workers_per_gpu)

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

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    model_cfg = {"hidden": args.hidden, "heads": args.heads, "layers": args.layers, "ship_buckets": len(SHIP_FRACTIONS)}
    learner_device = torch.device(args.learner_device)
    model = TinyPolicyValueNet(**model_cfg).to(learner_device)
    updater = PPOUpdater(model, PPOConfig(learning_rate=args.lr, entropy_coef=args.entropy_coef), device=str(learner_device))
    workers = [RolloutWorker.remote(model_cfg, args.episode_steps, args.opponent_mode, not args.no_numba) for _ in range(num_workers)]
    print(json.dumps({"ray_workers": num_workers, "visible_gpus": visible_gpu_count, "gpus_per_worker": args.gpus_per_worker}, ensure_ascii=False))

    best_winrate = -1.0
    phase = "random" if args.curriculum else args.opponent_mode
    latest_opponent_state = cpu_state_dict(model)
    for update in range(1, args.updates + 1):
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
        for rows, metrics in results:
            worker_metrics.append(metrics)
            for row in rows:
                buffer.add(**row)

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
        }

        should_eval = update % args.eval_interval == 0 or (args.eval_first and update == 1)
        if should_eval:
            ckpt_path = out_dir / "latest.pt"
            save_checkpoint(ckpt_path, model, args, update, summary)
            print(json.dumps({"event": "eval_start", "update": update, "games_each": args.eval_games}, ensure_ascii=False), flush=True)
            eval_random = run_matchups(
                lambda: TinyPPOAgent(ckpt_path, device=str(learner_device), deterministic=True),
                lambda: random_policy_agent,
                games=args.eval_games,
                seed=args.seed + 300_000 + update * 1_000,
                episode_steps=args.episode_steps,
                use_numba=not args.no_numba,
                progress=True,
                desc=f"eval-random u{update}",
            )
            print(json.dumps({"event": "eval_random_done", "update": update, "eval_vs_random": eval_random}, ensure_ascii=False), flush=True)
            eval_result = run_matchups(
                lambda: TinyPPOAgent(ckpt_path, device=str(learner_device), deterministic=True),
                lambda: nearest_planet_agent,
                games=args.eval_games,
                seed=args.seed + 500_000 + update * 1_000,
                episode_steps=args.episode_steps,
                use_numba=not args.no_numba,
                progress=True,
                desc=f"eval-nearest u{update}",
            )
            print(json.dumps({"event": "eval_nearest_done", "update": update, "eval_vs_nearest": eval_result}, ensure_ascii=False), flush=True)
            summary["eval_vs_random"] = eval_random
            summary["eval_vs_nearest"] = eval_result
            if eval_result["winrate"] > best_winrate:
                best_winrate = eval_result["winrate"]
                save_checkpoint(out_dir / "best.pt", model, args, update, summary)
            if args.curriculum and phase == "random" and eval_random["winrate"] >= args.random_winrate_threshold:
                phase = "latest"
                updater.cfg.entropy_coef = max(updater.cfg.entropy_coef, args.selfplay_entropy_coef)
                latest_opponent_state = cpu_state_dict(model)
                summary["phase_transition"] = {
                    "to": phase,
                    "reason": f"eval_vs_random winrate {eval_random['winrate']:.3f} >= {args.random_winrate_threshold:.3f}",
                    "entropy_coef": updater.cfg.entropy_coef,
                }
            if eval_result["winrate"] >= args.stop_winrate:
                summary["stop_reason"] = f"eval winrate {eval_result['winrate']:.3f} >= {args.stop_winrate:.3f}"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n")
                print(json.dumps(summary, ensure_ascii=False))
                break

        save_checkpoint(out_dir / "latest.pt", model, args, update, summary)
        if phase == "latest":
            latest_opponent_state = cpu_state_dict(model)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
