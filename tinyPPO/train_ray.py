from __future__ import annotations

import argparse
import json
import os
import shutil
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


def save_checkpoint_state(path: Path, state_dict: dict[str, torch.Tensor], model_cfg: dict, update: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "model": model_cfg,
            "update": update,
            "metrics": metrics,
        },
        path,
    )


def _eval_score(metrics: dict) -> tuple[float, float, float]:
    nearest = metrics.get("eval_vs_nearest", {})
    random_eval = metrics.get("eval_vs_random", {})
    return (
        float(nearest.get("winrate", -1.0)),
        float(random_eval.get("winrate", -1.0)),
        float(nearest.get("mean_reward", -999.0)),
    )


def update_top_checkpoints(
    out_dir: Path,
    state_dict: dict[str, torch.Tensor],
    model_cfg: dict,
    eval_update: int,
    metrics: dict,
    top_k: int,
) -> list[dict]:
    if top_k <= 0:
        return []
    top_dir = out_dir / "best_top"
    top_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = top_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = []

    nearest_wr, random_wr, nearest_reward = _eval_score(metrics)
    ckpt_name = f"eval_u{eval_update:06d}_nearest_{nearest_wr:.3f}_random_{random_wr:.3f}.pt"
    ckpt_path = top_dir / ckpt_name
    save_checkpoint_state(ckpt_path, state_dict, model_cfg, eval_update, metrics)
    manifest = [entry for entry in manifest if entry.get("path") != ckpt_name]
    manifest.append(
        {
            "path": ckpt_name,
            "update": int(eval_update),
            "nearest_winrate": nearest_wr,
            "random_winrate": random_wr,
            "nearest_mean_reward": nearest_reward,
        }
    )
    manifest.sort(key=lambda item: (item["nearest_winrate"], item["random_winrate"], item["nearest_mean_reward"], item["update"]), reverse=True)
    keep = manifest[:top_k]
    keep_names = {entry["path"] for entry in keep}
    for entry in manifest[top_k:]:
        stale = top_dir / entry["path"]
        if stale.exists():
            stale.unlink()
    for rank, entry in enumerate(keep, start=1):
        src = top_dir / entry["path"]
        if src.exists():
            shutil.copyfile(src, top_dir / f"rank{rank}.pt")
    manifest_path.write_text(json.dumps(keep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for rank_file in top_dir.glob("rank*.pt"):
        if rank_file.name.startswith("rank") and rank_file.name[4:-3].isdigit():
            rank = int(rank_file.name[4:-3])
            if rank > len(keep):
                rank_file.unlink()
    return keep


def _merge_eval_parts(parts: list[dict]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    keys = sorted({key for part in parts for key in part if key.startswith("eval_")})
    for key in keys:
        games = wins = losses = draws = reward_sum = 0.0
        for part in parts:
            data = part.get(key)
            if not isinstance(data, dict):
                continue
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
    parser.add_argument("--resume-checkpoint", default="", help="Load model weights and continue from checkpoint update + 1.")
    parser.add_argument("--gpu-ids", default="", help="Comma list to expose before ray.init, e.g. local '1,2,3,4,5,6,7' or remote '0,1,2,3,4,5,6,7,8'.")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0, help="0 means len(gpu_ids) * workers_per_gpu, or Ray's visible GPU count * workers_per_gpu.")
    parser.add_argument("--cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--gpus-per-worker", type=float, default=1.0 / 3.0)
    parser.add_argument("--learner-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--episodes-per-worker", type=int, default=2, help="Maximum episodes assigned to one rollout worker in one update.")
    parser.add_argument("--episodes-per-update", type=int, default=42, help="Fixed fresh rollout episodes per PPO update. 0 restores num_workers * episodes_per_worker.")
    parser.add_argument("--collect-overassign-factor", type=float, default=1.0, help="Assign extra rollout capacity and stop after episodes-per-update to avoid stragglers.")
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--opponent-mode", choices=["random", "self", "mix"], default="random")
    parser.add_argument("--curriculum", action="store_true", help="Train vs random first, then switch training opponent to latest checkpoint.")
    parser.add_argument("--random-winrate-threshold", type=float, default=0.90)
    parser.add_argument("--selfplay-entropy-coef", type=float, default=0.04)
    parser.add_argument("--latest-opponent-stochastic", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-first", action="store_true")
    parser.add_argument("--eval-aggression", type=float, default=0.0, help="Eval-only bias applied to both launch and ship amount. Positive is more aggressive.")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save latest.pt every N updates. 0 disables periodic latest checkpoints.")
    parser.add_argument("--eval-launch-bias", type=float, default=0.0, help="Eval-only logit bias for launch vs no-launch.")
    parser.add_argument("--eval-ship-bias", type=float, default=0.0, help="Eval-only logit-space bias for ship fraction mean.")
    parser.add_argument("--eval-launch-temperature", type=float, default=1.0)
    parser.add_argument("--eval-stochastic", action="store_true")
    parser.add_argument("--eval-stochastic-compare", action="store_true", help="Also log stochastic eval beside the main deterministic eval.")
    parser.add_argument("--sync-eval", action="store_true", help="Block training during eval instead of running Ray eval actors asynchronously.")
    parser.add_argument("--eval-workers", type=int, default=3)
    parser.add_argument("--gpus-per-eval-worker", type=float, default=1.0 / 3.0)
    parser.add_argument("--eval-gpu-ids", default="", help="Comma list forced inside eval actors, e.g. '0'. Use with --gpus-per-eval-worker 0 to keep eval on a GPU excluded from rollout Ray resources.")
    parser.add_argument("--eval-node-ip", default="", help="Pin async eval actors to this Ray node IP. Defaults to the driver node.")
    parser.add_argument("--cpus-per-eval-worker", type=float, default=1.0)
    parser.add_argument("--stop-eval-confirmations", type=int, default=2)
    parser.add_argument("--stop-winrate", type=float, default=0.55)
    parser.add_argument("--top-k-checkpoints", type=int, default=5, help="Keep this many eval-ranked checkpoints under best_top/.")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--action-slots", type=int, default=ACTION_SLOTS)
    parser.add_argument("--max-actions-per-source-safety", type=int, default=MAX_ACTIONS_PER_SOURCE_SAFETY)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--min-ppo-epochs", type=int, default=4, help="Minimum PPO epochs per rollout before KL early stopping can trigger.")
    parser.add_argument("--ppo-epochs", type=int, default=8, help="Maximum PPO epochs per rollout; KL early stopping may stop earlier after min-ppo-epochs.")
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--target-kl", type=float, default=0.01, help="Stop PPO epochs early when mean approx KL exceeds 1.5x this value. 0 disables.")
    parser.add_argument("--replay-updates", type=int, default=2, help="Keep this many previous update batches for age-decayed PPO replay. 0 disables replay.")
    parser.add_argument("--replay-ratio", type=float, default=0.25, help="Replay samples as a fraction of fresh rollout samples.")
    parser.add_argument("--replay-age-decay", type=float, default=0.50, help="Per-update replay loss weight decay.")
    parser.add_argument("--win-replay-weight", type=float, default=1.0, help="Multiplier for replay rows from winning self-generated episodes.")
    parser.add_argument("--loss-replay-weight", type=float, default=1.0, help="Multiplier for replay rows from losing self-generated episodes.")
    parser.add_argument("--draw-replay-weight", type=float, default=1.0, help="Multiplier for replay rows from drawn self-generated episodes.")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--swanlab-project", default="orbit-wars")
    parser.add_argument("--swanlab-experiment", default="tinyPPO-ray")
    parser.add_argument("--swanlab-mode", default="cloud")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--allow-no-swanlab", action="store_true")
    return parser


def rollout_assignments(
    num_workers: int,
    episodes_per_worker: int,
    episodes_per_update: int,
    update: int,
    overassign_factor: float = 1.0,
) -> list[tuple[int, int]]:
    max_episodes = max(1, num_workers) * max(1, episodes_per_worker)
    target = max_episodes if episodes_per_update <= 0 else int(episodes_per_update)
    if target <= 0:
        return []
    if target > max_episodes:
        raise ValueError(
            f"--episodes-per-update={target} exceeds worker capacity {max_episodes}; "
            f"increase --episodes-per-worker or --num-workers"
        )
    assigned_target = target
    if episodes_per_update > 0:
        assigned_target = min(max_episodes, max(target, int(np.ceil(target * max(1.0, overassign_factor)))))
    base, remainder = divmod(assigned_target, max(1, num_workers))
    counts = [base for _ in range(num_workers)]
    offset = (max(1, update) - 1) % max(1, num_workers)
    for i in range(remainder):
        counts[(offset + i) % num_workers] += 1
    return [(idx, count) for idx, count in enumerate(counts) if count > 0]


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
        def __init__(
            self,
            model_cfg: dict,
            episode_steps: int,
            use_numba: bool,
            max_actions_per_source: int,
            eval_gpu_ids: str = "",
            launch_bias: float = 0.0,
            ship_bias: float = 0.0,
            launch_temperature: float = 1.0,
            deterministic: bool = True,
        ):
            torch.set_num_threads(1)
            if eval_gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_gpu_ids)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = TinyPolicyValueNet(**model_cfg).to(self.device)
            self.episode_steps = episode_steps
            self.use_numba = use_numba
            self.max_actions_per_source = max_actions_per_source
            self.launch_bias = float(launch_bias)
            self.ship_bias = float(ship_bias)
            self.launch_temperature = float(launch_temperature)
            self.deterministic = bool(deterministic)

        def set_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
            self.model.load_state_dict(state_dict)
            self.model.eval()

        def model_agent(self, obs: dict, configuration=None) -> list[list]:
            actions, _row = sample_policy_action(
                self.model,
                obs,
                self.device,
                deterministic=self.deterministic,
                max_actions_per_source=self.max_actions_per_source,
                launch_bias=self.launch_bias,
                ship_bias=self.ship_bias,
                launch_temperature=self.launch_temperature,
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

        def evaluate(self, state_dict: dict[str, torch.Tensor], update: int, games: int, seed: int, stochastic_compare: bool = False) -> dict:
            self.set_weights(state_dict)
            summary = {
                "update": update,
                "eval_vs_random": self._run_matchups("random", games, seed),
                "eval_vs_nearest": self._run_matchups("nearest", games, seed + 100_000),
            }
            if stochastic_compare:
                previous = self.deterministic
                self.deterministic = False
                try:
                    summary["eval_stochastic_vs_random"] = self._run_matchups("random", games, seed + 200_000)
                    summary["eval_stochastic_vs_nearest"] = self._run_matchups("nearest", games, seed + 300_000)
                finally:
                    self.deterministic = previous
            return summary

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    swan = init_swanlab_or_none(args, "ray")
    model_cfg = {"hidden": args.hidden, "heads": args.heads, "layers": args.layers, "ship_buckets": 0, "action_slots": args.action_slots}
    learner_device = torch.device(args.learner_device)
    model = TinyPolicyValueNet(**model_cfg).to(learner_device)
    updater = PPOUpdater(
        model,
        PPOConfig(
            learning_rate=args.lr,
            entropy_coef=args.entropy_coef,
            min_epochs=args.min_ppo_epochs,
            epochs=args.ppo_epochs,
            batch_size=args.ppo_batch_size,
            target_kl=args.target_kl,
        ),
        device=str(learner_device),
    )
    start_update = 1
    resume_metrics = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        payload = torch.load(resume_path, map_location=learner_device, weights_only=True)
        ckpt_cfg = payload.get("model", {})
        if ckpt_cfg and ckpt_cfg != model_cfg:
            raise ValueError(f"Resume checkpoint model config {ckpt_cfg} does not match requested config {model_cfg}")
        model.load_state_dict(payload["state_dict"])
        start_update = int(payload.get("update", 0)) + 1
        resume_metrics = payload.get("metrics")
        print(
            json.dumps(
                {"event": "resume_loaded", "path": str(resume_path), "checkpoint_update": start_update - 1, "start_update": start_update},
                ensure_ascii=False,
            ),
            flush=True,
        )
    workers = [RolloutWorker.remote(model_cfg, args.episode_steps, args.opponent_mode, not args.no_numba) for _ in range(num_workers)]
    eval_actor_options = {"resources": {f"node:{eval_node_ip}": 0.001}}
    eval_launch_bias = args.eval_launch_bias + args.eval_aggression
    eval_ship_bias = args.eval_ship_bias + args.eval_aggression
    eval_workers = [] if args.sync_eval else [
        EvalWorker.options(**eval_actor_options).remote(
            model_cfg,
            args.episode_steps,
            not args.no_numba,
            args.max_actions_per_source_safety,
            "",
            eval_launch_bias,
            eval_ship_bias,
            args.eval_launch_temperature,
            not args.eval_stochastic,
        )
        if not args.eval_gpu_ids
        else EvalWorker.options(**eval_actor_options).remote(
            model_cfg,
            args.episode_steps,
            not args.no_numba,
            args.max_actions_per_source_safety,
            args.eval_gpu_ids,
            eval_launch_bias,
            eval_ship_bias,
            args.eval_launch_temperature,
            not args.eval_stochastic,
        )
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
                "episodes_per_update": args.episodes_per_update,
                "episodes_per_worker_cap": args.episodes_per_worker,
                "eval_launch_bias": eval_launch_bias,
                "eval_ship_bias": eval_ship_bias,
                "eval_launch_temperature": args.eval_launch_temperature,
                "eval_stochastic": args.eval_stochastic,
                "eval_stochastic_compare": args.eval_stochastic_compare,
                "collect_overassign_factor": args.collect_overassign_factor,
            },
            ensure_ascii=False,
        )
    )

    best_winrate = -1.0
    phase = "random" if args.curriculum else args.opponent_mode
    if isinstance(resume_metrics, dict):
        best_winrate = max(best_winrate, float(resume_metrics.get("eval_vs_nearest", {}).get("winrate", -1.0)))
        phase = str(resume_metrics.get("phase", phase))
    latest_opponent_state = cpu_state_dict(model)
    replay_batches: deque[tuple[int, list[dict]]] = deque(maxlen=max(0, args.replay_updates))
    pending_eval_refs: dict[object, int] = {}
    pending_eval_states: dict[int, dict[str, torch.Tensor]] = {}
    eval_parts: dict[int, list[dict]] = {}
    eval_expected_parts: dict[int, int] = {}
    nearest_stop_streak = 0
    stop_after_update: int | None = None
    for update in range(start_update, args.updates + 1):
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
                best_state = pending_eval_states.get(eval_update)
                best_metrics = {"async_eval_update": eval_update, **merged_eval}
                if best_state is not None:
                    top_manifest = update_top_checkpoints(out_dir, best_state, model_cfg, eval_update, best_metrics, args.top_k_checkpoints)
                    print(
                        json.dumps(
                            {"event": "top_checkpoints_updated", "update": eval_update, "top_k": len(top_manifest), "best": top_manifest[0] if top_manifest else None},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if merged_eval["eval_vs_nearest"]["winrate"] > best_winrate:
                    best_winrate = merged_eval["eval_vs_nearest"]["winrate"]
                    if best_state is not None:
                        save_checkpoint_state(out_dir / "best.pt", best_state, model_cfg, eval_update, best_metrics)
                    else:
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
                pending_eval_states.pop(eval_update, None)

        if stop_after_update is not None:
            print(json.dumps({"event": "stop_confirmed", "update": update, "confirmations": nearest_stop_streak}, ensure_ascii=False), flush=True)
            break

        target_fresh_episodes = int(sum(episodes for _wid, episodes in rollout_assignments(num_workers, args.episodes_per_worker, args.episodes_per_update, update, 1.0)))
        assignments = rollout_assignments(
            num_workers,
            args.episodes_per_worker,
            args.episodes_per_update,
            update,
            args.collect_overassign_factor,
        )
        assigned_episodes = int(sum(episodes for _wid, episodes in assignments))
        print(
            json.dumps(
                {
                    "event": "collect_start",
                    "update": update,
                    "phase": phase,
                    "workers": len(assignments),
                    "fresh_episodes_target": target_fresh_episodes,
                    "assigned_episodes": assigned_episodes,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        weights_ref = ray.put(cpu_state_dict(model))
        opponent_ref = ray.put(latest_opponent_state) if phase == "latest" else ray.put(None)
        futures = []
        future_to_worker: dict[object, int] = {}
        for wid, episodes in assignments:
            ref = workers[wid].collect.remote(
                weights_ref,
                opponent_ref,
                phase,
                not args.latest_opponent_stochastic,
                args.max_actions_per_source_safety,
                args.seed + update * 1_000_000 + wid * 10_000,
                episodes,
            )
            futures.append(ref)
            future_to_worker[ref] = wid
        results = []
        pending = list(futures)
        collected_episodes = 0
        straggler_restarts = 0
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
            done_ref = done[0]
            try:
                rows, metrics = ray.get(done_ref)
            except Exception as exc:
                stale_wid = future_to_worker.get(done_ref)
                if stale_wid is not None:
                    workers[stale_wid] = RolloutWorker.remote(model_cfg, args.episode_steps, args.opponent_mode, not args.no_numba)
                    straggler_restarts += 1
                print(
                    json.dumps(
                        {
                            "event": "worker_failed",
                            "update": update,
                            "worker": stale_wid,
                            "error": str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(episodes=collected_episodes, samples=sum(len(r) for r, _m in results), refresh=False)
                continue
            results.append((rows, metrics))
            collected_episodes += int(metrics.get("episodes", 0.0))
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(episodes=collected_episodes, samples=sum(len(r) for r, _m in results), refresh=False)
            else:
                print(
                    json.dumps(
                        {"event": "worker_done", "update": update, "done": len(results), "total": len(futures), "episodes": collected_episodes, "samples": len(rows)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if target_fresh_episodes > 0 and collected_episodes >= target_fresh_episodes:
                break
        if pending:
            stale_workers = sorted({future_to_worker[ref] for ref in pending if ref in future_to_worker})
            for ref in pending:
                try:
                    ray.cancel(ref, force=False)
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "event": "collect_cancel_failed",
                                "update": update,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            for wid in stale_workers:
                workers[wid] = RolloutWorker.remote(model_cfg, args.episode_steps, args.opponent_mode, not args.no_numba)
            straggler_restarts = len(stale_workers)
        if pbar is not None:
            pbar.close()
        print(
            json.dumps(
                {
                    "event": "collect_done",
                    "update": update,
                    "worker_results": len(results),
                    "episodes": collected_episodes,
                    "cancelled": len(pending),
                    "straggler_restarts": straggler_restarts,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

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
            args.win_replay_weight,
            args.loss_replay_weight,
            args.draw_replay_weight,
        )
        for row in replay_rows:
            buffer.add(**row)
        if args.replay_updates > 0:
            replay_batches.append((update, fresh_rows_for_replay))

        train_metrics = updater.update(buffer)
        print(json.dumps({"event": "update_done", "update": update, "samples": len(buffer), "train": train_metrics}, ensure_ascii=False), flush=True)
        summary = {
            "update": update,
            "workers": len(assignments),
            "ray_workers": num_workers,
            "episodes": collected_episodes,
            "assigned_episodes": assigned_episodes,
            "straggler_restarts": straggler_restarts,
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
            if args.checkpoint_interval > 0 and update % args.checkpoint_interval == 0:
                save_checkpoint(out_dir / "latest.pt", model, args, update, summary)
            eval_state = cpu_state_dict(model)
            eval_state_ref = ray.put(eval_state)
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
                    args.eval_gpu_ids,
                    eval_launch_bias,
                    eval_ship_bias,
                    args.eval_launch_temperature,
                    not args.eval_stochastic,
                )
                part = ray.get(temp_worker.evaluate.remote(eval_state_ref, update, args.eval_games, args.seed + 300_000 + update * 1_000, args.eval_stochastic_compare))
                merged_eval = _merge_eval_parts([part])
                summary.update(merged_eval)
                top_manifest = update_top_checkpoints(out_dir, eval_state, model_cfg, update, {"async_eval_update": update, **merged_eval}, args.top_k_checkpoints)
                if merged_eval["eval_vs_nearest"]["winrate"] > best_winrate:
                    best_winrate = merged_eval["eval_vs_nearest"]["winrate"]
                    save_checkpoint_state(out_dir / "best.pt", eval_state, model_cfg, update, {"async_eval_update": update, **merged_eval})
                print(
                    json.dumps(
                        {"event": "top_checkpoints_updated", "update": update, "top_k": len(top_manifest), "best": top_manifest[0] if top_manifest else None},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            else:
                pending_eval_states[update] = eval_state
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
                            args.eval_stochastic_compare,
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

        if args.checkpoint_interval > 0 and update % args.checkpoint_interval == 0 and not should_eval:
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
