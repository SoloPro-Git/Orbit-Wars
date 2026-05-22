from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training2 import make_fast_orbit_wars

from tinyPPO.agents import ACTION_SLOTS, MAX_ACTIONS_PER_SOURCE_SAFETY, SHIP_FRACTIONS, TinyPPOAgent, actions_from_decisions, nearest_planet_agent, random_policy_agent
from tinyPPO.eval import run_matchups
from tinyPPO.features import MAX_PLANETS, encode_obs, final_result
from tinyPPO.model import TinyPolicyValueNet
from tinyPPO.ppo import PPOConfig, PPOUpdater, RolloutBuffer, action_log_prob_entropy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_swanlab(args: argparse.Namespace, run_kind: str) -> Any | None:
    if getattr(args, "no_swanlab", False):
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
        config={f"tinyPPO_{run_kind}": vars(args)},
    )


def init_swanlab_or_none(args: argparse.Namespace, run_kind: str) -> Any | None:
    try:
        return _init_swanlab(args, run_kind)
    except Exception as exc:
        if getattr(args, "allow_no_swanlab", False):
            print(f"[SwanLab] init failed, continuing without SwanLab: {exc}", flush=True)
            return None
        raise RuntimeError(
            "SwanLab init failed. Set SWANLAB_API_KEY or training/config/swanlab_key.txt; "
            "for debugging use --allow-no-swanlab or --no-swanlab."
        ) from exc


def _flatten_metrics(prefix: str, obj: Any, out: dict[str, float]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten_metrics(f"{prefix}/{key}" if prefix else str(key), value, out)
        return
    if isinstance(obj, bool):
        out[prefix] = float(obj)
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)


def log_swanlab(swan: Any | None, summary: dict[str, Any], step: int) -> None:
    if swan is None:
        return
    log: dict[str, float] = {}
    _flatten_metrics("", summary, log)
    phase = str(summary.get("phase", ""))
    if phase:
        log["phase/is_random"] = float(phase == "random")
        log["phase/is_latest"] = float(phase == "latest")
        log["phase/is_self"] = float(phase == "self")
    swan.log(log, step=step)


def add_weighted_rows(buffer: RolloutBuffer, rows: list[dict], weight: float) -> None:
    for row in rows:
        item = dict(row)
        item["replay_weight"] = float(weight)
        buffer.add(**item)


def replay_rows_for_update(
    replay_batches: deque[tuple[int, list[dict]]],
    update: int,
    current_count: int,
    replay_ratio: float,
    replay_age_decay: float,
) -> tuple[list[dict], dict[str, float]]:
    if current_count <= 0 or replay_ratio <= 0.0 or not replay_batches:
        return [], {"replay_samples": 0.0, "replay_weight_mean": 0.0}
    target = int(current_count * replay_ratio)
    rows: list[dict] = []
    weights: list[float] = []
    batches = list(replay_batches)
    random.shuffle(batches)
    for batch_update, batch_rows in batches:
        age = max(1, update - batch_update)
        weight = float(replay_age_decay ** age)
        if weight <= 0.0:
            continue
        for row in batch_rows:
            item = dict(row)
            item["replay_weight"] = weight
            rows.append(item)
            weights.append(weight)
            if len(rows) >= target:
                return rows, {"replay_samples": float(len(rows)), "replay_weight_mean": float(np.mean(weights))}
    return rows, {"replay_samples": float(len(rows)), "replay_weight_mean": float(np.mean(weights)) if weights else 0.0}


def make_batch(enc, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "planets": torch.tensor(enc.planets[None], dtype=torch.float32, device=device),
        "pair_features": torch.tensor(enc.pair_features[None], dtype=torch.float32, device=device),
        "global_features": torch.tensor(enc.global_features[None], dtype=torch.float32, device=device),
        "planet_mask": torch.tensor(enc.planet_mask[None], dtype=torch.bool, device=device),
        "own_mask": torch.tensor(enc.own_mask[None], dtype=torch.bool, device=device),
    }


@torch.no_grad()
def sample_policy_action(
    model: TinyPolicyValueNet,
    obs: dict,
    device: torch.device,
    deterministic: bool = False,
    max_actions_per_source: int = MAX_ACTIONS_PER_SOURCE_SAFETY,
) -> tuple[list[list], dict]:
    player = int(obs.get("player", 0))
    enc = encode_obs(obs, player, players=2)
    batch = make_batch(enc, device)
    out = model(**batch)
    source_logits = out["source_logits"][0]
    target_logits = out["target_logits"][0]
    ship_logits = out["ship_logits"][0]

    action_slots = min(int(getattr(model, "action_slots", ACTION_SLOTS)), int(max_actions_per_source))
    launch_actions = torch.zeros((MAX_PLANETS, action_slots), dtype=torch.long, device=device)
    target_actions = torch.zeros((MAX_PLANETS, action_slots), dtype=torch.long, device=device)
    ship_actions = torch.zeros((MAX_PLANETS, action_slots), dtype=torch.long, device=device)
    launch_mask = torch.zeros((MAX_PLANETS, action_slots), dtype=torch.bool, device=device)

    source_slots: list[int] = []
    target_slots: list[int] = []
    ship_slots: list[int] = []
    for src in torch.where(batch["own_mask"][0])[0].tolist():
        for action_idx in range(action_slots):
            if deterministic:
                launch = int(torch.argmax(source_logits[src, action_idx]).item())
                target = int(torch.argmax(target_logits[src, action_idx]).item())
                ship = int(torch.argmax(ship_logits[src, action_idx, target]).item())
            else:
                launch = int(torch.distributions.Categorical(logits=source_logits[src, action_idx]).sample().item())
                target = int(torch.distributions.Categorical(logits=target_logits[src, action_idx]).sample().item())
                ship = int(torch.distributions.Categorical(logits=ship_logits[src, action_idx, target]).sample().item())
            launch_actions[src, action_idx] = launch
            target_actions[src, action_idx] = target
            ship_actions[src, action_idx] = ship
            if launch == 1:
                launch_mask[src, action_idx] = True
                source_slots.append(src)
                target_slots.append(target)
                ship_slots.append(ship)

    logprob, _entropy = action_log_prob_entropy(
        out,
        launch_actions[None],
        target_actions[None],
        ship_actions[None],
        batch["own_mask"],
        launch_mask[None],
    )
    value = out["value"][0]
    actions = actions_from_decisions(obs, player, np.asarray(source_slots), np.asarray(target_slots), np.asarray(ship_slots))
    row = {
        "player": player,
        "planets": enc.planets,
        "pair_features": enc.pair_features,
        "global_features": enc.global_features,
        "planet_mask": enc.planet_mask,
        "own_mask": enc.own_mask,
        "source_xy": enc.source_xy,
        "launch_actions": launch_actions.cpu().numpy(),
        "target_actions": target_actions.cpu().numpy(),
        "ship_actions": ship_actions.cpu().numpy(),
        "launch_mask": launch_mask.cpu().numpy(),
        "logprob": float(logprob.item()),
        "value": float(value.item()),
        "reward": 0.0,
        "done": False,
    }
    return actions, row


def collect_episode(
    model: TinyPolicyValueNet,
    seed: int,
    device: torch.device,
    episode_steps: int,
    opponent_mode: str,
    use_numba: bool,
    opponent_model: TinyPolicyValueNet | None = None,
    opponent_deterministic: bool = True,
    max_actions_per_source: int = MAX_ACTIONS_PER_SOURCE_SAFETY,
) -> tuple[list[dict], dict[str, float]]:
    env = make_fast_orbit_wars({"episodeSteps": episode_steps, "seed": seed}, keep_history=False, use_numba=use_numba)
    env.reset(2)
    controlled = {0}
    if opponent_mode == "self" or (opponent_mode == "mix" and random.random() < 0.5):
        controlled = {0, 1}
        opponent_kind = "self"
    elif opponent_mode == "latest":
        opponent_kind = "latest"
    else:
        opponent_kind = "random"

    by_player: dict[int, list[dict]] = {pid: [] for pid in controlled}
    launch_counts: list[int] = []
    while not env.done:
        actions = []
        for pid in range(2):
            obs = env.steps[-1][pid]["observation"]
            if pid in controlled:
                action, row = sample_policy_action(model, obs, device, deterministic=False, max_actions_per_source=max_actions_per_source)
                by_player[pid].append(row)
                launch_counts.append(len(action))
            elif opponent_kind == "latest":
                if opponent_model is None:
                    action = random_policy_agent(obs)
                else:
                    action, _row = sample_policy_action(
                        opponent_model,
                        obs,
                        device,
                        deterministic=opponent_deterministic,
                        max_actions_per_source=max_actions_per_source,
                    )
            else:
                action = random_policy_agent(obs)
            actions.append(action)
        env.step(actions)

    final_rows: list[dict] = []
    for pid, rows in by_player.items():
        if not rows:
            continue
        result = final_result(env.steps[-1][pid]["observation"], pid, players=2)
        rows[-1]["reward"] = result
        rows[-1]["done"] = True
        final_rows.extend(rows)
    final_obs0 = env.steps[-1][0]["observation"]
    return final_rows, {
        "result_p0": final_result(final_obs0, 0, players=2),
        "mean_launches": float(np.mean(launch_counts)) if launch_counts else 0.0,
        "opponent_self": float(opponent_kind == "self"),
        "opponent_latest": float(opponent_kind == "latest"),
    }


def save_checkpoint(path: Path, model: TinyPolicyValueNet, args: argparse.Namespace, update: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": {"hidden": args.hidden, "heads": args.heads, "layers": args.layers, "ship_buckets": len(SHIP_FRACTIONS), "action_slots": args.action_slots},
            "update": update,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="tinyPPO/runs/default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--opponent-mode", choices=["random", "self", "mix"], default="random")
    parser.add_argument("--curriculum", action="store_true", help="Train vs random first, then switch training opponent to latest checkpoint.")
    parser.add_argument("--random-winrate-threshold", type=float, default=0.90)
    parser.add_argument("--selfplay-entropy-coef", type=float, default=0.04)
    parser.add_argument("--latest-opponent-stochastic", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--eval-first", action="store_true")
    parser.add_argument("--stop-winrate", type=float, default=0.55)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--action-slots", type=int, default=ACTION_SLOTS)
    parser.add_argument("--max-actions-per-source-safety", type=int, default=MAX_ACTIONS_PER_SOURCE_SAFETY)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--replay-updates", type=int, default=0, help="Keep this many previous update batches for age-decayed PPO replay. 0 disables replay.")
    parser.add_argument("--replay-ratio", type=float, default=0.0, help="Replay samples as a fraction of fresh rollout samples.")
    parser.add_argument("--replay-age-decay", type=float, default=0.50, help="Per-update replay loss weight decay.")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--swanlab-project", default="orbit-wars")
    parser.add_argument("--swanlab-experiment", default="tinyPPO")
    parser.add_argument("--swanlab-mode", default="cloud")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--allow-no-swanlab", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    model = TinyPolicyValueNet(hidden=args.hidden, heads=args.heads, layers=args.layers, ship_buckets=len(SHIP_FRACTIONS), action_slots=args.action_slots).to(device)
    ppo_cfg = PPOConfig(learning_rate=args.lr, entropy_coef=args.entropy_coef)
    updater = PPOUpdater(model, ppo_cfg, device=str(device))
    log_path = out_dir / "train_log.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    swan = init_swanlab_or_none(args, "single")

    best_winrate = -1.0
    phase = "random" if args.curriculum else args.opponent_mode
    latest_opponent = TinyPolicyValueNet(hidden=args.hidden, heads=args.heads, layers=args.layers, ship_buckets=len(SHIP_FRACTIONS), action_slots=args.action_slots).to(device)
    latest_opponent.load_state_dict(model.state_dict())
    latest_opponent.eval()
    replay_batches: deque[tuple[int, list[dict]]] = deque(maxlen=max(0, args.replay_updates))
    for update in range(1, args.updates + 1):
        print(json.dumps({"event": "collect_start", "update": update, "phase": phase}, ensure_ascii=False), flush=True)
        buffer = RolloutBuffer()
        episode_metrics = []
        fresh_rows_for_replay: list[dict] = []
        rollout_mode = phase
        for ep in range(args.episodes_per_update):
            seed = args.seed + update * 10000 + ep
            rows, ep_metrics = collect_episode(
                model,
                seed,
                device,
                args.episode_steps,
                rollout_mode,
                not args.no_numba,
                opponent_model=latest_opponent if rollout_mode == "latest" else None,
                opponent_deterministic=not args.latest_opponent_stochastic,
                max_actions_per_source=args.max_actions_per_source_safety,
            )
            add_weighted_rows(buffer, rows, 1.0)
            fresh_rows_for_replay.extend(dict(row) for row in rows)
            episode_metrics.append(ep_metrics)

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

        print(json.dumps({"event": "collect_done", "update": update, "samples": len(buffer)}, ensure_ascii=False), flush=True)
        train_metrics = updater.update(buffer)
        print(json.dumps({"event": "update_done", "update": update, "train": train_metrics}, ensure_ascii=False), flush=True)
        summary = {
            "update": update,
            "samples": len(buffer),
            "train": train_metrics,
            "rollout_result_p0": float(np.mean([m["result_p0"] for m in episode_metrics])),
            "mean_launches": float(np.mean([m["mean_launches"] for m in episode_metrics])),
            "self_opponent_frac": float(np.mean([m["opponent_self"] for m in episode_metrics])),
            "latest_opponent_frac": float(np.mean([m["opponent_latest"] for m in episode_metrics])),
            "phase": phase,
            "replay": replay_metrics,
        }

        should_eval = update % args.eval_interval == 0 or (args.eval_first and update == 1)
        if should_eval:
            ckpt_path = out_dir / "latest.pt"
            save_checkpoint(ckpt_path, model, args, update, summary)
            print(json.dumps({"event": "eval_start", "update": update, "games_each": args.eval_games}, ensure_ascii=False), flush=True)
            eval_random = run_matchups(
                lambda: TinyPPOAgent(ckpt_path, device=str(device), deterministic=True),
                lambda: random_policy_agent,
                games=args.eval_games,
                seed=args.seed + 300000 + update * 1000,
                episode_steps=args.episode_steps,
                use_numba=not args.no_numba,
                progress=True,
                desc=f"eval-random u{update}",
            )
            print(json.dumps({"event": "eval_random_done", "update": update, "eval_vs_random": eval_random}, ensure_ascii=False), flush=True)
            eval_result = run_matchups(
                lambda: TinyPPOAgent(ckpt_path, device=str(device), deterministic=True),
                lambda: nearest_planet_agent,
                games=args.eval_games,
                seed=args.seed + 500000 + update * 1000,
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
                latest_opponent.load_state_dict(model.state_dict())
                latest_opponent.eval()
                summary["phase_transition"] = {
                    "to": phase,
                    "reason": f"eval_vs_random winrate {eval_random['winrate']:.3f} >= {args.random_winrate_threshold:.3f}",
                    "entropy_coef": updater.cfg.entropy_coef,
                }
            if eval_result["winrate"] >= args.stop_winrate:
                summary["stop_reason"] = f"eval winrate {eval_result['winrate']:.3f} >= {args.stop_winrate:.3f}"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n")
                log_swanlab(swan, summary, update)
                print(json.dumps(summary, ensure_ascii=False))
                break

        save_checkpoint(out_dir / "latest.pt", model, args, update, summary)
        if phase == "latest":
            latest_opponent.load_state_dict(model.state_dict())
            latest_opponent.eval()
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        log_swanlab(swan, summary, update)
        print(json.dumps(summary, ensure_ascii=False))

    if swan is not None:
        swan.finish()


if __name__ == "__main__":
    main()
