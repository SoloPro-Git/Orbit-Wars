from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyPPO.agents import _aim_angle_and_eta, _path_is_safe, candidate_target_mask
from tinyPPO.features import encode_obs, score
from tinyPPO.model import TinyPolicyValueNet
from tinyPPO.train import init_swanlab_or_none, log_swanlab
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent


@dataclass
class TargetRow:
    planets: np.ndarray
    pair_features: np.ndarray
    global_features: np.ndarray
    planet_mask: np.ndarray
    own_mask: np.ndarray
    source_idx: int
    candidates: np.ndarray
    choice: int
    old_logprob: float
    value: float
    ret: float


def _batch_from_row(row: TargetRow, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "planets": torch.tensor(row.planets[None], dtype=torch.float32, device=device),
        "pair_features": torch.tensor(row.pair_features[None], dtype=torch.float32, device=device),
        "global_features": torch.tensor(row.global_features[None], dtype=torch.float32, device=device),
        "planet_mask": torch.tensor(row.planet_mask[None], dtype=torch.bool, device=device),
        "own_mask": torch.tensor(row.own_mask[None], dtype=torch.bool, device=device),
    }


def _angle_diff(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def infer_target_index(obs: dict[str, Any], source: list, angle: float) -> int | None:
    planets = list(obs.get("planets", []))
    sx = float(source[2])
    sy = float(source[3])
    vx = math.cos(angle)
    vy = math.sin(angle)
    best: tuple[float, int] | None = None
    for idx, target in enumerate(planets):
        if int(target[0]) == int(source[0]):
            continue
        dx = float(target[2]) - sx
        dy = float(target[3]) - sy
        proj = dx * vx + dy * vy
        if proj <= 0.0:
            continue
        perp = abs(dx * vy - dy * vx)
        target_angle = math.atan2(dy, dx)
        score_value = perp + 4.0 * _angle_diff(angle, target_angle)
        if perp <= float(target[4]) + 6.0 and (best is None or score_value < best[0]):
            best = (score_value, idx)
    return None if best is None else best[1]


def target_gate_logits(target_logits: torch.Tensor, source_idx: int, candidates: np.ndarray, anchor_choice: int, regular_bias: float) -> torch.Tensor:
    candidate_tensor = torch.tensor(candidates.tolist(), dtype=torch.long, device=target_logits.device)
    logits = target_logits[source_idx, :, candidate_tensor].max(dim=0).values
    if 0 <= anchor_choice < logits.numel() and regular_bias:
        logits = logits.clone()
        logits[anchor_choice] += float(regular_bias)
    return logits


@torch.no_grad()
def target_gate_actions(
    model: TinyPolicyValueNet,
    obs: dict[str, Any],
    anchor_actions: list[list],
    device: torch.device,
    top_k: int,
    regular_bias: float,
    deterministic: bool,
) -> tuple[list[list], list[TargetRow], dict[str, float]]:
    player = int(obs.get("player", 0))
    planets = list(obs.get("planets", []))
    enc = encode_obs(obs, player, players=2)
    id_to_idx = {pid: idx for idx, pid in enumerate(enc.planet_ids)}
    mask = candidate_target_mask(obs, player, top_k=top_k)
    out = model(**{
        "planets": torch.tensor(enc.planets[None], dtype=torch.float32, device=device),
        "pair_features": torch.tensor(enc.pair_features[None], dtype=torch.float32, device=device),
        "global_features": torch.tensor(enc.global_features[None], dtype=torch.float32, device=device),
        "planet_mask": torch.tensor(enc.planet_mask[None], dtype=torch.bool, device=device),
        "own_mask": torch.tensor(enc.own_mask[None], dtype=torch.bool, device=device),
    })
    actions: list[list] = []
    rows: list[TargetRow] = []
    changed = 0
    attempted = 0
    fallback = 0
    for action in anchor_actions:
        src_idx = id_to_idx.get(int(action[0]), -1)
        if src_idx < 0 or src_idx >= len(planets):
            actions.append(action)
            fallback += 1
            continue
        source = planets[src_idx]
        ships = max(1, int(action[2]))
        anchor_target = infer_target_index(obs, source, float(action[1]))
        candidates = np.where(mask[src_idx])[0].astype(np.int64)
        if anchor_target is not None and 0 <= anchor_target < len(planets):
            candidates = np.asarray([anchor_target, *[idx for idx in candidates.tolist() if idx != anchor_target]], dtype=np.int64)
        candidates = candidates[: max(1, top_k)]
        if candidates.size <= 1:
            actions.append(action)
            fallback += 1
            continue
        anchor_choice = 0 if anchor_target is not None and int(candidates[0]) == int(anchor_target) else -1
        logits = target_gate_logits(out["target_logits"][0], src_idx, candidates, anchor_choice, regular_bias)
        dist = torch.distributions.Categorical(logits=logits)
        choice = int(torch.argmax(logits).item()) if deterministic else int(dist.sample().item())
        target_idx = int(candidates[choice])
        target = planets[target_idx]
        angle, eta = _aim_angle_and_eta(obs, source, target, ships)
        if not (math.isfinite(angle) and _path_is_safe(source, angle, ships, eta + 3)):
            actions.append(action)
            fallback += 1
            continue
        attempted += 1
        if anchor_choice >= 0 and choice != anchor_choice:
            changed += 1
        actions.append([int(source[0]), angle, ships])
        rows.append(
            TargetRow(
                planets=enc.planets,
                pair_features=enc.pair_features,
                global_features=enc.global_features,
                planet_mask=enc.planet_mask,
                own_mask=enc.own_mask,
                source_idx=src_idx,
                candidates=candidates,
                choice=choice,
                old_logprob=float(dist.log_prob(torch.tensor(choice, device=device)).item()),
                value=float(out["value"][0].item()),
                ret=0.0,
            )
        )
    stats = {
        "anchor_actions": float(len(anchor_actions)),
        "attempted": float(attempted),
        "changed": float(changed),
        "fallback": float(fallback),
    }
    return actions, rows, stats


def collect_episode(
    model: TinyPolicyValueNet,
    seed: int,
    device: torch.device,
    episode_steps: int,
    top_k: int,
    regular_bias: float,
    use_numba: bool,
) -> tuple[list[TargetRow], dict[str, float]]:
    env = make_fast_orbit_wars({"episodeSteps": episode_steps, "seed": seed}, keep_history=False, use_numba=use_numba)
    env.reset(2)
    anchor = make_rulebase_agent("regular")
    opponent = make_rulebase_agent("regular")
    rows: list[TargetRow] = []
    totals = {"anchor_actions": 0.0, "attempted": 0.0, "changed": 0.0, "fallback": 0.0}
    while not env.done:
        obs0 = env.steps[-1][0]["observation"]
        obs1 = env.steps[-1][1]["observation"]
        anchor_actions = list(anchor(obs0) or [])
        actions0, step_rows, stats = target_gate_actions(
            model,
            obs0,
            anchor_actions,
            device,
            top_k,
            regular_bias,
            deterministic=False,
        )
        rows.extend(step_rows)
        for key, value in stats.items():
            totals[key] += value
        env.step([actions0, opponent(obs1)])
    final = env.steps[-1]
    model_score = score(final[0]["observation"], 0)
    other_score = score(final[0]["observation"], 1)
    result = 1.0 if model_score > other_score else (-1.0 if model_score < other_score else 0.0)
    for row in rows:
        row.ret = result
    return rows, {
        "episodes": 1.0,
        "rows": float(len(rows)),
        "result": result,
        "attempt_frac": totals["attempted"] / max(1.0, totals["anchor_actions"]),
        "change_frac": totals["changed"] / max(1.0, totals["attempted"]),
        "fallback_frac": totals["fallback"] / max(1.0, totals["anchor_actions"]),
    }


def update_model(
    model: TinyPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    rows: list[TargetRow],
    device: torch.device,
    regular_bias: float,
    epochs: int,
    batch_size: int,
    clip_ratio: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    if not rows:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "updates": 0.0}
    returns = torch.tensor([row.ret for row in rows], dtype=torch.float32)
    old_values = torch.tensor([row.value for row in rows], dtype=torch.float32)
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6) if len(rows) > 1 else advantages
    old_logprobs = torch.tensor([row.old_logprob for row in rows], dtype=torch.float32)
    indices = np.arange(len(rows))
    stats: list[dict[str, float]] = []
    for _epoch in range(max(1, epochs)):
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            policy_losses = []
            value_losses = []
            entropies = []
            approx_kls = []
            clip_fracs = []
            for idx in batch_idx:
                row = rows[int(idx)]
                out = model(**_batch_from_row(row, device))
                logits = target_gate_logits(out["target_logits"][0], row.source_idx, row.candidates, 0, regular_bias)
                dist = torch.distributions.Categorical(logits=logits)
                choice = torch.tensor(int(row.choice), dtype=torch.long, device=device)
                logprob = dist.log_prob(choice)
                old_logprob = old_logprobs[int(idx)].to(device)
                adv = advantages[int(idx)].to(device)
                ratio = torch.exp(logprob - old_logprob)
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                policy_loss = -torch.min(ratio * adv, clipped)
                ret = returns[int(idx)].to(device)
                value = out["value"][0]
                value_loss = 0.5 * (value - ret).pow(2)
                entropy = dist.entropy()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                (loss / max(1, len(batch_idx))).backward()
                losses.append(float(loss.detach().cpu()))
                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                approx_kls.append(float((old_logprob - logprob).detach().cpu()))
                clip_fracs.append(float((torch.abs(ratio - 1.0) > clip_ratio).detach().cpu()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            stats.append(
                {
                    "loss": float(np.mean(losses)),
                    "policy_loss": float(np.mean(policy_losses)),
                    "value_loss": float(np.mean(value_losses)),
                    "entropy": float(np.mean(entropies)),
                    "approx_kl": float(np.mean(approx_kls)),
                    "clip_frac": float(np.mean(clip_fracs)),
                }
            )
    out = {key: float(np.mean([part[key] for part in stats])) for key in stats[0]}
    out["updates"] = float(len(stats))
    return out


def save_checkpoint(path: Path, model: TinyPolicyValueNet, model_cfg: dict[str, int], update: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model": model_cfg, "update": update, "metrics": metrics}, path)


@torch.no_grad()
def eval_vs_regular(
    ckpt_path: Path,
    games: int,
    seed: int,
    episode_steps: int,
    device: str,
    top_k: int,
    regular_bias: float,
    deterministic: bool,
    use_numba: bool,
) -> dict[str, float]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_cfg = {key: int(value) for key, value in payload.get("model", {}).items()}
    model = TinyPolicyValueNet(**model_cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    wins = losses = draws = 0
    change_fracs = []
    for i in range(games):
        model_seat = i % 2
        anchors = [make_rulebase_agent("regular"), make_rulebase_agent("regular")]
        opponents = [make_rulebase_agent("regular"), make_rulebase_agent("regular")]
        totals = {"anchor_actions": 0.0, "attempted": 0.0, "changed": 0.0, "fallback": 0.0}
        env = make_fast_orbit_wars({"episodeSteps": episode_steps, "seed": seed + i}, keep_history=False, use_numba=use_numba)
        env.reset(2)
        while not env.done:
            step_actions = []
            for pid in range(2):
                obs = env.steps[-1][pid]["observation"]
                if pid == model_seat:
                    anchor_actions = list(anchors[pid](obs) or [])
                    actions, _rows, stats = target_gate_actions(model, obs, anchor_actions, torch.device(device), top_k, regular_bias, deterministic)
                    for key, value in stats.items():
                        totals[key] += value
                    step_actions.append(actions)
                else:
                    step_actions.append(opponents[pid](obs))
            env.step(step_actions)
        obs = env.steps[-1][model_seat]["observation"]
        model_score = score(obs, model_seat)
        other_score = score(obs, 1 - model_seat)
        if model_score > other_score:
            wins += 1
        elif model_score < other_score:
            losses += 1
        else:
            draws += 1
        change_fracs.append(totals["changed"] / max(1.0, totals["attempted"]))
    return {
        "games": float(games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "winrate": wins / max(1, games),
        "nonloss": (wins + draws) / max(1, games),
        "mean_reward": (wins - losses) / max(1, games),
        "change_frac": float(np.mean(change_fracs)) if change_fracs else 0.0,
    }


def _merge_eval(parts: list[dict[str, float]]) -> dict[str, float]:
    games = sum(part["games"] for part in parts)
    wins = sum(part["wins"] for part in parts)
    losses = sum(part["losses"] for part in parts)
    draws = sum(part["draws"] for part in parts)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winrate": wins / max(1.0, games),
        "nonloss": (wins + draws) / max(1.0, games),
        "mean_reward": (wins - losses) / max(1.0, games),
        "change_frac": sum(part.get("change_frac", 0.0) * part["games"] for part in parts) / max(1.0, games),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--out-dir", default="tinyPPO/runs/target_gate")
    parser.add_argument("--init-checkpoint", default="tinyPPO/runs/bridge_gate_ultraconservative_regular_20260525/best.pt")
    parser.add_argument("--updates", type=int, default=100000)
    parser.add_argument("--episodes-per-update", type=int, default=40)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--gpus-per-worker", type=float, default=0.0)
    parser.add_argument("--learner-device", default="cuda:0")
    parser.add_argument("--worker-device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--regular-bias", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--clip-ratio", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.00005)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-workers", type=int, default=16)
    parser.add_argument("--stop-winrate", type=float, default=0.55)
    parser.add_argument("--stop-confirmations", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--swanlab-project", default="orbit-wars")
    parser.add_argument("--swanlab-experiment", default="tinyPPO-target-gate")
    parser.add_argument("--swanlab-mode", default="cloud")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--allow-no-swanlab", action="store_true")
    args = parser.parse_args()

    import ray

    ray.init(address=None if args.ray_address == "local" else args.ray_address, ignore_reinit_error=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    swan = init_swanlab_or_none(args, "target_gate")
    device = torch.device(args.learner_device)
    payload = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
    model_cfg = {key: int(value) for key, value in payload.get("model", {}).items()}
    model = TinyPolicyValueNet(**model_cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)
    save_checkpoint(out_dir / "init.pt", model, model_cfg, int(payload.get("update", 0)), {"source": args.init_checkpoint})

    @ray.remote(num_cpus=args.cpus_per_worker, num_gpus=args.gpus_per_worker)
    def collect_part(state: dict[str, torch.Tensor], seed: int, episodes: int) -> tuple[list[TargetRow], dict[str, float]]:
        torch.set_num_threads(1)
        local_device = torch.device(args.worker_device)
        local_model = TinyPolicyValueNet(**model_cfg).to(local_device)
        local_model.load_state_dict({key: value.to(local_device) for key, value in state.items()})
        local_model.eval()
        rows: list[TargetRow] = []
        metrics: list[dict[str, float]] = []
        for i in range(episodes):
            part_rows, part_metrics = collect_episode(
                local_model,
                seed + i,
                local_device,
                args.episode_steps,
                args.top_k,
                args.regular_bias,
                not args.no_numba,
            )
            rows.extend(part_rows)
            metrics.append(part_metrics)
        return rows, {
            "episodes": float(episodes),
            "rows": float(len(rows)),
            "result": float(np.mean([m["result"] for m in metrics])) if metrics else 0.0,
            "attempt_frac": float(np.mean([m["attempt_frac"] for m in metrics])) if metrics else 0.0,
            "change_frac": float(np.mean([m["change_frac"] for m in metrics])) if metrics else 0.0,
            "fallback_frac": float(np.mean([m["fallback_frac"] for m in metrics])) if metrics else 0.0,
        }

    @ray.remote(num_cpus=args.cpus_per_worker, num_gpus=0)
    def eval_part(path: str, games: int, seed: int, deterministic: bool) -> dict[str, float]:
        torch.set_num_threads(1)
        return eval_vs_regular(
            Path(path),
            games,
            seed,
            args.episode_steps,
            args.worker_device,
            args.top_k,
            args.regular_bias,
            deterministic,
            not args.no_numba,
        )

    best_winrate = -1.0
    stop_streak = 0
    log_path = out_dir / "train_log.jsonl"
    for update in range(1, args.updates + 1):
        state_ref = ray.put({key: value.detach().cpu() for key, value in model.state_dict().items()})
        refs = []
        remaining = args.episodes_per_update
        for worker_idx in range(args.workers):
            if remaining <= 0:
                break
            episodes = int(np.ceil(remaining / max(1, args.workers - worker_idx)))
            remaining -= episodes
            refs.append(collect_part.remote(state_ref, args.seed + update * 1_000_000 + worker_idx * 1000, episodes))
        rows: list[TargetRow] = []
        collect_metrics: list[dict[str, float]] = []
        while refs:
            ready, refs = ray.wait(refs, num_returns=1)
            part_rows, part_metrics = ray.get(ready[0])
            rows.extend(part_rows)
            collect_metrics.append(part_metrics)
            print(json.dumps({"event": "collect_progress", "update": update, "parts_done": len(collect_metrics), "rows": len(rows)}, ensure_ascii=False), flush=True)
        train_metrics = update_model(
            model,
            optimizer,
            rows,
            device,
            args.regular_bias,
            args.epochs,
            args.batch_size,
            args.clip_ratio,
            args.entropy_coef,
            args.value_coef,
            args.max_grad_norm,
        )
        summary: dict[str, Any] = {
            "event": "update_done",
            "update": update,
            "rows": len(rows),
            "collect": {
                "episodes": float(sum(m["episodes"] for m in collect_metrics)),
                "result": float(np.mean([m["result"] for m in collect_metrics])) if collect_metrics else 0.0,
                "attempt_frac": float(np.mean([m["attempt_frac"] for m in collect_metrics])) if collect_metrics else 0.0,
                "change_frac": float(np.mean([m["change_frac"] for m in collect_metrics])) if collect_metrics else 0.0,
                "fallback_frac": float(np.mean([m["fallback_frac"] for m in collect_metrics])) if collect_metrics else 0.0,
            },
            "train": train_metrics,
        }
        if args.checkpoint_interval > 0 and update % args.checkpoint_interval == 0:
            save_checkpoint(out_dir / "latest.pt", model, model_cfg, update, summary)
        if update % args.eval_interval == 0:
            eval_ckpt = out_dir / f"eval_u{update:06d}.pt"
            save_checkpoint(eval_ckpt, model, model_cfg, update, summary)
            eval_refs = []
            games_left = args.eval_games
            for i in range(args.eval_workers):
                if games_left <= 0:
                    break
                games = int(np.ceil(games_left / max(1, args.eval_workers - i)))
                games_left -= games
                eval_refs.append(eval_part.remote(str(eval_ckpt), games, args.seed + 11_000_000 + update * 10_000 + i * 1000, False))
            eval_metrics = _merge_eval([ray.get(ref) for ref in eval_refs])
            summary["eval_stochastic_vs_regular"] = eval_metrics
            if eval_metrics["winrate"] > best_winrate:
                best_winrate = eval_metrics["winrate"]
                save_checkpoint(out_dir / "best.pt", model, model_cfg, update, summary)
            stop_streak = stop_streak + 1 if eval_metrics["winrate"] >= args.stop_winrate else 0
            if stop_streak >= args.stop_confirmations:
                summary["stop_reason"] = f"eval winrate {eval_metrics['winrate']:.3f} >= {args.stop_winrate:.3f} for {stop_streak} evals"
                save_checkpoint(out_dir / "latest.pt", model, model_cfg, update, summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        log_swanlab(swan, summary, update)
        if "stop_reason" in summary:
            break
    if swan is not None:
        swan.finish()


if __name__ == "__main__":
    main()
