from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyPPO.bridge_agents import RegularSourceDropGateAgent, drop_gate_logits
from tinyPPO.features import MAX_PLANETS, encode_obs, score
from tinyPPO.model import TinyPolicyValueNet
from tinyPPO.train import init_swanlab_or_none, log_swanlab
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent


@dataclass
class GateRow:
    planets: np.ndarray
    pair_features: np.ndarray
    global_features: np.ndarray
    planet_mask: np.ndarray
    own_mask: np.ndarray
    source_indices: np.ndarray
    choice: int
    old_logprob: float
    value: float
    ret: float


def _batch_from_row(row: GateRow, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "planets": torch.tensor(row.planets[None], dtype=torch.float32, device=device),
        "pair_features": torch.tensor(row.pair_features[None], dtype=torch.float32, device=device),
        "global_features": torch.tensor(row.global_features[None], dtype=torch.float32, device=device),
        "planet_mask": torch.tensor(row.planet_mask[None], dtype=torch.bool, device=device),
        "own_mask": torch.tensor(row.own_mask[None], dtype=torch.bool, device=device),
    }


@torch.no_grad()
def gate_action(
    model: TinyPolicyValueNet,
    obs: dict[str, Any],
    anchor,
    device: torch.device,
    no_drop_bias: float,
    min_anchor_actions_to_filter: int,
    deterministic: bool,
) -> tuple[list[list], GateRow | None]:
    actions = list(anchor(obs) or [])
    if len(actions) < min_anchor_actions_to_filter:
        return actions, None
    player = int(obs.get("player", 0))
    enc = encode_obs(obs, player, players=2)
    id_to_idx = {pid: idx for idx, pid in enumerate(enc.planet_ids)}
    source_indices = np.asarray([id_to_idx.get(int(action[0]), -1) for action in actions], dtype=np.int64)
    if np.any(source_indices < 0):
        return actions, None
    out = model(**_batch_from_row(
        GateRow(
            enc.planets,
            enc.pair_features,
            enc.global_features,
            enc.planet_mask,
            enc.own_mask,
            source_indices,
            0,
            0.0,
            0.0,
            0.0,
        ),
        device,
    ))
    logits = drop_gate_logits(out["source_logits"][0], source_indices, no_drop_bias)
    dist = torch.distributions.Categorical(logits=logits)
    choice = int(torch.argmax(logits).item()) if deterministic else int(dist.sample().item())
    logprob = float(dist.log_prob(torch.tensor(choice, device=device)).item())
    value = float(out["value"][0].item())
    kept = actions if choice <= 0 else [action for idx, action in enumerate(actions) if idx != choice - 1]
    row = GateRow(
        planets=enc.planets,
        pair_features=enc.pair_features,
        global_features=enc.global_features,
        planet_mask=enc.planet_mask,
        own_mask=enc.own_mask,
        source_indices=source_indices,
        choice=choice,
        old_logprob=logprob,
        value=value,
        ret=0.0,
    )
    return kept, row


def collect_episode(
    model: TinyPolicyValueNet,
    seed: int,
    device: torch.device,
    episode_steps: int,
    no_drop_bias: float,
    min_anchor_actions_to_filter: int,
    use_numba: bool,
) -> tuple[list[GateRow], dict[str, float]]:
    env = make_fast_orbit_wars({"episodeSteps": episode_steps, "seed": seed}, keep_history=False, use_numba=use_numba)
    env.reset(2)
    gate_anchor = make_rulebase_agent("regular")
    opponent = make_rulebase_agent("regular")
    rows: list[GateRow] = []
    anchor_actions = kept_actions = dropped_actions = 0
    while not env.done:
        obs0 = env.steps[-1][0]["observation"]
        obs1 = env.steps[-1][1]["observation"]
        regular_actions = list(gate_anchor(obs0) or [])
        if len(regular_actions) >= min_anchor_actions_to_filter:
            actions0, row = gate_action(
                model,
                obs0,
                lambda _obs, _actions=regular_actions: _actions,
                device,
                no_drop_bias,
                min_anchor_actions_to_filter,
                deterministic=False,
            )
            if row is not None:
                rows.append(row)
        else:
            actions0 = regular_actions
            row = None
        anchor_actions += len(regular_actions)
        kept_actions += len(actions0)
        dropped_actions += max(0, len(regular_actions) - len(actions0))
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
        "anchor_actions": float(anchor_actions),
        "kept_actions": float(kept_actions),
        "dropped_actions": float(dropped_actions),
        "dropped_frac": dropped_actions / max(1, anchor_actions),
    }


def update_model(
    model: TinyPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    rows: list[GateRow],
    device: torch.device,
    no_drop_bias: float,
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
    stats: list[dict[str, float]] = []
    indices = np.arange(len(rows))
    for _epoch in range(max(1, epochs)):
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            losses = []
            policy_losses = []
            value_losses = []
            entropies = []
            approx_kls = []
            clip_fracs = []
            optimizer.zero_grad(set_to_none=True)
            for idx in batch_idx:
                row = rows[int(idx)]
                out = model(**_batch_from_row(row, device))
                logits = drop_gate_logits(out["source_logits"][0], row.source_indices, no_drop_bias)
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
    no_drop_bias: float,
    min_anchor_actions_to_filter: int,
    deterministic: bool,
    use_numba: bool,
) -> dict[str, float]:
    wins = losses = draws = 0
    dropped = anchor = 0.0
    for i in range(games):
        model_seat = i % 2
        agents = [make_rulebase_agent("regular"), make_rulebase_agent("regular")]
        agent = RegularSourceDropGateAgent(
            ckpt_path,
            device=device,
            no_drop_bias=no_drop_bias,
            min_anchor_actions_to_filter=min_anchor_actions_to_filter,
            deterministic=deterministic,
        )
        agents[model_seat] = agent
        env = make_fast_orbit_wars({"episodeSteps": episode_steps, "seed": seed + i}, keep_history=False, use_numba=use_numba)
        env.run(agents)
        obs = env.steps[-1][model_seat]["observation"]
        model_score = score(obs, model_seat)
        other_score = score(obs, 1 - model_seat)
        if model_score > other_score:
            wins += 1
        elif model_score < other_score:
            losses += 1
        else:
            draws += 1
        anchor += float(agent.stats["anchor_actions"])
        dropped += float(agent.stats["dropped_actions"])
    return {
        "games": float(games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "winrate": wins / max(1, games),
        "nonloss": (wins + draws) / max(1, games),
        "mean_reward": (wins - losses) / max(1, games),
        "dropped_frac": dropped / max(1.0, anchor),
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
        "dropped_frac": sum(part.get("dropped_frac", 0.0) * part["games"] for part in parts) / max(1.0, games),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--out-dir", default="tinyPPO/runs/bridge_gate")
    parser.add_argument("--init-checkpoint", default="tinyPPO/regular_bc.pt")
    parser.add_argument("--updates", type=int, default=100000)
    parser.add_argument("--episodes-per-update", type=int, default=40)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--gpus-per-worker", type=float, default=0.0)
    parser.add_argument("--learner-device", default="cuda:0")
    parser.add_argument("--worker-device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--no-drop-bias", type=float, default=2.0)
    parser.add_argument("--min-anchor-actions-to-filter", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--clip-ratio", type=float, default=0.10)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-workers", type=int, default=16)
    parser.add_argument("--stop-winrate", type=float, default=0.50)
    parser.add_argument("--stop-confirmations", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--swanlab-project", default="orbit-wars")
    parser.add_argument("--swanlab-experiment", default="tinyPPO-bridge-gate")
    parser.add_argument("--swanlab-mode", default="cloud")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument("--allow-no-swanlab", action="store_true")
    args = parser.parse_args()

    import ray

    ray.init(address=None if args.ray_address == "local" else args.ray_address, ignore_reinit_error=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    swan = init_swanlab_or_none(args, "bridge_gate")
    device = torch.device(args.learner_device)
    payload = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
    model_cfg = {key: int(value) for key, value in payload.get("model", {}).items()}
    model = TinyPolicyValueNet(**model_cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)
    save_checkpoint(out_dir / "init.pt", model, model_cfg, int(payload.get("update", 0)), {"source": args.init_checkpoint})

    @ray.remote(num_cpus=args.cpus_per_worker, num_gpus=args.gpus_per_worker)
    def collect_part(state: dict[str, torch.Tensor], seed: int, episodes: int) -> tuple[list[GateRow], dict[str, float]]:
        torch.set_num_threads(1)
        local_device = torch.device(args.worker_device)
        local_model = TinyPolicyValueNet(**model_cfg).to(local_device)
        local_model.load_state_dict({key: value.to(local_device) for key, value in state.items()})
        local_model.eval()
        rows: list[GateRow] = []
        metrics: list[dict[str, float]] = []
        for i in range(episodes):
            part_rows, part_metrics = collect_episode(
                local_model,
                seed + i,
                local_device,
                args.episode_steps,
                args.no_drop_bias,
                args.min_anchor_actions_to_filter,
                not args.no_numba,
            )
            rows.extend(part_rows)
            metrics.append(part_metrics)
        return rows, {
            "episodes": float(episodes),
            "rows": float(len(rows)),
            "result": float(np.mean([m["result"] for m in metrics])) if metrics else 0.0,
            "dropped_frac": float(np.mean([m["dropped_frac"] for m in metrics])) if metrics else 0.0,
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
            args.no_drop_bias,
            args.min_anchor_actions_to_filter,
            deterministic,
            not args.no_numba,
        )

    best_winrate = -1.0
    stop_streak = 0
    log_path = out_dir / "train_log.jsonl"
    for update in range(1, args.updates + 1):
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        state_ref = ray.put(state)
        refs = []
        remaining = args.episodes_per_update
        for worker_idx in range(args.workers):
            if remaining <= 0:
                break
            episodes = int(np.ceil(remaining / max(1, args.workers - worker_idx)))
            remaining -= episodes
            refs.append(collect_part.remote(state_ref, args.seed + update * 1_000_000 + worker_idx * 1000, episodes))
        rows: list[GateRow] = []
        collect_metrics: list[dict[str, float]] = []
        while refs:
            ready, refs = ray.wait(refs, num_returns=1)
            part_rows, part_metrics = ray.get(ready[0])
            rows.extend(part_rows)
            collect_metrics.append(part_metrics)
            print(
                json.dumps(
                    {"event": "collect_progress", "update": update, "parts_done": len(collect_metrics), "rows": len(rows)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        train_metrics = update_model(
            model,
            optimizer,
            rows,
            device,
            args.no_drop_bias,
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
                "dropped_frac": float(np.mean([m["dropped_frac"] for m in collect_metrics])) if collect_metrics else 0.0,
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
                eval_refs.append(eval_part.remote(str(eval_ckpt), games, args.seed + 9_000_000 + update * 10_000 + i * 1000, False))
            eval_parts = [ray.get(ref) for ref in eval_refs]
            eval_metrics = _merge_eval(eval_parts)
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
