from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from training.expert.action_labeling import infer_target_planet_id
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent

from tinyPPO.agents import ACTION_SLOTS, SHIP_BUCKET_MULTIPLIERS, candidate_target_mask, required_ships
from tinyPPO.features import MAX_PLANETS, encode_obs
from tinyPPO.model import TinyPolicyValueNet


@dataclass
class BCRow:
    planets: np.ndarray
    pair_features: np.ndarray
    global_features: np.ndarray
    planet_mask: np.ndarray
    own_mask: np.ndarray
    target_safety_mask: np.ndarray
    launch_actions: np.ndarray
    target_actions: np.ndarray
    ship_actions: np.ndarray
    launch_mask: np.ndarray


class LoggingRuleAgent:
    def __init__(self, player: int, rows: list[tuple[dict[str, Any], int, list[list]]], keep_noop_prob: float):
        self.player = player
        self.agent = make_rulebase_agent("regular")
        self.rows = rows
        self.keep_noop_prob = keep_noop_prob

    def __call__(self, obs: dict[str, Any], configuration=None) -> list[list]:
        del configuration
        action = self.agent(obs) or []
        if action or random.random() < self.keep_noop_prob:
            self.rows.append((obs, self.player, action))
        return action


def _bucket_for_action(obs: dict[str, Any], player: int, source: list, target: list, ships: int) -> int:
    needed = max(1, required_ships(obs, player, source, target))
    ratio = float(max(1, ships)) / float(needed)
    return int(min(range(len(SHIP_BUCKET_MULTIPLIERS)), key=lambda i: abs(SHIP_BUCKET_MULTIPLIERS[i] - ratio)))


def row_from_regular_action(obs: dict[str, Any], player: int, action: list[list], players: int) -> BCRow | None:
    enc = encode_obs(obs, player, players=players)
    planets = list(obs.get("planets", []))[:MAX_PLANETS]
    id_to_idx = {int(p[0]): i for i, p in enumerate(planets)}

    launch_actions = np.zeros((MAX_PLANETS, ACTION_SLOTS), dtype=np.int64)
    target_actions = np.zeros((MAX_PLANETS, ACTION_SLOTS), dtype=np.int64)
    ship_actions = np.zeros((MAX_PLANETS, ACTION_SLOTS), dtype=np.int64)
    launch_mask = np.zeros((MAX_PLANETS, ACTION_SLOTS), dtype=np.bool_)
    used_slots = np.zeros(MAX_PLANETS, dtype=np.int64)
    target_mask = candidate_target_mask(obs, player)

    labelled = 0
    skipped = 0
    for move in action:
        if not isinstance(move, list) or len(move) < 3:
            skipped += 1
            continue
        src_idx = id_to_idx.get(int(move[0]))
        if src_idx is None or src_idx >= len(planets):
            skipped += 1
            continue
        source = planets[src_idx]
        if int(source[1]) != player:
            skipped += 1
            continue
        slot = int(used_slots[src_idx])
        if slot >= ACTION_SLOTS:
            skipped += 1
            continue
        ships = max(1, int(float(move[2])))
        target_id = infer_target_planet_id(obs, int(move[0]), float(move[1]), ships)
        tgt_idx = id_to_idx.get(int(target_id)) if target_id is not None else None
        if tgt_idx is None or tgt_idx == src_idx or tgt_idx >= len(planets):
            skipped += 1
            continue
        target = planets[tgt_idx]
        launch_actions[src_idx, slot] = 1
        target_actions[src_idx, slot] = int(tgt_idx)
        ship_actions[src_idx, slot] = _bucket_for_action(obs, player, source, target, ships)
        launch_mask[src_idx, slot] = True
        target_mask[src_idx, tgt_idx] = True
        used_slots[src_idx] += 1
        labelled += 1

    if not action and not np.any(enc.own_mask):
        return None
    row = BCRow(
        planets=enc.planets.astype(np.float32, copy=False),
        pair_features=enc.pair_features.astype(np.float32, copy=False),
        global_features=enc.global_features.astype(np.float32, copy=False),
        planet_mask=enc.planet_mask.astype(np.bool_, copy=False),
        own_mask=enc.own_mask.astype(np.bool_, copy=False),
        target_safety_mask=target_mask.astype(np.bool_, copy=False),
        launch_actions=launch_actions,
        target_actions=target_actions,
        ship_actions=ship_actions,
        launch_mask=launch_mask,
    )
    row.labelled = labelled  # type: ignore[attr-defined]
    row.skipped = skipped  # type: ignore[attr-defined]
    return row


def _collect_one_game(
    seed: int,
    players: int,
    episode_steps: int,
    keep_noop_prob: float,
    sample_stride: int,
    rows_per_game: int,
    use_numba: bool,
) -> tuple[list[BCRow], dict[str, float]]:
    random.seed(seed)
    np.random.seed(seed)
    raw_rows: list[tuple[dict[str, Any], int, list[list]]] = []
    env = make_fast_orbit_wars(
        {"episodeSteps": episode_steps, "seed": seed},
        keep_history=False,
        use_numba=use_numba,
    )
    agents = [LoggingRuleAgent(pid, raw_rows, keep_noop_prob) for pid in range(players)]
    env.run(agents)
    sampled = raw_rows[:: max(1, sample_stride)]
    if rows_per_game > 0 and len(sampled) > rows_per_game:
        sampled = random.sample(sampled, rows_per_game)

    rows: list[BCRow] = []
    labelled_actions = 0
    skipped_actions = 0
    for obs, player, action in sampled:
        row = row_from_regular_action(obs, player, action, players=players)
        if row is None:
            continue
        rows.append(row)
        labelled_actions += int(getattr(row, "labelled", 0))
        skipped_actions += int(getattr(row, "skipped", 0))
    return rows, {
        "games": 1.0,
        "samples": float(len(rows)),
        "labelled_actions": float(labelled_actions),
        "skipped_actions": float(skipped_actions),
    }


def _players_list(raw: str) -> list[int]:
    players = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not players or any(p < 2 for p in players):
        raise ValueError(f"invalid players list: {raw!r}")
    return players


def collect_rows(args: argparse.Namespace) -> tuple[list[BCRow], dict[str, float]]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    rows: list[BCRow] = []
    labelled_actions = 0
    skipped_actions = 0

    players_values = _players_list(args.players_list)
    games_per_players = args.games_per_players if args.games_per_players > 0 else args.games
    jobs = [
        (args.seed + players * 1_000_000 + game, players)
        for players in players_values
        for game in range(games_per_players)
    ]

    def add_result(game_rows: list[BCRow], metrics: dict[str, float]) -> bool:
        nonlocal labelled_actions, skipped_actions
        rows.extend(game_rows)
        labelled_actions += int(metrics.get("labelled_actions", 0.0))
        skipped_actions += int(metrics.get("skipped_actions", 0.0))
        return bool(args.max_samples > 0 and len(rows) >= args.max_samples)

    iterator = jobs
    if args.progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="collect regular BC games", dynamic_ncols=True)

    if args.collect_workers <= 1:
        completed_games = 0
        for seed, players in iterator:
            game_rows, metrics = _collect_one_game(
                seed,
                players,
                args.episode_steps,
                args.keep_noop_prob,
                args.sample_stride,
                args.rows_per_game,
                not args.no_numba,
            )
            completed_games += 1
            if add_result(game_rows, metrics):
                break
        return rows[: args.max_samples if args.max_samples > 0 else None], {
            "games": float(completed_games),
            "samples": float(min(len(rows), args.max_samples) if args.max_samples > 0 else len(rows)),
            "labelled_actions": float(labelled_actions),
            "skipped_actions": float(skipped_actions),
            "players_modes": float(len(players_values)),
        }

    completed_games = 0
    with ProcessPoolExecutor(max_workers=args.collect_workers) as pool:
        futures = [
            pool.submit(
                _collect_one_game,
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
        future_iter = as_completed(futures)
        if args.progress:
            from tqdm.auto import tqdm

            future_iter = tqdm(future_iter, total=len(futures), desc="collect regular BC games", dynamic_ncols=True)
        for future in future_iter:
            game_rows, metrics = future.result()
            completed_games += 1
            if add_result(game_rows, metrics):
                for pending in futures:
                    pending.cancel()
                break
    return rows[: args.max_samples if args.max_samples > 0 else None], {
        "games": float(completed_games),
        "samples": float(min(len(rows), args.max_samples) if args.max_samples > 0 else len(rows)),
        "labelled_actions": float(labelled_actions),
        "skipped_actions": float(skipped_actions),
        "players_modes": float(len(players_values)),
    }


def stack_rows(rows: list[BCRow]) -> TensorDataset:
    arrays = {
        "planets": np.stack([r.planets for r in rows]),
        "pair_features": np.stack([r.pair_features for r in rows]),
        "global_features": np.stack([r.global_features for r in rows]),
        "planet_mask": np.stack([r.planet_mask for r in rows]),
        "own_mask": np.stack([r.own_mask for r in rows]),
        "target_safety_mask": np.stack([r.target_safety_mask for r in rows]),
        "launch_actions": np.stack([r.launch_actions for r in rows]),
        "target_actions": np.stack([r.target_actions for r in rows]),
        "ship_actions": np.stack([r.ship_actions for r in rows]),
        "launch_mask": np.stack([r.launch_mask for r in rows]),
    }
    tensors = []
    for key in (
        "planets",
        "pair_features",
        "global_features",
        "planet_mask",
        "own_mask",
        "target_safety_mask",
        "launch_actions",
        "target_actions",
        "ship_actions",
        "launch_mask",
    ):
        arr = arrays[key]
        dtype = torch.float32 if arr.dtype.kind == "f" else torch.bool if arr.dtype == np.bool_ else torch.long
        tensors.append(torch.tensor(arr, dtype=dtype))
    return TensorDataset(*tensors)


def unpack(batch: tuple[torch.Tensor, ...], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "planets",
        "pair_features",
        "global_features",
        "planet_mask",
        "own_mask",
        "target_safety_mask",
        "launch_actions",
        "target_actions",
        "ship_actions",
        "launch_mask",
    )
    return {k: v.to(device, non_blocking=True) for k, v in zip(keys, batch, strict=True)}


def _weighted_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits, targets, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def bc_loss(
    model: TinyPolicyValueNet,
    batch: dict[str, torch.Tensor],
    launch_pos_weight: float,
    target_loss_weight: float = 1.0,
    ship_loss_weight: float = 0.5,
    critical_action_weight: float = 0.0,
    target_loss_mask: str = "dataset",
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(batch["planets"], batch["pair_features"], batch["global_features"], batch["planet_mask"], batch["own_mask"])
    own_slots = batch["own_mask"][:, :, None].expand_as(batch["launch_actions"])
    launch_logits = out["source_logits"][own_slots]
    launch_targets = batch["launch_actions"][own_slots]
    launch_loss = F.cross_entropy(
        launch_logits,
        launch_targets,
        weight=torch.tensor([1.0, launch_pos_weight], dtype=torch.float32, device=launch_logits.device),
    )

    active = batch["launch_mask"] & own_slots
    target_loss = torch.tensor(0.0, device=launch_logits.device)
    ship_loss = torch.tensor(0.0, device=launch_logits.device)
    target_acc = torch.tensor(0.0, device=launch_logits.device)
    ship_acc = torch.tensor(0.0, device=launch_logits.device)
    action_weight_mean = torch.tensor(0.0, device=launch_logits.device)
    if active.any():
        if target_loss_mask == "all_planets":
            target_mask = batch["planet_mask"][:, None, :].expand(-1, batch["target_safety_mask"].shape[1], -1).clone()
            eye = torch.eye(target_mask.shape[1], dtype=torch.bool, device=target_mask.device)[None, :, :]
            target_mask = target_mask & ~eye
        elif target_loss_mask == "dataset":
            target_mask = batch["target_safety_mask"]
        else:
            raise ValueError(f"unsupported target_loss_mask: {target_loss_mask!r}")
        target_logits = out["target_logits"].masked_fill(~target_mask[:, :, None, :], -1e9)
        active_weights = torch.ones_like(batch["target_actions"][active], dtype=torch.float32, device=launch_logits.device)
        if critical_action_weight > 0.0:
            b, s, slot = torch.where(active)
            del slot
            t = batch["target_actions"][active]
            source_ship = batch["planets"][b, s, 6].clamp(0.0, 1.5)
            source_prod = batch["planets"][b, s, 7].clamp(0.0, 2.0)
            target_prod = batch["planets"][b, t, 7].clamp(0.0, 2.0)
            target_enemy = batch["pair_features"][b, s, t, 10].clamp(0.0, 1.0)
            target_neutral = batch["pair_features"][b, s, t, 9].clamp(0.0, 1.0)
            incoming_enemy = batch["planets"][b, t, 14].clamp(0.0, 2.0)
            importance = 0.35 * source_ship + 0.25 * source_prod + 0.35 * target_prod + 0.35 * target_enemy + 0.15 * target_neutral + 0.30 * incoming_enemy
            active_weights = active_weights + critical_action_weight * importance.clamp(0.0, 2.0)
        action_weight_mean = active_weights.mean()
        target_loss = _weighted_cross_entropy(target_logits[active], batch["target_actions"][active], active_weights)
        target_pred = target_logits[active].argmax(dim=-1)
        target_acc = (target_pred == batch["target_actions"][active]).float().mean()
        if "ship_logits" in out:
            ship_logits_all = out["ship_logits"]
            b, s, slot = torch.where(active)
            t = batch["target_actions"][active]
            ship_logits = ship_logits_all[b, s, slot, t]
            ship_loss = _weighted_cross_entropy(ship_logits, batch["ship_actions"][active], active_weights)
            ship_acc = (ship_logits.argmax(dim=-1) == batch["ship_actions"][active]).float().mean()

    loss = launch_loss + target_loss_weight * target_loss + ship_loss_weight * ship_loss
    launch_pred = out["source_logits"][own_slots].argmax(dim=-1)
    launch_acc = (launch_pred == launch_targets).float().mean()
    pos = launch_targets == 1
    pred_pos = launch_pred == 1
    true_count = batch["launch_actions"].masked_fill(~own_slots, 0).sum(dim=(1, 2)).float()
    pred_count = (out["source_logits"].argmax(dim=-1) == 1).masked_fill(~own_slots, False).sum(dim=(1, 2)).float()
    tp = (pred_pos & pos).float().sum()
    fp = (pred_pos & ~pos).float().sum()
    fn = (~pred_pos & pos).float().sum()
    launch_precision = tp / (tp + fp).clamp_min(1.0)
    launch_recall = (launch_pred[pos] == 1).float().mean() if pos.any() else torch.tensor(0.0, device=launch_logits.device)
    launch_f1 = 2.0 * launch_precision * launch_recall / (launch_precision + launch_recall).clamp_min(1e-6)
    launch_pred_rate = pred_pos.float().mean()
    launch_true_rate = pos.float().mean()
    action_count_mae = (pred_count - true_count).abs().mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "launch_loss": float(launch_loss.detach().cpu()),
        "target_loss": float(target_loss.detach().cpu()),
        "ship_loss": float(ship_loss.detach().cpu()),
        "launch_acc": float(launch_acc.detach().cpu()),
        "launch_precision": float(launch_precision.detach().cpu()),
        "launch_recall": float(launch_recall.detach().cpu()),
        "launch_f1": float(launch_f1.detach().cpu()),
        "launch_pred_rate": float(launch_pred_rate.detach().cpu()),
        "launch_true_rate": float(launch_true_rate.detach().cpu()),
        "action_count_mae": float(action_count_mae.detach().cpu()),
        "target_acc": float(target_acc.detach().cpu()),
        "ship_acc": float(ship_acc.detach().cpu()),
        "action_weight_mean": float(action_weight_mean.detach().cpu()),
    }


@torch.no_grad()
def evaluate_loader(
    model: TinyPolicyValueNet,
    loader: DataLoader,
    device: torch.device,
    launch_pos_weight: float,
    target_loss_weight: float = 1.0,
    ship_loss_weight: float = 0.5,
    critical_action_weight: float = 0.0,
    target_loss_mask: str = "dataset",
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        _, metrics = bc_loss(
            model,
            unpack(batch, device),
            launch_pos_weight,
            target_loss_weight,
            ship_loss_weight,
            critical_action_weight,
            target_loss_mask,
        )
        n = int(batch[0].shape[0])
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value * n
        count += n
    return {key: value / max(1, count) for key, value in sums.items()}


def train_bc(args: argparse.Namespace, dataset: TensorDataset) -> tuple[TinyPolicyValueNet, dict[str, float]]:
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = max(1, len(dataset) - val_size)
    train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.loader_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.loader_workers, pin_memory=device.type == "cuda")

    model_cfg = {
        "hidden": args.hidden,
        "heads": args.heads,
        "layers": args.layers,
        "ship_buckets": len(SHIP_BUCKET_MULTIPLIERS),
        "action_slots": ACTION_SLOTS,
        "source_target_summary": bool(args.source_target_summary),
    }
    model = TinyPolicyValueNet(**model_cfg).to(device)
    trainable_params = list(model.parameters())
    if args.trainable_modules == "target_head":
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.target_head.parameters():
            param.requires_grad_(True)
        trainable_params = list(model.target_head.parameters())
    elif args.trainable_modules != "all":
        raise ValueError(f"unsupported trainable_modules: {args.trainable_modules!r}")
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_metrics: dict[str, float] = {}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sums: dict[str, float] = {}
        train_count = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, metrics = bc_loss(
                model,
                unpack(batch, device),
                args.launch_pos_weight,
                args.target_loss_weight,
                args.ship_loss_weight,
                args.critical_action_weight,
                args.target_loss_mask,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            n = int(batch[0].shape[0])
            for key, value in metrics.items():
                train_sums[key] = train_sums.get(key, 0.0) + value * n
            train_count += n
        train_metrics = {f"train_{key}": value / max(1, train_count) for key, value in train_sums.items()}
        val_metrics = {
            f"val_{key}": value
            for key, value in evaluate_loader(
                model,
                val_loader,
                device,
                args.launch_pos_weight,
                args.target_loss_weight,
                args.ship_loss_weight,
                args.critical_action_weight,
                args.target_loss_mask,
            ).items()
        }
        merged = {"epoch": float(epoch), **train_metrics, **val_metrics}
        print(json.dumps(merged, ensure_ascii=True), flush=True)
        if best_state is None or merged["val_loss"] < best_metrics.get("val_loss", math.inf):
            best_metrics = merged
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.model_config = model_cfg  # type: ignore[attr-defined]
    return model, best_metrics


def save_checkpoint(path: Path, model: TinyPolicyValueNet, args: argparse.Namespace, collect_metrics: dict[str, float], train_metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = getattr(model, "model_config")
    payload = {
        "state_dict": model.state_dict(),
        "model": cfg,
        "update": 0,
        "metrics": {
            "kind": "regular_bc",
            "collect": collect_metrics,
            "train": train_metrics,
        },
        "args": vars(args),
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tinyPPO/regular_bc.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--games-per-players", type=int, default=0, help="Games for each value in --players-list. Defaults to --games.")
    parser.add_argument("--players-list", default="2", help="Comma list of player counts to collect, e.g. 2,4.")
    parser.add_argument("--collect-workers", type=int, default=1)
    parser.add_argument("--rows-per-game", type=int, default=0, help="Randomly cap labelled observation rows per game. 0 keeps all sampled rows.")
    parser.add_argument("--max-samples", type=int, default=1536, help="Global cap after collection. 0 disables the cap.")
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--keep-noop-prob", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=260525)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--launch-pos-weight", type=float, default=8.0)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--ship-loss-weight", type=float, default=0.5)
    parser.add_argument("--critical-action-weight", type=float, default=0.0)
    parser.add_argument("--target-loss-mask", choices=["dataset", "all_planets"], default="dataset")
    parser.add_argument("--trainable-modules", choices=["all", "target_head"], default="all")
    parser.add_argument("--val-frac", type=float, default=0.12)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--source-target-summary", action="store_true", help="Let the launch/source head see a pooled summary of source-target edge features.")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    rows, collect_metrics = collect_rows(args)
    if not rows:
        raise RuntimeError("no BC rows collected")
    print(json.dumps({"collect": collect_metrics}, ensure_ascii=True), flush=True)
    dataset = stack_rows(rows)
    model, train_metrics = train_bc(args, dataset)
    save_checkpoint(Path(args.out), model, args, collect_metrics, train_metrics)
    print(json.dumps({"saved": args.out, "metrics": train_metrics}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
