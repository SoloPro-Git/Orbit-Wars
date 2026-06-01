from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from tinyPPO.features import MAX_PLANETS
from tinyPPO.imitation_regular import BCRow
from tinyPPOv2.model import AutoregressivePolicyNet


def load_rows(path: Path, max_rows: int = 0, seed: int = 0) -> list[BCRow]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    rows = list(rows)
    if max_rows > 0 and len(rows) > max_rows:
        rng = random.Random(seed)
        rows = [rows[idx] for idx in sorted(rng.sample(range(len(rows)), max_rows))]
    return rows


def _row_actions(row: Any, max_actions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    active = np.argwhere(np.asarray(row.launch_mask, dtype=np.bool_))
    active = active[np.lexsort((active[:, 1], active[:, 0]))] if active.size else active
    truncated = max(0, int(active.shape[0]) - int(max_actions))
    active = active[:max_actions]
    source = np.full((max_actions,), MAX_PLANETS, dtype=np.int64)
    target = np.full((max_actions,), MAX_PLANETS, dtype=np.int64)
    ship = np.full((max_actions,), 0, dtype=np.int64)
    for out_idx, (source_idx, slot_idx) in enumerate(active):
        source[out_idx] = int(source_idx)
        target[out_idx] = int(row.target_actions[int(source_idx), int(slot_idx)])
        ship[out_idx] = int(row.ship_actions[int(source_idx), int(slot_idx)])
    return source, target, ship, int(active.shape[0]), truncated


def stack_rows_v2(rows: list[Any], max_actions: int) -> tuple[TensorDataset, dict[str, float]]:
    planets = np.stack([row.planets for row in rows]).astype(np.float32, copy=False)
    pair_features = np.stack([row.pair_features for row in rows]).astype(np.float32, copy=False)
    global_features = np.stack([row.global_features for row in rows]).astype(np.float32, copy=False)
    planet_mask = np.stack([row.planet_mask for row in rows]).astype(np.bool_, copy=False)
    own_mask = np.stack([row.own_mask for row in rows]).astype(np.bool_, copy=False)

    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ships: list[np.ndarray] = []
    lengths: list[int] = []
    truncated_total = 0
    for row in rows:
        source, target, ship, length, truncated = _row_actions(row, max_actions)
        sources.append(source)
        targets.append(target)
        ships.append(ship)
        lengths.append(length)
        truncated_total += truncated

    dataset = TensorDataset(
        torch.tensor(planets, dtype=torch.float32),
        torch.tensor(pair_features, dtype=torch.float32),
        torch.tensor(global_features, dtype=torch.float32),
        torch.tensor(planet_mask, dtype=torch.bool),
        torch.tensor(own_mask, dtype=torch.bool),
        torch.tensor(np.stack(sources), dtype=torch.long),
        torch.tensor(np.stack(targets), dtype=torch.long),
        torch.tensor(np.stack(ships), dtype=torch.long),
        torch.tensor(np.asarray(lengths), dtype=torch.long),
    )
    metrics = {
        "rows": float(len(rows)),
        "labelled_actions": float(sum(lengths) + truncated_total),
        "kept_actions": float(sum(lengths)),
        "truncated_actions": float(truncated_total),
        "max_actions": float(max_actions),
    }
    return dataset, metrics


def unpack(batch: tuple[torch.Tensor, ...], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "planets",
        "pair_features",
        "global_features",
        "planet_mask",
        "own_mask",
        "source_actions",
        "target_actions",
        "ship_actions",
        "action_lengths",
    )
    return {key: value.to(device, non_blocking=True) for key, value in zip(keys, batch, strict=True)}


def _teacher_inputs(batch: dict[str, torch.Tensor], max_actions: int, ship_buckets: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = batch["source_actions"].shape[0]
    none_source = torch.full((bsz, 1), MAX_PLANETS, dtype=torch.long, device=batch["source_actions"].device)
    none_ship = torch.full((bsz, 1), ship_buckets, dtype=torch.long, device=batch["source_actions"].device)
    prev_source = torch.cat([none_source, batch["source_actions"][:, :max_actions]], dim=1)
    prev_target = torch.cat([none_source, batch["target_actions"][:, :max_actions]], dim=1)
    prev_ship = torch.cat([none_ship, batch["ship_actions"][:, :max_actions].clamp(0, ship_buckets - 1)], dim=1)
    return prev_source, prev_target, prev_ship


def bc_loss_v2(
    model: AutoregressivePolicyNet,
    batch: dict[str, torch.Tensor],
    stop_loss_weight: float = 1.0,
    source_loss_weight: float = 1.0,
    target_loss_weight: float = 1.0,
    ship_loss_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    max_actions = int(model.max_actions)
    prev_source, prev_target, prev_ship = _teacher_inputs(batch, max_actions, int(model.ship_buckets))
    out = model(
        batch["planets"],
        batch["pair_features"],
        batch["global_features"],
        batch["planet_mask"],
        batch["own_mask"],
        prev_source=prev_source,
        prev_target=prev_target,
        prev_ship=prev_ship,
    )
    device = batch["source_actions"].device
    step_idx = torch.arange(max_actions + 1, device=device)[None, :]
    lengths = batch["action_lengths"].clamp(0, max_actions)
    continue_labels = (step_idx < lengths[:, None]).long()
    stop_loss = F.cross_entropy(out["continue_logits"].reshape(-1, 2), continue_labels.reshape(-1))

    active = step_idx[:, :max_actions] < lengths[:, None]
    if active.any():
        b, t = torch.where(active)
        source_logits = out["source_logits"][:, :max_actions][b, t]
        target_logits = out["target_logits"][:, :max_actions][b, t]
        ship_logits = out["ship_logits"][:, :max_actions][b, t]
        source_targets = batch["source_actions"][b, t]
        target_targets = batch["target_actions"][b, t]
        ship_targets = batch["ship_actions"][b, t].clamp(0, int(model.ship_buckets) - 1)
        source_loss = F.cross_entropy(source_logits, source_targets)
        target_loss = F.cross_entropy(target_logits, target_targets)
        ship_loss = F.cross_entropy(ship_logits, ship_targets)
        source_acc = (source_logits.argmax(dim=-1) == source_targets).float().mean()
        target_acc = (target_logits.argmax(dim=-1) == target_targets).float().mean()
        ship_acc = (ship_logits.argmax(dim=-1) == ship_targets).float().mean()
    else:
        zero = torch.tensor(0.0, device=device)
        source_loss = target_loss = ship_loss = zero
        source_acc = target_acc = ship_acc = zero

    pred_continue = out["continue_logits"].argmax(dim=-1)
    stop_acc = (pred_continue == continue_labels).float().mean()
    pred_lengths = pred_continue[:, :max_actions].sum(dim=1).float()
    count_mae = (pred_lengths - lengths.float()).abs().mean()
    loss = (
        stop_loss_weight * stop_loss
        + source_loss_weight * source_loss
        + target_loss_weight * target_loss
        + ship_loss_weight * ship_loss
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "stop_loss": float(stop_loss.detach().cpu()),
        "source_loss": float(source_loss.detach().cpu()),
        "target_loss": float(target_loss.detach().cpu()),
        "ship_loss": float(ship_loss.detach().cpu()),
        "stop_acc": float(stop_acc.detach().cpu()),
        "source_acc": float(source_acc.detach().cpu()),
        "target_acc": float(target_acc.detach().cpu()),
        "ship_acc": float(ship_acc.detach().cpu()),
        "action_count_mae": float(count_mae.detach().cpu()),
        "label_actions_per_row": float(lengths.float().mean().detach().cpu()),
        "pred_actions_per_row": float(pred_lengths.mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate(model: AutoregressivePolicyNet, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for raw_batch in loader:
        batch = unpack(raw_batch, device)
        _loss, metrics = bc_loss_v2(
            model,
            batch,
            stop_loss_weight=args.stop_loss_weight,
            source_loss_weight=args.source_loss_weight,
            target_loss_weight=args.target_loss_weight,
            ship_loss_weight=args.ship_loss_weight,
        )
        n = int(batch["planets"].shape[0])
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value) * n
        count += n
    return {key: value / max(1, count) for key, value in sums.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(Path(args.dataset_cache), max_rows=args.max_rows, seed=args.seed)
    dataset, data_metrics = stack_rows_v2(rows, args.max_actions)
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = max(1, len(dataset) - val_size)
    train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=device.type == "cuda",
    )
    model_cfg = {
        "hidden": args.hidden,
        "heads": args.heads,
        "layers": args.layers,
        "ship_buckets": args.ship_buckets,
        "max_actions": args.max_actions,
    }
    model = AutoregressivePolicyNet(**model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_metrics: dict[str, float] = {}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        count = 0
        for raw_batch in train_loader:
            batch = unpack(raw_batch, device)
            loss, metrics = bc_loss_v2(
                model,
                batch,
                stop_loss_weight=args.stop_loss_weight,
                source_loss_weight=args.source_loss_weight,
                target_loss_weight=args.target_loss_weight,
                ship_loss_weight=args.ship_loss_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            n = int(batch["planets"].shape[0])
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value) * n
            count += n
        train_metrics = {key: value / max(1, count) for key, value in sums.items()}
        val_metrics = evaluate(model, val_loader, device, args)
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_metrics = dict(val_metrics)
            torch.save({"model": model_cfg, "state_dict": model.state_dict(), "epoch": epoch}, out_path)
        if epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train": train_metrics,
                        "val": val_metrics,
                        "best_val_loss": best_val,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    return {"data": data_metrics, "best_val": best_metrics, "out": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaStar-lite autoregressive BC policy on regular-labelled rows.")
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ship-buckets", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-frac", type=float, default=0.08)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--stop-loss-weight", type=float, default=1.0)
    parser.add_argument("--source-loss-weight", type=float, default=1.0)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--ship-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=260606)
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
