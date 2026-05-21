"""Pretrain AlphaZeroLike proposal heads from training2 stage-1 JSONL rows."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from alphaZeroLike.compat import load_training2_stage1_backbone
from alphaZeroLike.features import GLOBAL_DIM, PLANET_DIM
from alphaZeroLike.model import AlphaZeroLikeNet


def _iter_jsonl_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_dir():
        return sorted(root.glob("*.jsonl"))
    return [root]


def _iter_rows(paths: list[Path], *, shuffle_files: bool, seed: int) -> Iterable[dict]:
    rng = random.Random(seed)
    while True:
        order = list(paths)
        if shuffle_files:
            rng.shuffle(order)
        for path in order:
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "proposal_valid" in row and "planets" in row and "global" in row:
                        yield row


def _has_active_launch(row: dict) -> bool:
    return any(float(send) > 0.5 and float(valid) > 0.5 for send, valid in zip(row.get("proposal_send", []), row.get("proposal_valid", [])))


def _next_balanced_batch(
    row_iter: Iterable[dict],
    batch_size: int,
    *,
    active_row_frac: float,
) -> list[dict]:
    active_target = int(round(batch_size * min(max(active_row_frac, 0.0), 1.0)))
    inactive_target = batch_size - active_target
    active_rows: list[dict] = []
    inactive_rows: list[dict] = []
    overflow_inactive: list[dict] = []
    max_scan = max(batch_size * 2000, 1)
    scanned = 0
    while scanned < max_scan and (len(active_rows) < active_target or len(inactive_rows) < inactive_target):
        scanned += 1
        row = next(row_iter)
        if _has_active_launch(row):
            if len(active_rows) < active_target:
                active_rows.append(row)
            elif len(inactive_rows) < inactive_target:
                inactive_rows.append(row)
        else:
            if len(inactive_rows) < inactive_target:
                inactive_rows.append(row)
            else:
                overflow_inactive.append(row)
    while len(active_rows) + len(inactive_rows) < batch_size:
        inactive_rows.append(overflow_inactive.pop() if overflow_inactive else next(row_iter))
    rows = active_rows + inactive_rows
    random.shuffle(rows)
    return rows


def _pad_planets(rows: list[dict], device: torch.device | str) -> torch.Tensor:
    max_entities = max(len(row["planets"]) for row in rows)
    out = np.zeros((len(rows), max_entities, PLANET_DIM), dtype=np.float32)
    for i, row in enumerate(rows):
        planets = np.asarray(row["planets"], dtype=np.float32)
        dims = min(planets.shape[1], PLANET_DIM)
        out[i, : planets.shape[0], :dims] = planets[:, :dims]
    return torch.tensor(out, dtype=torch.float32, device=device)


def _pad_global(rows: list[dict], device: torch.device | str) -> torch.Tensor:
    out = np.zeros((len(rows), GLOBAL_DIM), dtype=np.float32)
    for i, row in enumerate(rows):
        glob = np.asarray(row["global"], dtype=np.float32)
        dims = min(glob.shape[0], GLOBAL_DIM)
        out[i, :dims] = glob[:dims]
    return torch.tensor(out, dtype=torch.float32, device=device)


def _pad_float(rows: list[dict], key: str, n_entities: int, device: torch.device | str) -> torch.Tensor:
    out = torch.zeros((len(rows), n_entities), dtype=torch.float32, device=device)
    for i, row in enumerate(rows):
        vals = row.get(key, [])[:n_entities]
        if vals:
            out[i, : len(vals)] = torch.tensor(vals, dtype=torch.float32, device=device)
    return out


def _pad_long(rows: list[dict], key: str, n_entities: int, device: torch.device | str) -> torch.Tensor:
    out = torch.zeros((len(rows), n_entities), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        vals = row.get(key, [])[:n_entities]
        if vals:
            out[i, : len(vals)] = torch.tensor(vals, dtype=torch.long, device=device)
    return out


def proposal_pretrain_batch(
    model: AlphaZeroLikeNet,
    opt: torch.optim.Optimizer,
    rows: list[dict],
    *,
    device: torch.device | str,
) -> dict[str, float]:
    planets = _pad_planets(rows, device)
    glob = _pad_global(rows, device)
    n_entities = planets.size(1)

    entity_valid = (planets.abs().sum(dim=-1) > 0.0).float()
    prop_valid = _pad_float(rows, "proposal_valid", n_entities, device) * entity_valid
    ship_valid = planets[..., 6] > 0.0
    source_valid = (prop_valid > 0.0) & ship_valid
    target_valid = planets[..., 17] > 0.5
    if not bool(target_valid.any()):
        target_valid = entity_valid > 0.0
    proposal = model.proposal(planets, glob, source_mask=source_valid, target_mask=target_valid)
    prop_send = _pad_float(rows, "proposal_send", n_entities, device)
    prop_target = _pad_long(rows, "proposal_target", n_entities, device).clamp(max=max(n_entities - 1, 0))
    prop_ship = _pad_float(rows, "proposal_ship_ratio", n_entities, device)

    valid_denom = prop_valid.sum().clamp(min=1.0)
    active = prop_valid * prop_send
    active_denom = active.sum().clamp(min=1.0)

    send_loss = F.binary_cross_entropy_with_logits(
        proposal["send_logits"],
        prop_send,
        weight=prop_valid,
        reduction="sum",
    ) / valid_denom
    target_loss_flat = F.cross_entropy(
        proposal["target_logits"].reshape(-1, n_entities),
        prop_target.reshape(-1),
        reduction="none",
    ).reshape_as(prop_send)
    target_loss = (target_loss_flat * active).sum() / active_denom
    ship_pred = torch.sigmoid(proposal["ship_logits"])
    ship_loss = (F.smooth_l1_loss(ship_pred, prop_ship, reduction="none") * active).sum() / active_denom
    loss = send_loss + target_loss + ship_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()

    with torch.no_grad():
        send_acc = (((proposal["send_logits"].sigmoid() >= 0.5).float() == prop_send).float() * prop_valid).sum() / valid_denom
        target_pred = proposal["target_logits"].argmax(dim=-1)
        target_acc = ((target_pred == prop_target).float() * active).sum() / active_denom
        ship_mae = ((ship_pred - prop_ship).abs() * active).sum() / active_denom
    return {
        "loss": float(loss.item()),
        "send_loss": float(send_loss.item()),
        "target_loss": float(target_loss.item()),
        "ship_loss": float(ship_loss.item()),
        "send_acc": float(send_acc.item()),
        "target_acc": float(target_acc.item()),
        "ship_mae": float(ship_mae.item()),
        "active": float(active.sum().item()),
    }


def _freeze_except_proposal(model: AlphaZeroLikeNet) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("proposal_")


def _load_model(args: argparse.Namespace, device: str) -> AlphaZeroLikeNet:
    model = AlphaZeroLikeNet(
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(json.dumps({"resume": args.resume}, ensure_ascii=False), flush=True)
    elif args.init_from_training2:
        report = load_training2_stage1_backbone(model, args.init_from_training2, map_location=device)
        print(
            json.dumps(
                {
                    "init_from_training2": args.init_from_training2,
                    "loaded_tensors": len(report["loaded"]),
                    "skipped_shape_mismatch": len(report["skipped_shape_mismatch"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training2/stage1_regular_10k_shuffle")
    parser.add_argument("--out", default="alphaZeroLike/checkpoints/latest.pt")
    parser.add_argument("--resume", default="alphaZeroLike/checkpoints/latest.pt")
    parser.add_argument("--init-from-training2", default="training2/checkpoints/stage1_tactical_entities_20260519/latest.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--active-row-frac", type=float, default=0.75)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--shuffle-files", action="store_true", default=True)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = _iter_jsonl_paths(args.data)
    if not paths:
        raise FileNotFoundError(args.data)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = _load_model(args, device)
    if not args.train_backbone:
        _freeze_except_proposal(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    row_iter = _iter_rows(paths, shuffle_files=args.shuffle_files, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "stage": "proposal_pretrain_stage1",
                "data_files": len(paths),
                "device": device,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "active_row_frac": args.active_row_frac,
                "train_backbone": bool(args.train_backbone),
                "out": str(out),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    ema: dict[str, float] = {}
    for step in range(1, args.steps + 1):
        rows = _next_balanced_batch(row_iter, args.batch_size, active_row_frac=args.active_row_frac)
        metrics = proposal_pretrain_batch(model, opt, rows, device=device)
        for key, value in metrics.items():
            ema[key] = value if key not in ema else 0.95 * ema[key] + 0.05 * value
        if step % args.log_every == 0 or step == 1:
            print(json.dumps({"step": step, **{f"train/{k}": v for k, v in ema.items()}}, ensure_ascii=False), flush=True)
        if args.save_every > 0 and step % args.save_every == 0:
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "step": step}, out)

    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "step": args.steps}, out)
    print(json.dumps({"saved": str(out), "steps": args.steps, "final_loss": ema.get("loss", math.nan)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
