"""Training utilities for search-improved AlphaZero-like samples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import numpy as np

from alphaZeroLike.features import encode_search_position
from alphaZeroLike.model import AlphaZeroLikeNet
from alphaZeroLike.compat import load_training2_stage1_backbone


def _batch_rows(rows: list[dict], device: torch.device | str):
    max_candidates = max(len(row["policy_target"]) for row in rows)
    encoded = [
        encode_search_position(
            row["obs"],
            int(row["player"]),
            row["candidates"],
            max_candidates=max_candidates,
            max_moves=int(row.get("max_moves", 16)),
        )
        for row in rows
    ]
    policy_targets = np.zeros((len(rows), max_candidates), dtype=np.float32)
    for i, row in enumerate(rows):
        target = np.asarray(row["policy_target"], dtype=np.float32)
        policy_targets[i, : len(target)] = target
    return (
        torch.tensor(np.stack([e.planet_features for e in encoded]), dtype=torch.float32, device=device),
        torch.tensor(np.stack([e.global_features for e in encoded]), dtype=torch.float32, device=device),
        torch.tensor(np.stack([e.move_features for e in encoded]), dtype=torch.float32, device=device),
        torch.tensor(np.stack([e.move_source_indices for e in encoded]), dtype=torch.long, device=device),
        torch.tensor(np.stack([e.move_target_indices for e in encoded]), dtype=torch.long, device=device),
        torch.tensor(np.stack([e.move_mask for e in encoded]), dtype=torch.float32, device=device),
        torch.tensor(np.stack([e.candidate_mask for e in encoded]), dtype=torch.float32, device=device),
        torch.tensor(policy_targets, dtype=torch.float32, device=device),
        torch.tensor([row["value"] for row in rows], dtype=torch.float32, device=device),
    )


def train_batch(
    model: AlphaZeroLikeNet,
    opt: torch.optim.Optimizer,
    rows: list[dict],
    *,
    device: torch.device | str = "cpu",
    value_weight: float = 1.0,
    entropy_weight: float = 0.01,
    proposal_weight: float = 0.25,
) -> dict[str, float]:
    (
        planets,
        glob,
        moves,
        sources,
        targets,
        move_mask,
        candidate_mask,
        policy_target,
        value_target,
    ) = _batch_rows(rows, device)
    logits, value = model(planets, glob, moves, sources, targets, move_mask, candidate_mask)
    logp = torch.log_softmax(logits, dim=-1)
    policy_loss = -(policy_target * logp).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, value_target)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * logp).sum(dim=-1).mean()
    target_entropy = -(policy_target * torch.log(policy_target.clamp_min(1e-8))).sum(dim=-1).mean()
    target_max = policy_target.max(dim=-1).values.mean()
    policy_max = probs.max(dim=-1).values.mean()
    candidate_count = candidate_mask.sum(dim=-1).mean()
    proposal_loss = torch.tensor(0.0, device=device)
    proposal_send_acc = torch.tensor(0.0, device=device)
    if any("proposal_valid" in row for row in rows):
        n_entities = planets.size(1)

        def pad_float(key: str) -> torch.Tensor:
            out = torch.zeros((len(rows), n_entities), dtype=torch.float32, device=device)
            for i, row in enumerate(rows):
                vals = row.get(key, [])[:n_entities]
                if vals:
                    out[i, : len(vals)] = torch.tensor(vals, dtype=torch.float32, device=device)
            return out

        def pad_long(key: str) -> torch.Tensor:
            out = torch.zeros((len(rows), n_entities), dtype=torch.long, device=device)
            for i, row in enumerate(rows):
                vals = row.get(key, [])[:n_entities]
                if vals:
                    out[i, : len(vals)] = torch.tensor(vals, dtype=torch.long, device=device)
            return out

        entity_valid = (planets.abs().sum(dim=-1) > 0.0).float()
        prop_valid = pad_float("proposal_valid") * entity_valid
        ship_valid = planets[..., 6] > 0.0
        source_valid = (prop_valid > 0.0) & ship_valid
        target_valid = planets[..., 17] > 0.5
        if not bool(target_valid.any()):
            target_valid = entity_valid > 0.0
        proposal = model.proposal(planets, glob, source_mask=source_valid, target_mask=target_valid)
        prop_send = pad_float("proposal_send")
        prop_target = pad_long("proposal_target")
        prop_ship = pad_float("proposal_ship_ratio")
        valid_denom = prop_valid.sum().clamp(min=1.0)
        active = prop_valid * prop_send
        active_denom = active.sum().clamp(min=1.0)
        send_loss = F.binary_cross_entropy_with_logits(
            proposal["send_logits"],
            prop_send,
            weight=prop_valid,
            reduction="sum",
        ) / valid_denom
        target_loss = F.cross_entropy(
            proposal["target_logits"].reshape(-1, n_entities),
            prop_target.reshape(-1),
            reduction="none",
        ).reshape_as(prop_send)
        target_loss = (target_loss * active).sum() / active_denom
        ship_loss = (
            F.smooth_l1_loss(torch.sigmoid(proposal["ship_logits"]), prop_ship, reduction="none") * active
        ).sum() / active_denom
        proposal_loss = send_loss + target_loss + ship_loss
        proposal_send_acc = (((proposal["send_logits"].sigmoid() >= 0.5).float() == prop_send).float() * prop_valid).sum() / valid_denom

    loss = policy_loss + value_weight * value_loss + proposal_weight * proposal_loss - entropy_weight * entropy

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "proposal_loss": float(proposal_loss.item()),
        "proposal_weight": float(proposal_weight),
        "proposal_send_acc": float(proposal_send_acc.item()),
        "entropy": float(entropy.item()),
        "target_entropy": float(target_entropy.item()),
        "target_max": float(target_max.item()),
        "policy_max": float(policy_max.item()),
        "candidate_count": float(candidate_count.item()),
        "value_pred_mean": float(value.mean().item()),
        "value_target_mean": float(value_target.mean().item()),
    }


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--out", default="alphaZeroLike/checkpoints/latest.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--proposal-weight", type=float, default=0.25)
    parser.add_argument("--resume", help="Optional alphaZeroLike checkpoint to continue training.")
    parser.add_argument(
        "--init-from-training2",
        help="Optional training2 stage1 checkpoint; loads compatible backbone/value tensors.",
    )
    args = parser.parse_args()

    rows = [row for path in args.data for row in iter_jsonl(path)]
    model = AlphaZeroLikeNet().to(args.device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print({"resume": args.resume})
    elif args.init_from_training2:
        report = load_training2_stage1_backbone(model, args.init_from_training2, map_location=args.device)
        print(
            {
                "init_from_training2": args.init_from_training2,
                "loaded_tensors": len(report["loaded"]),
                "skipped_shape_mismatch": len(report["skipped_shape_mismatch"]),
                "missing_source": len(report["missing_source"]),
            }
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(rows)
        for start in range(0, len(rows), args.batch_size):
            metrics = train_batch(
                model,
                opt,
                rows[start : start + args.batch_size],
                device=args.device,
                value_weight=args.value_weight,
                entropy_weight=args.entropy_weight,
                proposal_weight=args.proposal_weight,
            )
        print({"epoch": epoch, **metrics})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, out)


if __name__ == "__main__":
    main()
