from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tinyPPO.diagnose_bc_cache import (
    _filter_rows,
    _load_rows,
    _prf,
    _row_action_count,
    _target_mask,
    _unpack,
)
from tinyPPO.imitation_regular import stack_rows
from tinyPPO.model import TinyPolicyValueNet


def _gap_bucket(row: Any) -> str:
    gap = getattr(row, "dagger_action_gap", None)
    if gap is None:
        model_count = getattr(row, "dagger_model_action_count", None)
        label_count = getattr(row, "dagger_regular_label_action_count", None)
        if model_count is None or label_count is None:
            return "missing"
        gap = int(label_count) - int(model_count)
    gap = int(gap)
    if gap < 0:
        return "regular_lt_model"
    if gap > 0:
        return "regular_gt_model"
    return "equal"


def _turn_bucket(row: Any) -> str:
    turn = getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", None))
    if turn is None:
        return "missing"
    turn = int(turn)
    if turn < 50:
        return "000_049"
    if turn < 150:
        return "050_149"
    if turn < 250:
        return "150_249"
    if turn < 350:
        return "250_349"
    return "350_plus"


def _outcome_bucket(row: Any) -> str:
    value = getattr(row, "dagger_model_outcome", None)
    if value is not None:
        return str(value)
    reward = getattr(row, "dagger_model_final_reward", None)
    if reward is None:
        return "missing"
    reward = float(reward)
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def _ckpt_bucket(row: Any) -> str:
    return Path(str(getattr(row, "dagger_checkpoint", "unknown"))).stem


def _row_key(row: Any, fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        if field == "gap":
            value = _gap_bucket(row)
        elif field == "turn":
            value = _turn_bucket(row)
        elif field == "outcome":
            value = _outcome_bucket(row)
        elif field == "ckpt":
            value = _ckpt_bucket(row)
        else:
            value = str(getattr(row, field, "missing"))
        parts.append(f"{field}={value}")
    return "|".join(parts) if parts else "all"


def _sample_rows(rows: list[Any], max_rows: int, seed: int) -> list[Any]:
    if max_rows <= 0 or max_rows >= len(rows):
        return rows
    rng = random.Random(seed)
    return [rows[idx] for idx in sorted(rng.sample(range(len(rows)), max_rows))]


@torch.no_grad()
def _evaluate_rows(
    model: TinyPolicyValueNet,
    rows: list[Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    dataset = stack_rows(rows)
    loader = DataLoader(Subset(dataset, range(len(dataset))), batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)

    sums = {
        "rows": 0.0,
        "own_slots": 0.0,
        "label_actions": 0.0,
        "pred_actions": 0.0,
        "count_abs": 0.0,
        "launch_tp": 0.0,
        "launch_correct": 0.0,
        "target_correct": 0.0,
        "target_pair_correct": 0.0,
        "ship_correct": 0.0,
        "label_target_covered": 0.0,
    }
    target_rank_hits = {k: 0.0 for k in args.target_topk}

    for raw_batch in loader:
        batch = _unpack(raw_batch, device)
        out = model(
            batch["planets"],
            batch["pair_features"],
            batch["global_features"],
            batch["planet_mask"],
            batch["own_mask"],
        )
        source_logits = out["source_logits"]
        if args.launch_bias:
            source_logits = source_logits.clone()
            source_logits[..., 1] += float(args.launch_bias)
        target_logits = out["target_logits"]
        if "target_pair_logits" in out and abs(args.target_pair_weight) > 1e-9:
            target_logits = target_logits + float(args.target_pair_weight) * out["target_pair_logits"][:, :, None, :]
        mask = _target_mask(batch, args.target_loss_mask)
        target_logits = target_logits.masked_fill(~mask[:, :, None, :], -1e9)

        own_slots = batch["own_mask"][:, :, None].expand_as(batch["launch_actions"])
        label_launch = batch["launch_actions"]
        pred_launch = source_logits.argmax(dim=-1)
        active = (label_launch == 1) & own_slots
        pred_active = (pred_launch == 1) & own_slots

        row_true = active.sum(dim=(1, 2)).float()
        row_pred = pred_active.sum(dim=(1, 2)).float()
        sums["rows"] += float(label_launch.shape[0])
        sums["own_slots"] += float(own_slots.sum().item())
        sums["label_actions"] += float(active.sum().item())
        sums["pred_actions"] += float(pred_active.sum().item())
        sums["count_abs"] += float((row_pred - row_true).abs().sum().item())
        sums["launch_tp"] += float((pred_active & active).sum().item())
        sums["launch_correct"] += float(((pred_launch == label_launch) & own_slots).sum().item())

        if active.any():
            b_idx, src_idx, slot_idx = torch.where(active)
            label_targets = batch["target_actions"][b_idx, src_idx, slot_idx]
            active_target_logits = target_logits[b_idx, src_idx, slot_idx]
            pred_targets = active_target_logits.argmax(dim=-1)
            sums["target_correct"] += float((pred_targets == label_targets).sum().item())

            pair_logits = out.get("target_pair_logits")
            if pair_logits is None:
                pair_logits = out["target_logits"].max(dim=2).values
            pair_valid = batch["own_mask"][:, :, None] & batch["planet_mask"][:, None, :]
            eye = torch.eye(pair_valid.shape[1], dtype=torch.bool, device=pair_valid.device)[None, :, :]
            pair_valid = pair_valid & ~eye
            pred_pair_targets = pair_logits.masked_fill(~pair_valid, -1e9)[b_idx, src_idx].argmax(dim=-1)
            sums["target_pair_correct"] += float((pred_pair_targets == label_targets).sum().item())
            sums["label_target_covered"] += float(mask[b_idx, src_idx, label_targets].sum().item())

            for k in args.target_topk:
                top = active_target_logits.topk(min(int(k), active_target_logits.shape[-1]), dim=-1).indices
                target_rank_hits[k] += float((top == label_targets[:, None]).any(dim=-1).sum().item())
            if "ship_logits" in out:
                ship_logits = out["ship_logits"][b_idx, src_idx, slot_idx, label_targets]
                pred_ship = ship_logits.argmax(dim=-1)
                label_ship = batch["ship_actions"][b_idx, src_idx, slot_idx]
                sums["ship_correct"] += float((pred_ship == label_ship).sum().item())

    launch_precision, launch_recall, launch_f1 = _prf(
        sums["launch_tp"],
        sums["pred_actions"],
        sums["label_actions"],
    )
    rows_count = max(1.0, sums["rows"])
    label_actions = max(1.0, sums["label_actions"])
    own_slots = max(1.0, sums["own_slots"])
    result = {
        "rows": sums["rows"],
        "label_actions": sums["label_actions"],
        "label_actions_per_row": sums["label_actions"] / rows_count,
        "pred_actions_per_row": sums["pred_actions"] / rows_count,
        "density_ratio": sums["pred_actions"] / max(1e-6, sums["label_actions"]),
        "action_count_mae": sums["count_abs"] / rows_count,
        "launch_acc": sums["launch_correct"] / own_slots,
        "launch_precision": launch_precision,
        "launch_recall": launch_recall,
        "launch_f1": launch_f1,
        "target_acc": sums["target_correct"] / label_actions,
        "target_pair_acc": sums["target_pair_correct"] / label_actions,
        "ship_acc": sums["ship_correct"] / label_actions if sums["ship_correct"] > 0 else 0.0,
        "label_target_covered": sums["label_target_covered"] / label_actions,
    }
    for k, value in sorted(target_rank_hits.items()):
        result[f"target_top{k}_acc"] = value / label_actions
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a BC checkpoint by DAgger cache buckets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=260526)
    parser.add_argument("--max-rows-per-group", type=int, default=4096)
    parser.add_argument("--min-group-rows", type=int, default=32)
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--target-loss-mask", choices=["dataset", "all_planets"], default="all_planets")
    parser.add_argument("--target-topk", type=int, nargs="*", default=[1, 3, 5])
    parser.add_argument("--group-by", nargs="+", default=["gap"])
    parser.add_argument("--min-actions-per-row", type=int, default=-1)
    parser.add_argument("--max-actions-per-row", type=int, default=-1)
    parser.add_argument("--min-turn", type=int, default=-1)
    parser.add_argument("--max-turn", type=int, default=-1)
    parser.add_argument("--min-final-reward", type=float, default=-999.0)
    parser.add_argument("--max-final-reward", type=float, default=999.0)
    parser.add_argument("--min-abs-action-gap", type=int, default=-1)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows, cache_metadata = _load_rows(Path(args.cache))
    rows, filter_metrics = _filter_rows(
        rows,
        args.min_actions_per_row,
        args.max_actions_per_row,
        args.min_turn,
        args.max_turn,
        args.min_final_reward,
        args.max_final_reward,
        args.min_abs_action_gap,
    )
    buckets: dict[str, list[Any]] = {}
    for row in rows:
        buckets.setdefault(_row_key(row, args.group_by), []).append(row)

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TinyPolicyValueNet(**payload.get("model", {})).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    groups: dict[str, dict[str, float]] = {}
    for index, (key, group_rows) in enumerate(sorted(buckets.items())):
        if len(group_rows) < args.min_group_rows:
            continue
        sampled = _sample_rows(group_rows, args.max_rows_per_group, args.seed + index)
        metrics = _evaluate_rows(model, sampled, args)
        metrics["all_rows_in_group"] = float(len(group_rows))
        metrics["all_label_actions_in_group"] = float(sum(_row_action_count(row) for row in group_rows))
        groups[key] = metrics

    result = {
        "checkpoint": args.checkpoint,
        "cache": args.cache,
        "cache_rows_after_filter": float(len(rows)),
        "cache_metrics": cache_metadata.get("metrics", {}),
        "row_filter": filter_metrics,
        "group_by": args.group_by,
        "max_rows_per_group": float(args.max_rows_per_group),
        "min_group_rows": float(args.min_group_rows),
        "target_loss_mask": args.target_loss_mask,
        "launch_bias": float(args.launch_bias),
        "target_pair_weight": float(args.target_pair_weight),
        "groups": groups,
    }
    encoded = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
