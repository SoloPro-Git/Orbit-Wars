from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tinyPPO.features import MAX_PLANETS
from tinyPPO.imitation_regular import stack_rows
from tinyPPO.model import TinyPolicyValueNet


def _load_rows(path: Path) -> tuple[list[Any], dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        metadata = {key: value for key, value in payload.items() if key != "rows"}
        rows = payload.get("rows")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise TypeError(f"expected list cache at {path}, got {type(rows)!r}")
    return rows, metadata


def _sample_indices(total: int, max_rows: int, seed: int) -> list[int]:
    if max_rows <= 0 or max_rows >= total:
        return list(range(total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), max_rows))


def _row_action_count(row: Any) -> int:
    return int(np.asarray(row.launch_mask).sum())


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _filter_rows(
    rows: list[Any],
    min_actions: int,
    max_actions: int,
    min_turn: int,
    max_turn: int,
    min_final_reward: float,
    max_final_reward: float,
    min_abs_action_gap: int,
) -> tuple[list[Any], dict[str, float]]:
    if (
        min_actions < 0
        and max_actions < 0
        and min_turn < 0
        and max_turn < 0
        and min_final_reward <= -999.0
        and max_final_reward >= 999.0
        and min_abs_action_gap < 0
    ):
        return rows, {}
    filtered: list[Any] = []
    before_actions = 0
    kept_actions = 0
    missing_turn = 0
    missing_final_reward = 0
    missing_action_gap = 0
    for row in rows:
        action_count = _row_action_count(row)
        before_actions += action_count
        if min_actions >= 0 and action_count < min_actions:
            continue
        if max_actions >= 0 and action_count > max_actions:
            continue
        turn = _maybe_float(getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", None)))
        if min_turn >= 0 or max_turn >= 0:
            if turn is None:
                missing_turn += 1
                continue
            if min_turn >= 0 and turn < min_turn:
                continue
            if max_turn >= 0 and turn > max_turn:
                continue
        final_reward = _maybe_float(getattr(row, "dagger_model_final_reward", None))
        if min_final_reward > -999.0 or max_final_reward < 999.0:
            if final_reward is None:
                missing_final_reward += 1
                continue
            if final_reward < min_final_reward or final_reward > max_final_reward:
                continue
        if min_abs_action_gap >= 0:
            model_actions = _maybe_float(getattr(row, "dagger_model_action_count", None))
            label_actions = _maybe_float(getattr(row, "dagger_regular_label_action_count", None))
            if model_actions is None or label_actions is None:
                missing_action_gap += 1
                continue
            if abs(label_actions - model_actions) < min_abs_action_gap:
                continue
        filtered.append(row)
        kept_actions += action_count
    metrics = {
        "filter_min_actions_per_row": float(min_actions),
        "filter_max_actions_per_row": float(max_actions),
        "filter_min_turn": float(min_turn),
        "filter_max_turn": float(max_turn),
        "filter_min_final_reward": float(min_final_reward),
        "filter_max_final_reward": float(max_final_reward),
        "filter_min_abs_action_gap": float(min_abs_action_gap),
        "missing_turn_for_filter": float(missing_turn),
        "missing_final_reward_for_filter": float(missing_final_reward),
        "missing_action_gap_for_filter": float(missing_action_gap),
        "rows_before_filter": float(len(rows)),
        "rows_after_filter": float(len(filtered)),
        "actions_before_filter": float(before_actions),
        "actions_after_filter": float(kept_actions),
        "filter_keep_frac": float(len(filtered) / len(rows)) if rows else 0.0,
        "filter_actions_per_row": float(kept_actions / len(filtered)) if filtered else 0.0,
    }
    return filtered, metrics


def _prf(tp: float, pred: float, true: float) -> tuple[float, float, float]:
    precision = tp / pred if pred > 0 else (1.0 if true == 0 else 0.0)
    recall = tp / true if true > 0 else (1.0 if pred == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _target_mask(batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "dataset":
        return batch["target_safety_mask"]
    if mode != "all_planets":
        raise ValueError(f"unsupported target mask mode for encoded cache: {mode!r}")
    planet_mask = batch["planet_mask"]
    mask = planet_mask[:, None, :].expand(-1, MAX_PLANETS, -1).clone()
    eye = torch.eye(MAX_PLANETS, dtype=torch.bool, device=planet_mask.device)[None]
    return mask & ~eye


def _unpack(raw_batch: tuple[torch.Tensor, ...], device: torch.device) -> dict[str, torch.Tensor]:
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
    return {key: value.to(device, non_blocking=True) for key, value in zip(keys, raw_batch, strict=True)}


@torch.no_grad()
def evaluate_cache(args: argparse.Namespace) -> dict[str, Any]:
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
    selected = _sample_indices(len(rows), args.max_rows, args.seed)
    dataset = stack_rows([rows[idx] for idx in selected])
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    loader = DataLoader(Subset(dataset, range(len(dataset))), batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg = payload.get("model", {})
    model = TinyPolicyValueNet(**model_cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

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
            masked_pair_logits = pair_logits.masked_fill(~pair_valid, -1e9)
            pred_pair_targets = masked_pair_logits[b_idx, src_idx].argmax(dim=-1)
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
    label_actions = max(1.0, sums["label_actions"])
    own_slots = max(1.0, sums["own_slots"])
    result = {
        "checkpoint": args.checkpoint,
        "cache": args.cache,
        "cache_rows": float(len(rows)),
        "cache_metrics": cache_metadata.get("metrics", {}),
        "row_filter": filter_metrics,
        "sampled_rows": float(len(selected)),
        "target_loss_mask": args.target_loss_mask,
        "launch_bias": float(args.launch_bias),
        "target_pair_weight": float(args.target_pair_weight),
        "label_actions_per_row": sums["label_actions"] / max(1.0, sums["rows"]),
        "pred_actions_per_row": sums["pred_actions"] / max(1.0, sums["rows"]),
        "density_ratio": sums["pred_actions"] / max(1e-6, sums["label_actions"]),
        "action_count_mae": sums["count_abs"] / max(1.0, sums["rows"]),
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
    parser = argparse.ArgumentParser(description="Diagnose a BC checkpoint on encoded regular/DAgger cache rows.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rows", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=260526)
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--target-loss-mask", choices=["dataset", "all_planets"], default="all_planets")
    parser.add_argument("--target-topk", type=int, nargs="*", default=[1, 3, 5])
    parser.add_argument("--min-actions-per-row", type=int, default=-1)
    parser.add_argument("--max-actions-per-row", type=int, default=-1)
    parser.add_argument("--min-turn", type=int, default=-1)
    parser.add_argument("--max-turn", type=int, default=-1)
    parser.add_argument("--min-final-reward", type=float, default=-999.0)
    parser.add_argument("--max-final-reward", type=float, default=999.0)
    parser.add_argument("--min-abs-action-gap", type=int, default=-1)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = evaluate_cache(args)
    encoded = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
