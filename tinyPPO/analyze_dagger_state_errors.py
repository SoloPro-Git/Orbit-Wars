from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyPPO.features import MAX_PLANETS
from tinyPPO.model import TinyPolicyValueNet

ACTION_SLOTS = 3


def _load_rows(path: Path) -> tuple[list[Any], dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        rows = payload.get("rows")
        metadata = {key: value for key, value in payload.items() if key != "rows"}
    else:
        rows = payload
        metadata = {}
    if not isinstance(rows, list):
        raise TypeError(f"unsupported cache format in {path}")
    return rows, metadata


def _row_batch(rows: list[Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "planets": torch.tensor(np.stack([row.planets for row in rows]), dtype=torch.float32, device=device),
        "pair_features": torch.tensor(np.stack([row.pair_features for row in rows]), dtype=torch.float32, device=device),
        "global_features": torch.tensor(np.stack([row.global_features for row in rows]), dtype=torch.float32, device=device),
        "planet_mask": torch.tensor(np.stack([row.planet_mask for row in rows]), dtype=torch.bool, device=device),
        "own_mask": torch.tensor(np.stack([row.own_mask for row in rows]), dtype=torch.bool, device=device),
    }


def _target_mask(row: Any, mode: str) -> np.ndarray:
    if mode == "dataset":
        return np.asarray(row.target_safety_mask, dtype=np.bool_).copy()
    if mode == "all_planets":
        planet_mask = np.asarray(row.planet_mask, dtype=np.bool_)
        own_mask = np.asarray(row.own_mask, dtype=np.bool_)
        mask = np.zeros((MAX_PLANETS, MAX_PLANETS), dtype=np.bool_)
        for source_idx in np.where(own_mask & planet_mask)[0]:
            mask[int(source_idx)] = planet_mask
            mask[int(source_idx), int(source_idx)] = False
        return mask
    raise ValueError(f"unsupported target mask mode: {mode!r}")


def _turn_bucket(row: Any, width: int) -> str:
    turn = int(getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", -1)))
    if turn < 0:
        return "unknown"
    lo = (turn // max(1, width)) * max(1, width)
    return f"{lo}-{lo + max(1, width) - 1}"


def _ckpt_bucket(row: Any) -> str:
    checkpoint = str(getattr(row, "dagger_checkpoint", "unknown"))
    return Path(checkpoint).name or "unknown"


def _final_reward_bucket(row: Any) -> str:
    reward = getattr(row, "dagger_model_final_reward", None)
    if reward is None:
        return "unknown"
    reward = float(reward)
    if reward > 0.0:
        return "win"
    if reward < 0.0:
        return "loss"
    return "draw"


def _action_gap(row: Any) -> int | None:
    label_count = getattr(row, "dagger_regular_label_action_count", None)
    model_count = getattr(row, "dagger_model_action_count", None)
    if label_count is None or model_count is None:
        return None
    return int(label_count) - int(model_count)


def _gap_bucket(row: Any) -> str:
    gap = _action_gap(row)
    if gap is None:
        return "unknown"
    if gap <= -1:
        return "regular_lt_model"
    if gap >= 1:
        return "regular_gt_model"
    return "equal"


def _prf(tp: float, pred: float, true: float) -> tuple[float, float, float]:
    precision = tp / pred if pred > 0 else (1.0 if true == 0 else 0.0)
    recall = tp / true if true > 0 else (1.0 if pred == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _new_stats() -> dict[str, float]:
    return defaultdict(float)


def _add(stats: dict[str, float], key: str, value: float) -> None:
    stats[key] += float(value)


def _summarize(stats: dict[str, float]) -> dict[str, float]:
    rows = max(1.0, stats["rows"])
    label_actions = max(1.0, stats["label_actions"])
    own_slots = max(1.0, stats["own_slots"])
    precision, recall, f1 = _prf(stats["launch_tp"], stats["pred_actions"], stats["label_actions"])
    return {
        "rows": stats["rows"],
        "label_actions": stats["label_actions"],
        "pred_actions": stats["pred_actions"],
        "recorded_model_actions": stats["recorded_model_actions"],
        "label_actions_per_row": stats["label_actions"] / rows,
        "pred_actions_per_row": stats["pred_actions"] / rows,
        "recorded_model_actions_per_row": stats["recorded_model_actions"] / rows,
        "pred_density_ratio": stats["pred_actions"] / max(1e-6, stats["label_actions"]),
        "recorded_density_ratio": stats["recorded_model_actions"] / max(1e-6, stats["label_actions"]),
        "pred_count_mae": stats["pred_count_abs"] / rows,
        "recorded_count_mae": stats["recorded_count_abs"] / rows,
        "launch_acc": stats["launch_correct"] / own_slots,
        "launch_precision": precision,
        "launch_recall": recall,
        "launch_f1": f1,
        "target_acc": stats["target_correct"] / label_actions,
        "target_pair_acc": stats["target_pair_correct"] / label_actions,
        "target_top3_acc": stats["target_top3"] / label_actions,
        "target_top5_acc": stats["target_top5"] / label_actions,
        "label_target_covered": stats["label_target_covered"] / label_actions,
        "score_gap_mean": stats["score_gap_sum"] / rows,
        "owned_planets_mean": stats["owned_planets_sum"] / rows,
        "final_reward_mean": stats["final_reward_sum"] / rows,
    }


def _bucket_names(row: Any, width: int) -> dict[str, str]:
    turn = _turn_bucket(row, width)
    gap = _gap_bucket(row)
    outcome = str(getattr(row, "dagger_model_outcome", _final_reward_bucket(row)))
    reward = _final_reward_bucket(row)
    ckpt = _ckpt_bucket(row)
    return {
        "turn": turn,
        "gap": gap,
        "outcome": outcome,
        "reward": reward,
        "checkpoint": ckpt,
        "turn_gap": f"{turn}/{gap}",
        "checkpoint_gap": f"{ckpt}/{gap}",
    }


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rows, metadata = _load_rows(Path(args.cache))
    if args.max_rows > 0 and args.max_rows < len(rows):
        rng = random.Random(args.seed)
        rows = [rows[idx] for idx in sorted(rng.sample(range(len(rows)), args.max_rows))]

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TinyPolicyValueNet(**payload.get("model", {})).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    overall = _new_stats()
    groups: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(_new_stats))

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch = _row_batch(batch_rows, device)
        out = model(**batch)
        source_logits = out["source_logits"]
        if args.launch_bias:
            source_logits = source_logits.clone()
            source_logits[..., 1] += float(args.launch_bias)
        pred_launch_batch = source_logits.argmax(dim=-1).detach().cpu().numpy()

        target_logits = out["target_logits"]
        if "target_pair_logits" in out and abs(args.target_pair_weight) > 1e-9:
            target_logits = target_logits + float(args.target_pair_weight) * out["target_pair_logits"][:, :, None, :]
        target_logits_np = target_logits.detach().cpu().numpy()
        pair_logits = out.get("target_pair_logits")
        pair_logits_np = pair_logits.detach().cpu().numpy() if pair_logits is not None else None

        for local_idx, row in enumerate(batch_rows):
            label_launch = np.asarray(row.launch_actions, dtype=np.int64)
            own_slots = np.asarray(row.own_mask, dtype=np.bool_)[:, None] & np.ones((MAX_PLANETS, ACTION_SLOTS), dtype=np.bool_)
            active = (label_launch == 1) & own_slots
            pred_active = (pred_launch_batch[local_idx] == 1) & own_slots
            label_count = int(active.sum())
            pred_count = int(pred_active.sum())
            recorded_count = int(getattr(row, "dagger_model_action_count", pred_count))
            row_stats = _new_stats()
            for stats in [overall, row_stats]:
                _add(stats, "rows", 1.0)
                _add(stats, "own_slots", float(own_slots.sum()))
                _add(stats, "label_actions", float(label_count))
                _add(stats, "pred_actions", float(pred_count))
                _add(stats, "recorded_model_actions", float(recorded_count))
                _add(stats, "pred_count_abs", abs(pred_count - label_count))
                _add(stats, "recorded_count_abs", abs(recorded_count - label_count))
                _add(stats, "launch_tp", float((pred_active & active).sum()))
                _add(stats, "launch_correct", float(((pred_launch_batch[local_idx] == label_launch) & own_slots).sum()))
                _add(stats, "score_gap_sum", float(getattr(row, "dagger_score_gap", 0.0)))
                _add(stats, "owned_planets_sum", float(getattr(row, "dagger_owned_planets", 0.0)))
                _add(stats, "final_reward_sum", float(getattr(row, "dagger_model_final_reward", 0.0)))

            if label_count > 0:
                mask_np = _target_mask(row, args.target_loss_mask)
                active_indices = np.argwhere(active)
                for source_idx_raw, slot_raw in active_indices:
                    source_idx = int(source_idx_raw)
                    slot = int(slot_raw)
                    label_target = int(row.target_actions[source_idx, slot])
                    valid = mask_np[source_idx]
                    if label_target < 0 or label_target >= MAX_PLANETS or not bool(valid[label_target]):
                        continue
                    logits = target_logits_np[local_idx, source_idx, slot].copy()
                    logits[~valid] = -1e9
                    order = np.argsort(-logits)
                    pred_target = int(order[0])
                    pair_pred = pred_target
                    if pair_logits_np is not None:
                        pair_valid = np.asarray(row.planet_mask, dtype=np.bool_).copy()
                        pair_valid[source_idx] = False
                        pair_scores = pair_logits_np[local_idx, source_idx].copy()
                        pair_scores[~pair_valid] = -1e9
                        pair_pred = int(np.argmax(pair_scores))
                    for stats in [overall, row_stats]:
                        _add(stats, "target_correct", float(pred_target == label_target))
                        _add(stats, "target_pair_correct", float(pair_pred == label_target))
                        _add(stats, "target_top3", float(label_target in set(order[:3].tolist())))
                        _add(stats, "target_top5", float(label_target in set(order[:5].tolist())))
                        _add(stats, "label_target_covered", 1.0)

            for group_name, bucket in _bucket_names(row, args.turn_bucket_width).items():
                for key, value in row_stats.items():
                    groups[group_name][bucket][key] += value

    result: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "cache": args.cache,
        "cache_rows": float(len(rows)),
        "source_cache_metrics": metadata.get("metrics", {}),
        "target_loss_mask": args.target_loss_mask,
        "launch_bias": float(args.launch_bias),
        "target_pair_weight": float(args.target_pair_weight),
        "overall": _summarize(overall),
        "groups": {},
    }
    for group_name, buckets in sorted(groups.items()):
        items: list[tuple[str, dict[str, float]]] = []
        for bucket, stats in buckets.items():
            summary = _summarize(stats)
            if summary["rows"] >= args.min_bucket_rows and summary["label_actions"] >= args.min_bucket_actions:
                items.append((bucket, summary))
        items.sort(key=lambda item: (item[1]["launch_f1"], item[1]["target_pair_acc"], item[0]))
        result["groups"][group_name] = {bucket: summary for bucket, summary in items[: args.max_buckets_per_group]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze where a checkpoint disagrees with regular labels on DAgger model-rollout states.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260530)
    parser.add_argument("--launch-bias", type=float, default=-0.05)
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--target-loss-mask", choices=["dataset", "all_planets"], default="all_planets")
    parser.add_argument("--turn-bucket-width", type=int, default=50)
    parser.add_argument("--min-bucket-rows", type=int, default=50)
    parser.add_argument("--min-bucket-actions", type=int, default=50)
    parser.add_argument("--max-buckets-per-group", type=int, default=24)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = analyze(args)
    encoded = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
