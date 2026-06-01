from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load_rows(paths: list[Path]) -> tuple[list[Any], dict[str, Any]]:
    rows: list[Any] = []
    metrics: dict[str, Any] = {"input_caches": [str(path) for path in paths]}
    for path in paths:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        part_rows = list(payload["rows"] if isinstance(payload, dict) else payload)
        rows.extend(part_rows)
        if isinstance(payload, dict):
            part_metrics = payload.get("metrics", {})
            for key, value in part_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(metrics.get(key, 0.0)) + float(value)
    metrics["input_rows"] = float(len(rows))
    return rows, metrics


def _row_action_count(row: Any) -> int:
    labelled = getattr(row, "labelled", None)
    if labelled is not None:
        return int(labelled)
    return int(np.asarray(getattr(row, "launch_mask")).sum())


def _turn_bucket(row: Any, width: int) -> str:
    turn = int(getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", -1)))
    if turn < 0:
        return "unknown"
    lo = (turn // max(1, width)) * max(1, width)
    return f"{lo}-{lo + max(1, width) - 1}"


def _action_gap(row: Any) -> int | None:
    gap = getattr(row, "dagger_action_gap", None)
    if gap is None:
        label_count = getattr(row, "dagger_regular_label_action_count", None)
        model_count = getattr(row, "dagger_model_action_count", None)
        if label_count is None or model_count is None:
            return None
        gap = int(label_count) - int(model_count)
    return int(gap)


def _gap_bucket(row: Any) -> str:
    gap = _action_gap(row)
    if gap is None:
        return "unknown"
    if gap <= -1:
        return "regular_lt_model"
    if gap >= 1:
        return "regular_gt_model"
    return "equal"


def _ckpt_bucket(row: Any) -> str:
    checkpoint = str(getattr(row, "dagger_checkpoint", "unknown"))
    name = Path(checkpoint).name
    return name or "unknown"


def _final_reward(row: Any) -> float | None:
    reward = getattr(row, "dagger_model_final_reward", None)
    if reward is None:
        return None
    return float(reward)


def _reward_bucket(row: Any) -> str:
    reward = _final_reward(row)
    if reward is None:
        return "unknown"
    if reward > 0.0:
        return "win"
    if reward < 0.0:
        return "loss"
    return "draw"


def _build_bins(rows: list[Any], args: argparse.Namespace) -> dict[tuple[str, str, str], list[int]]:
    bins: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    allowed_gap_buckets = {
        item.strip()
        for item in str(getattr(args, "gap_buckets", "")).split(",")
        if item.strip()
    }
    for index, row in enumerate(rows):
        actions = _row_action_count(row)
        if args.min_actions_per_row >= 0 and actions < args.min_actions_per_row:
            continue
        if args.max_actions_per_row >= 0 and actions > args.max_actions_per_row:
            continue
        turn = int(getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", -1)))
        if args.min_turn >= 0 and turn < args.min_turn:
            continue
        if args.max_turn >= 0 and turn > args.max_turn:
            continue
        gap = _action_gap(row)
        if args.min_action_gap >= 0 and (gap is None or gap < args.min_action_gap):
            continue
        if args.max_action_gap >= 0 and (gap is None or gap > args.max_action_gap):
            continue
        final_reward = _final_reward(row)
        if args.min_final_reward > -1.0e30 and (final_reward is None or final_reward < args.min_final_reward):
            continue
        if args.max_final_reward < 1.0e30 and (final_reward is None or final_reward > args.max_final_reward):
            continue
        gap_bucket = _gap_bucket(row)
        if allowed_gap_buckets and gap_bucket not in allowed_gap_buckets:
            continue
        key = (_turn_bucket(row, args.turn_bucket_width), gap_bucket, _ckpt_bucket(row))
        bins[key].append(index)
    return bins


def _parse_gap_bucket_quotas(spec: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid gap bucket quota {item!r}; expected bucket=actions")
        bucket, value = item.split("=", 1)
        bucket = bucket.strip()
        if bucket not in {"regular_lt_model", "regular_gt_model", "equal"}:
            raise ValueError(f"unsupported gap bucket quota {bucket!r}")
        quota = int(value)
        if quota < 0:
            raise ValueError(f"negative gap bucket quota for {bucket!r}: {quota}")
        quotas[bucket] = quota
    return quotas


def _parse_gap_bucket_sample_weights(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid gap bucket sample weight {item!r}; expected bucket=weight")
        bucket, value = item.split("=", 1)
        bucket = bucket.strip()
        if bucket not in {"regular_lt_model", "regular_gt_model", "equal"}:
            raise ValueError(f"unsupported gap bucket sample weight {bucket!r}")
        weight = float(value)
        if weight < 0.0:
            raise ValueError(f"negative gap bucket sample weight for {bucket!r}: {weight}")
        weights[bucket] = weight
    return weights


def _select_balanced(rows: list[Any], bins: dict[tuple[str, str, str], list[int]], args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    rng = random.Random(args.seed)
    shuffled_bins: dict[tuple[str, str, str], list[int]] = {}
    for key, indices in bins.items():
        copied = list(indices)
        rng.shuffle(copied)
        shuffled_bins[key] = copied

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    selected_actions = 0
    selected_gap_actions: Counter = Counter()
    per_bin_actions: Counter = Counter()
    per_bin_rows: Counter = Counter()
    target_total = int(args.max_labelled_actions)
    target_per_bin = max(1, target_total // max(1, len(shuffled_bins)))
    gap_bucket_quotas = _parse_gap_bucket_quotas(getattr(args, "gap_bucket_action_quotas", ""))
    if gap_bucket_quotas:
        target_total = min(target_total, sum(gap_bucket_quotas.values()))

    for key in sorted(shuffled_bins):
        gap_bucket = key[1]
        gap_quota = gap_bucket_quotas.get(gap_bucket)
        if gap_bucket_quotas and gap_quota is None:
            continue
        for index in shuffled_bins[key]:
            actions = _row_action_count(rows[index])
            if selected_actions + actions > target_total:
                break
            if gap_quota is not None and selected_gap_actions[gap_bucket] + actions > gap_quota:
                break
            if per_bin_actions[key] + actions > target_per_bin and per_bin_rows[key] > 0:
                break
            selected_indices.append(index)
            selected_set.add(index)
            selected_actions += actions
            selected_gap_actions[gap_bucket] += actions
            per_bin_actions[key] += actions
            per_bin_rows[key] += 1

    if selected_actions < target_total:
        leftovers = [index for indices in shuffled_bins.values() for index in indices if index not in selected_set]
        rng.shuffle(leftovers)
        for index in leftovers:
            actions = _row_action_count(rows[index])
            gap_bucket = _gap_bucket(rows[index])
            gap_quota = gap_bucket_quotas.get(gap_bucket)
            if gap_bucket_quotas and gap_quota is None:
                continue
            if selected_actions + actions > target_total:
                continue
            if gap_quota is not None and selected_gap_actions[gap_bucket] + actions > gap_quota:
                continue
            selected_indices.append(index)
            selected_set.add(index)
            selected_actions += actions
            selected_gap_actions[gap_bucket] += actions
            key = (_turn_bucket(rows[index], args.turn_bucket_width), _gap_bucket(rows[index]), _ckpt_bucket(rows[index]))
            per_bin_actions[key] += actions
            per_bin_rows[key] += 1
            if selected_actions >= target_total:
                break

    rng.shuffle(selected_indices)
    gap_bucket_sample_weights = _parse_gap_bucket_sample_weights(getattr(args, "gap_bucket_sample_weights", ""))
    selected_rows = [rows[index] for index in selected_indices]
    if gap_bucket_sample_weights:
        for row in selected_rows:
            bucket = _gap_bucket(row)
            if bucket in gap_bucket_sample_weights:
                setattr(row, "sample_weight", float(gap_bucket_sample_weights[bucket]))
                setattr(row, "dagger_loss_weight_override", float(gap_bucket_sample_weights[bucket]))
    metrics = {
        "selected_rows": float(len(selected_rows)),
        "selected_labelled_actions": float(selected_actions),
        "bins_available": float(len(bins)),
        "target_actions_per_bin": float(target_per_bin),
        "turn_buckets": dict(Counter(_turn_bucket(row, args.turn_bucket_width) for row in selected_rows)),
        "gap_buckets": dict(Counter(_gap_bucket(row) for row in selected_rows)),
        "gap_bucket_actions": dict(selected_gap_actions),
        "gap_bucket_action_quotas": dict(gap_bucket_quotas),
        "gap_bucket_sample_weights": dict(gap_bucket_sample_weights),
        "action_gap_buckets": dict(Counter(str(_action_gap(row)) for row in selected_rows)),
        "reward_buckets": dict(Counter(_reward_bucket(row) for row in selected_rows)),
        "checkpoint_buckets": dict(Counter(_ckpt_bucket(row) for row in selected_rows)),
        "bin_rows": {"/".join(key): float(value) for key, value in per_bin_rows.items()},
        "bin_actions": {"/".join(key): float(value) for key, value in per_bin_actions.items()},
    }
    return selected_rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced DAgger cache from model-rollout states relabelled by regular.")
    parser.add_argument("--in-cache", action="append", required=True, help="Input pickle cache. Can be repeated.")
    parser.add_argument("--out-cache", required=True)
    parser.add_argument("--max-labelled-actions", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=260529)
    parser.add_argument("--turn-bucket-width", type=int, default=50)
    parser.add_argument("--min-turn", type=int, default=-1)
    parser.add_argument("--max-turn", type=int, default=-1)
    parser.add_argument("--min-action-gap", type=int, default=-1, help="Minimum regular_action_count - model_action_count.")
    parser.add_argument("--max-action-gap", type=int, default=-1, help="Maximum regular_action_count - model_action_count.")
    parser.add_argument("--min-final-reward", type=float, default=-1.0e30)
    parser.add_argument("--max-final-reward", type=float, default=1.0e30)
    parser.add_argument("--min-actions-per-row", type=int, default=-1)
    parser.add_argument("--max-actions-per-row", type=int, default=-1)
    parser.add_argument(
        "--gap-buckets",
        default="",
        help="Optional comma list from regular_lt_model,regular_gt_model,equal. Empty keeps all gap buckets.",
    )
    parser.add_argument(
        "--gap-bucket-action-quotas",
        default="",
        help="Optional comma list of per-gap action quotas, e.g. regular_lt_model=17671,equal=8000.",
    )
    parser.add_argument(
        "--gap-bucket-sample-weights",
        default="",
        help="Optional comma list of per-gap row sample weights, e.g. equal=0.25.",
    )
    args = parser.parse_args()

    rows, input_metrics = _load_rows([Path(path) for path in args.in_cache])
    bins = _build_bins(rows, args)
    selected_rows, balance_metrics = _select_balanced(rows, bins, args)
    metrics = {
        **input_metrics,
        **balance_metrics,
        "max_labelled_actions": float(args.max_labelled_actions),
        "seed": float(args.seed),
        "turn_bucket_width": float(args.turn_bucket_width),
        "min_turn": float(args.min_turn),
        "max_turn": float(args.max_turn),
        "min_action_gap": float(args.min_action_gap),
        "max_action_gap": float(args.max_action_gap),
        "min_final_reward": float(args.min_final_reward),
        "max_final_reward": float(args.max_final_reward),
        "min_actions_per_row": float(args.min_actions_per_row),
        "max_actions_per_row": float(args.max_actions_per_row),
        "gap_buckets_filter": args.gap_buckets,
        "gap_bucket_action_quotas_filter": args.gap_bucket_action_quotas,
        "gap_bucket_sample_weights_filter": args.gap_bucket_sample_weights,
    }
    out_path = Path(args.out_cache)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump({"rows": selected_rows, "metrics": metrics, "args": vars(args)}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(out_path)
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_cache": str(out_path), "summary": str(summary_path), "metrics": metrics}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
