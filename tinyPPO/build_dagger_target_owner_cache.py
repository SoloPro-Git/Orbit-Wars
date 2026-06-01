from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OWNER_NAMES = ("own", "neutral", "enemy")


def _load_rows(paths: list[Path]) -> tuple[list[Any], dict[str, Any]]:
    rows: list[Any] = []
    metrics: dict[str, Any] = {"input_caches": [str(path) for path in paths]}
    for path in paths:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict):
            part_rows = list(payload.get("rows", []))
            part_metrics = payload.get("metrics", {})
            for key, value in part_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(metrics.get(key, 0.0)) + float(value)
        else:
            part_rows = list(payload)
        rows.extend(part_rows)
    metrics["input_rows"] = float(len(rows))
    return rows, metrics


def _row_action_count(row: Any) -> int:
    labelled = getattr(row, "labelled", None)
    if labelled is not None:
        return int(labelled)
    return int(np.asarray(getattr(row, "launch_mask")).sum())


def _turn(row: Any) -> int:
    return int(getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", -1)))


def _turn_bucket(row: Any, width: int) -> str:
    turn = _turn(row)
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
    return Path(checkpoint).name or "unknown"


def _reward_bucket(row: Any) -> str:
    reward = getattr(row, "dagger_model_final_reward", None)
    if reward is None:
        return "unknown"
    reward = float(reward)
    if reward > 0.0:
        return "win"
    if reward < 0.0:
        return "loss"
    return "draw"


def _owner_counts(row: Any) -> Counter[str]:
    planets = np.asarray(row.planets, dtype=np.float32)
    launch_mask = np.asarray(getattr(row, "launch_mask", getattr(row, "launch_actions")), dtype=np.bool_)
    target_actions = np.asarray(row.target_actions, dtype=np.int64)
    counts: Counter[str] = Counter()
    for source_idx, slot_idx in np.argwhere(launch_mask):
        target_idx = int(target_actions[int(source_idx), int(slot_idx)])
        if target_idx < 0 or target_idx >= planets.shape[0]:
            counts["invalid"] += 1
            continue
        owner_idx = int(np.argmax(planets[target_idx, :3]))
        if 0 <= owner_idx < len(OWNER_NAMES):
            counts[OWNER_NAMES[owner_idx]] += 1
        else:
            counts["invalid"] += 1
    return counts


def _parse_allowed(spec: str) -> set[str]:
    return {item.strip() for item in str(spec or "").split(",") if item.strip()}


def _passes(row: Any, args: argparse.Namespace, owner_counts: Counter[str]) -> bool:
    actions = _row_action_count(row)
    if args.min_actions_per_row >= 0 and actions < args.min_actions_per_row:
        return False
    if args.max_actions_per_row >= 0 and actions > args.max_actions_per_row:
        return False
    turn = _turn(row)
    if args.min_turn >= 0 and turn < args.min_turn:
        return False
    if args.max_turn >= 0 and turn > args.max_turn:
        return False
    gap = _action_gap(row)
    if args.min_action_gap >= 0 and (gap is None or gap < args.min_action_gap):
        return False
    if args.max_action_gap >= 0 and (gap is None or gap > args.max_action_gap):
        return False
    allowed_gaps = _parse_allowed(args.gap_buckets)
    if allowed_gaps and _gap_bucket(row) not in allowed_gaps:
        return False
    reward = getattr(row, "dagger_model_final_reward", None)
    if args.min_final_reward > -1.0e30 and (reward is None or float(reward) < args.min_final_reward):
        return False
    if args.max_final_reward < 1.0e30 and (reward is None or float(reward) > args.max_final_reward):
        return False
    total = sum(owner_counts[name] for name in OWNER_NAMES)
    if args.min_owner_actions > 0 and owner_counts[args.owner] < args.min_owner_actions:
        return False
    if args.min_owner_frac > 0.0 and total > 0 and owner_counts[args.owner] / total < args.min_owner_frac:
        return False
    if args.max_other_owner_frac < 1.0 and total > 0:
        other = total - owner_counts[args.owner]
        if other / total > args.max_other_owner_frac:
            return False
    return True


def build(args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    rows, source_metrics = _load_rows([Path(path) for path in args.in_cache])
    candidates: list[tuple[int, Any, Counter[str]]] = []
    reject_counts: Counter[str] = Counter()
    owner_action_counts: Counter[str] = Counter()
    rows_by_owner_count: Counter[str] = Counter()

    for row in rows:
        counts = _owner_counts(row)
        for owner, value in counts.items():
            owner_action_counts[owner] += int(value)
        rows_by_owner_count[str(counts.get(args.owner, 0))] += 1
        if _passes(row, args, counts):
            candidates.append((_row_action_count(row), row, counts))
        else:
            reject_counts["filtered"] += 1

    rng = random.Random(args.seed)
    by_key: dict[tuple[str, str, str], list[tuple[int, Any, Counter[str]]]] = defaultdict(list)
    for item in candidates:
        row = item[1]
        by_key[(_turn_bucket(row, args.turn_bucket_width), _gap_bucket(row), _ckpt_bucket(row))].append(item)
    for bucket in by_key.values():
        rng.shuffle(bucket)

    selected: list[Any] = []
    selected_actions = 0
    selected_owner_actions = 0
    selected_key_actions: Counter[tuple[str, str, str]] = Counter()
    target_actions = int(args.max_labelled_actions)
    target_per_key = max(1, target_actions // max(1, len(by_key))) if target_actions > 0 else 10**12
    for key in sorted(by_key):
        for actions, row, counts in by_key[key]:
            if target_actions > 0 and selected_actions + actions > target_actions:
                break
            if selected_key_actions[key] + actions > target_per_key and selected_key_actions[key] > 0:
                break
            selected.append(row)
            selected_actions += actions
            selected_owner_actions += int(counts[args.owner])
            selected_key_actions[key] += actions

    if target_actions <= 0 or selected_actions < target_actions:
        selected_ids = {id(row) for row in selected}
        leftovers = [item for bucket in by_key.values() for item in bucket if id(item[1]) not in selected_ids]
        rng.shuffle(leftovers)
        for actions, row, counts in leftovers:
            if target_actions > 0 and selected_actions + actions > target_actions:
                continue
            selected.append(row)
            selected_actions += actions
            selected_owner_actions += int(counts[args.owner])
            if target_actions > 0 and selected_actions >= target_actions:
                break

    rng.shuffle(selected)
    if args.weight >= 0.0:
        for row in selected:
            setattr(row, "sample_weight", float(args.weight))
            setattr(row, "dagger_loss_weight_override", float(args.weight))
            setattr(row, "dagger_source_name", f"{args.owner}_target_filter")

    metrics: dict[str, Any] = {
        "source_metrics": source_metrics,
        "input_rows": float(len(rows)),
        "candidate_rows": float(len(candidates)),
        "selected_rows": float(len(selected)),
        "selected_labelled_actions": float(selected_actions),
        "selected_owner": args.owner,
        "selected_owner_actions": float(selected_owner_actions),
        "selected_owner_action_frac": float(selected_owner_actions / max(1, selected_actions)),
        "owner_action_counts": dict(owner_action_counts),
        "rows_by_selected_owner_action_count": dict(rows_by_owner_count),
        "turn_buckets": dict(Counter(_turn_bucket(row, args.turn_bucket_width) for row in selected)),
        "gap_buckets": dict(Counter(_gap_bucket(row) for row in selected)),
        "reward_buckets": dict(Counter(_reward_bucket(row) for row in selected)),
        "checkpoint_buckets": dict(Counter(_ckpt_bucket(row) for row in selected)),
        "bin_actions": {"/".join(key): float(value) for key, value in selected_key_actions.items()},
        "reject_counts": dict(reject_counts),
        "weight": float(args.weight),
        "filters": {
            "min_turn": args.min_turn,
            "max_turn": args.max_turn,
            "owner": args.owner,
            "min_owner_actions": args.min_owner_actions,
            "min_owner_frac": args.min_owner_frac,
            "max_other_owner_frac": args.max_other_owner_frac,
            "gap_buckets": args.gap_buckets,
            "min_action_gap": args.min_action_gap,
            "max_action_gap": args.max_action_gap,
            "max_labelled_actions": args.max_labelled_actions,
        },
    }
    return selected, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DAgger cache focused on regular-labelled target owner buckets.")
    parser.add_argument("--in-cache", action="append", required=True)
    parser.add_argument("--out-cache", required=True)
    parser.add_argument("--owner", choices=OWNER_NAMES, default="enemy")
    parser.add_argument("--min-owner-actions", type=int, default=1)
    parser.add_argument("--min-owner-frac", type=float, default=0.0)
    parser.add_argument("--max-other-owner-frac", type=float, default=1.0)
    parser.add_argument("--min-turn", type=int, default=-1)
    parser.add_argument("--max-turn", type=int, default=-1)
    parser.add_argument("--min-actions-per-row", type=int, default=-1)
    parser.add_argument("--max-actions-per-row", type=int, default=-1)
    parser.add_argument("--min-action-gap", type=int, default=-1)
    parser.add_argument("--max-action-gap", type=int, default=-1)
    parser.add_argument("--gap-buckets", default="")
    parser.add_argument("--min-final-reward", type=float, default=-1.0e30)
    parser.add_argument("--max-final-reward", type=float, default=1.0e30)
    parser.add_argument("--max-labelled-actions", type=int, default=20000)
    parser.add_argument("--turn-bucket-width", type=int, default=50)
    parser.add_argument("--weight", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=260601)
    args = parser.parse_args()

    selected, metrics = build(args)
    out_path = Path(args.out_cache)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump({"rows": selected, "metrics": metrics}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
