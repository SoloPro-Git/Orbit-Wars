from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyPPO.agents import ACTION_SLOTS
from tinyPPO.features import MAX_PLANETS
from tinyPPO.model import TinyPolicyValueNet


def _load_rows(path: str | Path) -> list[Any]:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    rows = payload.get("rows") if isinstance(payload, dict) else payload[0] if isinstance(payload, tuple) else payload
    if not isinstance(rows, list):
        raise TypeError(f"unsupported cache format in {path!s}")
    return rows


def _sample_rows(rows: list[Any], max_rows: int, seed: int) -> list[Any]:
    if max_rows <= 0 or max_rows >= len(rows):
        return rows
    rng = random.Random(seed)
    return [rows[idx] for idx in sorted(rng.sample(range(len(rows)), max_rows))]


def _owner_bucket(row: Any, target_idx: int) -> str:
    planets = np.asarray(row.planets)
    if target_idx < 0 or target_idx >= planets.shape[0]:
        return "invalid"
    if planets[target_idx, 0] > 0.5:
        return "own"
    if planets[target_idx, 1] > 0.5:
        return "neutral"
    if planets[target_idx, 2] > 0.5:
        return "enemy"
    return "masked"


def _turn_bucket(row: Any) -> str:
    turn = int(getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", 0)))
    lo = (max(0, turn) // 50) * 50
    return f"{lo:03d}-{lo + 49:03d}"


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


def _label_source_targets(row: Any) -> Counter[tuple[int, int]]:
    keys: Counter[tuple[int, int]] = Counter()
    active = np.argwhere(np.asarray(row.launch_mask, dtype=np.bool_))
    for source_idx, slot in active:
        source_idx = int(source_idx)
        slot = int(slot)
        if slot >= ACTION_SLOTS:
            continue
        target_idx = int(row.target_actions[source_idx, slot])
        if 0 <= target_idx < MAX_PLANETS:
            keys[(source_idx, target_idx)] += 1
    return keys


def _owner_counts(row: Any, keys: Counter[tuple[int, int]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for (_source_idx, target_idx), value in keys.items():
        counts[_owner_bucket(row, int(target_idx))] += int(value)
    return counts


def _source_counts(keys: Counter[tuple[int, int]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for (source_idx, _target_idx), value in keys.items():
        counts[int(source_idx)] += int(value)
    return counts


def _prf(pred: Counter, true: Counter) -> tuple[float, float, float]:
    tp = sum((pred & true).values())
    pred_n = sum(pred.values())
    true_n = sum(true.values())
    precision = tp / pred_n if pred_n else (1.0 if true_n == 0 else 0.0)
    recall = tp / true_n if true_n else (1.0 if pred_n == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _empty_group() -> dict[str, float | Counter[str]]:
    return {
        "rows": 0.0,
        "true_actions": 0.0,
        "pred_actions": 0.0,
        "matched_source_targets": 0.0,
        "source_f1_sum": 0.0,
        "source_target_f1_sum": 0.0,
        "action_count_abs_sum": 0.0,
        "rows_pred_gt_true": 0.0,
        "rows_pred_lt_true": 0.0,
        "rows_pred_eq_true": 0.0,
        "true_owner_counts": Counter(),
        "pred_owner_counts": Counter(),
        "matched_owner_counts": Counter(),
        "missed_owner_counts": Counter(),
        "extra_owner_counts": Counter(),
    }


def _add_counter_field(group: dict[str, float | Counter[str]], key: str, value: Counter[str]) -> None:
    counter = group[key]
    if not isinstance(counter, Counter):
        raise TypeError(f"group field {key!r} is not a Counter")
    counter.update(value)


def _add_group(group: dict[str, float | Counter[str]], row: Any, true_keys: Counter, pred_keys: Counter) -> None:
    true_count = int(sum(true_keys.values()))
    pred_count = int(sum(pred_keys.values()))
    matched = true_keys & pred_keys
    missed = true_keys - pred_keys
    extra = pred_keys - true_keys
    true_source = _source_counts(true_keys)
    pred_source = _source_counts(pred_keys)
    _sp, _sr, source_f1 = _prf(pred_source, true_source)
    _tp, _tr, source_target_f1 = _prf(pred_keys, true_keys)

    group["rows"] = float(group["rows"]) + 1.0
    group["true_actions"] = float(group["true_actions"]) + float(true_count)
    group["pred_actions"] = float(group["pred_actions"]) + float(pred_count)
    group["matched_source_targets"] = float(group["matched_source_targets"]) + float(sum(matched.values()))
    group["source_f1_sum"] = float(group["source_f1_sum"]) + source_f1
    group["source_target_f1_sum"] = float(group["source_target_f1_sum"]) + source_target_f1
    group["action_count_abs_sum"] = float(group["action_count_abs_sum"]) + abs(float(pred_count - true_count))
    group["rows_pred_gt_true"] = float(group["rows_pred_gt_true"]) + float(pred_count > true_count)
    group["rows_pred_lt_true"] = float(group["rows_pred_lt_true"]) + float(pred_count < true_count)
    group["rows_pred_eq_true"] = float(group["rows_pred_eq_true"]) + float(pred_count == true_count)
    _add_counter_field(group, "true_owner_counts", _owner_counts(row, true_keys))
    _add_counter_field(group, "pred_owner_counts", _owner_counts(row, pred_keys))
    _add_counter_field(group, "matched_owner_counts", _owner_counts(row, matched))
    _add_counter_field(group, "missed_owner_counts", _owner_counts(row, missed))
    _add_counter_field(group, "extra_owner_counts", _owner_counts(row, extra))


def _counter_rates(counter: Counter[str], denom: float) -> dict[str, float]:
    return {key: float(value) / max(1.0, denom) for key, value in sorted(counter.items())}


def _finish_group(group: dict[str, float | Counter[str]]) -> dict[str, Any]:
    rows = max(1.0, float(group["rows"]))
    true_actions = float(group["true_actions"])
    pred_actions = float(group["pred_actions"])
    matched = float(group["matched_source_targets"])
    out = {
        "rows": float(group["rows"]),
        "regular_actions_per_state": true_actions / rows,
        "model_actions_per_state": pred_actions / rows,
        "source_f1": float(group["source_f1_sum"]) / rows,
        "source_target_f1": float(group["source_target_f1_sum"]) / rows,
        "action_count_mae": float(group["action_count_abs_sum"]) / rows,
        "rows_pred_gt_true_rate": float(group["rows_pred_gt_true"]) / rows,
        "rows_pred_lt_true_rate": float(group["rows_pred_lt_true"]) / rows,
        "rows_pred_eq_true_rate": float(group["rows_pred_eq_true"]) / rows,
        "micro_source_target_precision": matched / max(1.0, pred_actions),
        "micro_source_target_recall": matched / max(1.0, true_actions),
        "true_owner_counts": dict(sorted(cast_counter(group["true_owner_counts"]).items())),
        "pred_owner_counts": dict(sorted(cast_counter(group["pred_owner_counts"]).items())),
        "matched_owner_counts": dict(sorted(cast_counter(group["matched_owner_counts"]).items())),
        "missed_owner_counts": dict(sorted(cast_counter(group["missed_owner_counts"]).items())),
        "extra_owner_counts": dict(sorted(cast_counter(group["extra_owner_counts"]).items())),
        "missed_owner_rate_per_true_action": _counter_rates(cast_counter(group["missed_owner_counts"]), true_actions),
        "extra_owner_rate_per_pred_action": _counter_rates(cast_counter(group["extra_owner_counts"]), pred_actions),
    }
    precision = out["micro_source_target_precision"]
    recall = out["micro_source_target_recall"]
    out["micro_source_target_f1"] = 2.0 * precision * recall / max(1e-9, precision + recall)
    return out


def cast_counter(value: float | Counter[str]) -> Counter[str]:
    if not isinstance(value, Counter):
        raise TypeError("expected Counter")
    return value


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    rows = _sample_rows(_load_rows(args.cache), args.max_rows, args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TinyPolicyValueNet(**payload.get("model", {})).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    total = _empty_group()
    by_turn: dict[str, dict[str, float | Counter[str]]] = defaultdict(_empty_group)
    by_source_name: dict[str, dict[str, float | Counter[str]]] = defaultdict(_empty_group)

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch = _row_batch(batch_rows, device)
        out = model(**batch)
        source_logits = out["source_logits"] / max(1e-4, float(args.launch_temperature))
        if abs(args.launch_bias) > 1e-9:
            source_logits = source_logits.clone()
            source_logits[..., 1] += float(args.launch_bias)
        target_logits = out["target_logits"]
        if "target_pair_logits" in out and abs(args.target_pair_weight) > 1e-9:
            target_logits = target_logits + float(args.target_pair_weight) * out["target_pair_logits"][:, :, None, :]

        for local_idx, row in enumerate(batch_rows):
            target_mask_np = _target_mask(row, args.target_mask_mode)
            target_mask = torch.tensor(target_mask_np, dtype=torch.bool, device=device)
            masked_targets = target_logits[local_idx].masked_fill(~target_mask[:, None, :], -1e9)
            pred_launch = torch.argmax(source_logits[local_idx], dim=-1)
            own = np.asarray(row.own_mask, dtype=np.bool_) & np.asarray(row.planet_mask, dtype=np.bool_)
            pred_keys: Counter[tuple[int, int]] = Counter()
            for source_idx in np.where(own)[0].tolist():
                for slot in range(min(ACTION_SLOTS, pred_launch.shape[1])):
                    if int(pred_launch[source_idx, slot].item()) != 1:
                        continue
                    target_idx = int(torch.argmax(masked_targets[source_idx, slot]).item())
                    if 0 <= target_idx < MAX_PLANETS and bool(target_mask_np[source_idx, target_idx]):
                        pred_keys[(int(source_idx), target_idx)] += 1
            true_keys = _label_source_targets(row)
            _add_group(total, row, true_keys, pred_keys)
            _add_group(by_turn[_turn_bucket(row)], row, true_keys, pred_keys)
            _add_group(by_source_name[str(getattr(row, "dagger_source_name", "unknown"))], row, true_keys, pred_keys)

    result = {
        "cache": args.cache,
        "checkpoint": args.checkpoint,
        "rows": float(len(rows)),
        "launch_bias": float(args.launch_bias),
        "launch_temperature": float(args.launch_temperature),
        "target_pair_weight": float(args.target_pair_weight),
        "target_mask_mode": args.target_mask_mode,
        "summary": _finish_group(total),
        "by_turn": {key: _finish_group(value) for key, value in sorted(by_turn.items())},
        "by_source_name": {key: _finish_group(value) for key, value in sorted(by_source_name.items())},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model-vs-regular action errors on regular-labelled DAgger rows.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=270527)
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--launch-temperature", type=float, default=1.0)
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--target-mask-mode", choices=["dataset", "all_planets"], default="all_planets")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    result = diagnose(args)
    encoded = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
