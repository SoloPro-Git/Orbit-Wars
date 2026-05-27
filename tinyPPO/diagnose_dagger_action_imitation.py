from __future__ import annotations

import argparse
import json
import pickle
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
    if isinstance(payload, dict):
        rows = payload.get("rows")
    elif isinstance(payload, tuple):
        rows = payload[0]
    else:
        rows = payload
    if not isinstance(rows, list):
        raise TypeError(f"unsupported cache format in {path!s}")
    return rows


def _keys_from_label(row: Any) -> tuple[Counter, Counter, Counter]:
    source_keys: Counter = Counter()
    source_target_keys: Counter = Counter()
    action_keys: Counter = Counter()
    active = np.argwhere(np.asarray(row.launch_mask, dtype=np.bool_))
    for source_idx, slot in active:
        source_idx = int(source_idx)
        slot = int(slot)
        target_idx = int(row.target_actions[source_idx, slot])
        ship_idx = int(row.ship_actions[source_idx, slot])
        if target_idx < 0 or target_idx >= MAX_PLANETS:
            continue
        source_keys[source_idx] += 1
        source_target_keys[(source_idx, target_idx)] += 1
        action_keys[(source_idx, target_idx, ship_idx)] += 1
    return source_keys, source_target_keys, action_keys


def _prf(pred: Counter, true: Counter) -> tuple[float, float, float]:
    tp = sum((pred & true).values())
    pred_n = sum(pred.values())
    true_n = sum(true.values())
    precision = tp / pred_n if pred_n else (1.0 if true_n == 0 else 0.0)
    recall = tp / true_n if true_n else (1.0 if pred_n == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _turn_bucket(row: Any) -> str:
    turn = int(getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", 0)))
    lo = (max(0, turn) // 50) * 50
    return f"{lo:03d}-{lo + 49:03d}"


def _group_key(row: Any) -> str:
    return "|".join(
        [
            str(getattr(row, "dagger_model_outcome", "unknown")),
            _turn_bucket(row),
            str(getattr(row, "dagger_source_name", "unknown")),
        ]
    )


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


@torch.no_grad()
def _predict_keys(
    model: TinyPolicyValueNet,
    rows: list[Any],
    device: torch.device,
    *,
    batch_size: int,
    launch_bias: float,
    launch_temperature: float,
    target_pair_weight: float,
    target_mask_mode: str,
) -> list[tuple[Counter, Counter, Counter, float]]:
    out_rows: list[tuple[Counter, Counter, Counter, float]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = _row_batch(batch_rows, device)
        out = model(**batch)
        source_logits = out["source_logits"] / max(1e-4, float(launch_temperature))
        if abs(launch_bias) > 1e-9:
            source_logits = source_logits.clone()
            source_logits[..., 1] += float(launch_bias)
        target_logits = out["target_logits"]
        if "target_pair_logits" in out and abs(target_pair_weight) > 1e-9:
            target_logits = target_logits + float(target_pair_weight) * out["target_pair_logits"][:, :, None, :]
        ship_logits = out.get("ship_logits")
        if ship_logits is None:
            raise RuntimeError("diagnose_dagger_action_imitation requires bucket ship logits")

        for local_idx, row in enumerate(batch_rows):
            mask_np = _target_mask(row, target_mask_mode)
            mask = torch.tensor(mask_np, dtype=torch.bool, device=device)
            masked_target_logits = target_logits[local_idx].masked_fill(~mask[:, None, :], -1e9)
            pred_launch = torch.argmax(source_logits[local_idx], dim=-1)
            pred_source: Counter = Counter()
            pred_source_target: Counter = Counter()
            pred_action: Counter = Counter()
            own = np.asarray(row.own_mask, dtype=np.bool_) & np.asarray(row.planet_mask, dtype=np.bool_)
            for source_idx in np.where(own)[0].tolist():
                for slot in range(min(ACTION_SLOTS, pred_launch.shape[1])):
                    if int(pred_launch[source_idx, slot].item()) != 1:
                        continue
                    target_idx = int(torch.argmax(masked_target_logits[source_idx, slot]).item())
                    if target_idx < 0 or target_idx >= MAX_PLANETS or not bool(mask_np[source_idx, target_idx]):
                        continue
                    ship_idx = int(torch.argmax(ship_logits[local_idx, source_idx, slot, target_idx]).item())
                    pred_source[source_idx] += 1
                    pred_source_target[(source_idx, target_idx)] += 1
                    pred_action[(source_idx, target_idx, ship_idx)] += 1
            out_rows.append((pred_source, pred_source_target, pred_action, float(sum(pred_source.values()))))
    return out_rows


def _empty_group() -> dict[str, float]:
    return {
        "rows": 0.0,
        "true_actions": 0.0,
        "pred_actions": 0.0,
        "source_f1_sum": 0.0,
        "source_target_f1_sum": 0.0,
        "action_f1_sum": 0.0,
        "action_count_abs_sum": 0.0,
    }


def _add_group(stats: dict[str, float], true_count: int, pred_count: int, source_f1: float, st_f1: float, action_f1: float) -> None:
    stats["rows"] += 1.0
    stats["true_actions"] += float(true_count)
    stats["pred_actions"] += float(pred_count)
    stats["source_f1_sum"] += float(source_f1)
    stats["source_target_f1_sum"] += float(st_f1)
    stats["action_f1_sum"] += float(action_f1)
    stats["action_count_abs_sum"] += abs(float(pred_count - true_count))


def _finish_group(stats: dict[str, float]) -> dict[str, float]:
    rows = max(1.0, stats["rows"])
    return {
        "rows": float(stats["rows"]),
        "regular_actions_per_state": stats["true_actions"] / rows,
        "model_actions_per_state": stats["pred_actions"] / rows,
        "source_f1": stats["source_f1_sum"] / rows,
        "source_target_f1": stats["source_target_f1_sum"] / rows,
        "action_f1": stats["action_f1_sum"] / rows,
        "action_count_mae": stats["action_count_abs_sum"] / rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--launch-temperature", type=float, default=1.0)
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--target-mask-mode", choices=["dataset", "all_planets"], default="dataset")
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = _load_rows(args.cache)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TinyPolicyValueNet(**payload.get("model", {})).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    predictions = _predict_keys(
        model,
        rows,
        device,
        batch_size=args.batch_size,
        launch_bias=args.launch_bias,
        launch_temperature=args.launch_temperature,
        target_pair_weight=args.target_pair_weight,
        target_mask_mode=args.target_mask_mode,
    )

    total = _empty_group()
    groups: dict[str, dict[str, float]] = defaultdict(_empty_group)
    micro_pred_source: Counter = Counter()
    micro_true_source: Counter = Counter()
    micro_pred_st: Counter = Counter()
    micro_true_st: Counter = Counter()
    micro_pred_action: Counter = Counter()
    micro_true_action: Counter = Counter()
    rows_with_no_pred = 0
    rows_with_no_label = 0

    for row_idx, (row, pred) in enumerate(zip(rows, predictions, strict=False)):
        true_source, true_st, true_action = _keys_from_label(row)
        pred_source, pred_st, pred_action, pred_count_float = pred
        pred_count = int(pred_count_float)
        true_count = int(sum(true_source.values()))
        rows_with_no_pred += int(pred_count == 0)
        rows_with_no_label += int(true_count == 0)
        _sp, _sr, sf = _prf(pred_source, true_source)
        _tp, _tr, tf = _prf(pred_st, true_st)
        _ap, _ar, af = _prf(pred_action, true_action)
        _add_group(total, true_count, pred_count, sf, tf, af)
        _add_group(groups[_group_key(row)], true_count, pred_count, sf, tf, af)
        micro_pred_source.update({(row_idx, key): value for key, value in pred_source.items()})
        micro_true_source.update({(row_idx, key): value for key, value in true_source.items()})
        micro_pred_st.update({(row_idx, key): value for key, value in pred_st.items()})
        micro_true_st.update({(row_idx, key): value for key, value in true_st.items()})
        micro_pred_action.update({(row_idx, key): value for key, value in pred_action.items()})
        micro_true_action.update({(row_idx, key): value for key, value in true_action.items()})

    micro_sp, micro_sr, micro_sf = _prf(micro_pred_source, micro_true_source)
    micro_tp, micro_tr, micro_tf = _prf(micro_pred_st, micro_true_st)
    micro_ap, micro_ar, micro_af = _prf(micro_pred_action, micro_true_action)
    result = {
        "cache": args.cache,
        "checkpoint": args.checkpoint,
        "rows": float(len(rows)),
        "launch_bias": args.launch_bias,
        "launch_temperature": args.launch_temperature,
        "target_pair_weight": args.target_pair_weight,
        "target_mask_mode": args.target_mask_mode,
        **_finish_group(total),
        "rows_with_no_pred_rate": rows_with_no_pred / max(1, len(rows)),
        "rows_with_no_label_rate": rows_with_no_label / max(1, len(rows)),
        "micro_source_precision": micro_sp,
        "micro_source_recall": micro_sr,
        "micro_source_f1": micro_sf,
        "micro_source_target_precision": micro_tp,
        "micro_source_target_recall": micro_tr,
        "micro_source_target_f1": micro_tf,
        "micro_action_precision": micro_ap,
        "micro_action_recall": micro_ar,
        "micro_action_f1": micro_af,
        "top_groups": {
            key: _finish_group(value)
            for key, value in sorted(groups.items(), key=lambda kv: (-kv[1]["rows"], kv[0]))[: max(0, args.top)]
        },
    }
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
