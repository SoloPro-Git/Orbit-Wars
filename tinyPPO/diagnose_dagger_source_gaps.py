from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyPPO.model import TinyPolicyValueNet


def _load_rows(path: str | Path) -> list[Any]:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        rows = obj.get("rows")
    elif isinstance(obj, tuple):
        rows = obj[0]
    else:
        rows = obj
    if not isinstance(rows, list):
        raise TypeError(f"unsupported cache format in {path!s}")
    return rows


def _bucket(value: float, edges: list[float], labels: list[str]) -> str:
    for edge, label in zip(edges, labels, strict=False):
        if value < edge:
            return label
    return labels[-1]


def _turn_bucket(turn: int) -> str:
    lo = (max(0, int(turn)) // 50) * 50
    return f"{lo:03d}-{lo + 49:03d}"


def _target_kind(row: Any, source_idx: int) -> str:
    slots = np.where(np.asarray(row.launch_mask[source_idx], dtype=np.bool_))[0]
    kinds: list[str] = []
    for slot in slots:
        target_idx = int(row.target_actions[source_idx, int(slot)])
        if target_idx < 0 or target_idx >= row.planets.shape[0] or not bool(row.planet_mask[target_idx]):
            kinds.append("invalid")
        elif bool(row.pair_features[source_idx, target_idx, 10] > 0.5):
            kinds.append("enemy")
        elif bool(row.pair_features[source_idx, target_idx, 9] > 0.5):
            kinds.append("neutral")
        elif bool(row.pair_features[source_idx, target_idx, 11] > 0.5):
            kinds.append("friendly")
        else:
            kinds.append("other")
    if not kinds:
        return "none"
    if len(set(kinds)) == 1:
        return kinds[0]
    priority = ["enemy", "neutral", "friendly", "other", "invalid"]
    return "mixed_" + next(kind for kind in priority if kind in kinds)


def _source_record(row: Any, source_idx: int, prob: float, pred: bool, labelled: bool) -> dict[str, Any]:
    ships_norm = float(row.planets[source_idx, 6])
    ships_est = float(np.expm1(ships_norm * np.log(1000.0)))
    production = float(row.planets[source_idx, 7] * 5.0)
    incoming_friend = float(row.planets[source_idx, 13] * 500.0)
    incoming_enemy = float(row.planets[source_idx, 14] * 500.0)
    slots = int(np.asarray(row.launch_mask[source_idx], dtype=np.bool_).sum()) if labelled else 0
    return {
        "prob": float(prob),
        "pred": bool(pred),
        "labelled": bool(labelled),
        "slots": float(slots),
        "turn": int(getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", 0))),
        "outcome": str(getattr(row, "dagger_model_outcome", "unknown")),
        "source": str(getattr(row, "dagger_source_name", "unknown")),
        "bucket": str(getattr(row, "dagger_bucket_name", "unknown")),
        "score_gap": float(getattr(row, "dagger_score_gap", 0.0)),
        "action_gap": float(getattr(row, "dagger_action_gap", 0.0)),
        "owned_planets": float(getattr(row, "dagger_owned_planets", 0.0)),
        "ships": ships_est,
        "production": production,
        "incoming_friend": incoming_friend,
        "incoming_enemy": incoming_enemy,
        "target_kind": _target_kind(row, source_idx) if labelled else "unlabelled",
        "ships_bucket": _bucket(ships_est, [25, 75, 150, 300, 600], ["<25", "25-74", "75-149", "150-299", "300-599", "600+"]),
        "prod_bucket": _bucket(production, [1.5, 2.5, 3.5, 4.5], ["0-1", "2", "3", "4", "5+"]),
        "incoming_enemy_bucket": _bucket(incoming_enemy, [1, 25, 100, 300], ["0", "1-24", "25-99", "100-299", "300+"]),
    }


def _empty_stats() -> dict[str, float]:
    return {
        "items": 0.0,
        "slots": 0.0,
        "prob_sum": 0.0,
        "pred_sum": 0.0,
        "low_prob_lt_025": 0.0,
        "low_prob_lt_050": 0.0,
        "ships_sum": 0.0,
        "production_sum": 0.0,
        "incoming_enemy_sum": 0.0,
        "score_gap_sum": 0.0,
        "action_gap_sum": 0.0,
    }


def _add(stats: dict[str, float], rec: dict[str, Any]) -> None:
    stats["items"] += 1.0
    stats["slots"] += float(rec["slots"])
    stats["prob_sum"] += float(rec["prob"])
    stats["pred_sum"] += float(rec["pred"])
    stats["low_prob_lt_025"] += float(float(rec["prob"]) < 0.25)
    stats["low_prob_lt_050"] += float(float(rec["prob"]) < 0.50)
    stats["ships_sum"] += float(rec["ships"])
    stats["production_sum"] += float(rec["production"])
    stats["incoming_enemy_sum"] += float(rec["incoming_enemy"])
    stats["score_gap_sum"] += float(rec["score_gap"])
    stats["action_gap_sum"] += float(rec["action_gap"])


def _finalize(stats: dict[str, float]) -> dict[str, float]:
    n = max(1.0, float(stats["items"]))
    return {
        "items": float(stats["items"]),
        "slots": float(stats["slots"]),
        "prob_mean": float(stats["prob_sum"]) / n,
        "pred_rate": float(stats["pred_sum"]) / n,
        "low_prob_lt_025_rate": float(stats["low_prob_lt_025"]) / n,
        "low_prob_lt_050_rate": float(stats["low_prob_lt_050"]) / n,
        "ships_mean": float(stats["ships_sum"]) / n,
        "production_mean": float(stats["production_sum"]) / n,
        "incoming_enemy_mean": float(stats["incoming_enemy_sum"]) / n,
        "score_gap_mean": float(stats["score_gap_sum"]) / n,
        "action_gap_mean": float(stats["action_gap_sum"]) / n,
    }


def _top(items: dict[str, dict[str, float]], limit: int) -> dict[str, dict[str, float]]:
    pairs = sorted(items.items(), key=lambda kv: (-kv[1]["items"], kv[0]))
    return {key: _finalize(value) for key, value in pairs[:limit]}


@torch.no_grad()
def _source_probs(model: TinyPolicyValueNet, rows: list[Any], device: torch.device, batch_size: int) -> list[np.ndarray]:
    out_probs: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = {
            "planets": torch.tensor(np.stack([r.planets for r in batch_rows]), dtype=torch.float32, device=device),
            "pair_features": torch.tensor(np.stack([r.pair_features for r in batch_rows]), dtype=torch.float32, device=device),
            "global_features": torch.tensor(np.stack([r.global_features for r in batch_rows]), dtype=torch.float32, device=device),
            "planet_mask": torch.tensor(np.stack([r.planet_mask for r in batch_rows]), dtype=torch.bool, device=device),
            "own_mask": torch.tensor(np.stack([r.own_mask for r in batch_rows]), dtype=torch.bool, device=device),
        }
        logits = model(**batch)["source_logits"]
        slot_probs = torch.softmax(logits, dim=-1)[..., 1].clamp(0.0, 1.0)
        probs = 1.0 - torch.prod(1.0 - slot_probs, dim=-1)
        out_probs.extend(probs.detach().cpu().numpy())
    return out_probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int, default=0)
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

    probs_by_row = _source_probs(model, rows, device, args.batch_size)
    labelled = _empty_stats()
    unlabelled_own = _empty_stats()
    missed = _empty_stats()
    hit = _empty_stats()
    by_labelled_group: dict[str, dict[str, float]] = defaultdict(_empty_stats)
    by_missed_group: dict[str, dict[str, float]] = defaultdict(_empty_stats)

    rows_with_missed = 0
    total_regular_sources = 0
    total_pred_sources = 0
    total_overlap_sources = 0

    group_keys = ["target_kind", "ships_bucket", "prod_bucket", "incoming_enemy_bucket", "outcome", "source"]
    for row, row_probs in zip(rows, probs_by_row, strict=False):
        launch_by_source = np.asarray(row.launch_mask, dtype=np.bool_).any(axis=1)
        pred_by_source = np.asarray(row_probs >= float(args.threshold), dtype=np.bool_)
        own_mask = np.asarray(row.own_mask, dtype=np.bool_)
        valid = own_mask & np.asarray(row.planet_mask, dtype=np.bool_)
        regular_sources = int((launch_by_source & valid).sum())
        pred_sources = int((pred_by_source & valid).sum())
        overlap_sources = int((launch_by_source & pred_by_source & valid).sum())
        total_regular_sources += regular_sources
        total_pred_sources += pred_sources
        total_overlap_sources += overlap_sources
        row_missed = False
        for source_idx in np.where(valid)[0]:
            rec = _source_record(
                row,
                int(source_idx),
                float(row_probs[source_idx]),
                bool(pred_by_source[source_idx]),
                bool(launch_by_source[source_idx]),
            )
            if rec["labelled"]:
                _add(labelled, rec)
                if rec["pred"]:
                    _add(hit, rec)
                else:
                    _add(missed, rec)
                    row_missed = True
                for key in group_keys:
                    _add(by_labelled_group[f"{key}:{rec[key]}"], rec)
                    if not rec["pred"]:
                        _add(by_missed_group[f"{key}:{rec[key]}"], rec)
            else:
                _add(unlabelled_own, rec)
        rows_with_missed += int(row_missed)

    summary = {
        "cache": args.cache,
        "checkpoint": args.checkpoint,
        "rows": len(rows),
        "threshold": float(args.threshold),
        "regular_sources": float(total_regular_sources),
        "pred_sources": float(total_pred_sources),
        "overlap_sources": float(total_overlap_sources),
        "source_precision": float(total_overlap_sources) / max(1.0, float(total_pred_sources)),
        "source_recall": float(total_overlap_sources) / max(1.0, float(total_regular_sources)),
        "rows_with_missed_source_rate": float(rows_with_missed) / max(1.0, float(len(rows))),
        "labelled_regular_sources": _finalize(labelled),
        "hit_regular_sources": _finalize(hit),
        "missed_regular_sources": _finalize(missed),
        "unlabelled_own_sources": _finalize(unlabelled_own),
        "labelled_groups_top": _top(by_labelled_group, args.top),
        "missed_groups_top": _top(by_missed_group, args.top),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
