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


def _load_rows(path: str | Path) -> tuple[list[Any], dict[str, Any]]:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        metadata = {key: value for key, value in payload.items() if key != "rows"}
        rows = payload.get("rows")
    elif isinstance(payload, tuple):
        rows = payload[0]
    else:
        rows = payload
    if not isinstance(rows, list):
        raise TypeError(f"unsupported cache format in {path!s}")
    return rows, metadata


def _sample_rows(rows: list[Any], max_rows: int, seed: int) -> list[Any]:
    if max_rows <= 0 or max_rows >= len(rows):
        return rows
    rng = random.Random(seed)
    return [rows[idx] for idx in sorted(rng.sample(range(len(rows)), max_rows))]


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


def _empty_stats() -> dict[str, Any]:
    return {
        "actions": 0,
        "covered": 0,
        "rank_sum": 0.0,
        "reciprocal_rank_sum": 0.0,
        "margin_sum": 0.0,
        "rank_values": [],
        "top_hits": defaultdict(int),
    }


def _add_rank(stats: dict[str, Any], rank: int, margin: float, topk: list[int]) -> None:
    stats["actions"] += 1
    stats["covered"] += 1
    stats["rank_sum"] += float(rank)
    stats["reciprocal_rank_sum"] += 1.0 / float(rank)
    stats["margin_sum"] += float(margin)
    stats["rank_values"].append(int(rank))
    for k in topk:
        if rank <= k:
            stats["top_hits"][int(k)] += 1


def _add_uncovered(stats: dict[str, Any]) -> None:
    stats["actions"] += 1


def _finish_stats(stats: dict[str, Any], topk: list[int]) -> dict[str, float]:
    actions = max(1, int(stats["actions"]))
    covered = int(stats["covered"])
    ranks = np.asarray(stats["rank_values"], dtype=np.float32)
    out = {
        "actions": float(stats["actions"]),
        "covered": float(covered),
        "covered_rate": float(covered / actions),
        "mean_rank": float(stats["rank_sum"] / max(1, covered)),
        "mrr": float(stats["reciprocal_rank_sum"] / max(1, covered)),
        "mean_top1_margin_over_label": float(stats["margin_sum"] / max(1, covered)),
    }
    if covered:
        out.update(
            {
                "median_rank": float(np.percentile(ranks, 50)),
                "p75_rank": float(np.percentile(ranks, 75)),
                "p90_rank": float(np.percentile(ranks, 90)),
                "p95_rank": float(np.percentile(ranks, 95)),
            }
        )
    else:
        out.update({"median_rank": 0.0, "p75_rank": 0.0, "p90_rank": 0.0, "p95_rank": 0.0})
    for k in topk:
        out[f"top{k}_rate"] = float(stats["top_hits"][int(k)] / max(1, covered))
    return out


def _top_owner(row: Any, source_idx: int, logits: torch.Tensor, mask: np.ndarray) -> str:
    masked = logits.masked_fill(~torch.tensor(mask[source_idx], dtype=torch.bool, device=logits.device), -1e9)
    top_idx = int(torch.argmax(masked).item())
    return _owner_bucket(row, top_idx)


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    rows, metadata = _load_rows(args.cache)
    rows = _sample_rows(rows, args.max_rows, args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TinyPolicyValueNet(**payload.get("model", {})).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    topk = sorted({int(k) for k in args.topk if int(k) > 0})
    total = _empty_stats()
    by_owner: dict[str, dict[str, Any]] = defaultdict(_empty_stats)
    by_turn: dict[str, dict[str, Any]] = defaultdict(_empty_stats)
    top_owner_counts: dict[str, int] = defaultdict(int)
    regular_owner_counts: dict[str, int] = defaultdict(int)
    rows_with_labels = 0

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch = _row_batch(batch_rows, device)
        out = model(**batch)
        target_logits = out["target_logits"]
        if "target_pair_logits" in out and abs(args.target_pair_weight) > 1e-9:
            target_logits = target_logits + float(args.target_pair_weight) * out["target_pair_logits"][:, :, None, :]

        for local_idx, row in enumerate(batch_rows):
            mask_np = _target_mask(row, args.target_mask_mode)
            active = np.argwhere(np.asarray(row.launch_mask, dtype=np.bool_))
            if len(active) > 0:
                rows_with_labels += 1
            for source_idx_raw, slot_raw in active:
                source_idx = int(source_idx_raw)
                slot = int(slot_raw)
                if slot >= ACTION_SLOTS:
                    continue
                label_idx = int(row.target_actions[source_idx, slot])
                owner = _owner_bucket(row, label_idx)
                turn = _turn_bucket(row)
                regular_owner_counts[owner] += 1
                if label_idx < 0 or label_idx >= MAX_PLANETS or not bool(mask_np[source_idx, label_idx]):
                    _add_uncovered(total)
                    _add_uncovered(by_owner[owner])
                    _add_uncovered(by_turn[turn])
                    continue
                logits = target_logits[local_idx, source_idx, slot]
                valid_mask = torch.tensor(mask_np[source_idx], dtype=torch.bool, device=device)
                label_score = logits[label_idx]
                rank = int((logits[valid_mask] > label_score).sum().item()) + 1
                other_logits = logits.masked_fill(~valid_mask, -1e9).clone()
                other_logits[label_idx] = -1e9
                margin = float(torch.max(other_logits).item() - label_score.item())
                _add_rank(total, rank, margin, topk)
                _add_rank(by_owner[owner], rank, margin, topk)
                _add_rank(by_turn[turn], rank, margin, topk)
                top_owner_counts[_top_owner(row, source_idx, logits, mask_np)] += 1

    result = {
        "checkpoint": args.checkpoint,
        "cache": args.cache,
        "cache_metrics": metadata.get("metrics", {}),
        "rows": float(len(rows)),
        "rows_with_labels": float(rows_with_labels),
        "target_mask_mode": args.target_mask_mode,
        "target_pair_weight": float(args.target_pair_weight),
        "summary": _finish_stats(total, topk),
        "regular_target_owner_counts": dict(sorted(regular_owner_counts.items())),
        "model_top1_owner_counts": dict(sorted(top_owner_counts.items())),
        "by_regular_target_owner": {
            key: _finish_stats(value, topk) for key, value in sorted(by_owner.items())
        },
        "by_turn": {
            key: _finish_stats(value, topk) for key, value in sorted(by_turn.items())
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank regular-labelled DAgger targets under a checkpoint's target logits.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260527)
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--target-mask-mode", choices=["dataset", "all_planets"], default="all_planets")
    parser.add_argument("--topk", type=int, nargs="*", default=[1, 3, 5, 10])
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
