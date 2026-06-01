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


def _row_action_count(row: Any) -> int:
    labelled = getattr(row, "labelled", None)
    if labelled is not None:
        return int(labelled)
    return int(np.asarray(row.launch_mask, dtype=np.bool_).sum())


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


def _ckpt_bucket(row: Any) -> str:
    checkpoint = str(getattr(row, "dagger_checkpoint", "unknown"))
    name = Path(checkpoint).name
    return name or "unknown"


@torch.no_grad()
def select_agreement_rows(args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    rows, metadata = _load_rows(Path(args.cache))
    if args.max_input_rows > 0 and args.max_input_rows < len(rows):
        rng = random.Random(args.seed)
        rows = [rows[idx] for idx in sorted(rng.sample(range(len(rows)), args.max_input_rows))]

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TinyPolicyValueNet(**payload.get("model", {})).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    candidates: list[tuple[Any, int, str, str, str]] = []
    stats: dict[str, Any] = {
        "cache": args.cache,
        "checkpoint": args.checkpoint,
        "target_mask_mode": args.target_mask_mode,
        "target_pair_weight": float(args.target_pair_weight),
        "max_rank": float(args.max_rank),
        "max_margin": float(args.max_margin),
        "rows_seen": float(len(rows)),
        "rows_with_actions": 0.0,
        "actions_seen": 0.0,
        "candidate_rows": 0.0,
        "candidate_actions": 0.0,
        "selected_rows": 0.0,
        "selected_actions": 0.0,
        "reject_counts": {},
        "turn_bucket_counts": {},
        "gap_bucket_counts": {},
        "checkpoint_bucket_counts": {},
        "source_cache_metrics": metadata.get("metrics", {}),
        "agreement_loss_weight": float(args.weight),
    }
    reject_counts: Counter[str] = Counter()
    candidate_actions = 0

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch = _row_batch(batch_rows, device)
        out = model(**batch)
        target_logits = out["target_logits"]
        if "target_pair_logits" in out and abs(args.target_pair_weight) > 1e-9:
            target_logits = target_logits + float(args.target_pair_weight) * out["target_pair_logits"][:, :, None, :]

        for local_idx, row in enumerate(batch_rows):
            action_count = _row_action_count(row)
            if action_count <= 0:
                continue
            stats["rows_with_actions"] += 1.0
            stats["actions_seen"] += float(action_count)
            if args.max_actions_per_row >= 0 and action_count > args.max_actions_per_row:
                reject_counts["too_many_actions"] += 1
                continue
            if args.min_actions_per_row >= 0 and action_count < args.min_actions_per_row:
                reject_counts["too_few_actions"] += 1
                continue
            turn = int(getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", -1)))
            if args.min_turn >= 0 and turn < args.min_turn:
                reject_counts["too_early"] += 1
                continue
            if args.max_turn >= 0 and turn > args.max_turn:
                reject_counts["too_late"] += 1
                continue

            mask_np = _target_mask(row, args.target_mask_mode)
            active = np.argwhere(np.asarray(row.launch_mask, dtype=np.bool_))
            row_max_rank = 1
            row_max_margin = -1e9
            ok = True
            for source_idx_raw, slot_raw in active:
                source_idx = int(source_idx_raw)
                slot = int(slot_raw)
                if slot >= ACTION_SLOTS:
                    continue
                label_idx = int(row.target_actions[source_idx, slot])
                if label_idx < 0 or label_idx >= MAX_PLANETS or not bool(mask_np[source_idx, label_idx]):
                    ok = False
                    reject_counts["label_not_valid"] += 1
                    break
                logits = target_logits[local_idx, source_idx, slot]
                valid_mask = torch.tensor(mask_np[source_idx], dtype=torch.bool, device=device)
                label_score = logits[label_idx]
                rank = int((logits[valid_mask] > label_score).sum().item()) + 1
                other_logits = logits.masked_fill(~valid_mask, -1e9).clone()
                other_logits[label_idx] = -1e9
                margin = float(torch.max(other_logits).item() - label_score.item())
                row_max_rank = max(row_max_rank, rank)
                row_max_margin = max(row_max_margin, margin)
                if args.max_rank > 0 and rank > args.max_rank:
                    ok = False
                    reject_counts["rank_too_low"] += 1
                    break
                if margin > args.max_margin:
                    ok = False
                    reject_counts["margin_too_low"] += 1
                    break
            if not ok:
                continue
            setattr(row, "dagger_source_name", "agreement")
            setattr(row, "dagger_agreement_max_rank", int(row_max_rank))
            setattr(row, "dagger_agreement_max_margin", float(row_max_margin))
            setattr(row, "dagger_loss_weight_override", float(args.weight))
            candidates.append((row, action_count, _turn_bucket(row, args.turn_bucket_width), _gap_bucket(row), _ckpt_bucket(row)))
            candidate_actions += action_count

    stats["candidate_rows"] = float(len(candidates))
    stats["candidate_actions"] = float(candidate_actions)
    rng = random.Random(args.seed)
    by_key: dict[tuple[str, str, str], list[tuple[Any, int, str, str, str]]] = defaultdict(list)
    for item in candidates:
        by_key[(item[2], item[3], item[4])].append(item)
    for bucket in by_key.values():
        rng.shuffle(bucket)

    selected: list[Any] = []
    selected_actions = 0
    selected_per_key: Counter[tuple[str, str, str]] = Counter()
    target_per_key = max(1, int(args.max_actions) // max(1, len(by_key))) if args.max_actions > 0 else 10**12
    for key in sorted(by_key):
        for row, actions, _turn, _gap, _ckpt in by_key[key]:
            if args.max_actions > 0 and selected_actions + actions > int(args.max_actions):
                break
            if selected_per_key[key] + actions > target_per_key and selected_per_key[key] > 0:
                break
            selected.append(row)
            selected_actions += actions
            selected_per_key[key] += actions

    if args.max_actions <= 0 or selected_actions < int(args.max_actions):
        selected_ids = {id(row) for row in selected}
        leftovers = [item for bucket in by_key.values() for item in bucket if id(item[0]) not in selected_ids]
        rng.shuffle(leftovers)
        for row, actions, _turn, _gap, _ckpt in leftovers:
            if args.max_actions > 0 and selected_actions + actions > int(args.max_actions):
                continue
            selected.append(row)
            selected_actions += actions
            if args.max_actions > 0 and selected_actions >= int(args.max_actions):
                break

    rng.shuffle(selected)
    stats["selected_rows"] = float(len(selected))
    stats["selected_actions"] = float(selected_actions)
    stats["reject_counts"] = dict(sorted(reject_counts.items()))
    stats["turn_bucket_counts"] = dict(Counter(_turn_bucket(row, args.turn_bucket_width) for row in selected))
    stats["gap_bucket_counts"] = dict(Counter(_gap_bucket(row) for row in selected))
    stats["checkpoint_bucket_counts"] = dict(Counter(_ckpt_bucket(row) for row in selected))
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a low-conflict DAgger cache whose regular targets agree with an anchor checkpoint.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--target-mask-mode", choices=["dataset", "all_planets"], default="all_planets")
    parser.add_argument("--max-rank", type=int, default=1)
    parser.add_argument("--max-margin", type=float, default=0.0)
    parser.add_argument("--max-actions", type=int, default=12000)
    parser.add_argument("--max-input-rows", type=int, default=0)
    parser.add_argument("--min-actions-per-row", type=int, default=-1)
    parser.add_argument("--max-actions-per-row", type=int, default=8)
    parser.add_argument("--min-turn", type=int, default=-1)
    parser.add_argument("--max-turn", type=int, default=-1)
    parser.add_argument("--turn-bucket-width", type=int, default=50)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260530)
    args = parser.parse_args()

    rows, metrics = select_agreement_rows(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump({"rows": rows, "metrics": metrics, "args": vars(args)}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    summary = out.with_suffix(out.suffix + ".summary.json")
    summary.write_text(json.dumps(metrics, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "dagger_agreement_cache_saved", "path": str(out), "summary": str(summary), "metrics": metrics}, ensure_ascii=True, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
