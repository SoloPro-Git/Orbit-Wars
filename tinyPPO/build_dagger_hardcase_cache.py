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
    with path.open("rb") as f:
        payload = pickle.load(f)
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        metadata = {key: value for key, value in payload.items() if key != "rows"}
        rows = payload.get("rows")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise TypeError(f"unsupported cache format in {path!s}")
    return rows, metadata


def _row_action_count(row: Any) -> int:
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


def _reason_priority(reasons: set[str]) -> str:
    order = ["enemy_neutral_to_own", "owner_mismatch", "rank_gt_max", "margin_gt_min"]
    for item in order:
        if item in reasons:
            return item
    return "selected"


@torch.no_grad()
def select_hard_cases(args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    rows, cache_metadata = _load_rows(Path(args.cache))
    if args.max_input_rows > 0 and args.max_input_rows < len(rows):
        rng = random.Random(args.seed)
        rows = [rows[idx] for idx in sorted(rng.sample(range(len(rows)), args.max_input_rows))]

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TinyPolicyValueNet(**payload.get("model", {})).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    selected: list[tuple[Any, int, int, str, Counter[str]]] = []
    metrics: dict[str, Any] = {
        "cache": args.cache,
        "checkpoint": args.checkpoint,
        "target_mask_mode": args.target_mask_mode,
        "target_pair_weight": float(args.target_pair_weight),
        "rows_seen": float(len(rows)),
        "rows_with_actions": 0.0,
        "actions_seen": 0.0,
        "selected_rows_before_cap": 0.0,
        "selected_actions_before_cap": 0.0,
        "selected_rows": 0.0,
        "selected_actions": 0.0,
        "reason_counts": {},
        "primary_reason_counts": {},
        "regular_owner_counts": {},
        "model_top1_owner_counts": {},
        "selected_regular_owner_counts": {},
        "selected_model_top1_owner_counts": {},
        "turn_bucket_counts": {},
        "source_counts": {},
        "source_cache_metrics": cache_metadata.get("metrics", {}),
        "hardcase_loss_weight": float(args.weight),
    }
    reason_counts: Counter[str] = Counter()
    primary_reason_counts: Counter[str] = Counter()
    regular_owner_counts: Counter[str] = Counter()
    model_owner_counts: Counter[str] = Counter()
    selected_regular_owner_counts: Counter[str] = Counter()
    selected_model_owner_counts: Counter[str] = Counter()
    turn_bucket_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

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
            metrics["rows_with_actions"] += 1.0
            metrics["actions_seen"] += float(action_count)
            mask_np = _target_mask(row, args.target_mask_mode)
            active = np.argwhere(np.asarray(row.launch_mask, dtype=np.bool_))
            row_reasons: set[str] = set()
            row_model_owner_counts: Counter[str] = Counter()
            row_rank = 0
            row_margin = 0
            for source_idx_raw, slot_raw in active:
                source_idx = int(source_idx_raw)
                slot = int(slot_raw)
                if slot >= ACTION_SLOTS:
                    continue
                label_idx = int(row.target_actions[source_idx, slot])
                label_owner = _owner_bucket(row, label_idx)
                regular_owner_counts[label_owner] += 1
                if label_idx < 0 or label_idx >= MAX_PLANETS or not bool(mask_np[source_idx, label_idx]):
                    continue
                logits = target_logits[local_idx, source_idx, slot]
                valid_mask = torch.tensor(mask_np[source_idx], dtype=torch.bool, device=device)
                masked_logits = logits.masked_fill(~valid_mask, -1e9)
                pred_idx = int(torch.argmax(masked_logits).item())
                pred_owner = _owner_bucket(row, pred_idx)
                model_owner_counts[pred_owner] += 1
                row_model_owner_counts[pred_owner] += 1
                label_score = logits[label_idx]
                rank = int((logits[valid_mask] > label_score).sum().item()) + 1
                other_logits = masked_logits.clone()
                other_logits[label_idx] = -1e9
                margin = float(torch.max(other_logits).item() - label_score.item())
                row_rank = max(row_rank, rank)
                row_margin = max(row_margin, int(margin > args.min_margin))
                if pred_owner != label_owner:
                    row_reasons.add("owner_mismatch")
                if label_owner in {"enemy", "neutral"} and pred_owner == "own":
                    row_reasons.add("enemy_neutral_to_own")
                if args.max_rank > 0 and rank > args.max_rank:
                    row_reasons.add("rank_gt_max")
                if margin > args.min_margin:
                    row_reasons.add("margin_gt_min")
            if not row_reasons:
                continue
            for reason in row_reasons:
                reason_counts[reason] += 1
            primary = _reason_priority(row_reasons)
            primary_reason_counts[primary] += 1
            row.dagger_source_name = str(getattr(row, "dagger_source_name", "hardcase"))  # type: ignore[attr-defined]
            row.dagger_hardcase_reason = primary  # type: ignore[attr-defined]
            row.dagger_hardcase_reasons = ",".join(sorted(row_reasons))  # type: ignore[attr-defined]
            row.dagger_hardcase_max_rank = int(row_rank)  # type: ignore[attr-defined]
            row.dagger_hardcase_margin_hit = int(row_margin)  # type: ignore[attr-defined]
            row.dagger_loss_weight_override = float(args.weight)  # type: ignore[attr-defined]
            selected.append((row, action_count, row_rank, primary, row_model_owner_counts))

    metrics["selected_rows_before_cap"] = float(len(selected))
    metrics["selected_actions_before_cap"] = float(sum(item[1] for item in selected))
    rng = random.Random(args.seed)
    rng.shuffle(selected)
    capped: list[Any] = []
    actions = 0
    for row, action_count, _rank, primary, row_model_owner_counts in selected:
        if args.max_actions > 0 and actions + action_count > args.max_actions and capped:
            continue
        capped.append(row)
        actions += action_count
        turn_bucket_counts[_turn_bucket(row)] += 1
        source_counts[str(getattr(row, "dagger_source_name", "unknown"))] += 1
        selected_model_owner_counts.update(row_model_owner_counts)
        active = np.argwhere(np.asarray(row.launch_mask, dtype=np.bool_))
        for source_idx_raw, slot_raw in active:
            source_idx = int(source_idx_raw)
            slot = int(slot_raw)
            label_idx = int(row.target_actions[source_idx, slot])
            selected_regular_owner_counts[_owner_bucket(row, label_idx)] += 1
        if args.max_actions > 0 and actions >= args.max_actions:
            break
    metrics["selected_rows"] = float(len(capped))
    metrics["selected_actions"] = float(actions)
    metrics["reason_counts"] = dict(sorted(reason_counts.items()))
    metrics["primary_reason_counts"] = dict(sorted(primary_reason_counts.items()))
    metrics["regular_owner_counts"] = dict(sorted(regular_owner_counts.items()))
    metrics["model_top1_owner_counts"] = dict(sorted(model_owner_counts.items()))
    metrics["selected_regular_owner_counts"] = dict(sorted(selected_regular_owner_counts.items()))
    metrics["selected_model_top1_owner_counts"] = dict(sorted(selected_model_owner_counts.items()))
    metrics["turn_bucket_counts"] = dict(sorted(turn_bucket_counts.items()))
    metrics["source_counts"] = dict(sorted(source_counts.items()))
    return capped, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DAgger hard-case cache from rows where the checkpoint misranks regular targets.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--target-mask-mode", choices=["dataset", "all_planets"], default="all_planets")
    parser.add_argument("--max-rank", type=int, default=3, help="Keep rows containing a regular target ranked worse than this. <=0 disables rank selection.")
    parser.add_argument("--min-margin", type=float, default=0.0, help="Keep rows where top1 logit exceeds the regular label by more than this margin.")
    parser.add_argument("--max-actions", type=int, default=12000)
    parser.add_argument("--max-input-rows", type=int, default=0)
    parser.add_argument("--weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=260527)
    args = parser.parse_args()

    rows, metrics = select_hard_cases(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"rows": rows, "metrics": metrics, "args": vars(args)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    print(json.dumps({"event": "dagger_hardcase_cache_saved", "path": str(out), "metrics": metrics}, ensure_ascii=True, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
