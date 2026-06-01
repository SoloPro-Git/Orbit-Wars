from __future__ import annotations

import argparse
import json
import math
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


def _turn_bucket(row: Any, width: int) -> str:
    turn = int(getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", -1)))
    if turn < 0:
        return "unknown"
    lo = (turn // max(1, width)) * max(1, width)
    return f"{lo}-{lo + max(1, width) - 1}"


def _ckpt_bucket(row: Any) -> str:
    checkpoint = str(getattr(row, "dagger_checkpoint", "unknown"))
    return Path(checkpoint).name or "unknown"


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


def _ship_bucket(norm_ship: float) -> str:
    # Feature is log1p(ships) / log(1000).
    ships = math.expm1(float(norm_ship) * math.log(1000.0))
    if ships < 20:
        return "ships<20"
    if ships < 50:
        return "ships20-49"
    if ships < 100:
        return "ships50-99"
    if ships < 200:
        return "ships100-199"
    return "ships>=200"


def _prod_bucket(norm_prod: float) -> str:
    prod = float(norm_prod) * 5.0
    if prod < 1.0:
        return "prod<1"
    if prod < 2.0:
        return "prod1"
    if prod < 4.0:
        return "prod2-3"
    return "prod>=4"


def _incoming_bucket(friend_norm: float, enemy_norm: float) -> str:
    friend = float(friend_norm) * 500.0
    enemy = float(enemy_norm) * 500.0
    if friend < 1.0 and enemy < 1.0:
        return "no_incoming"
    if enemy > friend + 10.0:
        return "enemy_in"
    if friend > enemy + 10.0:
        return "friend_in"
    return "mixed_in"


def _target_owner_bucket(row: Any, source_idx: int, slot: int) -> str:
    if int(row.launch_actions[source_idx, slot]) != 1:
        return "no_label"
    target_idx = int(row.target_actions[source_idx, slot])
    if target_idx < 0 or target_idx >= MAX_PLANETS:
        return "invalid"
    planets = np.asarray(row.planets, dtype=np.float32)
    if planets[target_idx, 0] > 0.5:
        return "own_target"
    if planets[target_idx, 1] > 0.5:
        return "neutral_target"
    if planets[target_idx, 2] > 0.5:
        return "enemy_target"
    return "unknown_target"


def _new_stats() -> dict[str, float]:
    return defaultdict(float)


def _add_slot(stats: dict[str, float], label: bool, pred: bool) -> None:
    stats["slots"] += 1.0
    stats["label_pos"] += float(label)
    stats["pred_pos"] += float(pred)
    stats["tp"] += float(label and pred)
    stats["fn"] += float(label and not pred)
    stats["fp"] += float(pred and not label)
    stats["tn"] += float((not label) and (not pred))


def _summarize(stats: dict[str, float]) -> dict[str, float]:
    slots = max(1.0, stats["slots"])
    label_pos = stats["label_pos"]
    pred_pos = stats["pred_pos"]
    precision = stats["tp"] / pred_pos if pred_pos > 0 else (1.0 if label_pos == 0 else 0.0)
    recall = stats["tp"] / label_pos if label_pos > 0 else (1.0 if pred_pos == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "slots": stats["slots"],
        "label_pos": label_pos,
        "pred_pos": pred_pos,
        "tp": stats["tp"],
        "fn": stats["fn"],
        "fp": stats["fp"],
        "label_rate": label_pos / slots,
        "pred_rate": pred_pos / slots,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fn_share": stats["fn"] / max(1.0, label_pos),
        "fp_share": stats["fp"] / max(1.0, pred_pos),
    }


def _bucket_values(row: Any, source_idx: int, slot: int) -> dict[str, str]:
    planets = np.asarray(row.planets, dtype=np.float32)
    source = planets[source_idx]
    turn = _turn_bucket(row, 50)
    gap = _gap_bucket(row)
    ships = _ship_bucket(float(source[6]))
    prod = _prod_bucket(float(source[7]))
    incoming = _incoming_bucket(float(source[13]), float(source[14]))
    target_owner = _target_owner_bucket(row, source_idx, slot)
    slot_name = f"slot{slot}"
    return {
        "turn": turn,
        "gap": gap,
        "turn_gap": f"{turn}/{gap}",
        "checkpoint": _ckpt_bucket(row),
        "checkpoint_gap": f"{_ckpt_bucket(row)}/{gap}",
        "slot": slot_name,
        "ships": ships,
        "prod": prod,
        "incoming": incoming,
        "target_owner": target_owner,
        "ships_target_owner": f"{ships}/{target_owner}",
        "prod_target_owner": f"{prod}/{target_owner}",
        "turn_gap_target_owner": f"{turn}/{gap}/{target_owner}",
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

        for local_idx, row in enumerate(batch_rows):
            label_launch = np.asarray(row.launch_actions, dtype=np.int64)
            own_mask = np.asarray(row.own_mask, dtype=np.bool_)
            planet_mask = np.asarray(row.planet_mask, dtype=np.bool_)
            for source_idx in np.where(own_mask & planet_mask)[0]:
                for slot in range(ACTION_SLOTS):
                    label = bool(label_launch[source_idx, slot] == 1)
                    pred = bool(pred_launch_batch[local_idx, source_idx, slot] == 1)
                    _add_slot(overall, label, pred)
                    for group_name, bucket in _bucket_values(row, int(source_idx), slot).items():
                        _add_slot(groups[group_name][bucket], label, pred)

    result: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "cache": args.cache,
        "cache_rows": float(len(rows)),
        "source_cache_metrics": metadata.get("metrics", {}),
        "launch_bias": float(args.launch_bias),
        "overall": _summarize(overall),
        "groups": {},
    }
    for group_name, buckets in sorted(groups.items()):
        summaries = []
        for bucket, stats in buckets.items():
            summary = _summarize(stats)
            if summary["slots"] >= args.min_bucket_slots and summary["label_pos"] >= args.min_bucket_labels:
                summaries.append((bucket, summary))
        if args.sort_by == "fn":
            summaries.sort(key=lambda item: (-item[1]["fn"], item[0]))
        elif args.sort_by == "recall":
            summaries.sort(key=lambda item: (item[1]["recall"], -item[1]["label_pos"], item[0]))
        elif args.sort_by == "f1":
            summaries.sort(key=lambda item: (item[1]["f1"], -item[1]["label_pos"], item[0]))
        else:
            summaries.sort(key=lambda item: (item[0],))
        result["groups"][group_name] = {bucket: summary for bucket, summary in summaries[: args.max_buckets_per_group]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Break down DAgger source launch errors by source feature buckets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=260530)
    parser.add_argument("--launch-bias", type=float, default=-0.05)
    parser.add_argument("--min-bucket-slots", type=float, default=100.0)
    parser.add_argument("--min-bucket-labels", type=float, default=20.0)
    parser.add_argument("--max-buckets-per-group", type=int, default=12)
    parser.add_argument("--sort-by", choices=["fn", "recall", "f1", "name"], default="fn")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    result = analyze(args)
    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
