from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _row_action_count(row: Any) -> int:
    return int(np.asarray(row.launch_mask).sum())


def _turn(row: Any) -> int | None:
    value = getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", None))
    return None if value is None else int(value)


def _outcome(row: Any) -> str | None:
    value = getattr(row, "dagger_model_outcome", None)
    return None if value is None else str(value)


def _parse_source(raw: str) -> tuple[Path, str]:
    parts = raw.split("|")
    if len(parts) != 2:
        raise ValueError(f"--source must be path|name; got {raw!r}")
    return Path(parts[0]), parts[1]


def _parse_bucket(raw: str) -> dict[str, Any]:
    parts = raw.split("|")
    if len(parts) != 4:
        raise ValueError("--bucket must be outcome|min_turn|max_turn|max_actions")
    outcome, min_turn, max_turn, max_actions = parts
    return {
        "outcome": outcome,
        "min_turn": int(min_turn),
        "max_turn": int(max_turn),
        "max_actions": int(max_actions),
        "name": f"{outcome}:{min_turn}-{max_turn}",
    }


def _bucket_name(row: Any, buckets: list[dict[str, Any]]) -> str | None:
    outcome = _outcome(row)
    turn = _turn(row)
    if outcome is None or turn is None:
        return None
    for bucket in buckets:
        if outcome != bucket["outcome"]:
            continue
        if turn < bucket["min_turn"] or turn > bucket["max_turn"]:
            continue
        return str(bucket["name"])
    return None


def _pick_bucket(rows: list[Any], max_actions: int, rng: random.Random) -> tuple[list[Any], int]:
    rng.shuffle(rows)
    picked: list[Any] = []
    actions = 0
    for row in rows:
        action_count = _row_action_count(row)
        if action_count <= 0:
            continue
        if max_actions > 0 and actions + action_count > max_actions and picked:
            continue
        picked.append(row)
        actions += action_count
        if max_actions > 0 and actions >= max_actions:
            break
    return picked, actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", action="append", required=True, help="path|name")
    parser.add_argument("--bucket", action="append", required=True, help="outcome|min_turn|max_turn|max_actions")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260527)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bucket_specs = [_parse_bucket(raw) for raw in args.bucket]
    bucket_caps = {str(spec["name"]): int(spec["max_actions"]) for spec in bucket_specs}
    candidates: dict[str, list[Any]] = defaultdict(list)
    metrics: dict[str, Any] = {
        "sources": {},
        "buckets": {},
        "samples": 0.0,
        "labelled_actions": 0.0,
        "weighted_labelled_actions": 0.0,
        "weight": float(args.weight),
    }

    for raw_source in args.source:
        path, source_name = _parse_source(raw_source)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        source_metrics = {
            "path": str(path),
            "rows": 0.0,
            "labelled_actions": 0.0,
            "candidate_rows": 0.0,
            "candidate_actions": 0.0,
        }
        for row in payload["rows"]:
            action_count = _row_action_count(row)
            source_metrics["rows"] += 1.0
            source_metrics["labelled_actions"] += float(action_count)
            name = _bucket_name(row, bucket_specs)
            if name is None or action_count <= 0:
                continue
            row.dagger_source_name = source_name  # type: ignore[attr-defined]
            row.dagger_bucket_name = name  # type: ignore[attr-defined]
            row.dagger_loss_weight_override = float(args.weight)  # type: ignore[attr-defined]
            candidates[name].append(row)
            source_metrics["candidate_rows"] += 1.0
            source_metrics["candidate_actions"] += float(action_count)
        metrics["sources"][source_name] = source_metrics

    all_rows: list[Any] = []
    for spec in bucket_specs:
        name = str(spec["name"])
        picked, actions = _pick_bucket(candidates.get(name, []), bucket_caps[name], rng)
        all_rows.extend(picked)
        metrics["buckets"][name] = {
            "candidate_rows": float(len(candidates.get(name, []))),
            "picked_rows": float(len(picked)),
            "picked_actions": float(actions),
            "max_actions": float(bucket_caps[name]),
        }
        metrics["labelled_actions"] += float(actions)
    rng.shuffle(all_rows)
    metrics["samples"] = float(len(all_rows))
    metrics["weighted_labelled_actions"] = float(metrics["labelled_actions"]) * float(args.weight)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"rows": all_rows, "metrics": metrics, "args": vars(args)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    print(json.dumps({"event": "dagger_bucket_mix_cache_saved", "path": str(out), "metrics": metrics}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
