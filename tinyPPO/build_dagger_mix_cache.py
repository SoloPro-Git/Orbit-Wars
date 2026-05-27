from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np


def _row_action_count(row: Any) -> int:
    return int(np.asarray(row.launch_mask).sum())


def _parse_source(raw: str) -> dict[str, Any]:
    parts = raw.split("|")
    if len(parts) != 7:
        raise ValueError(
            "--source must be path|name|weight|max_actions|min_turn|max_turn|outcomes; "
            f"got {raw!r}"
        )
    path, name, weight, max_actions, min_turn, max_turn, outcomes = parts
    return {
        "path": Path(path),
        "name": name,
        "weight": float(weight),
        "max_actions": int(max_actions),
        "min_turn": int(min_turn),
        "max_turn": int(max_turn),
        "outcomes": {item for item in outcomes.split(",") if item},
    }


def _keep_row(row: Any, spec: dict[str, Any]) -> bool:
    min_turn = int(spec["min_turn"])
    max_turn = int(spec["max_turn"])
    if min_turn >= 0 or max_turn >= 0:
        turn = getattr(row, "dagger_turn_index", getattr(row, "dagger_obs_step", None))
        if turn is None:
            return False
        turn = int(turn)
        if min_turn >= 0 and turn < min_turn:
            return False
        if max_turn >= 0 and turn > max_turn:
            return False
    outcomes = spec["outcomes"]
    if outcomes:
        outcome = getattr(row, "dagger_model_outcome", None)
        if outcome is None or str(outcome) not in outcomes:
            return False
    return _row_action_count(row) > 0


def _sample_rows(rows: list[Any], spec: dict[str, Any], rng: random.Random) -> tuple[list[Any], dict[str, float]]:
    candidates = [row for row in rows if _keep_row(row, spec)]
    rng.shuffle(candidates)
    max_actions = int(spec["max_actions"])
    picked: list[Any] = []
    actions = 0
    for row in candidates:
        action_count = _row_action_count(row)
        if max_actions > 0 and actions + action_count > max_actions and picked:
            continue
        row.dagger_source_name = str(spec["name"])  # type: ignore[attr-defined]
        row.dagger_loss_weight_override = float(spec["weight"])  # type: ignore[attr-defined]
        picked.append(row)
        actions += action_count
        if max_actions > 0 and actions >= max_actions:
            break
    return picked, {
        "candidate_rows": float(len(candidates)),
        "picked_rows": float(len(picked)),
        "picked_actions": float(actions),
        "weight": float(spec["weight"]),
        "max_actions": float(max_actions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--seed", type=int, default=260527)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_rows: list[Any] = []
    metrics: dict[str, Any] = {
        "sources": {},
        "samples": 0.0,
        "labelled_actions": 0.0,
        "weighted_labelled_actions": 0.0,
    }
    for raw_source in args.source:
        spec = _parse_source(raw_source)
        if not spec["path"].exists():
            raise FileNotFoundError(spec["path"])
        with spec["path"].open("rb") as fh:
            payload = pickle.load(fh)
        picked, source_metrics = _sample_rows(list(payload["rows"]), spec, rng)
        all_rows.extend(picked)
        metrics["sources"][spec["name"]] = {
            **source_metrics,
            "path": str(spec["path"]),
        }
        metrics["labelled_actions"] += source_metrics["picked_actions"]
        metrics["weighted_labelled_actions"] += source_metrics["picked_actions"] * float(spec["weight"])
    metrics["samples"] = float(len(all_rows))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"rows": all_rows, "metrics": metrics, "args": vars(args)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(out)
    print(json.dumps({"event": "dagger_mix_cache_saved", "path": str(out), "metrics": metrics}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
