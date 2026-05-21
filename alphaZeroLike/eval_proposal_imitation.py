"""Offline imitation eval for the AlphaZeroLike proposal head."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from alphaZeroLike.features import GLOBAL_DIM, PLANET_DIM
from alphaZeroLike.model import AlphaZeroLikeNet
from alphaZeroLike.train_proposal_stage1 import _has_active_launch, _iter_jsonl_paths


def _iter_rows_once(paths: list[Path], *, seed: int, max_rows_scan: int) -> Iterable[dict]:
    rng = random.Random(seed)
    order = list(paths)
    rng.shuffle(order)
    scanned = 0
    for path in order:
        with path.open() as f:
            for line in f:
                if max_rows_scan > 0 and scanned >= max_rows_scan:
                    return
                scanned += 1
                if not line.strip():
                    continue
                row = json.loads(line)
                if "proposal_valid" in row and "planets" in row and "global" in row:
                    yield row


def _sample_rows(paths: list[Path], *, seed: int, samples: int, active_frac: float, max_rows_scan: int) -> list[dict]:
    rng = random.Random(seed)
    active_target = int(round(samples * min(max(active_frac, 0.0), 1.0)))
    inactive_target = samples - active_target
    active: list[dict] = []
    inactive: list[dict] = []
    seen_active = 0
    seen_inactive = 0
    for row in _iter_rows_once(paths, seed=seed, max_rows_scan=max_rows_scan):
        if _has_active_launch(row):
            seen_active += 1
            if len(active) < active_target:
                active.append(row)
            else:
                j = rng.randrange(seen_active)
                if j < active_target:
                    active[j] = row
        else:
            seen_inactive += 1
            if len(inactive) < inactive_target:
                inactive.append(row)
            else:
                j = rng.randrange(seen_inactive)
                if j < inactive_target:
                    inactive[j] = row
    rows = active + inactive
    rng.shuffle(rows)
    return rows[:samples]


def _pad_planets(rows: list[dict], device: torch.device | str) -> torch.Tensor:
    max_entities = max(len(row["planets"]) for row in rows)
    out = np.zeros((len(rows), max_entities, PLANET_DIM), dtype=np.float32)
    for i, row in enumerate(rows):
        planets = np.asarray(row["planets"], dtype=np.float32)
        dims = min(planets.shape[1], PLANET_DIM)
        out[i, : planets.shape[0], :dims] = planets[:, :dims]
    return torch.tensor(out, dtype=torch.float32, device=device)


def _pad_global(rows: list[dict], device: torch.device | str) -> torch.Tensor:
    out = np.zeros((len(rows), GLOBAL_DIM), dtype=np.float32)
    for i, row in enumerate(rows):
        glob = np.asarray(row["global"], dtype=np.float32)
        dims = min(glob.shape[0], GLOBAL_DIM)
        out[i, :dims] = glob[:dims]
    return torch.tensor(out, dtype=torch.float32, device=device)


def _pad_float(rows: list[dict], key: str, n_entities: int, device: torch.device | str) -> torch.Tensor:
    out = torch.zeros((len(rows), n_entities), dtype=torch.float32, device=device)
    for i, row in enumerate(rows):
        vals = row.get(key, [])[:n_entities]
        if vals:
            out[i, : len(vals)] = torch.tensor(vals, dtype=torch.float32, device=device)
    return out


def _pad_long(rows: list[dict], key: str, n_entities: int, device: torch.device | str) -> torch.Tensor:
    out = torch.zeros((len(rows), n_entities), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        vals = row.get(key, [])[:n_entities]
        if vals:
            out[i, : len(vals)] = torch.tensor(vals, dtype=torch.long, device=device)
    return out


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def _load_model(checkpoint: Path, device: str) -> AlphaZeroLikeNet:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model = AlphaZeroLikeNet(
        d_model=int(ckpt_args.get("d_model", 192)),
        nhead=int(ckpt_args.get("nhead", 6)),
        layers=int(ckpt_args.get("layers", 4)),
        dropout=float(ckpt_args.get("dropout", 0.10)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def _batch_eval(model: AlphaZeroLikeNet, rows: list[dict], *, device: str, send_threshold: float) -> dict:
    planets = _pad_planets(rows, device)
    glob = _pad_global(rows, device)
    n_entities = planets.size(1)
    entity_valid = planets.abs().sum(dim=-1) > 0.0
    prop_valid = (_pad_float(rows, "proposal_valid", n_entities, device) > 0.5) & entity_valid
    source_valid = prop_valid & (planets[..., 6] > 0.0)
    target_valid = (planets[..., 17] > 0.5) & entity_valid
    prop_send = _pad_float(rows, "proposal_send", n_entities, device)
    prop_target = _pad_long(rows, "proposal_target", n_entities, device).clamp(max=max(n_entities - 1, 0))
    prop_ship = _pad_float(rows, "proposal_ship_ratio", n_entities, device)
    active = prop_valid & (prop_send > 0.5)

    with torch.no_grad():
        pred = model.proposal(planets, glob, source_mask=source_valid, target_mask=target_valid)
        send_prob = torch.sigmoid(pred["send_logits"])
        pred_send = (send_prob >= send_threshold) & prop_valid
        target_logits = pred["target_logits"]
        target_pred = target_logits.argmax(dim=-1)
        ship_pred = torch.sigmoid(pred["ship_logits"])
        topk = torch.topk(target_logits, k=min(5, n_entities), dim=-1).indices

    valid_count = int(prop_valid.sum().item())
    active_count = int(active.sum().item())
    pred_count = int(pred_send.sum().item())
    true_count = int(active.sum().item())
    true_positive_sources = int((pred_send & active).sum().item())
    send_correct = int(((pred_send == active) & prop_valid).sum().item())
    target_correct = int(((target_pred == prop_target) & active).sum().item())
    top3_correct = int(((topk[:, :, : min(3, topk.size(-1))] == prop_target.unsqueeze(-1)).any(dim=-1) & active).sum().item())
    top5_correct = int(((topk == prop_target.unsqueeze(-1)).any(dim=-1) & active).sum().item())
    ship_abs = ((ship_pred - prop_ship).abs() * active.float()).sum().item()
    ship_within_005 = int((((ship_pred - prop_ship).abs() <= 0.05) & active).sum().item())
    ship_within_010 = int((((ship_pred - prop_ship).abs() <= 0.10) & active).sum().item())
    invalid_source_predictions = int(((send_prob >= send_threshold) & ~source_valid).sum().item())
    invalid_target_predictions = int(((target_pred == torch.arange(n_entities, device=device).view(1, -1)) & pred_send).sum().item())

    pair_jaccards: list[float] = []
    source_jaccards: list[float] = []
    exact_pair_sets = 0
    exact_source_sets = 0
    pred_action_sizes: list[int] = []
    true_action_sizes: list[int] = []
    active_row_target_acc: list[float] = []
    examples: list[dict] = []
    target_pred_cpu = target_pred.cpu()
    pred_send_cpu = pred_send.cpu()
    active_cpu = active.cpu()
    prop_target_cpu = prop_target.cpu()
    prop_ship_cpu = prop_ship.cpu()
    ship_pred_cpu = ship_pred.cpu()
    send_prob_cpu = send_prob.cpu()

    for i, row in enumerate(rows):
        true_sources = {j for j in range(n_entities) if bool(active_cpu[i, j])}
        pred_sources = {j for j in range(n_entities) if bool(pred_send_cpu[i, j])}
        true_pairs = {(j, int(prop_target_cpu[i, j])) for j in true_sources}
        pred_pairs = {(j, int(target_pred_cpu[i, j])) for j in pred_sources}
        source_union = true_sources | pred_sources
        pair_union = true_pairs | pred_pairs
        source_j = _safe_div(len(true_sources & pred_sources), len(source_union))
        pair_j = _safe_div(len(true_pairs & pred_pairs), len(pair_union))
        source_jaccards.append(source_j)
        pair_jaccards.append(pair_j)
        exact_source_sets += int(true_sources == pred_sources)
        exact_pair_sets += int(true_pairs == pred_pairs)
        pred_action_sizes.append(len(pred_sources))
        true_action_sizes.append(len(true_sources))
        if true_sources:
            active_row_target_acc.append(_safe_div(len(true_pairs & pred_pairs), len(true_sources)))
        if len(examples) < 8 and true_sources:
            examples.append(
                {
                    "row": i,
                    "true": [
                        {
                            "src_idx": int(src),
                            "target_idx": int(prop_target_cpu[i, src]),
                            "ship_ratio": round(float(prop_ship_cpu[i, src]), 3),
                        }
                        for src in sorted(true_sources)
                    ],
                    "pred": [
                        {
                            "src_idx": int(src),
                            "target_idx": int(target_pred_cpu[i, src]),
                            "ship_ratio": round(float(ship_pred_cpu[i, src]), 3),
                            "send_prob": round(float(send_prob_cpu[i, src]), 3),
                        }
                        for src in sorted(pred_sources)
                    ],
                    "source_jaccard": round(source_j, 3),
                    "pair_jaccard": round(pair_j, 3),
                }
            )

    return {
        "rows": len(rows),
        "valid_sources": valid_count,
        "active_sources": active_count,
        "predicted_sources": pred_count,
        "send_acc": _safe_div(send_correct, valid_count),
        "source_precision": _safe_div(true_positive_sources, pred_count),
        "source_recall": _safe_div(true_positive_sources, true_count),
        "source_f1": _safe_div(2 * true_positive_sources, pred_count + true_count),
        "target_top1": _safe_div(target_correct, active_count),
        "target_top3": _safe_div(top3_correct, active_count),
        "target_top5": _safe_div(top5_correct, active_count),
        "ship_mae": _safe_div(ship_abs, active_count),
        "ship_within_005": _safe_div(ship_within_005, active_count),
        "ship_within_010": _safe_div(ship_within_010, active_count),
        "invalid_source_predictions": invalid_source_predictions,
        "invalid_target_predictions": invalid_target_predictions,
        "exact_source_set": _safe_div(exact_source_sets, len(rows)),
        "exact_pair_set": _safe_div(exact_pair_sets, len(rows)),
        "mean_source_jaccard": _mean(source_jaccards),
        "mean_pair_jaccard": _mean(pair_jaccards),
        "active_row_pair_recall": _mean(active_row_target_acc),
        "pred_action_size_mean": _mean(pred_action_sizes),
        "true_action_size_mean": _mean(true_action_sizes),
        "pred_action_size_p90": _percentile(pred_action_sizes, 90),
        "true_action_size_p90": _percentile(true_action_sizes, 90),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/alphaZeroLike/proposal_regular_20260521")
    parser.add_argument("--checkpoint", default="alphaZeroLike/checkpoints/latest.pt")
    parser.add_argument("--out", default="alphaZeroLike/logs/proposal_imitation_eval_latest.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--active-row-frac", type=float, default=0.75)
    parser.add_argument("--max-rows-scan", type=int, default=200000)
    parser.add_argument("--send-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    paths = _iter_jsonl_paths(args.data)
    if not paths:
        raise FileNotFoundError(args.data)
    rows = _sample_rows(
        paths,
        seed=args.seed,
        samples=args.samples,
        active_frac=args.active_row_frac,
        max_rows_scan=args.max_rows_scan,
    )
    if not rows:
        raise RuntimeError("no eval rows loaded")
    model = _load_model(Path(args.checkpoint), device)
    parts = []
    for start in range(0, len(rows), args.batch_size):
        parts.append(_batch_eval(model, rows[start : start + args.batch_size], device=device, send_threshold=args.send_threshold))

    total_rows = sum(p["rows"] for p in parts)
    totals = {
        "rows": total_rows,
        "valid_sources": sum(p["valid_sources"] for p in parts),
        "active_sources": sum(p["active_sources"] for p in parts),
        "predicted_sources": sum(p["predicted_sources"] for p in parts),
        "invalid_source_predictions": sum(p["invalid_source_predictions"] for p in parts),
        "invalid_target_predictions": sum(p["invalid_target_predictions"] for p in parts),
    }
    weighted_keys = [
        "send_acc",
        "source_precision",
        "source_recall",
        "source_f1",
        "target_top1",
        "target_top3",
        "target_top5",
        "ship_mae",
        "ship_within_005",
        "ship_within_010",
    ]
    row_weighted_keys = [
        "exact_source_set",
        "exact_pair_set",
        "mean_source_jaccard",
        "mean_pair_jaccard",
        "active_row_pair_recall",
        "pred_action_size_mean",
        "true_action_size_mean",
        "pred_action_size_p90",
        "true_action_size_p90",
    ]
    denom_by_key = {
        "send_acc": "valid_sources",
        "source_precision": "predicted_sources",
        "source_recall": "active_sources",
        "source_f1": None,
        "target_top1": "active_sources",
        "target_top3": "active_sources",
        "target_top5": "active_sources",
        "ship_mae": "active_sources",
        "ship_within_005": "active_sources",
        "ship_within_010": "active_sources",
    }
    metrics = dict(totals)
    for key in weighted_keys:
        denom_name = denom_by_key[key]
        if denom_name is None:
            metrics[key] = _safe_div(
                2 * metrics["source_precision"] * metrics["source_recall"],
                metrics["source_precision"] + metrics["source_recall"],
            )
            continue
        denom = sum(p[denom_name] for p in parts)
        metrics[key] = _safe_div(sum(p[key] * p[denom_name] for p in parts), denom)
    for key in row_weighted_keys:
        metrics[key] = _safe_div(sum(p[key] * p["rows"] for p in parts), total_rows)
    examples = []
    for part in parts:
        examples.extend(part["examples"])
        if len(examples) >= 8:
            break

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "device": device,
        "samples_requested": args.samples,
        "max_rows_scan": args.max_rows_scan,
        "active_row_frac": args.active_row_frac,
        "send_threshold": args.send_threshold,
        "metrics": metrics,
        "examples": examples[:8],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
