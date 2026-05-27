from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from tinyPPO.bridge_agents import drop_gate_logits
from tinyPPO.model import TinyPolicyValueNet


@dataclass
class GateExample:
    planets: np.ndarray
    pair_features: np.ndarray
    global_features: np.ndarray
    planet_mask: np.ndarray
    own_mask: np.ndarray
    source_indices: np.ndarray
    choice: int


def _load_rows(path: Path) -> list[Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        rows = payload.get("rows")
    elif isinstance(payload, tuple):
        rows = payload[0]
    else:
        rows = payload
    if not isinstance(rows, list):
        raise TypeError(f"{path} does not contain a row list")
    return rows


def _source_indices_from_row(row: Any) -> np.ndarray:
    active = np.argwhere(np.asarray(row.launch_mask, dtype=np.bool_))
    if active.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return active[:, 0].astype(np.int64, copy=False)


def _batch_from_rows(rows: list[Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "planets": torch.tensor(np.stack([row.planets for row in rows]), dtype=torch.float32, device=device),
        "pair_features": torch.tensor(np.stack([row.pair_features for row in rows]), dtype=torch.float32, device=device),
        "global_features": torch.tensor(np.stack([row.global_features for row in rows]), dtype=torch.float32, device=device),
        "planet_mask": torch.tensor(np.stack([row.planet_mask for row in rows]), dtype=torch.bool, device=device),
        "own_mask": torch.tensor(np.stack([row.own_mask for row in rows]), dtype=torch.bool, device=device),
    }


@torch.no_grad()
def build_examples(
    rows: list[Any],
    teacher: TinyPolicyValueNet,
    device: torch.device,
    *,
    threshold: float,
    min_anchor_actions_to_filter: int,
    min_keep_actions: int,
    max_rows: int,
    seed: int,
) -> tuple[list[GateExample], dict[str, float]]:
    indexed = list(range(len(rows)))
    random.Random(seed).shuffle(indexed)
    if max_rows > 0:
        indexed = indexed[:max_rows]
    examples: list[GateExample] = []
    skipped_no_action = skipped_too_few = skipped_bad = 0
    keep = drop = 0
    teacher.eval()
    for raw_idx in indexed:
        row = rows[raw_idx]
        source_indices = _source_indices_from_row(row)
        if len(source_indices) == 0:
            skipped_no_action += 1
            continue
        if len(source_indices) < min_anchor_actions_to_filter:
            skipped_too_few += 1
            continue
        if len(source_indices) <= min_keep_actions:
            skipped_too_few += 1
            continue
        batch = _batch_from_rows([row], device)
        out = teacher(**batch)
        probs = torch.softmax(out["source_logits"][0], dim=-1)[..., 1]
        source_probs = probs.max(dim=-1).values.detach().cpu().numpy()
        action_scores = [(idx, float(source_probs[int(src)])) for idx, src in enumerate(source_indices)]
        candidates = [(score, idx) for idx, score in action_scores if score < threshold]
        if candidates:
            choice = int(min(candidates)[1] + 1)
            drop += 1
        else:
            choice = 0
            keep += 1
        try:
            examples.append(
                GateExample(
                    planets=np.asarray(row.planets, dtype=np.float32),
                    pair_features=np.asarray(row.pair_features, dtype=np.float32),
                    global_features=np.asarray(row.global_features, dtype=np.float32),
                    planet_mask=np.asarray(row.planet_mask, dtype=np.bool_),
                    own_mask=np.asarray(row.own_mask, dtype=np.bool_),
                    source_indices=source_indices,
                    choice=choice,
                )
            )
        except Exception:
            skipped_bad += 1
    metrics = {
        "rows_seen": float(len(indexed)),
        "examples": float(len(examples)),
        "teacher_keep": float(keep),
        "teacher_drop": float(drop),
        "teacher_drop_rate": drop / max(1, keep + drop),
        "skipped_no_action": float(skipped_no_action),
        "skipped_too_few": float(skipped_too_few),
        "skipped_bad": float(skipped_bad),
    }
    return examples, metrics


def _tensor_dataset(examples: list[GateExample]) -> TensorDataset:
    max_actions = max(len(ex.source_indices) for ex in examples)
    source_indices = np.full((len(examples), max_actions), -1, dtype=np.int64)
    action_mask = np.zeros((len(examples), max_actions + 1), dtype=np.bool_)
    choices = np.zeros((len(examples),), dtype=np.int64)
    for i, ex in enumerate(examples):
        n = len(ex.source_indices)
        source_indices[i, :n] = ex.source_indices
        action_mask[i, : n + 1] = True
        choices[i] = ex.choice
    return TensorDataset(
        torch.tensor(np.stack([ex.planets for ex in examples]), dtype=torch.float32),
        torch.tensor(np.stack([ex.pair_features for ex in examples]), dtype=torch.float32),
        torch.tensor(np.stack([ex.global_features for ex in examples]), dtype=torch.float32),
        torch.tensor(np.stack([ex.planet_mask for ex in examples]), dtype=torch.bool),
        torch.tensor(np.stack([ex.own_mask for ex in examples]), dtype=torch.bool),
        torch.tensor(source_indices, dtype=torch.long),
        torch.tensor(action_mask, dtype=torch.bool),
        torch.tensor(choices, dtype=torch.long),
    )


def _forward_loss(model: TinyPolicyValueNet, batch: tuple[torch.Tensor, ...], device: torch.device, no_drop_bias: float) -> tuple[torch.Tensor, dict[str, float]]:
    planets, pair_features, global_features, planet_mask, own_mask, source_indices, action_mask, choices = [x.to(device) for x in batch]
    out = model(planets, pair_features, global_features, planet_mask, own_mask)
    losses = []
    correct = 0
    drop_pred = 0
    drop_true = 0
    for i in range(planets.shape[0]):
        valid_sources = source_indices[i][source_indices[i] >= 0].detach().cpu().numpy()
        logits = drop_gate_logits(out["source_logits"][i], valid_sources, no_drop_bias)
        logits = logits.masked_fill(~action_mask[i, : logits.shape[0]], -1e9)
        choice = choices[i]
        losses.append(F.cross_entropy(logits[None], choice[None]))
        pred = int(torch.argmax(logits).item())
        true = int(choice.item())
        correct += int(pred == true)
        drop_pred += int(pred > 0)
        drop_true += int(true > 0)
    loss = torch.stack(losses).mean()
    n = max(1, int(planets.shape[0]))
    return loss, {
        "acc": correct / n,
        "pred_drop_rate": drop_pred / n,
        "true_drop_rate": drop_true / n,
    }


@torch.no_grad()
def evaluate(model: TinyPolicyValueNet, loader: DataLoader, device: torch.device, no_drop_bias: float) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        loss, metrics = _forward_loss(model, batch, device, no_drop_bias)
        n = int(batch[0].shape[0])
        sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu()) * n
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value * n
        count += n
    return {key: value / max(1, count) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dagger-cache", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-threshold", type=float, default=0.25)
    parser.add_argument("--no-drop-bias", type=float, default=2.0)
    parser.add_argument("--min-anchor-actions-to-filter", type=int, default=4)
    parser.add_argument("--min-keep-actions", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--trainable-modules", choices=["source_head", "all"], default="source_head")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
    model_cfg = {key: int(value) if isinstance(value, (int, float, bool)) else value for key, value in payload.get("model", {}).items()}
    model = TinyPolicyValueNet(**model_cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    teacher = TinyPolicyValueNet(**model_cfg).to(device)
    teacher.load_state_dict(payload["state_dict"])

    rows = _load_rows(Path(args.dagger_cache))
    examples, build_metrics = build_examples(
        rows,
        teacher,
        device,
        threshold=args.teacher_threshold,
        min_anchor_actions_to_filter=args.min_anchor_actions_to_filter,
        min_keep_actions=args.min_keep_actions,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    if not examples:
        raise RuntimeError("no gate examples built")
    dataset = _tensor_dataset(examples)
    generator = torch.Generator().manual_seed(args.seed)
    val_size = max(1, int(len(dataset) * args.val_frac))
    train_size = max(1, len(dataset) - val_size)
    train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    trainable_params = list(model.parameters())
    if args.trainable_modules == "source_head":
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.source_head.parameters():
            param.requires_grad_(True)
        model.slot_embed.requires_grad_(True)
        trainable_params = list(model.source_head.parameters()) + [model.slot_embed]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    log_rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sums: dict[str, float] = {}
        train_count = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, metrics = _forward_loss(model, batch, device, args.no_drop_bias)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            opt.step()
            n = int(batch[0].shape[0])
            train_sums["loss"] = train_sums.get("loss", 0.0) + float(loss.detach().cpu()) * n
            for key, value in metrics.items():
                train_sums[key] = train_sums.get(key, 0.0) + value * n
            train_count += n
        train_metrics = {key: value / max(1, train_count) for key, value in train_sums.items()}
        val_metrics = evaluate(model, val_loader, device, args.no_drop_bias)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        log_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = {
        "source": args.init_checkpoint,
        "dagger_cache": args.dagger_cache,
        "teacher_threshold": args.teacher_threshold,
        "no_drop_bias": args.no_drop_bias,
        "min_anchor_actions_to_filter": args.min_anchor_actions_to_filter,
        "build": build_metrics,
        "best_val_loss": best_val,
        "final_val": evaluate(model, val_loader, device, args.no_drop_bias),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model": model_cfg, "metrics": final_metrics}, out)
    with out.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": final_metrics, "log": log_rows}, f, ensure_ascii=True, indent=2)
    print(json.dumps({"event": "saved", "out": str(out), "metrics": final_metrics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
