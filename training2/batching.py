"""Batch padding helpers for variable-size Orbit Wars maps."""
from __future__ import annotations

import torch


def pad_planets(rows: list) -> torch.Tensor:
    tensors = [
        row if isinstance(row, torch.Tensor) else torch.tensor(row, dtype=torch.float32)
        for row in rows
    ]
    max_planets = max(t.size(0) for t in tensors)
    feat_dim = tensors[0].size(-1)
    out = torch.zeros(len(tensors), max_planets, feat_dim, dtype=torch.float32)
    for i, t in enumerate(tensors):
        out[i, : t.size(0)] = t
    return out


def collate_rows(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "planets": pad_planets([row["planets"] for row in batch]),
        "global": torch.stack([row["global"] for row in batch]),
        "candidates": torch.stack([row["candidates"] for row in batch]),
        "mask": torch.stack([row["mask"] for row in batch]),
        "target": torch.stack([row["target"] for row in batch]),
        "value": torch.stack([row["value"] for row in batch]),
    }

