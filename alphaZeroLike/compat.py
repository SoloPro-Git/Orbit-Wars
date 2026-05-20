"""Compatibility helpers for bootstrapping from ``training2`` checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def load_training2_stage1_backbone(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, list[str]]:
    """Load compatible backbone/value tensors from a training2 checkpoint.

    ``training2`` and ``alphaZeroLike`` intentionally have different action
    encoders and policy heads.  The transferable pieces are the planet
    projection, transformer encoder, global projection, and value head when
    their tensor shapes match.
    """

    ckpt = torch.load(Path(checkpoint), map_location=map_location, weights_only=False)
    source = ckpt.get("model_state_dict", ckpt)
    current = model.state_dict()
    allowed_prefixes = ("planet_proj.", "encoder.", "global_proj.", "value.")
    loaded = {}
    skipped_shape = []
    skipped_prefix = []
    missing_source = []

    for key, value in current.items():
        if not key.startswith(allowed_prefixes):
            skipped_prefix.append(key)
            continue
        source_value = source.get(key)
        if source_value is None:
            missing_source.append(key)
            continue
        if tuple(source_value.shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        loaded[key] = source_value

    merged = dict(current)
    merged.update(loaded)
    model.load_state_dict(merged, strict=True)
    return {
        "loaded": sorted(loaded),
        "skipped_shape_mismatch": sorted(skipped_shape),
        "missing_source": sorted(missing_source),
        "skipped_new_head": sorted(skipped_prefix),
    }
