from __future__ import annotations

import torch
from torch import nn

from tinyPPO.features import GLOBAL_FEAT_DIM, MAX_PLANETS, PAIR_FEAT_DIM, PLANET_FEAT_DIM


class TinyPolicyValueNet(nn.Module):
    """Small geometry-first actor critic.

    The policy scores every source-target edge. It outputs launch/no-launch per
    owned source, target logits per source, and ship bucket logits conditioned
    on the selected source-target pair. No angle head exists; callers compute
    angle from map coordinates.
    """

    def __init__(self, hidden: int = 64, heads: int = 4, layers: int = 1, ship_buckets: int = 4):
        super().__init__()
        self.ship_buckets = ship_buckets
        self.planet_in = nn.Sequential(
            nn.Linear(PLANET_FEAT_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        del heads, layers
        self.global_in = nn.Sequential(nn.Linear(GLOBAL_FEAT_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.source_context = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.source_head = nn.Linear(hidden, 2)
        self.edge = nn.Sequential(
            nn.Linear(hidden * 3 + PAIR_FEAT_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.target_head = nn.Linear(hidden, 1)
        self.ship_head = nn.Linear(hidden, ship_buckets)
        self.value = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(
        self,
        planets: torch.Tensor,
        pair_features: torch.Tensor,
        global_features: torch.Tensor,
        planet_mask: torch.Tensor,
        own_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = self.planet_in(planets)
        g = self.global_in(global_features)
        source_context = self.source_context(torch.cat([x, g[:, None, :].expand(-1, MAX_PLANETS, -1)], dim=-1))
        source_logits = self.source_head(source_context)

        src = source_context[:, :, None, :].expand(-1, -1, MAX_PLANETS, -1)
        tgt = x[:, None, :, :].expand(-1, MAX_PLANETS, -1, -1)
        glob = g[:, None, None, :].expand(-1, MAX_PLANETS, MAX_PLANETS, -1)
        edge_hidden = self.edge(torch.cat([src, tgt, glob, pair_features], dim=-1))
        target_logits = self.target_head(edge_hidden).squeeze(-1)
        ship_logits = self.ship_head(edge_hidden)

        target_logits = target_logits.masked_fill(~planet_mask[:, None, :], -1e9)
        eye = torch.eye(MAX_PLANETS, dtype=torch.bool, device=planets.device)[None, :, :]
        target_logits = target_logits.masked_fill(eye, -1e9)

        source_logits = source_logits.masked_fill(~own_mask[:, :, None], -1e9)
        target_logits = target_logits.masked_fill(~own_mask[:, :, None], -1e9)
        ship_logits = ship_logits.masked_fill(~own_mask[:, :, None, None], -1e9)
        ship_logits = ship_logits.masked_fill(~planet_mask[:, None, :, None], -1e9)

        valid = planet_mask.float().unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        value = self.value(torch.cat([pooled, g], dim=-1)).squeeze(-1)
        return {
            "source_logits": source_logits,
            "target_logits": target_logits,
            "ship_logits": ship_logits,
            "value": value,
        }
