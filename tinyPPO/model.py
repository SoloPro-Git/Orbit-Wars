from __future__ import annotations

import torch
from torch import nn

from tinyPPO.features import GLOBAL_FEAT_DIM, MAX_PLANETS, PAIR_FEAT_DIM, PLANET_FEAT_DIM


class TinyPolicyValueNet(nn.Module):
    """Small geometry-first actor critic.

    The policy scores every source-target edge. It outputs launch/no-launch per
    owned source and target logits per source. Ship amount can be either the
    legacy continuous fraction Beta distribution or categorical buckets around
    a rule-computed required ship count. No angle head exists; callers compute
    angle from map coordinates.
    """

    def __init__(self, hidden: int = 64, heads: int = 4, layers: int = 1, ship_buckets: int = 0, action_slots: int = 3):
        super().__init__()
        self.ship_buckets = ship_buckets
        self.action_slots = action_slots
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
        self.slot_embed = nn.Parameter(torch.zeros(action_slots, hidden))
        self.edge = nn.Sequential(
            nn.Linear(hidden * 3 + PAIR_FEAT_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.target_head = nn.Linear(hidden * 2, 1)
        ship_out = ship_buckets if ship_buckets > 0 else 2
        self.ship_head = nn.Linear(hidden * 2, ship_out)
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
        slot_context = source_context[:, :, None, :] + self.slot_embed[None, None, :, :]
        source_logits = self.source_head(slot_context)

        src = source_context[:, :, None, :].expand(-1, -1, MAX_PLANETS, -1)
        tgt = x[:, None, :, :].expand(-1, MAX_PLANETS, -1, -1)
        glob = g[:, None, None, :].expand(-1, MAX_PLANETS, MAX_PLANETS, -1)
        edge_hidden = self.edge(torch.cat([src, tgt, glob, pair_features], dim=-1))
        edge_by_slot = edge_hidden[:, :, None, :, :].expand(-1, -1, self.action_slots, -1, -1)
        slot_by_target = slot_context[:, :, :, None, :].expand(-1, -1, -1, MAX_PLANETS, -1)
        slot_edge = torch.cat([edge_by_slot, slot_by_target], dim=-1)
        target_logits = self.target_head(slot_edge).squeeze(-1)
        ship_raw = self.ship_head(slot_edge)

        target_logits = target_logits.masked_fill(~planet_mask[:, None, None, :], -1e9)
        eye = torch.eye(MAX_PLANETS, dtype=torch.bool, device=planets.device)[None, :, None, :]
        target_logits = target_logits.masked_fill(eye, -1e9)

        source_logits = source_logits.masked_fill(~own_mask[:, :, None, None], -1e9)
        target_logits = target_logits.masked_fill(~own_mask[:, :, None, None], -1e9)
        if self.ship_buckets > 0:
            ship_logits = ship_raw.masked_fill(~own_mask[:, :, None, None, None], -1e9)
            ship_logits = ship_logits.masked_fill(~planet_mask[:, None, None, :, None], -1e9)
            ship_params = None
        else:
            ship_params = torch.nn.functional.softplus(ship_raw) + 1.0
            ship_params = ship_params.masked_fill(~own_mask[:, :, None, None, None], 1.0)
            ship_params = ship_params.masked_fill(~planet_mask[:, None, None, :, None], 1.0)
            ship_logits = None

        valid = planet_mask.float().unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        value = self.value(torch.cat([pooled, g], dim=-1)).squeeze(-1)
        out = {
            "source_logits": source_logits,
            "target_logits": target_logits,
            "value": value,
        }
        if ship_logits is not None:
            out["ship_logits"] = ship_logits
        if ship_params is not None:
            out["ship_params"] = ship_params
        return out
