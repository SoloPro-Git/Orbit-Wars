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

    def __init__(
        self,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 1,
        ship_buckets: int = 0,
        action_slots: int = 3,
        source_target_summary: bool = False,
        target_pair_head: bool = False,
        target_pair_adapter: bool = False,
        target_pair_owner_head: bool = False,
    ):
        super().__init__()
        self.ship_buckets = ship_buckets
        self.action_slots = action_slots
        self.source_target_summary = source_target_summary
        self.use_target_pair_head = bool(target_pair_head)
        self.use_target_pair_adapter = bool(target_pair_adapter)
        self.use_target_pair_owner_head = bool(target_pair_owner_head)
        self.context_layers = max(0, int(layers) - 1)
        self.planet_in = nn.Sequential(
            nn.Linear(PLANET_FEAT_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        if self.context_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=max(1, int(heads)),
                dim_feedforward=hidden * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.planet_context = nn.TransformerEncoder(encoder_layer, num_layers=self.context_layers)
        else:
            self.planet_context = None
        self.global_in = nn.Sequential(nn.Linear(GLOBAL_FEAT_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.source_context = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        source_head_in = hidden * 2 if source_target_summary else hidden
        self.source_head = nn.Linear(source_head_in, 2)
        self.slot_embed = nn.Parameter(torch.zeros(action_slots, hidden))
        self.edge = nn.Sequential(
            nn.Linear(hidden * 3 + PAIR_FEAT_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        if self.use_target_pair_head and self.use_target_pair_adapter:
            self.target_pair_edge = nn.Sequential(
                nn.Linear(hidden * 3 + PAIR_FEAT_DIM, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
        else:
            self.target_pair_edge = None
        self.target_head = nn.Linear(hidden * 2, 1)
        self.target_pair_head = nn.Linear(hidden, 1) if self.use_target_pair_head else None
        self.target_pair_owner_head = nn.Linear(hidden, 3) if self.use_target_pair_head and self.use_target_pair_owner_head else None
        if self.target_pair_owner_head is not None:
            nn.init.zeros_(self.target_pair_owner_head.weight)
            nn.init.zeros_(self.target_pair_owner_head.bias)
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
        if self.planet_context is not None:
            x = self.planet_context(x, src_key_padding_mask=~planet_mask)
            x = x.masked_fill(~planet_mask[:, :, None], 0.0)
        g = self.global_in(global_features)
        source_context = self.source_context(torch.cat([x, g[:, None, :].expand(-1, MAX_PLANETS, -1)], dim=-1))
        src = source_context[:, :, None, :].expand(-1, -1, MAX_PLANETS, -1)
        tgt = x[:, None, :, :].expand(-1, MAX_PLANETS, -1, -1)
        glob = g[:, None, None, :].expand(-1, MAX_PLANETS, MAX_PLANETS, -1)
        edge_input = torch.cat([src, tgt, glob, pair_features], dim=-1)
        edge_hidden = self.edge(edge_input)
        target_summary = None
        if self.source_target_summary:
            target_valid = planet_mask[:, None, :].expand(-1, MAX_PLANETS, -1)
            eye_sources = torch.eye(MAX_PLANETS, dtype=torch.bool, device=planets.device)[None, :, :]
            target_valid = target_valid & ~eye_sources
            masked_edges = edge_hidden.masked_fill(~target_valid[..., None], -1e9)
            target_summary = masked_edges.max(dim=2).values
            target_summary = torch.where(torch.isfinite(target_summary), target_summary, torch.zeros_like(target_summary))
        slot_context = source_context[:, :, None, :] + self.slot_embed[None, None, :, :]
        if self.source_target_summary:
            target_slot_context = target_summary[:, :, None, :].expand(-1, -1, self.action_slots, -1)
            source_slot_context = torch.cat([slot_context, target_slot_context], dim=-1)
        else:
            source_slot_context = slot_context
        source_logits = self.source_head(source_slot_context)

        edge_by_slot = edge_hidden[:, :, None, :, :].expand(-1, -1, self.action_slots, -1, -1)
        slot_by_target = slot_context[:, :, :, None, :].expand(-1, -1, -1, MAX_PLANETS, -1)
        slot_edge = torch.cat([edge_by_slot, slot_by_target], dim=-1)
        target_logits = self.target_head(slot_edge).squeeze(-1)
        if self.target_pair_head is not None:
            pair_hidden = self.target_pair_edge(edge_input) if self.target_pair_edge is not None else edge_hidden
            target_pair_logits = self.target_pair_head(pair_hidden).squeeze(-1)
            if self.target_pair_owner_head is not None:
                target_owner = planets[..., :3].argmax(dim=-1)
                source_owner_logits = self.target_pair_owner_head(source_context)
                owner_index = target_owner[:, None, :, None].expand(-1, MAX_PLANETS, -1, 1)
                owner_bias = torch.gather(
                    source_owner_logits[:, :, None, :].expand(-1, -1, MAX_PLANETS, -1),
                    dim=-1,
                    index=owner_index,
                ).squeeze(-1)
                target_pair_logits = target_pair_logits + owner_bias
        else:
            target_pair_logits = None
        ship_raw = self.ship_head(slot_edge)

        target_logits = target_logits.masked_fill(~planet_mask[:, None, None, :], -1e9)
        if target_pair_logits is not None:
            target_pair_logits = target_pair_logits.masked_fill(~planet_mask[:, None, :], -1e9)
        eye = torch.eye(MAX_PLANETS, dtype=torch.bool, device=planets.device)[None, :, None, :]
        target_logits = target_logits.masked_fill(eye, -1e9)
        if target_pair_logits is not None:
            pair_eye = torch.eye(MAX_PLANETS, dtype=torch.bool, device=planets.device)[None, :, :]
            target_pair_logits = target_pair_logits.masked_fill(pair_eye, -1e9)

        source_logits = source_logits.masked_fill(~own_mask[:, :, None, None], -1e9)
        target_logits = target_logits.masked_fill(~own_mask[:, :, None, None], -1e9)
        if target_pair_logits is not None:
            target_pair_logits = target_pair_logits.masked_fill(~own_mask[:, :, None], -1e9)
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
        if target_pair_logits is not None:
            out["target_pair_logits"] = target_pair_logits
        return out
