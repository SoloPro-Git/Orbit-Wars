"""Candidate-ranking policy/value network."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from training2.features import ACTION_DIM, PLANET_DIM


class CandidatePolicyValueNet(nn.Module):
    def __init__(
        self,
        d_model: int = 192,
        nhead: int = 6,
        layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.planet_proj = nn.Linear(PLANET_DIM, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers, norm=nn.LayerNorm(d_model))
        self.global_proj = nn.Sequential(nn.Linear(8, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.action_proj = nn.Sequential(nn.Linear(ACTION_DIM, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.policy = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        planets: torch.Tensor,
        global_features: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        planet_mask = planets[..., -1] <= 0.0
        planet_emb = self.encoder(self.planet_proj(planets), src_key_padding_mask=planet_mask)
        valid = (~planet_mask).float().unsqueeze(-1)
        pooled = (planet_emb * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        glob = self.global_proj(global_features)
        action_emb = self.action_proj(candidates)
        ctx = torch.cat([pooled, glob], dim=-1)
        ctx_expanded = ctx.unsqueeze(1).expand(-1, candidates.size(1), -1)
        logits = self.policy(torch.cat([ctx_expanded, action_emb], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(candidate_mask <= 0.0, -1e9)
        value = self.value(ctx).squeeze(-1)
        return logits, value


def masked_policy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)

