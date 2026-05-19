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
        self.proposal_send = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.proposal_target = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 64))
        self.proposal_ship = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.proposal_angle = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def encode_context(
        self,
        planets: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        planet_mask = planets.abs().sum(dim=-1) <= 0.0
        planet_emb = self.encoder(self.planet_proj(planets), src_key_padding_mask=planet_mask)
        valid = (~planet_mask).float().unsqueeze(-1)
        pooled = (planet_emb * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        glob = self.global_proj(global_features)
        ctx = torch.cat([pooled, glob], dim=-1)
        return planet_emb, glob, ctx, planet_mask

    def proposal(
        self,
        planets: torch.Tensor,
        global_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        planet_emb, glob, _, planet_mask = self.encode_context(planets, global_features)
        glob_expanded = glob.unsqueeze(1).expand(-1, planet_emb.size(1), -1)
        per_planet = torch.cat([planet_emb, glob_expanded], dim=-1)
        target_logits = self.proposal_target(per_planet)[..., : planet_emb.size(1)]
        target_logits = target_logits.masked_fill(planet_mask.unsqueeze(1), -1e9)
        return {
            "send_logits": self.proposal_send(per_planet).squeeze(-1).masked_fill(planet_mask, -1e9),
            "target_logits": target_logits,
            "ship_logits": self.proposal_ship(per_planet).squeeze(-1),
            "angle_offsets": torch.tanh(self.proposal_angle(per_planet).squeeze(-1)),
            "planet_mask": planet_mask,
        }

    def forward(
        self,
        planets: torch.Tensor,
        global_features: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        return_proposal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        planet_emb, glob, ctx, _ = self.encode_context(planets, global_features)
        action_emb = self.action_proj(candidates)
        ctx_expanded = ctx.unsqueeze(1).expand(-1, candidates.size(1), -1)
        logits = self.policy(torch.cat([ctx_expanded, action_emb], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(candidate_mask <= 0.0, -1e9)
        value = self.value(ctx).squeeze(-1)
        if return_proposal:
            return logits, value, self.proposal(planets, global_features)
        return logits, value


def masked_policy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def load_compatible_state_dict(model: nn.Module, state_dict: dict, *, strict: bool = False) -> dict[str, list[str]]:
    """Load checkpoint weights while skipping tensors whose shapes changed."""
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    skipped = [
        key
        for key, value in state_dict.items()
        if key in current and tuple(value.shape) != tuple(current[key].shape)
    ]
    missing, unexpected = model.load_state_dict(compatible, strict=strict)
    return {
        "skipped_shape_mismatch": skipped,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
