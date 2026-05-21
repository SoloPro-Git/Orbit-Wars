"""Policy/value network with move-set candidate embeddings."""

from __future__ import annotations

import torch
from torch import nn

from alphaZeroLike.features import GLOBAL_DIM, MOVE_DIM, PLANET_DIM


class AlphaZeroLikeNet(nn.Module):
    def __init__(
        self,
        d_model: int = 192,
        nhead: int = 6,
        layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
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
        self.global_proj = nn.Sequential(nn.Linear(GLOBAL_DIM, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.move_proj = nn.Sequential(
            nn.Linear(MOVE_DIM + d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.empty_action = nn.Parameter(torch.zeros(d_model))
        self.action_norm = nn.LayerNorm(d_model * 2)
        self.policy = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.proposal_send = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.proposal_target = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 64))
        self.proposal_ship = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.value = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

    def encode_context(
        self,
        planets: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        planet_mask = planets.abs().sum(dim=-1) <= 0.0
        planet_emb = self.encoder(self.planet_proj(planets), src_key_padding_mask=planet_mask)
        valid = (~planet_mask).float().unsqueeze(-1)
        pooled = (planet_emb * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        glob = self.global_proj(global_features)
        return planet_emb, glob, torch.cat([pooled, glob], dim=-1)

    def proposal(
        self,
        planets: torch.Tensor,
        global_features: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        planet_mask = planets.abs().sum(dim=-1) <= 0.0
        planet_emb, glob, _ = self.encode_context(planets, global_features)
        glob_expanded = glob.unsqueeze(1).expand(-1, planet_emb.size(1), -1)
        per_entity = torch.cat([planet_emb, glob_expanded], dim=-1)
        valid_sources = ~planet_mask if source_mask is None else source_mask.bool() & ~planet_mask
        valid_targets = ~planet_mask if target_mask is None else target_mask.bool() & ~planet_mask
        target_logits = self.proposal_target(per_entity)[..., : planet_emb.size(1)]
        target_logits = target_logits.masked_fill(~valid_targets.unsqueeze(1), -1e9)
        eye = torch.eye(planet_emb.size(1), dtype=torch.bool, device=planet_emb.device).unsqueeze(0)
        target_logits = target_logits.masked_fill(eye, -1e9)
        return {
            "send_logits": self.proposal_send(per_entity).squeeze(-1).masked_fill(~valid_sources, -1e9),
            "target_logits": target_logits,
            "ship_logits": self.proposal_ship(per_entity).squeeze(-1),
            "entity_mask": planet_mask,
            "source_mask": valid_sources,
            "target_mask": valid_targets,
        }

    @staticmethod
    def _gather_planet(planet_emb: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, entities, dim = planet_emb.shape
        safe = indices.clamp(min=0, max=max(entities - 1, 0))
        expanded = safe.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, dim)
        gathered = torch.gather(
            planet_emb.unsqueeze(1).expand(-1, indices.size(1), -1, -1),
            2,
            expanded,
        ).squeeze(2)
        return gathered * (indices >= 0).unsqueeze(-1).float()

    def encode_actions(
        self,
        planet_emb: torch.Tensor,
        move_features: torch.Tensor,
        move_source_indices: torch.Tensor,
        move_target_indices: torch.Tensor,
        move_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidates, moves, _ = move_features.shape
        flat_sources = move_source_indices.reshape(batch, candidates * moves)
        flat_targets = move_target_indices.reshape(batch, candidates * moves)
        src = self._gather_planet(planet_emb, flat_sources).reshape(batch, candidates, moves, -1)
        tgt = self._gather_planet(planet_emb, flat_targets).reshape(batch, candidates, moves, -1)
        move_in = torch.cat([move_features, src, tgt], dim=-1)
        move_emb = self.move_proj(move_in)
        masked = move_emb * move_mask.unsqueeze(-1)
        count = move_mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        mean_emb = masked.sum(dim=2) / count
        max_emb = move_emb.masked_fill(move_mask.unsqueeze(-1) <= 0.0, -1e9).max(dim=2).values
        max_emb = torch.where(torch.isfinite(max_emb), max_emb, self.empty_action.view(1, 1, -1))
        return self.action_norm(torch.cat([mean_emb, max_emb], dim=-1))

    def forward(
        self,
        planets: torch.Tensor,
        global_features: torch.Tensor,
        move_features: torch.Tensor,
        move_source_indices: torch.Tensor,
        move_target_indices: torch.Tensor,
        move_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        planet_emb, _, ctx = self.encode_context(planets, global_features)
        action_emb = self.encode_actions(
            planet_emb,
            move_features,
            move_source_indices,
            move_target_indices,
            move_mask,
        )
        ctx_expanded = ctx.unsqueeze(1).expand(-1, action_emb.size(1), -1)
        logits = self.policy(torch.cat([ctx_expanded, action_emb], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(candidate_mask <= 0.0, -1e9)
        value = self.value(ctx).squeeze(-1)
        return logits, value
