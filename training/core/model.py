"""Orbit Wars RL neural network model.

Transformer-based architecture that processes variable-length planet and fleet
inputs with cross-attention fusion, producing per-planet action distributions
and a global ranking value estimate.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    return {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
    }[name.lower()]


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float = 0.0,
    activation: nn.Module | None = None,
) -> nn.Sequential:
    """Two-layer MLP with LayerNorm and optional dropout."""
    act = activation or nn.GELU()
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        act,
        nn.LayerNorm(hidden_dim),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for variable-length sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [1, max_len, d_model] so it broadcasts over batch
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Input Projection
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    """Project raw observation features into the shared d_model space."""

    def __init__(self, d_model: int, d_planet: int = 25, d_fleet: int = 13, d_global: int = 8):
        super().__init__()
        self.planet_projection = nn.Linear(d_planet, d_model)
        self.fleet_projection = nn.Linear(d_fleet, d_model)
        self.global_projection = _build_mlp(d_global, d_model, d_model)

    def forward(
        self,
        planet_features: torch.Tensor,
        fleet_features: Optional[torch.Tensor],
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
            planet_emb: [batch, N_planets, d_model]
            fleet_emb:  [batch, N_fleets, d_model] or None
            global_emb: [batch, d_model]
        """
        planet_emb = self.planet_projection(planet_features)
        fleet_emb = self.fleet_projection(fleet_features) if fleet_features is not None else None
        global_emb = self.global_projection(global_features)
        return planet_emb, fleet_emb, global_emb


# ---------------------------------------------------------------------------
# Transformer Encoders
# ---------------------------------------------------------------------------

class PlanetEncoder(nn.Module):
    """Transformer encoder for planet set."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * config.mlp_ratio,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.planet_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x:             [batch, N_planets, d_model]
        padding_mask:  [batch, N_planets] — True for positions to *ignore*
        Returns:       [batch, N_planets, d_model]
        """
        return self.encoder(x, src_key_padding_mask=padding_mask)


class FleetEncoder(nn.Module):
    """Transformer encoder for fleet set. Handles the empty-fleet case."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * config.mlp_ratio,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.fleet_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x:             [batch, N_fleets, d_model]  (N_fleets may be 0)
        padding_mask:  [batch, N_fleets]
        Returns:       [batch, N_fleets, d_model]
        """
        if x.size(1) == 0:
            return x
        return self.encoder(x, src_key_padding_mask=padding_mask)


# ---------------------------------------------------------------------------
# Fusion Layer (Cross-Attention)
# ---------------------------------------------------------------------------

class FusionLayer(nn.Module):
    """Cross-attention fusion: queries attend over a context sequence."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(config.d_model)
        self.norm_kv = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.mlp_ratio),
            _get_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * config.mlp_ratio, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.norm_ffn = nn.LayerNorm(config.d_model)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        query_padding_mask: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        queries:            [batch, N_q, d_model]
        context:            [batch, N_c, d_model]
        query_padding_mask: [batch, N_q]   True = ignore
        context_padding_mask: [batch, N_c] True = ignore
        Returns:            [batch, N_q, d_model]
        """
        q = self.norm_q(queries)
        kv = self.norm_kv(context)

        attn_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=context_padding_mask,
        )
        queries = queries + attn_out
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


# ---------------------------------------------------------------------------
# Policy / Value / Opponent Heads
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """Per-planet action head: target logits + ship fraction."""

    def __init__(self, config: ModelConfig, n_planets: int = 32):
        super().__init__()
        d = config.d_model
        # Takes the concatenation of [owned_planet_emb, context_pool] per owned planet
        self.target_mlp = _build_mlp(
            d * 2, d, n_planets, dropout=config.dropout,
            activation=_get_activation(config.activation),
        )
        self.ships_mlp = _build_mlp(
            d * 2, d, 1, dropout=config.dropout,
            activation=_get_activation(config.activation),
        )

    def forward(
        self,
        owned_emb: torch.Tensor,
        global_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        owned_emb:       [batch, N_owned, d_model]
        global_context:  [batch, d_model]
        Returns:
            target_logits: [batch, N_owned, n_planets]
            num_ships:     [batch, N_owned, 1]  (sigmoid applied)
        """
        # Expand global to match owned: [batch, N_owned, d_model]
        ctx = global_context.unsqueeze(1).expand_as(owned_emb)
        combined = torch.cat([owned_emb, ctx], dim=-1)  # [batch, N_owned, d*2]

        target_logits = self.target_mlp(combined)
        num_ships = torch.sigmoid(self.ships_mlp(combined))
        return target_logits, num_ships


class ValueHead(nn.Module):
    """Global value estimate: ranking probability distribution."""

    def __init__(self, config: ModelConfig, max_players: int = 4):
        super().__init__()
        d = config.d_model
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            _get_activation(config.activation),
            nn.LayerNorm(d),
            nn.Dropout(config.dropout),
            nn.Linear(d, max_players),
        )
        self.max_players = max_players

    def forward(
        self,
        planet_emb: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        num_players: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        planet_emb:   [batch, N_planets, d_model]
        padding_mask: [batch, N_planets] True = pad
        num_players:  [batch]  (used to mask out invalid ranks)
        Returns:      [batch, max_players]  softmax ranking probabilities
        """
        # Mean-pool over valid (non-padded) planets
        if padding_mask is not None:
            mask_inv = (~padding_mask).float().unsqueeze(-1)  # [B, N, 1]
            pooled = (planet_emb * mask_inv).sum(dim=1) / mask_inv.sum(dim=1).clamp(min=1)
        else:
            pooled = planet_emb.mean(dim=1)

        logits = self.mlp(pooled)  # [batch, max_players]

        # Mask out ranks beyond num_players for each sample
        if num_players is not None:
            max_p = self.max_players
            player_mask = torch.arange(max_p, device=logits.device).unsqueeze(0)  # [1, max_p]
            player_mask = player_mask >= num_players.unsqueeze(1)  # [batch, max_p]
            logits = logits.masked_fill(player_mask, float("-inf"))

        return F.softmax(logits, dim=-1)


class OpponentHead(nn.Module):
    """Predict opponent actions from enemy planet embeddings."""

    def __init__(self, config: ModelConfig, n_planets: int = 32):
        super().__init__()
        d = config.d_model
        self.target_mlp = _build_mlp(
            d * 2, d, n_planets, dropout=config.dropout,
            activation=_get_activation(config.activation),
        )
        self.ships_mlp = _build_mlp(
            d * 2, d, 1, dropout=config.dropout,
            activation=_get_activation(config.activation),
        )

    def forward(
        self,
        enemy_emb: torch.Tensor,
        global_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        enemy_emb:       [batch, N_enemy, d_model]
        global_context:  [batch, d_model]
        Returns:
            opp_target_logits: [batch, N_enemy, n_planets]
            opp_num_ships:     [batch, N_enemy, 1]
        """
        ctx = global_context.unsqueeze(1).expand_as(enemy_emb)
        combined = torch.cat([enemy_emb, ctx], dim=-1)

        opp_target_logits = self.target_mlp(combined)
        opp_num_ships = torch.sigmoid(self.ships_mlp(combined))
        return opp_target_logits, opp_num_ships


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class OrbitWarsModel(nn.Module):
    """Full model: encode planets & fleets, fuse, then predict actions."""

    def __init__(self, config: ModelConfig, n_planets: int = 32, max_players: int = 4):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Input projection (match feature_engineering dimensions)
        from core.feature_engineering import D_PLANET, D_FLEET, D_GLOBAL
        self.input_proj = InputProjection(
            d_model=config.d_model, d_planet=D_PLANET, d_fleet=D_FLEET, d_global=D_GLOBAL
        )

        # Positional encoding (shared)
        self.pos_enc = PositionalEncoding(config.d_model, dropout=config.dropout)

        # Encoders
        self.planet_encoder = PlanetEncoder(config)
        self.fleet_encoder = FleetEncoder(config)

        # Global embedding: combine fleet pool + global features
        self.global_combine = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            _get_activation(config.activation),
            nn.LayerNorm(config.d_model),
        )

        # Fusion layers
        self.fusion_layers = nn.ModuleList(
            [FusionLayer(config) for _ in range(config.fusion_layers)]
        )

        # Heads
        self.policy_head = PolicyHead(config, n_planets=n_planets)
        self.value_head = ValueHead(config, max_players=max_players)
        self.opponent_head: Optional[OpponentHead] = None
        if config.use_opponent_head:
            self.opponent_head = OpponentHead(config, n_planets=n_planets)

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Helper: build padding mask from all-zero feature rows
    # ------------------------------------------------------------------
    @staticmethod
    def _build_padding_mask(features: torch.Tensor) -> torch.Tensor:
        """Return True where the entire feature row is zero (i.e. padded)."""
        return (features == 0).all(dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        planet_features: torch.Tensor,
        fleet_features: Optional[torch.Tensor],
        global_features: torch.Tensor,
        owned_mask: torch.Tensor,
        enemy_mask: torch.Tensor,
        num_players: torch.Tensor,
        planet_ships: Optional[torch.Tensor] = None,
        fleet_padding_mask: Optional[torch.Tensor] = None,
        planet_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            planet_features:   [batch, N_planets, D_planet]
            fleet_features:    [batch, N_fleets, D_fleet]  (N_fleets can be 0)
            global_features:   [batch, D_global]
            owned_mask:        [batch, N_planets] bool, True = owned by current player
            enemy_mask:        [batch, N_planets] bool, True = enemy planet
            num_players:       [batch] int
            planet_ships:      [batch, N_planets]  current ship counts (for scaling)
            fleet_padding_mask: [batch, N_fleets]  True = pad  (optional, auto-built)
            planet_padding_mask:[batch, N_planets] True = pad  (optional, auto-built)

        Returns:
            target_logits:     [batch, N_owned, N_planets]
            num_ships:         [batch, N_owned, 1]  (sigmoid, in [0, 1])
            value:             [batch, max_players]  (ranking probability)
            opp_target_logits: [batch, N_enemy, N_planets]  (or None)
            opp_num_ships:     [batch, N_enemy, 1]          (or None)
        """
        B = planet_features.size(0)

        # --- Auto-build padding masks if not provided ---
        if planet_padding_mask is None:
            planet_padding_mask = self._build_padding_mask(planet_features)

        # --- 1. Input projection ---
        planet_emb, fleet_emb, global_emb = self.input_proj(
            planet_features, fleet_features, global_features
        )

        # --- 2. Add positional encoding ---
        planet_emb = self.pos_enc(planet_emb)
        if fleet_emb is not None and fleet_emb.size(1) > 0:
            fleet_emb = self.pos_enc(fleet_emb)

        # --- 3. Encode ---
        planet_emb = self.planet_encoder(planet_emb, padding_mask=planet_padding_mask)

        # Fleet encoding (skip if no fleets)
        fleet_pool = torch.zeros(B, self.d_model, device=planet_features.device)
        if fleet_emb is not None and fleet_emb.size(1) > 0:
            if fleet_padding_mask is None:
                fleet_padding_mask = self._build_padding_mask(fleet_features)
            fleet_emb = self.fleet_encoder(fleet_emb, padding_mask=fleet_padding_mask)
            # Mean-pool fleet embeddings (over valid fleets)
            valid_fleet_mask = (~fleet_padding_mask).float().unsqueeze(-1)  # [B, Nf, 1]
            n_valid = valid_fleet_mask.sum(dim=1).clamp(min=1)
            fleet_pool = (fleet_emb * valid_fleet_mask).sum(dim=1) / n_valid

        # --- 4. Build global context ---
        global_context = self.global_combine(
            torch.cat([global_emb, fleet_pool], dim=-1)
        )  # [B, d_model]

        # --- 5. Build context sequence for fusion ---
        # Context = all planet embeddings + global embedding
        global_ctx_tokens = global_context.unsqueeze(1)  # [B, 1, d_model]

        if fleet_emb is not None and fleet_emb.size(1) > 0:
            context_seq = torch.cat([planet_emb, fleet_emb, global_ctx_tokens], dim=1)
            # context padding mask: planet mask | fleet mask | global (always valid)
            global_pad = torch.zeros(B, 1, dtype=torch.bool, device=planet_features.device)
            context_pad_mask = torch.cat(
                [planet_padding_mask, fleet_padding_mask, global_pad], dim=1
            )
        else:
            context_seq = torch.cat([planet_emb, global_ctx_tokens], dim=1)
            global_pad = torch.zeros(B, 1, dtype=torch.bool, device=planet_features.device)
            context_pad_mask = torch.cat(
                [planet_padding_mask, global_pad], dim=1
            )

        # --- 6. Fusion (cross-attention) ---
        # Queries for owned planets
        owned_emb = self._gather_by_mask(planet_emb, owned_mask)  # [B, N_owned, d]
        owned_pad_mask = self._gather_mask_by_bool(
            planet_padding_mask, owned_mask
        )  # [B, N_owned]

        fused = owned_emb
        for fusion_layer in self.fusion_layers:
            fused = fusion_layer(
                fused, context_seq,
                query_padding_mask=owned_pad_mask,
                context_padding_mask=context_pad_mask,
            )

        # --- 7. Policy head ---
        target_logits, num_ships_raw = self.policy_head(fused, global_context)

        # Scale num_ships by actual ship counts if provided
        if planet_ships is not None:
            owned_ships = self._gather_by_mask(
                planet_ships.unsqueeze(-1), owned_mask
            )  # [B, N_owned, 1]
            num_ships_out = num_ships_raw * owned_ships
        else:
            num_ships_out = num_ships_raw

        # --- 8. Value head ---
        value = self.value_head(planet_emb, padding_mask=planet_padding_mask, num_players=num_players)

        # --- 9. Opponent head (optional) ---
        opp_target_logits = None
        opp_num_ships = None
        if self.opponent_head is not None and enemy_mask.any():
            enemy_emb = self._gather_by_mask(planet_emb, enemy_mask)  # [B, N_enemy, d]

            # Fuse enemy embeddings too
            enemy_pad_mask = self._gather_mask_by_bool(
                planet_padding_mask, enemy_mask
            )
            enemy_fused = enemy_emb
            for fusion_layer in self.fusion_layers:
                enemy_fused = fusion_layer(
                    enemy_fused, context_seq,
                    query_padding_mask=enemy_pad_mask,
                    context_padding_mask=context_pad_mask,
                )

            opp_target_logits, opp_num_ships_raw = self.opponent_head(enemy_fused, global_context)

            if planet_ships is not None:
                enemy_ships = self._gather_by_mask(
                    planet_ships.unsqueeze(-1), enemy_mask
                )
                opp_num_ships = opp_num_ships_raw * enemy_ships
            else:
                opp_num_ships = opp_num_ships_raw

        return target_logits, num_ships_out, value, opp_target_logits, opp_num_ships

    # ------------------------------------------------------------------
    # Masking utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _gather_by_mask(
        seq: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Gather elements from seq where mask is True.
        Pads with zeros so all samples have the same leading dimension.

        seq:  [B, N, ...]
        mask: [B, N]  bool
        Returns: [B, K, ...]  where K = max number of True entries in the batch
        """
        B, N = mask.shape
        rest = list(seq.shape[2:])

        # Build a fixed-size output by scattering into a new tensor
        counts = mask.sum(dim=1)  # [B]
        K = int(counts.max().item())
        if K == 0:
            return torch.zeros(B, 0, *rest, device=seq.device, dtype=seq.dtype)

        # Index gather approach
        idx = mask.float().cumsum(dim=1).long()  # [B, N]  1-based positions
        out = torch.zeros(B, K, *rest, device=seq.device, dtype=seq.dtype)
        # Place elements where mask is True into the right slot (idx-1)
        valid = mask.unsqueeze(-1).expand_as(seq)  # [B, N, ...]
        slot_idx = (idx - 1).unsqueeze(-1).expand_as(seq)  # [B, N, ...]
        slot_idx = slot_idx.clamp(min=0)  # avoid -1 for non-owned
        out.scatter_add_(1, slot_idx, seq * valid.float())
        return out

    @staticmethod
    def _gather_mask_by_bool(
        padding_mask: torch.Tensor, bool_mask: torch.Tensor
    ) -> torch.Tensor:
        """Gather padding-mask entries corresponding to bool_mask == True.
        All positions not selected become True (padded) in the output.

        padding_mask: [B, N]  True = pad
        bool_mask:    [B, N]  True = select
        Returns:      [B, K]
        """
        B, N = bool_mask.shape
        counts = bool_mask.sum(dim=1)
        K = int(counts.max().item())
        if K == 0:
            return torch.ones(B, 0, dtype=torch.bool, device=bool_mask.device)

        # Gather the padding values for selected positions
        idx = bool_mask.float().cumsum(dim=1).long()
        out = torch.ones(B, K, dtype=torch.long, device=bool_mask.device)
        sel_padding = (padding_mask & bool_mask).long()
        slot_idx = (idx - 1).clamp(min=0)
        out.scatter_add_(1, slot_idx, sel_padding)
        # Convert back to bool, re-mask slots beyond each sample's count
        out = out.bool()
        positions = torch.arange(K, device=bool_mask.device).unsqueeze(0)  # [1, K]
        beyond = positions >= counts.unsqueeze(1)  # [B, K]
        out = out | beyond
        return out
