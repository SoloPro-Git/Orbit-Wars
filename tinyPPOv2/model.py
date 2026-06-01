from __future__ import annotations

import torch
from torch import nn

from tinyPPO.features import GLOBAL_FEAT_DIM, MAX_PLANETS, PAIR_FEAT_DIM, PLANET_FEAT_DIM


class AutoregressivePolicyNet(nn.Module):
    """Entity encoder plus autoregressive action decoder.

    This is an AlphaStar-lite policy head for imitation experiments. It predicts
    a sequence of actions with teacher forcing:

    continue/stop -> source -> target -> ship bucket

    Unlike TinyPolicyValueNet's independent source-slot heads, later actions can
    condition on earlier selected source/target/ship decisions.
    """

    def __init__(
        self,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 2,
        ship_buckets: int = 5,
        max_actions: int = 24,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.ship_buckets = int(ship_buckets)
        self.max_actions = int(max_actions)
        context_layers = max(0, int(layers) - 1)
        self.planet_in = nn.Sequential(
            nn.Linear(PLANET_FEAT_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        if context_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=max(1, int(heads)),
                dim_feedforward=hidden * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.planet_context = nn.TransformerEncoder(encoder_layer, num_layers=context_layers)
        else:
            self.planet_context = None
        self.global_in = nn.Sequential(nn.Linear(GLOBAL_FEAT_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.source_context = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.edge = nn.Sequential(
            nn.Linear(hidden * 3 + PAIR_FEAT_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.init_decoder = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())
        self.start_embed = nn.Parameter(torch.zeros(hidden))
        self.source_embed = nn.Embedding(MAX_PLANETS + 1, hidden)
        self.target_embed = nn.Embedding(MAX_PLANETS + 1, hidden)
        self.ship_embed = nn.Embedding(max(1, int(ship_buckets)) + 1, hidden)
        self.step_embed = nn.Parameter(torch.zeros(max_actions + 1, hidden))
        self.decoder = nn.GRUCell(hidden, hidden)
        self.continue_head = nn.Linear(hidden, 2)
        self.source_head = nn.Linear(hidden * 2, 1)
        self.target_head = nn.Linear(hidden * 3, 1)
        self.ship_head = nn.Linear(hidden * 3, ship_buckets)
        self.value = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def encode(
        self,
        planets: torch.Tensor,
        pair_features: torch.Tensor,
        global_features: torch.Tensor,
        planet_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = self.planet_in(planets)
        if self.planet_context is not None:
            x = self.planet_context(x, src_key_padding_mask=~planet_mask)
            x = x.masked_fill(~planet_mask[:, :, None], 0.0)
        g = self.global_in(global_features)
        src_ctx = self.source_context(torch.cat([x, g[:, None, :].expand(-1, MAX_PLANETS, -1)], dim=-1))
        src = src_ctx[:, :, None, :].expand(-1, -1, MAX_PLANETS, -1)
        tgt = x[:, None, :, :].expand(-1, MAX_PLANETS, -1, -1)
        glob = g[:, None, None, :].expand(-1, MAX_PLANETS, MAX_PLANETS, -1)
        edge_hidden = self.edge(torch.cat([src, tgt, glob, pair_features], dim=-1))
        valid = planet_mask.float().unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return {
            "planet_hidden": x,
            "source_hidden": src_ctx,
            "edge_hidden": edge_hidden,
            "global_hidden": g,
            "pooled": pooled,
        }

    def forward(
        self,
        planets: torch.Tensor,
        pair_features: torch.Tensor,
        global_features: torch.Tensor,
        planet_mask: torch.Tensor,
        own_mask: torch.Tensor,
        prev_source: torch.Tensor | None = None,
        prev_target: torch.Tensor | None = None,
        prev_ship: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        enc = self.encode(planets, pair_features, global_features, planet_mask)
        bsz = planets.shape[0]
        steps = self.max_actions + 1 if prev_source is None else int(prev_source.shape[1])
        hidden = self.init_decoder(torch.cat([enc["pooled"], enc["global_hidden"]], dim=-1))
        prev_input = self.start_embed[None, :].expand(bsz, -1)
        cont_logits: list[torch.Tensor] = []
        source_logits: list[torch.Tensor] = []
        target_logits: list[torch.Tensor] = []
        ship_logits: list[torch.Tensor] = []

        none_source = torch.full((bsz,), MAX_PLANETS, dtype=torch.long, device=planets.device)

        for step in range(steps):
            step_input = prev_input + self.step_embed[min(step, self.max_actions)][None, :]
            hidden = self.decoder(step_input, hidden)
            cont_logits.append(self.continue_head(hidden))

            src_in = torch.cat([enc["source_hidden"], hidden[:, None, :].expand(-1, MAX_PLANETS, -1)], dim=-1)
            src_logits = self.source_head(src_in).squeeze(-1).masked_fill(~own_mask, -1e9)
            source_logits.append(src_logits)

            if prev_source is None:
                teacher_source = src_logits.argmax(dim=-1)
            else:
                action_step = min(step + 1, prev_source.shape[1] - 1)
                teacher_source = prev_source[:, action_step].clamp(0, MAX_PLANETS)
            source_for_gather = torch.where(teacher_source < MAX_PLANETS, teacher_source, none_source)
            source_safe = source_for_gather.clamp(0, MAX_PLANETS - 1)
            chosen_source = enc["source_hidden"][torch.arange(bsz, device=planets.device), source_safe]
            pair_for_source = enc["edge_hidden"][torch.arange(bsz, device=planets.device), source_safe]
            tgt_in = torch.cat(
                [
                    pair_for_source,
                    chosen_source[:, None, :].expand(-1, MAX_PLANETS, -1),
                    hidden[:, None, :].expand(-1, MAX_PLANETS, -1),
                ],
                dim=-1,
            )
            tgt_logits = self.target_head(tgt_in).squeeze(-1)
            tgt_logits = tgt_logits.masked_fill(~planet_mask, -1e9)
            tgt_logits.scatter_(1, source_safe[:, None], -1e9)
            target_logits.append(tgt_logits)

            if prev_target is None:
                teacher_target = tgt_logits.argmax(dim=-1)
            else:
                action_step = min(step + 1, prev_target.shape[1] - 1)
                teacher_target = prev_target[:, action_step].clamp(0, MAX_PLANETS)
            target_safe = teacher_target.clamp(0, MAX_PLANETS - 1)
            chosen_edge = enc["edge_hidden"][
                torch.arange(bsz, device=planets.device),
                source_safe,
                target_safe,
            ]
            ship_in = torch.cat([chosen_edge, chosen_source, hidden], dim=-1)
            ship_logits.append(self.ship_head(ship_in))

            if prev_source is None or prev_target is None or prev_ship is None:
                prev_input = chosen_source + enc["planet_hidden"][torch.arange(bsz, device=planets.device), target_safe]
            else:
                src_emb = self.source_embed(prev_source[:, step].clamp(0, MAX_PLANETS))
                tgt_emb = self.target_embed(prev_target[:, step].clamp(0, MAX_PLANETS))
                shp_emb = self.ship_embed(prev_ship[:, step].clamp(0, self.ship_buckets))
                prev_input = src_emb + tgt_emb + shp_emb

        value = self.value(torch.cat([enc["pooled"], enc["global_hidden"]], dim=-1)).squeeze(-1)
        return {
            "continue_logits": torch.stack(cont_logits, dim=1),
            "source_logits": torch.stack(source_logits, dim=1),
            "target_logits": torch.stack(target_logits, dim=1),
            "ship_logits": torch.stack(ship_logits, dim=1),
            "value": value,
        }
