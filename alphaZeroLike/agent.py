"""Kaggle-style agent wrapper for the AlphaZero-like model."""

from __future__ import annotations

from pathlib import Path

import torch

from alphaZeroLike.candidates import CandidateConfig, CandidateGenerator
from alphaZeroLike.compat import load_training2_stage1_backbone
from alphaZeroLike.features import encode_search_position
from alphaZeroLike.model import AlphaZeroLikeNet
from alphaZeroLike.proposal import ProposalConfig, proposals_from_model


class AlphaZeroLikeAgent:
    def __init__(
        self,
        checkpoint: str | None = None,
        init_from_training2: str | None = None,
        device: str = "cpu",
        candidate_config: CandidateConfig | None = None,
        use_model_proposals: bool = True,
        proposal_config: ProposalConfig | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = AlphaZeroLikeNet().to(self.device)
        if checkpoint:
            ckpt = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif init_from_training2:
            load_training2_stage1_backbone(self.model, init_from_training2, map_location=self.device)
        self.model.eval()
        self.generator = CandidateGenerator(candidate_config)
        self.use_model_proposals = use_model_proposals
        self.proposal_config = proposal_config or ProposalConfig()

    def act(self, obs: dict) -> list[list]:
        player = int(obs.get("player", 0))
        extra = (
            proposals_from_model(obs, player, self.model, self.device, self.proposal_config)
            if self.use_model_proposals
            else []
        )
        candidates, _ = self.generator(obs, extra_candidates=extra)
        if not candidates:
            return []
        encoded = encode_search_position(obs, player, candidates)
        with torch.no_grad():
            logits, _ = self.model(
                torch.tensor(encoded.planet_features, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(encoded.global_features, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(encoded.move_features, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(encoded.move_source_indices, dtype=torch.long, device=self.device).unsqueeze(0),
                torch.tensor(encoded.move_target_indices, dtype=torch.long, device=self.device).unsqueeze(0),
                torch.tensor(encoded.move_mask, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(encoded.candidate_mask, dtype=torch.float32, device=self.device).unsqueeze(0),
            )
        idx = int(logits.argmax(dim=-1).item())
        return candidates[min(idx, len(candidates) - 1)]
