"""Kaggle-style agent that reranks regular-rulebase candidates with a model."""
from __future__ import annotations

from pathlib import Path

import torch

from training2.candidates import build_candidates
from training2.features import encode_position
from training2.model import CandidatePolicyValueNet
from training2.rulebase_bridge import make_rulebase_agent


class CandidateModelAgent:
    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cpu",
        max_candidates: int = 32,
        oracle: str = "rl_informed_regular",
    ) -> None:
        self.device = torch.device(device)
        self.max_candidates = max_candidates
        self.rulebase = make_rulebase_agent(oracle)
        self.model = CandidatePolicyValueNet().to(self.device)
        if checkpoint:
            ckpt = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()

    def act(self, obs: dict) -> list[list]:
        player = int(obs.get("player", 0))
        candidates, _ = build_candidates(obs, self.rulebase, max_candidates=self.max_candidates)
        if not candidates:
            return []
        enc = encode_position(obs, player, candidates, max_candidates=self.max_candidates)
        with torch.no_grad():
            logits, _ = self.model(
                torch.tensor(enc.planet_features, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(enc.global_features, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(enc.candidate_features, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(enc.candidate_mask, dtype=torch.float32, device=self.device).unsqueeze(0),
            )
        idx = int(logits.argmax(-1).item())
        return candidates[idx] if idx < len(candidates) else candidates[0]

