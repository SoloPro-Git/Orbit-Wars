from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinyPPO.agents import actions_from_decisions
from tinyPPO.features import MAX_PLANETS, encode_obs
from tinyPPOv2.model import AutoregressivePolicyNet


class AutoregressiveAgent:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        deterministic: bool = True,
        continue_bias: float = 0.0,
        temperature: float = 1.0,
        max_actions: int | None = None,
    ):
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        cfg = dict(payload.get("model", {}))
        self.model = AutoregressivePolicyNet(**cfg).to(device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.continue_bias = float(continue_bias)
        self.temperature = max(1e-4, float(temperature))
        self.max_actions = int(max_actions or cfg.get("max_actions", self.model.max_actions))

    @torch.no_grad()
    def __call__(self, obs: dict[str, Any], configuration=None) -> list[list]:
        del configuration
        player = int(obs.get("player", 0))
        enc = encode_obs(obs, player, players=2)
        batch = {
            "planets": torch.tensor(enc.planets[None], dtype=torch.float32, device=self.device),
            "pair_features": torch.tensor(enc.pair_features[None], dtype=torch.float32, device=self.device),
            "global_features": torch.tensor(enc.global_features[None], dtype=torch.float32, device=self.device),
            "planet_mask": torch.tensor(enc.planet_mask[None], dtype=torch.bool, device=self.device),
            "own_mask": torch.tensor(enc.own_mask[None], dtype=torch.bool, device=self.device),
        }
        out = self.model(**batch)
        source_slots: list[int] = []
        target_slots: list[int] = []
        ship_slots: list[int] = []
        steps = min(self.max_actions, int(out["source_logits"].shape[1]))
        for step in range(steps):
            continue_logits = out["continue_logits"][0, step] / self.temperature
            if self.continue_bias:
                continue_logits = continue_logits.clone()
                continue_logits[1] += self.continue_bias
            if self.deterministic:
                keep_going = int(torch.argmax(continue_logits).item()) == 1
            else:
                keep_going = int(torch.distributions.Categorical(logits=continue_logits).sample().item()) == 1
            if not keep_going:
                break
            source_logits = out["source_logits"][0, step] / self.temperature
            target_logits = out["target_logits"][0, step] / self.temperature
            if self.deterministic:
                source_idx = int(torch.argmax(source_logits).item())
                target_idx = int(torch.argmax(target_logits).item())
                ship_idx = int(torch.argmax(out["ship_logits"][0, step]).item())
            else:
                source_idx = int(torch.distributions.Categorical(logits=source_logits).sample().item())
                target_idx = int(torch.distributions.Categorical(logits=target_logits).sample().item())
                ship_idx = int(torch.distributions.Categorical(logits=out["ship_logits"][0, step]).sample().item())
            if 0 <= source_idx < MAX_PLANETS and 0 <= target_idx < MAX_PLANETS:
                source_slots.append(source_idx)
                target_slots.append(target_idx)
                ship_slots.append(ship_idx)
        return actions_from_decisions(
            obs,
            player,
            np.asarray(source_slots, dtype=np.int64),
            np.asarray(target_slots, dtype=np.int64),
            np.asarray(ship_slots, dtype=np.int64),
            ship_mode="required_bucket",
        )
