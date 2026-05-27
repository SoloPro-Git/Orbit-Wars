"""Kaggle-style agent wrapper for the AlphaZero-like model."""

from __future__ import annotations

from pathlib import Path
import random

import torch
import torch.nn.functional as F

from alphaZeroLike.candidates import CandidateConfig, CandidateGenerator
from alphaZeroLike.compat import load_training2_stage1_backbone
from alphaZeroLike.features import encode_global, encode_planets, encode_search_position
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
        rulebase_anchor: str = "disabled",
        anchor_min_prob: float = 0.55,
        anchor_min_margin: float = 0.10,
        proposal_explore_prob: float = 0.0,
        proposal_explore_top_k: int = 8,
        proposal_explore_only_model: bool = True,
        proposal_explore_max_extra_moves: int = 1,
        proposal_explore_max_ship_ratio: float = 1.25,
        proposal_explore_max_extra_ships: int = 20,
        proposal_explore_min_anchor_source_jaccard: float = 0.0,
        proposal_explore_project_anchor_sources: bool = False,
        proposal_explore_project_anchor_source_probs: bool = False,
        proposal_explore_max_anchor_source_drops: int | None = None,
        proposal_explore_min_anchor_moves: int = 1,
        seed: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.rulebase_anchor = rulebase_anchor
        self.anchor_min_prob = float(anchor_min_prob)
        self.anchor_min_margin = float(anchor_min_margin)
        self.proposal_explore_prob = max(0.0, min(float(proposal_explore_prob), 1.0))
        self.proposal_explore_top_k = max(1, int(proposal_explore_top_k))
        self.proposal_explore_only_model = bool(proposal_explore_only_model)
        self.proposal_explore_max_extra_moves = max(0, int(proposal_explore_max_extra_moves))
        self.proposal_explore_max_ship_ratio = max(0.0, float(proposal_explore_max_ship_ratio))
        self.proposal_explore_max_extra_ships = max(0, int(proposal_explore_max_extra_ships))
        self.proposal_explore_min_anchor_source_jaccard = max(
            0.0,
            min(float(proposal_explore_min_anchor_source_jaccard), 1.0),
        )
        self.proposal_explore_project_anchor_sources = bool(proposal_explore_project_anchor_sources)
        self.proposal_explore_project_anchor_source_probs = bool(proposal_explore_project_anchor_source_probs)
        self.proposal_explore_max_anchor_source_drops = (
            None if proposal_explore_max_anchor_source_drops is None else max(0, int(proposal_explore_max_anchor_source_drops))
        )
        self.proposal_explore_min_anchor_moves = max(1, int(proposal_explore_min_anchor_moves))
        self.last_explore_attempted = False
        self.last_explored = False
        self.last_explore_blocked = False
        self.last_explore_anchor_moves = 0
        self.last_explore_selected_moves = 0
        self.last_selected_idx = 0
        self.last_anchor_prob = 1.0
        self.rng = random.Random(seed)
        self.generator = CandidateGenerator(candidate_config)
        self.use_model_proposals = use_model_proposals
        self.proposal_config = proposal_config or ProposalConfig()
        needs_model = self.rulebase_anchor != "always" or self.use_model_proposals
        self.model = AlphaZeroLikeNet().to(self.device) if needs_model else None
        if self.model is not None:
            if checkpoint:
                ckpt = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            elif init_from_training2:
                load_training2_stage1_backbone(self.model, init_from_training2, map_location=self.device)
            self.model.eval()

    def act(self, obs: dict) -> list[list]:
        self.last_explore_attempted = False
        self.last_explored = False
        self.last_explore_blocked = False
        self.last_explore_anchor_moves = 0
        self.last_explore_selected_moves = 0
        self.last_selected_idx = 0
        self.last_anchor_prob = 1.0
        player = int(obs.get("player", 0))
        use_direct_source_projection = (
            self.proposal_explore_project_anchor_source_probs
            and self.proposal_explore_prob > 0.0
            and self.use_model_proposals
            and self.model is not None
        )
        extra = (
            proposals_from_model(obs, player, self.model, self.device, self.proposal_config)
            if self.use_model_proposals and self.model is not None and not use_direct_source_projection
            else []
        )
        candidates, _ = self.generator(obs, extra_candidates=extra)
        if not candidates:
            return []
        if self.proposal_explore_prob > 0.0 and len(candidates) > 1 and self.rng.random() < self.proposal_explore_prob:
            self.last_explore_attempted = True
            if use_direct_source_projection:
                selected = self._proposal_anchor_source_subset(obs, player, candidates[0])
                self.last_explore_anchor_moves = len(candidates[0])
                self.last_explore_selected_moves = len(selected)
                if selected and self._canonical_action(selected) != self._canonical_action(candidates[0]):
                    self.last_explored = True
                    return selected
                self.last_explore_blocked = True
                if self.rulebase_anchor == "always":
                    return candidates[0]
            extra_keys = {self._canonical_action(action) for action in extra}
            explore_indices = [
                idx
                for idx in range(1, len(candidates))
                if not self.proposal_explore_only_model or self._canonical_action(candidates[idx]) in extra_keys
            ][: self.proposal_explore_top_k]
            safe_indices = [
                idx
                for idx in explore_indices
                if self._safe_explore_action(candidates[0], candidates[idx])
            ]
            if safe_indices:
                self.last_explored = True
                selected = candidates[self.rng.choice(safe_indices)]
                if self.proposal_explore_project_anchor_sources:
                    selected = self._project_to_anchor_sources(candidates[0], selected)
                    if not selected:
                        self.last_explored = False
                        self.last_explore_blocked = True
                    else:
                        return selected
                else:
                    return selected
            self.last_explore_blocked = True
        if self.rulebase_anchor == "always":
            return candidates[0]
        assert self.model is not None
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
            probs = F.softmax(logits[0, : len(candidates)], dim=-1)
        idx = int(probs.argmax(dim=-1).item())
        self.last_selected_idx = idx
        self.last_anchor_prob = float(probs[0].item())
        if self.rulebase_anchor == "confidence" and idx != 0:
            top2 = torch.topk(probs, k=min(2, len(candidates))).values
            top_prob = float(top2[0].item())
            margin = float((top2[0] - top2[1]).item()) if len(top2) > 1 else top_prob
            if top_prob < self.anchor_min_prob or margin < self.anchor_min_margin:
                idx = 0
                self.last_selected_idx = idx
        return candidates[min(idx, len(candidates) - 1)]

    def _safe_explore_action(self, anchor: list[list], candidate: list[list]) -> bool:
        anchor_moves = len(anchor)
        candidate_moves = len(candidate)
        if candidate_moves > anchor_moves + self.proposal_explore_max_extra_moves:
            return False
        if self.proposal_explore_min_anchor_source_jaccard > 0.0:
            anchor_sources = self._action_sources(anchor)
            candidate_sources = self._action_sources(candidate)
            union = anchor_sources | candidate_sources
            jaccard = len(anchor_sources & candidate_sources) / max(len(union), 1)
            if jaccard < self.proposal_explore_min_anchor_source_jaccard:
                return False
        anchor_ships = self._action_ship_count(anchor)
        candidate_ships = self._action_ship_count(candidate)
        max_ships = max(
            anchor_ships + self.proposal_explore_max_extra_ships,
            int(anchor_ships * self.proposal_explore_max_ship_ratio),
        )
        return candidate_ships <= max_ships

    @staticmethod
    def _action_ship_count(action: list[list]) -> int:
        total = 0
        for move in action:
            if len(move) >= 3:
                total += max(0, int(move[2]))
        return total

    @staticmethod
    def _action_sources(action: list[list]) -> set[int]:
        return {int(move[0]) for move in action if len(move) >= 3}

    @staticmethod
    def _project_to_anchor_sources(anchor: list[list], candidate: list[list]) -> list[list]:
        candidate_sources = {int(move[0]) for move in candidate if len(move) >= 3}
        return [move for move in anchor if len(move) >= 3 and int(move[0]) in candidate_sources]

    def _proposal_anchor_source_subset(self, obs: dict, player: int, anchor: list[list]) -> list[list]:
        if self.model is None or not anchor:
            return []
        if len(anchor) < self.proposal_explore_min_anchor_moves:
            return []
        planets = obs.get("planets", [])
        if not planets:
            return []
        id_to_idx = {int(planet[0]): idx for idx, planet in enumerate(planets)}
        planet_features = torch.tensor(encode_planets(obs, player), dtype=torch.float32, device=self.device).unsqueeze(0)
        global_features = torch.tensor(encode_global(obs, player), dtype=torch.float32, device=self.device).unsqueeze(0)
        source_mask = torch.zeros((1, planet_features.size(1)), dtype=torch.bool, device=self.device)
        target_mask = torch.zeros((1, planet_features.size(1)), dtype=torch.bool, device=self.device)
        for idx, planet in enumerate(planets[: planet_features.size(1)]):
            target_mask[0, idx] = True
            source_mask[0, idx] = int(planet[1]) == player and float(planet[5]) >= 1.0
        with torch.no_grad():
            pred = self.model.proposal(
                planet_features,
                global_features,
                source_mask=source_mask,
                target_mask=target_mask,
            )
            send_prob = torch.sigmoid(pred["send_logits"])[0].detach().cpu()
        threshold = float(self.proposal_config.send_threshold)
        scored_moves: list[tuple[float, list]] = []
        for move in anchor:
            if len(move) < 3:
                continue
            src_idx = id_to_idx.get(int(move[0]))
            if src_idx is None or src_idx >= len(send_prob):
                continue
            scored_moves.append((float(send_prob[src_idx]), move))
        selected = [move for prob, move in scored_moves if prob >= threshold]
        if not selected and scored_moves:
            best_prob, best_move = max(scored_moves, key=lambda row: row[0])
            if best_prob >= threshold * 0.85:
                selected = [best_move]
        if (
            self.proposal_explore_max_anchor_source_drops is not None
            and len(anchor) - len(selected) > self.proposal_explore_max_anchor_source_drops
        ):
            keep_count = max(0, len(anchor) - self.proposal_explore_max_anchor_source_drops)
            selected_keys = {self._canonical_move(move) for move in selected}
            for _, move in sorted(scored_moves, key=lambda row: row[0], reverse=True):
                if len(selected) >= keep_count:
                    break
                key = self._canonical_move(move)
                if key not in selected_keys:
                    selected.append(move)
                    selected_keys.add(key)
        return selected

    @staticmethod
    def _canonical_move(move: list) -> tuple:
        return (int(move[0]), round(float(move[1]), 6), int(move[2]))

    @staticmethod
    def _canonical_action(action: list[list]) -> tuple:
        return tuple((int(a[0]), round(float(a[1]), 6), int(a[2])) for a in action if len(a) >= 3)
