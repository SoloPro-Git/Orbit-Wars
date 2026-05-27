"""Shallow PUCT search over generated candidate action sets."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from alphaZeroLike.candidates import CandidateGenerator
from alphaZeroLike.env_clone import clone_fast_env
from alphaZeroLike.features import encode_search_position, margin_value, result_value
from alphaZeroLike.proposal import ProposalConfig, proposals_from_model


@dataclass
class MCTSConfig:
    simulations: int = 32
    c_puct: float = 1.5
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    rollout_depth: int = 8
    max_candidates: int = 64
    max_moves: int = 16
    value_mode: str = "rank"
    margin_scale: float = 50.0
    root_eval_mode: str = "puct"
    max_root_evals: int = 0


@dataclass
class SearchResult:
    actions: list[list[list]]
    priors: np.ndarray
    visits: np.ndarray
    q_values: np.ndarray
    policy_target: np.ndarray
    selected_index: int
    value: float


def _raw_obs(env, player: int) -> dict:
    return env.steps[-1][player]["observation"]


def _to_tensors(encoded, device: torch.device | str) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor(encoded.planet_features, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(encoded.global_features, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(encoded.move_features, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(encoded.move_source_indices, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(encoded.move_target_indices, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(encoded.move_mask, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(encoded.candidate_mask, dtype=torch.float32, device=device).unsqueeze(0),
    )


def evaluate_candidates(
    model,
    obs: dict,
    player: int,
    candidates: list[list[list]],
    *,
    device: torch.device | str = "cpu",
    max_candidates: int = 64,
    max_moves: int = 16,
) -> tuple[np.ndarray, float]:
    encoded = encode_search_position(
        obs,
        player,
        candidates,
        max_candidates=max_candidates,
        max_moves=max_moves,
    )
    with torch.no_grad():
        logits, value = model(*_to_tensors(encoded, device))
        valid = int(encoded.candidate_mask.sum())
        probs = torch.softmax(logits[0, :valid], dim=-1).detach().cpu().numpy()
    return probs.astype(np.float64), float(value[0].item())


class ShallowPUCTSearch:
    """Root PUCT search with short environment rollouts.

    This is not a full AlphaZero tree yet.  It is the smallest useful bridge:
    network priors choose root candidates, cloned fast environments estimate
    candidate value, and visit counts become the policy target.
    """

    def __init__(
        self,
        model,
        candidate_generator: CandidateGenerator | None = None,
        cfg: MCTSConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        opponent_factory: Callable[[], Callable[[dict], list[list]]] | None = None,
        proposal_cfg: ProposalConfig | None = None,
    ) -> None:
        self.model = model
        self.candidate_generator = candidate_generator or CandidateGenerator()
        self.cfg = cfg or MCTSConfig()
        self.device = device
        self.opponent_factory = opponent_factory
        self.proposal_cfg = proposal_cfg or ProposalConfig(enabled=False)

    def search(self, env, player: int, *, add_noise: bool = True, rng: random.Random | None = None) -> SearchResult:
        rng = rng or random.Random()
        obs = _raw_obs(env, player)
        extra = proposals_from_model(obs, player, self.model, self.device, self.proposal_cfg)
        candidates, _ = self.candidate_generator(obs, extra_candidates=extra)
        if not candidates:
            empty = np.zeros(0, dtype=np.float64)
            return SearchResult([], empty, empty, empty, empty, 0, 0.0)

        priors, root_value = evaluate_candidates(
            self.model,
            obs,
            player,
            candidates,
            device=self.device,
            max_candidates=self.cfg.max_candidates,
            max_moves=self.cfg.max_moves,
        )
        priors = priors[: len(candidates)]
        if add_noise and len(priors) > 1 and self.cfg.dirichlet_frac > 0:
            noise = np.fromiter(
                (rng.gammavariate(self.cfg.dirichlet_alpha, 1.0) for _ in priors),
                dtype=np.float64,
                count=len(priors),
            )
            noise /= max(float(noise.sum()), 1e-12)
            priors = (1.0 - self.cfg.dirichlet_frac) * priors + self.cfg.dirichlet_frac * noise
        priors /= max(float(priors.sum()), 1e-12)

        if self.cfg.root_eval_mode == "exhaustive":
            opponents = {
                pid: self.opponent_factory()
                for pid in range(getattr(env, "num_agents", 0))
                if pid != player and self.opponent_factory is not None
            }
            eval_count = len(candidates)
            if self.cfg.max_root_evals > 0:
                eval_count = min(eval_count, int(self.cfg.max_root_evals))
            order = sorted(range(len(candidates)), key=lambda i: float(priors[i]), reverse=True)[:eval_count]
            visits = np.zeros(len(candidates), dtype=np.float64)
            values = np.zeros(len(candidates), dtype=np.float64)
            for idx in order:
                values[idx] = self._rollout_value(env, player, candidates[idx], opponents)
                visits[idx] = 1.0
            q_values = np.divide(values, np.maximum(visits, 1.0))
            selected = max(order, key=lambda i: (float(q_values[i]), float(priors[i]))) if order else int(np.argmax(priors))
            policy = np.zeros(len(candidates), dtype=np.float64)
            policy[selected] = 1.0
            return SearchResult(candidates, priors, visits, q_values, policy, int(selected), root_value)

        visits = np.zeros(len(candidates), dtype=np.float64)
        values = np.zeros(len(candidates), dtype=np.float64)
        opponents = {
            pid: self.opponent_factory()
            for pid in range(getattr(env, "num_agents", 0))
            if pid != player and self.opponent_factory is not None
        }

        sims = max(1, int(self.cfg.simulations))
        for _ in range(sims):
            total = float(visits.sum())
            q = np.divide(values, np.maximum(visits, 1.0))
            u = self.cfg.c_puct * priors * math.sqrt(total + 1.0) / (1.0 + visits)
            idx = int(np.argmax(q + u))
            value = self._rollout_value(env, player, candidates[idx], opponents)
            visits[idx] += 1.0
            values[idx] += value

        q_values = np.divide(values, np.maximum(visits, 1.0))
        policy = visits.copy()
        if self.cfg.temperature <= 1e-6:
            selected = int(np.argmax(policy))
            policy[:] = 0.0
            policy[selected] = 1.0
        else:
            policy = np.power(policy, 1.0 / self.cfg.temperature)
            policy /= max(float(policy.sum()), 1e-12)
            selected = int(rng.choices(range(len(candidates)), weights=policy.tolist(), k=1)[0])
        return SearchResult(candidates, priors, visits, q_values, policy, selected, root_value)

    def policy_action(self, env, player: int, *, rng: random.Random | None = None) -> SearchResult:
        rng = rng or random.Random()
        obs = _raw_obs(env, player)
        extra = proposals_from_model(obs, player, self.model, self.device, self.proposal_cfg)
        candidates, _ = self.candidate_generator(obs, extra_candidates=extra)
        if not candidates:
            empty = np.zeros(0, dtype=np.float64)
            return SearchResult([], empty, empty, empty, empty, 0, 0.0)

        priors, root_value = evaluate_candidates(
            self.model,
            obs,
            player,
            candidates,
            device=self.device,
            max_candidates=self.cfg.max_candidates,
            max_moves=self.cfg.max_moves,
        )
        priors = priors[: len(candidates)]
        priors /= max(float(priors.sum()), 1e-12)
        selected = int(rng.choices(range(len(candidates)), weights=priors.tolist(), k=1)[0])
        zeros = np.zeros(len(candidates), dtype=np.float64)
        return SearchResult(candidates, priors, zeros, zeros, priors.copy(), selected, root_value)

    def _rollout_value(
        self,
        env,
        player: int,
        first_action: list[list],
        opponents: dict[int, Callable[[dict], list[list]]],
    ) -> float:
        sim = clone_fast_env(env)
        for depth in range(max(1, self.cfg.rollout_depth)):
            actions = []
            for pid in range(sim.num_agents):
                obs = _raw_obs(sim, pid)
                if pid == player:
                    if depth == 0:
                        actions.append(first_action)
                    else:
                        candidates, _ = self.candidate_generator(obs)
                        if not candidates:
                            actions.append([])
                        else:
                            priors, _ = evaluate_candidates(
                                self.model,
                                obs,
                                pid,
                                candidates,
                                device=self.device,
                                max_candidates=self.cfg.max_candidates,
                                max_moves=self.cfg.max_moves,
                            )
                            actions.append(candidates[int(np.argmax(priors[: len(candidates)]))])
                else:
                    agent = opponents.get(pid)
                    actions.append(agent(obs) if agent is not None else [])
            sim.step(actions)
            if all(state.get("status") != "ACTIVE" for state in sim.steps[-1]):
                break
        final_obs = _raw_obs(sim, player)
        if self.cfg.value_mode == "margin_tanh":
            return margin_value(final_obs, player, self.cfg.margin_scale)
        return result_value(final_obs, player)
