"""Fast rule-fusion selector for regular-family configs.

This is the pragmatic counterpart to the rollout planner: instead of spending
time on online rollouts every turn, it switches between regular-family configs
that have shown mode-specific signs of life in fast-simulator probes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any

from rulebase.kaggle_public_rl_informed_strategies.rl_informed_agent import (
    RLInformedPublicRuleAgent,
)
from rulebase.kaggle_public_rl_informed_strategies.strategy_config import REGULAR_CONFIG

from .configs import config_params


_ALLOWED_AGENT_KWARGS = {field.name for field in dataclass_fields(RLInformedPublicRuleAgent)}


@dataclass(frozen=True, slots=True)
class RuleFusionConfig:
    two_player_config: str = "p2move_regular"
    four_player_config: str = "regular"
    fallback_config: str = "regular"


@dataclass(slots=True)
class RuleFusionSelector:
    config: RuleFusionConfig = field(default_factory=RuleFusionConfig)
    _agents: dict[str, RLInformedPublicRuleAgent] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agents = {}
        for name in {
            self.config.two_player_config,
            self.config.four_player_config,
            self.config.fallback_config,
            "regular",
        }:
            self._agents[name] = self._make_agent(name)

    def act(self, obs: Any, configuration: Any = None) -> list:
        name = self._select_config_name(obs)
        agent = self._agents.get(name) or self._agents[self.config.fallback_config]
        try:
            return agent.act(obs)
        except Exception:
            return []

    def _select_config_name(self, obs: Any) -> str:
        active_players = _infer_num_agents(obs)
        if active_players <= 2:
            return self.config.two_player_config
        if active_players >= 4:
            return self.config.four_player_config
        return self.config.fallback_config

    @staticmethod
    def _make_agent(name: str) -> RLInformedPublicRuleAgent:
        params = config_params(name) or REGULAR_CONFIG.to_agent_kwargs()
        return RLInformedPublicRuleAgent(
            **{key: value for key, value in dict(params).items() if key in _ALLOWED_AGENT_KWARGS}
        )


def _obs_get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _infer_num_agents(obs: Any) -> int:
    max_player = int(_obs_get(obs, "player", 0))
    for planet in _obs_get(obs, "planets", []) or []:
        owner = int(planet[1])
        if owner >= 0:
            max_player = max(max_player, owner)
    for fleet in _obs_get(obs, "fleets", []) or []:
        owner = int(fleet[1])
        if owner >= 0:
            max_player = max(max_player, owner)
    return max(2, max_player + 1)
