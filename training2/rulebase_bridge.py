"""Bridge to the strongest local rulebase agent.

The bridge creates fresh stateful rule agents per episode/player. Reusing one
global instance across games leaks trajectory state and makes supervised labels
noisy.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Callable


def make_rulebase_agent(name: str = "rl_informed_regular") -> Callable[[dict], list[list]]:
    if name == "public_exact":
        from rulebase.kaggle_public_strategies.public_rule_agent import PublicRuleAgent

        agent = PublicRuleAgent()
        return lambda obs: agent.act(obs)

    if name in {"rl_informed_regular", "regular"}:
        from rulebase.kaggle_public_rl_informed_strategies.rl_informed_agent import (
            RLInformedPublicRuleAgent,
        )
        from rulebase.kaggle_public_rl_informed_strategies.strategy_config import REGULAR_CONFIG

        allowed = {field.name for field in fields(RLInformedPublicRuleAgent)}
        kwargs = {
            key: value
            for key, value in REGULAR_CONFIG.to_agent_kwargs().items()
            if key in allowed
        }
        agent = RLInformedPublicRuleAgent(**kwargs)
        return lambda obs: agent.act(obs)

    raise ValueError(f"unknown rulebase agent: {name}")
