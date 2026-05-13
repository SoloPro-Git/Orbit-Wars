"""Kaggle-style entrypoint for the structured public strategy package."""

from __future__ import annotations

try:
    from .rl_informed_agent import RLInformedPublicRuleAgent
    from .strategy_config import REGULAR_CONFIG
except ImportError:  # Allows running this file directly after copying beside modules.
    from rl_informed_agent import RLInformedPublicRuleAgent
    from strategy_config import REGULAR_CONFIG

_AGENT = RLInformedPublicRuleAgent(**REGULAR_CONFIG.to_agent_kwargs())


def agent(obs, configuration=None):
    try:
        return _AGENT.act(obs)
    except Exception:
        return []
