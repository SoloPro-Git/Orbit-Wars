"""Kaggle-style entrypoint for the structured public strategy package."""

from __future__ import annotations

from dataclasses import fields

try:
    from .rl_informed_agent import RLInformedPublicRuleAgent
    from .strategy_config import REGULAR_CONFIG
except ImportError:  # Allows running this file directly after copying beside modules.
    from rl_informed_agent import RLInformedPublicRuleAgent
    from strategy_config import REGULAR_CONFIG

_ALLOWED_KWARGS = {field.name for field in fields(RLInformedPublicRuleAgent)}
_AGENT = RLInformedPublicRuleAgent(
    **{
        key: value
        for key, value in REGULAR_CONFIG.to_agent_kwargs().items()
        if key in _ALLOWED_KWARGS
    }
)


def agent(obs, configuration=None):
    try:
        return _AGENT.act(obs)
    except Exception:
        return []
