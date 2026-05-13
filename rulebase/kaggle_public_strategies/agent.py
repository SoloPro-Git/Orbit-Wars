"""Kaggle-style entrypoint for the structured public strategy package."""

from __future__ import annotations

try:
    from .public_rule_agent import PublicRuleAgent
except ImportError:  # Allows running this file directly after copying beside modules.
    from public_rule_agent import PublicRuleAgent

_AGENT = PublicRuleAgent()


def agent(obs, configuration=None):
    try:
        return _AGENT.act(obs)
    except Exception:
        return []
