"""Kaggle-style entrypoint for the rulebase MCTS rollout selector."""

from __future__ import annotations

try:
    from .fusion import RuleFusionConfig, RuleFusionSelector
except ImportError:  # Allows copying this directory into a flat submission.
    from fusion import RuleFusionConfig, RuleFusionSelector


_AGENT = RuleFusionSelector(RuleFusionConfig())


def agent(obs, configuration=None):
    try:
        return _AGENT.act(obs, configuration)
    except Exception:
        return []
