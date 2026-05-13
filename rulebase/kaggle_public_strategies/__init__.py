"""Structured rule strategies distilled from public Kaggle Orbit Wars notebooks."""

from .agent import agent
from .public_rule_agent import PublicRuleAgent

__all__ = ["agent", "PublicRuleAgent"]
