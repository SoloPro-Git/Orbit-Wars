"""Tiny PPO baseline for Orbit Wars.

The package is intentionally small: fast simulator first, 2P first, model
chooses targets/ship buckets per owned source planet, and geometry computes
angles outside the network.
"""

from tinyPPO.agents import TinyPPOAgent, nearest_planet_agent, random_policy_agent

__all__ = ["TinyPPOAgent", "nearest_planet_agent", "random_policy_agent"]
