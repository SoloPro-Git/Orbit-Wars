"""Candidate action generation.

The first candidate is always the oracle rulebase move list. Additional
candidates are no-op and rulebase sub-actions. This keeps the policy close to
known legal/tactical behavior while allowing the model to suppress or combine
parts of the rulebase plan.
"""
from __future__ import annotations

from collections.abc import Callable


def _canonical(action: list[list]) -> tuple:
    return tuple((int(a[0]), round(float(a[1]), 6), int(a[2])) for a in action if len(a) >= 3)


def build_candidates(
    obs: dict,
    rulebase_agent: Callable[[dict], list[list]],
    max_candidates: int = 32,
    include_noop: bool = True,
) -> tuple[list[list[list]], int]:
    oracle = rulebase_agent(obs) or []
    candidates: list[list[list]] = []
    seen: set[tuple] = set()

    def add(action: list[list]) -> None:
        key = _canonical(action)
        if key in seen:
            return
        seen.add(key)
        candidates.append([[int(a[0]), float(a[1]), int(a[2])] for a in action if len(a) >= 3])

    add(oracle)
    oracle_index = 0
    if include_noop:
        add([])
    for move in oracle:
        add([move])
    for i in range(len(oracle)):
        add([m for j, m in enumerate(oracle) if j != i])
    return candidates[:max_candidates], oracle_index

