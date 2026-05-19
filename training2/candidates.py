"""Candidate action generation."""
from __future__ import annotations

from collections.abc import Callable
import random


def _canonical(action: list[list]) -> tuple:
    return tuple((int(a[0]), round(float(a[1]), 6), int(a[2])) for a in action if len(a) >= 3)


def build_candidates(
    obs: dict,
    rulebase_agent: Callable[[dict], list[list]],
    max_candidates: int = 32,
    include_noop: bool = True,
    extra_candidates: list[list[list]] | None = None,
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
    for candidate in extra_candidates or []:
        add(candidate)
    return candidates[:max_candidates], oracle_index


def shuffle_candidates(
    candidates: list[list[list]],
    target_index: int,
    rng: random.Random,
) -> tuple[list[list[list]], int]:
    """Shuffle candidates and return the target candidate's new index."""
    if not candidates:
        return candidates, target_index
    indexed = list(enumerate(candidates))
    rng.shuffle(indexed)
    shuffled = [candidate for _, candidate in indexed]
    new_target = next((i for i, (old_idx, _) in enumerate(indexed) if old_idx == target_index), 0)
    return shuffled, new_target
