"""Candidate generation for the AlphaZero-like prototype."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from training2.candidates import build_candidates as _training2_build_candidates
from training2.rulebase_bridge import make_rulebase_agent


@dataclass
class CandidateConfig:
    max_candidates: int = 64
    include_noop: bool = True
    include_heuristics: bool = True
    oracle: str = "rl_informed_regular"
    use_rulebase: bool = True


class CandidateGenerator:
    def __init__(
        self,
        cfg: CandidateConfig | None = None,
        rulebase_agent: Callable[[dict], list[list]] | None = None,
    ) -> None:
        self.cfg = cfg or CandidateConfig()
        self.rulebase = rulebase_agent or (
            make_rulebase_agent(self.cfg.oracle) if self.cfg.use_rulebase else (lambda obs: [])
        )

    def __call__(
        self,
        obs: dict,
        extra_candidates: list[list[list]] | None = None,
    ) -> tuple[list[list[list]], int]:
        return _training2_build_candidates(
            obs,
            self.rulebase,
            max_candidates=self.cfg.max_candidates,
            include_noop=self.cfg.include_noop,
            extra_candidates=extra_candidates,
            include_heuristics=self.cfg.include_heuristics,
        )
