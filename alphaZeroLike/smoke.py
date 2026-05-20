"""Smoke test for the AlphaZero-like prototype."""

from __future__ import annotations

from alphaZeroLike.candidates import CandidateGenerator
from alphaZeroLike.features import encode_search_position
from alphaZeroLike.mcts import MCTSConfig, ShallowPUCTSearch, _raw_obs
from alphaZeroLike.model import AlphaZeroLikeNet
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent


def main() -> None:
    env = make_fast_orbit_wars({"episodeSteps": 80, "seed": 0}, keep_history=False, use_numba=False)
    env.reset(2)
    model = AlphaZeroLikeNet(d_model=64, nhead=4, layers=1, dropout=0.0).eval()
    obs = _raw_obs(env, 0)
    candidates, _ = CandidateGenerator()(obs)
    enc = encode_search_position(obs, 0, candidates, max_candidates=64, max_moves=16)
    search = ShallowPUCTSearch(
        model,
        CandidateGenerator(),
        MCTSConfig(simulations=2, rollout_depth=1, max_candidates=64, max_moves=16),
        opponent_factory=lambda: make_rulebase_agent("rl_informed_regular"),
    )
    result = search.search(env, 0)
    print(
        {
            "candidates": len(candidates),
            "planet_shape": tuple(enc.planet_features.shape),
            "move_shape": tuple(enc.move_features.shape),
            "visits": result.visits.tolist(),
            "selected": result.selected_index,
        }
    )


if __name__ == "__main__":
    main()
