"""Small end-to-end smoke test for training2."""
from __future__ import annotations

from kaggle_environments import make
import torch

from training2.candidates import build_candidates
from training2.features import encode_position
from training2.model import CandidatePolicyValueNet
from training2.rulebase_bridge import make_rulebase_agent


def main() -> None:
    env = make("orbit_wars", configuration={"episodeSteps": 500}, debug=True)
    env.reset(4)
    obs = env.steps[-1][0]["observation"]
    agent = make_rulebase_agent()
    candidates, oracle_idx = build_candidates(obs, agent)
    encoded = encode_position(obs, 0, candidates)
    model = CandidatePolicyValueNet(d_model=48, nhead=3, layers=1)
    logits, value = model(
        torch.tensor(encoded.planet_features).unsqueeze(0),
        torch.tensor(encoded.global_features).unsqueeze(0),
        torch.tensor(encoded.candidate_features).unsqueeze(0),
        torch.tensor(encoded.candidate_mask).unsqueeze(0),
    )
    print(
        {
            "num_candidates": len(candidates),
            "oracle_idx": oracle_idx,
            "logits_shape": tuple(logits.shape),
            "value_shape": tuple(value.shape),
        }
    )


if __name__ == "__main__":
    main()

