"""Generate rulebase imitation data for training2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_environments import make

from training2.candidates import build_candidates
from training2.features import encode_position, result_value
from training2.rulebase_bridge import make_rulebase_agent


def _raw_obs(env, player: int) -> dict:
    return env.steps[-1][player]["observation"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--players", type=int, default=4, choices=[2, 4])
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--oracle", default="rl_informed_regular")
    parser.add_argument("--out", default="data/training2/rulebase.jsonl")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    with out.open("w") as f:
        for episode in range(args.episodes):
            env = make("orbit_wars", configuration={"episodeSteps": 500, "seed": episode}, debug=True)
            env.reset(args.players)
            agents = {pid: make_rulebase_agent(args.oracle) for pid in range(args.players)}
            pending: list[dict] = []

            for _ in range(500):
                step_actions = []
                for pid in range(args.players):
                    obs = _raw_obs(env, pid)
                    candidates, oracle_idx = build_candidates(
                        obs,
                        agents[pid],
                        max_candidates=args.max_candidates,
                    )
                    enc = encode_position(obs, pid, candidates, max_candidates=args.max_candidates)
                    pending.append(
                        {
                            "player": pid,
                            "planets": enc.planet_features.tolist(),
                            "global": enc.global_features.tolist(),
                            "candidates": enc.candidate_features.tolist(),
                            "candidate_mask": enc.candidate_mask.tolist(),
                            "target": oracle_idx,
                        }
                    )
                    step_actions.append(candidates[oracle_idx] if candidates else [])
                env.step(step_actions)
                if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                    break

            final_obs = [_raw_obs(env, pid) for pid in range(args.players)]
            for row in pending:
                row["value"] = result_value(final_obs[row["player"]], row["player"])
                f.write(json.dumps(row, separators=(",", ":")) + "\n")

            rows.append({"episode": episode, "samples": len(pending)})
            print(f"episode={episode} samples={len(pending)} total={sum(r['samples'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()

