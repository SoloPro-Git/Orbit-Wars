"""Self-play data generation using shallow PUCT search."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from alphaZeroLike.candidates import CandidateConfig, CandidateGenerator
from alphaZeroLike.compat import load_training2_stage1_backbone
from alphaZeroLike.features import result_value
from alphaZeroLike.mcts import MCTSConfig, ShallowPUCTSearch, _raw_obs
from alphaZeroLike.model import AlphaZeroLikeNet
from alphaZeroLike.proposal import ProposalConfig
from alphaZeroLike.proposal import proposal_labels
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent


def generate_game(
    model: AlphaZeroLikeNet,
    *,
    seed: int,
    players: int = 2,
    episode_steps: int = 500,
    use_numba: bool = True,
    opponent_oracle: str = "rl_informed_regular",
    device: str = "cpu",
    mcts_cfg: MCTSConfig | None = None,
    candidate_cfg: CandidateConfig | None = None,
    proposal_cfg: ProposalConfig | None = None,
) -> list[dict]:
    env = make_fast_orbit_wars({"episodeSteps": episode_steps, "seed": seed}, keep_history=False, use_numba=use_numba)
    env.reset(players)
    rng = random.Random(seed)
    search = ShallowPUCTSearch(
        model,
        CandidateGenerator(candidate_cfg),
        mcts_cfg or MCTSConfig(),
        device=device,
        opponent_factory=lambda: make_rulebase_agent(opponent_oracle),
        proposal_cfg=proposal_cfg or ProposalConfig(enabled=True),
    )
    opponents = {pid: make_rulebase_agent(opponent_oracle) for pid in range(players)}
    pending: list[dict] = []
    model_pid = seed % players

    for _ in range(episode_steps):
        actions = []
        for pid in range(players):
            obs = _raw_obs(env, pid)
            if pid == model_pid:
                result = search.search(env, pid, add_noise=True, rng=rng)
                actions.append(result.actions[result.selected_index] if result.actions else [])
                pending.append(
                    {
                        "obs": obs,
                        "player": pid,
                        "candidates": result.actions,
                        "policy_target": result.policy_target.tolist(),
                        "root_value": result.value,
                        "max_moves": (mcts_cfg or MCTSConfig()).max_moves,
                        **proposal_labels(
                            obs,
                            pid,
                            result.actions[result.selected_index] if result.actions else [],
                        ),
                    }
                )
            else:
                actions.append(opponents[pid](obs) or [])
        env.step(actions)
        if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
            break

    final_obs = _raw_obs(env, model_pid)
    value = result_value(final_obs, model_pid)
    for row in pending:
        row["value"] = value
    return pending


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--init-from-training2",
        help="Optional training2 stage1 checkpoint; used only when --checkpoint is not set.",
    )
    parser.add_argument("--out", default="data/alphaZeroLike/selfplay.jsonl")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--opponent-oracle", default="rl_informed_regular")
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--rollout-depth", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-rulebase-candidates", action="store_true")
    parser.add_argument("--no-heuristics", action="store_true")
    parser.add_argument("--proposal-num-candidates", type=int, default=16)
    parser.add_argument("--proposal-num-full-actions", type=int, default=8)
    parser.add_argument("--proposal-send-threshold", type=float, default=0.45)
    parser.add_argument("--proposal-max-sources", type=int, default=4)
    args = parser.parse_args()

    model = AlphaZeroLikeNet().to(args.device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif args.init_from_training2:
        report = load_training2_stage1_backbone(model, args.init_from_training2, map_location=args.device)
        print({"init_from_training2": args.init_from_training2, "loaded_tensors": len(report["loaded"])})
    model.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mcts_cfg = MCTSConfig(simulations=args.simulations, rollout_depth=args.rollout_depth)
    candidate_cfg = CandidateConfig(
        use_rulebase=not args.no_rulebase_candidates,
        include_heuristics=not args.no_heuristics,
    )
    proposal_cfg = ProposalConfig(
        enabled=True,
        num_candidates=args.proposal_num_candidates,
        num_full_actions=args.proposal_num_full_actions,
        send_threshold=args.proposal_send_threshold,
        max_sources=args.proposal_max_sources,
    )
    with out.open("w") as f:
        for game in range(args.games):
            rows = generate_game(
                model,
                seed=args.seed + game,
                players=args.players,
                episode_steps=args.episode_steps,
                use_numba=not args.no_numba,
                opponent_oracle=args.opponent_oracle,
                device=args.device,
                mcts_cfg=mcts_cfg,
                candidate_cfg=candidate_cfg,
                proposal_cfg=proposal_cfg,
            )
            for row in rows:
                f.write(json.dumps(row) + "\n")
            print({"game": game, "rows": len(rows)})


if __name__ == "__main__":
    main()
