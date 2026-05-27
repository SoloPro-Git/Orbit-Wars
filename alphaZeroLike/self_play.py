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


class OpponentPool:
    def __init__(
        self,
        *,
        rulebase_oracle: str = "rl_informed_regular",
        checkpoint_paths: list[str] | None = None,
        rulebase_weight: float = 1.0,
        checkpoint_weight: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.rulebase_oracle = rulebase_oracle
        self.checkpoint_paths = [str(p) for p in checkpoint_paths or [] if p]
        self.rulebase_weight = max(float(rulebase_weight), 0.0)
        self.checkpoint_weight = max(float(checkpoint_weight), 0.0)
        self.device = device
        self._model_agents: dict[str, object] = {}

    def make_agent(self, rng: random.Random):
        total = self.rulebase_weight + (self.checkpoint_weight if self.checkpoint_paths else 0.0)
        if total <= 0.0 or not self.checkpoint_paths or rng.random() < self.rulebase_weight / total:
            return make_rulebase_agent(self.rulebase_oracle)
        path = rng.choice(self.checkpoint_paths)
        agent = self._model_agents.get(path)
        if agent is None:
            from alphaZeroLike.agent import AlphaZeroLikeAgent

            agent = AlphaZeroLikeAgent(
                checkpoint=path,
                device=self.device,
                use_model_proposals=True,
                proposal_config=ProposalConfig(enabled=True, num_candidates=16, num_full_actions=8),
            )
            self._model_agents[path] = agent
        return agent.act


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
    opponent_checkpoints: list[str] | None = None,
    opponent_rulebase_weight: float = 1.0,
    opponent_checkpoint_weight: float = 0.0,
    opponent_device: str = "cpu",
    model_pid: int | None = None,
    search_stride: int = 1,
) -> list[dict]:
    env = make_fast_orbit_wars({"episodeSteps": episode_steps, "seed": seed}, keep_history=False, use_numba=use_numba)
    env.reset(players)
    rng = random.Random(seed)
    opponent_pool = OpponentPool(
        rulebase_oracle=opponent_oracle,
        checkpoint_paths=opponent_checkpoints,
        rulebase_weight=opponent_rulebase_weight,
        checkpoint_weight=opponent_checkpoint_weight,
        device=opponent_device,
    )
    search = ShallowPUCTSearch(
        model,
        CandidateGenerator(candidate_cfg),
        mcts_cfg or MCTSConfig(),
        device=device,
        opponent_factory=lambda: opponent_pool.make_agent(rng),
        proposal_cfg=proposal_cfg or ProposalConfig(enabled=True),
    )
    opponents = {pid: opponent_pool.make_agent(rng) for pid in range(players)}
    pending: list[dict] = []
    model_pid = int(seed % players if model_pid is None else model_pid) % players
    search_stride = max(1, int(search_stride))
    model_turn = 0

    for _ in range(episode_steps):
        actions = []
        for pid in range(players):
            obs = _raw_obs(env, pid)
            if pid == model_pid:
                do_search = model_turn % search_stride == 0
                model_turn += 1
                result = search.search(env, pid, add_noise=True, rng=rng) if do_search else search.policy_action(env, pid, rng=rng)
                actions.append(result.actions[result.selected_index] if result.actions else [])
                if do_search:
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
    parser.add_argument("--opponent-checkpoint", action="append", default=[])
    parser.add_argument("--opponent-rulebase-weight", type=float, default=1.0)
    parser.add_argument("--opponent-checkpoint-weight", type=float, default=0.0)
    parser.add_argument("--opponent-device", default="cpu")
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--rollout-depth", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search-stride", type=int, default=1)
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
                opponent_checkpoints=args.opponent_checkpoint,
                opponent_rulebase_weight=args.opponent_rulebase_weight,
                opponent_checkpoint_weight=args.opponent_checkpoint_weight,
                opponent_device=args.opponent_device,
                search_stride=args.search_stride,
            )
            for row in rows:
                f.write(json.dumps(row) + "\n")
            print({"game": game, "rows": len(rows)})


if __name__ == "__main__":
    main()
