from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent

from tinyPPO.agents import TinyPPOAgent, nearest_planet_agent, random_policy_agent
from tinyPPO.features import score


def run_matchups(
    agent_factory: Callable[[], Callable],
    opponent_factory: Callable[[], Callable],
    games: int = 20,
    seed: int = 1000,
    episode_steps: int = 500,
    use_numba: bool = True,
    progress: bool = False,
    desc: str = "eval",
) -> dict[str, float]:
    iterator = range(games)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=games, desc=desc, dynamic_ncols=True, leave=False)
        except Exception:
            pass
    wins = losses = draws = 0
    rewards: list[float] = []
    for i in iterator:
        model_seat = i % 2
        agents = [opponent_factory(), opponent_factory()]
        agents[model_seat] = agent_factory()
        env = make_fast_orbit_wars(
            {"episodeSteps": episode_steps, "seed": seed + i},
            keep_history=False,
            use_numba=use_numba,
        )
        env.run(agents)
        final = env.steps[-1]
        obs = final[model_seat]["observation"]
        model_score = score(obs, model_seat)
        other_score = score(obs, 1 - model_seat)
        if model_score > other_score:
            wins += 1
            rewards.append(1.0)
        elif model_score < other_score:
            losses += 1
            rewards.append(-1.0)
        else:
            draws += 1
            rewards.append(0.0)
    return {
        "games": float(games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "winrate": wins / max(1, games),
        "nonloss": (wins + draws) / max(1, games),
        "mean_reward": sum(rewards) / max(1, len(rewards)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", choices=["nearest", "random", "regular"], default="nearest")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--aggression", type=float, default=0.0, help="Convenience bias applied to both launch and ship fraction. Positive is more aggressive.")
    parser.add_argument("--launch-bias", type=float, default=0.0, help="Logit bias for launch vs no-launch. Positive launches more often.")
    parser.add_argument("--ship-bias", type=float, default=0.0, help="Logit-space bias for ship fraction mean. Positive sends more ships.")
    parser.add_argument("--launch-temperature", type=float, default=1.0, help="Temperature for launch/no-launch logits before bias.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of deterministic argmax/mean.")
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if args.opponent == "regular":
        opponent_factory = lambda: make_rulebase_agent("regular")
    else:
        opponent = nearest_planet_agent if args.opponent == "nearest" else random_policy_agent
        opponent_factory = lambda: opponent
    launch_bias = args.launch_bias + args.aggression
    ship_bias = args.ship_bias + args.aggression
    result = run_matchups(
        lambda: TinyPPOAgent(
            ckpt,
            device=args.device,
            deterministic=not args.stochastic,
            launch_bias=launch_bias,
            ship_bias=ship_bias,
            launch_temperature=args.launch_temperature,
        ),
        opponent_factory,
        games=args.games,
        seed=args.seed,
        episode_steps=args.episode_steps,
        use_numba=not args.no_numba,
    )
    print(result)


if __name__ == "__main__":
    main()
