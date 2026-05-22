from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from training2 import make_fast_orbit_wars

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
    parser.add_argument("--opponent", choices=["nearest", "random"], default="nearest")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    opponent = nearest_planet_agent if args.opponent == "nearest" else random_policy_agent
    result = run_matchups(
        lambda: TinyPPOAgent(ckpt, device=args.device, deterministic=True),
        lambda: opponent,
        games=args.games,
        seed=args.seed,
        episode_steps=args.episode_steps,
        use_numba=not args.no_numba,
    )
    print(result)


if __name__ == "__main__":
    main()
