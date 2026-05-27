from __future__ import annotations

import argparse
import json
from pathlib import Path

from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent

from tinyPPO.agents import TinyPPOAgent
from tinyPPO.bridge_agents import RegularSourceBridgeAgent, RegularSourceDropGateAgent, aggregate_bridge_stats, parse_float_list, parse_int_list
from tinyPPO.features import score


def _run_agent_pair(agent_factory, opponent_factory, games: int, seed: int, episode_steps: int, use_numba: bool, progress: bool, desc: str) -> dict[str, float]:
    iterator = range(games)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=games, desc=desc, dynamic_ncols=True, leave=False)
        except Exception:
            pass

    wins = losses = draws = 0
    agents_seen: list[RegularSourceBridgeAgent | RegularSourceDropGateAgent] = []
    rewards: list[float] = []
    for i in iterator:
        model_seat = i % 2
        agents = [opponent_factory(), opponent_factory()]
        model_agent = agent_factory()
        if isinstance(model_agent, (RegularSourceBridgeAgent, RegularSourceDropGateAgent)):
            agents_seen.append(model_agent)
        agents[model_seat] = model_agent
        env = make_fast_orbit_wars({"episodeSteps": episode_steps, "seed": seed + i}, keep_history=False, use_numba=use_numba)
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

    result = {
        "games": float(games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "winrate": wins / max(1, games),
        "nonloss": (wins + draws) / max(1, games),
        "mean_reward": sum(rewards) / max(1, len(rewards)),
    }
    if agents_seen:
        result.update({f"bridge_{key}": value for key, value in aggregate_bridge_stats(agents_seen).items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=930000)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thresholds", default="0.2,0.3,0.4")
    parser.add_argument("--apply-probs", default="0.0,0.02,0.05,0.1")
    parser.add_argument("--max-source-drops-list", default="-1,1,2")
    parser.add_argument("--max-drop-frac", type=float, default=1.0)
    parser.add_argument("--min-keep-actions", type=int, default=1)
    parser.add_argument("--min-anchor-actions-to-filter", type=int, default=1)
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--launch-temperature", type=float, default=1.0)
    parser.add_argument("--reduce", choices=["noisy_or", "max", "mean"], default="noisy_or")
    parser.add_argument("--include-pure", action="store_true")
    parser.add_argument("--include-drop-gate", action="store_true")
    parser.add_argument("--drop-gate-biases", default="2.0")
    parser.add_argument("--drop-gate-deterministic", action="store_true")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    opponent_factory = lambda: make_rulebase_agent("regular")
    rows: list[dict[str, float | str | int | None]] = []

    anchor = _run_agent_pair(
        lambda: make_rulebase_agent("regular"),
        opponent_factory,
        games=args.games,
        seed=args.seed,
        episode_steps=args.episode_steps,
        use_numba=not args.no_numba,
        progress=args.progress,
        desc="anchor_regular",
    )
    anchor.update({"variant": "regular_anchor", "threshold": None, "apply_prob": 0.0, "max_source_drops": None})
    rows.append(anchor)

    if args.include_pure:
        pure = _run_agent_pair(
            lambda: TinyPPOAgent(ckpt, device=args.device, deterministic=False),
            opponent_factory,
            games=args.games,
            seed=args.seed,
            episode_steps=args.episode_steps,
            use_numba=not args.no_numba,
            progress=args.progress,
            desc="pure_tinyppo",
        )
        pure.update({"variant": "pure_tinyppo_stochastic", "threshold": None, "apply_prob": 1.0, "max_source_drops": None})
        rows.append(pure)

    for apply_prob in parse_float_list(args.apply_probs):
        for threshold in parse_float_list(args.thresholds):
            for max_drops in parse_int_list(args.max_source_drops_list):
                result = _run_agent_pair(
                    lambda ap=apply_prob, th=threshold, md=max_drops: RegularSourceBridgeAgent(
                        ckpt,
                        device=args.device,
                        threshold=th,
                        apply_prob=ap,
                        max_source_drops=md,
                        max_drop_frac=args.max_drop_frac,
                        min_keep_actions=args.min_keep_actions,
                        min_anchor_actions_to_filter=args.min_anchor_actions_to_filter,
                        launch_bias=args.launch_bias,
                        launch_temperature=args.launch_temperature,
                        reduce=args.reduce,
                    ),
                    opponent_factory,
                    games=args.games,
                    seed=args.seed,
                    episode_steps=args.episode_steps,
                    use_numba=not args.no_numba,
                    progress=args.progress,
                    desc=f"bridge_p{apply_prob:g}_t{threshold:g}_d{max_drops if max_drops is not None else 'all'}",
                )
                result.update(
                    {
                        "variant": "regular_source_bridge",
                        "threshold": threshold,
                        "apply_prob": apply_prob,
                        "max_source_drops": max_drops,
                    }
                )
                rows.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)

    if args.include_drop_gate:
        for bias in parse_float_list(args.drop_gate_biases):
            result = _run_agent_pair(
                lambda b=bias: RegularSourceDropGateAgent(
                    ckpt,
                    device=args.device,
                    no_drop_bias=b,
                    min_anchor_actions_to_filter=args.min_anchor_actions_to_filter,
                    deterministic=args.drop_gate_deterministic,
                ),
                opponent_factory,
                games=args.games,
                seed=args.seed,
                episode_steps=args.episode_steps,
                use_numba=not args.no_numba,
                progress=args.progress,
                desc=f"drop_gate_b{bias:g}",
            )
            result.update({"variant": "regular_source_drop_gate", "threshold": None, "apply_prob": 1.0, "max_source_drops": 1, "no_drop_bias": bias})
            rows.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    rows.sort(key=lambda row: (float(row["nonloss"]), float(row["winrate"]), float(row["mean_reward"])), reverse=True)
    print("\nTop variants:")
    for row in rows[:10]:
        print(
            f"{row['variant']} p={row['apply_prob']} th={row['threshold']} drops={row['max_source_drops']} "
            f"W/L/D={int(row['wins'])}/{int(row['losses'])}/{int(row['draws'])} "
            f"wr={float(row['winrate']):.3f} nonloss={float(row['nonloss']):.3f} "
            f"kept={float(row.get('bridge_kept_frac', 1.0)):.3f} attempted={float(row.get('bridge_attempt_frac', 0.0)):.3f}"
        )


if __name__ == "__main__":
    main()
