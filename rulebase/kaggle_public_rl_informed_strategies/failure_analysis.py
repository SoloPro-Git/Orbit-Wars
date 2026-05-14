"""Replay-level diagnostics for champion-vs-candidate losses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KAGGLE_ENVS_LOG_LEVEL", "ERROR")

from kaggle_environments import make

ROOT = Path("/data2/solo/Orbit-Wars")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulebase.kaggle_public_rl_informed_strategies.rl_informed_agent import RLInformedPublicRuleAgent
from rulebase.kaggle_public_rl_informed_strategies.state import parse_observation
from rulebase.kaggle_public_rl_informed_strategies.strategy_config import (
    ABLATION_SUITES,
    CHAMPION_OPPONENT_VARIANTS,
)

OUT_DIR = ROOT / "rulebase/kaggle_public_rl_informed_strategies/experiments"


def resolve_params(name: str) -> dict:
    if name in CHAMPION_OPPONENT_VARIANTS and CHAMPION_OPPONENT_VARIANTS[name] is not None:
        return dict(CHAMPION_OPPONENT_VARIANTS[name])
    for variants in ABLATION_SUITES.values():
        if name in variants:
            return dict(variants[name])
    raise ValueError(f"Unknown config variant: {name}")


def make_agent(params: dict):
    instance = RLInformedPublicRuleAgent(**params)

    def agent(obs, configuration=None):
        return instance.act(obs)

    return agent


def owner_stats(local, player_id: int) -> dict:
    mine = [p for p in local.planets if p.owner == player_id]
    enemy = [p for p in local.planets if p.owner not in (-1, player_id)]
    my_fleets = [f for f in local.fleets if f.owner == player_id]
    enemy_fleets = [f for f in local.fleets if f.owner not in (-1, player_id)]
    return {
        "planets": len(mine),
        "production": sum(p.production for p in mine),
        "planet_ships": sum(p.ships for p in mine),
        "fleet_ships": sum(f.ships for f in my_fleets),
        "total_ships": sum(p.ships for p in mine) + sum(f.ships for f in my_fleets),
        "enemy_planets": len(enemy),
        "enemy_production": sum(p.production for p in enemy),
        "enemy_planet_ships": sum(p.ships for p in enemy),
        "enemy_fleet_ships": sum(f.ships for f in enemy_fleets),
        "enemy_total_ships": sum(p.ships for p in enemy) + sum(f.ships for f in enemy_fleets),
    }


def summarize_timeline(env, variant_position: int) -> dict:
    rows = []
    prev_owners: dict[int, int] | None = None
    variant_player = None

    for step_idx in range(1, len(env.steps)):
        obs = env.steps[step_idx][variant_position].observation
        local = parse_observation(obs)
        if variant_player is None:
            variant_player = local.player

        stats = owner_stats(local, variant_player)
        stats["step"] = step_idx
        stats["prod_diff"] = stats["production"] - stats["enemy_production"]
        stats["planet_diff"] = stats["planets"] - stats["enemy_planets"]
        stats["ship_diff"] = stats["total_ships"] - stats["enemy_total_ships"]

        curr_owners = {p.id: p.owner for p in local.planets}
        stats["lost_planets"] = 0
        stats["lost_high_prod"] = 0
        stats["captured_neutral"] = 0
        stats["captured_neutral_prod"] = 0.0
        stats["enemy_captured_neutral"] = 0
        stats["enemy_captured_neutral_prod"] = 0.0

        if prev_owners is not None:
            by_id = {p.id: p for p in local.planets}
            for pid, owner in curr_owners.items():
                prev_owner = prev_owners.get(pid)
                if prev_owner is None or prev_owner == owner:
                    continue
                planet = by_id[pid]
                if prev_owner == variant_player and owner != variant_player:
                    stats["lost_planets"] += 1
                    if planet.production >= 3.0:
                        stats["lost_high_prod"] += 1
                elif prev_owner == -1 and owner == variant_player:
                    stats["captured_neutral"] += 1
                    stats["captured_neutral_prod"] += planet.production
                elif prev_owner == -1 and owner != -1:
                    stats["enemy_captured_neutral"] += 1
                    stats["enemy_captured_neutral_prod"] += planet.production

        prev_owners = curr_owners
        rows.append(stats)

    first_prod_deficit = next((r["step"] for r in rows if r["prod_diff"] <= -3), None)
    first_planet_deficit = next((r["step"] for r in rows if r["planet_diff"] <= -1), None)
    first_ship_deficit = next((r["step"] for r in rows if r["ship_diff"] <= -25), None)
    high_prod_loss_steps = [r["step"] for r in rows if r["lost_high_prod"]]

    return {
        "final": rows[-1] if rows else {},
        "first_prod_deficit": first_prod_deficit,
        "first_planet_deficit": first_planet_deficit,
        "first_ship_deficit": first_ship_deficit,
        "high_prod_loss_steps": high_prod_loss_steps,
        "early_neutral_prod": sum(r["captured_neutral_prod"] for r in rows if r["step"] <= 60),
        "enemy_early_neutral_prod": sum(r["enemy_captured_neutral_prod"] for r in rows if r["step"] <= 60),
        "total_lost_planets": sum(r["lost_planets"] for r in rows),
        "total_lost_high_prod": sum(r["lost_high_prod"] for r in rows),
        "timeline": rows,
    }


def run_game(seed: int, variant_as_p0: bool, variant_params: dict, opponent_params: dict) -> dict:
    variant = make_agent(variant_params)
    opponent = make_agent(opponent_params)
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)

    if variant_as_p0:
        env.run([variant, opponent])
        variant_position = 0
        variant_reward = env.steps[-1][0].reward
        opponent_reward = env.steps[-1][1].reward
        seat = "variant_p0"
    else:
        env.run([opponent, variant])
        variant_position = 1
        opponent_reward = env.steps[-1][0].reward
        variant_reward = env.steps[-1][1].reward
        seat = "variant_p1"

    if variant_reward > opponent_reward:
        outcome = "win"
    elif variant_reward < opponent_reward:
        outcome = "loss"
    else:
        outcome = "draw"

    return {
        "seed": seed,
        "seat": seat,
        "outcome": outcome,
        "variant_reward": variant_reward,
        "opponent_reward": opponent_reward,
        "steps": len(env.steps),
        "diagnostics": summarize_timeline(env, variant_position),
    }


def load_loss_tasks(results_json: Path, variant: str) -> list[tuple[int, bool]]:
    payload = json.loads(results_json.read_text())
    tasks = []
    for row in payload["games"]:
        if row["variant"] == variant and row["outcome"] == "loss":
            tasks.append((int(row["seed"]), row["seat"] == "variant_p0"))
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--opponent", default="regular")
    parser.add_argument("--results-json", type=Path)
    parser.add_argument("--seeds", nargs="*", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant_params = resolve_params(args.variant)
    opponent_params = resolve_params(args.opponent)

    if args.results_json:
        tasks = load_loss_tasks(args.results_json, args.variant)
    else:
        seeds = args.seeds or [42]
        tasks = [(seed, True) for seed in seeds]

    rows = [
        run_game(seed, variant_as_p0, variant_params, opponent_params)
        for seed, variant_as_p0 in tasks
    ]
    summary = {
        "variant": args.variant,
        "opponent": args.opponent,
        "games": len(rows),
        "outcomes": {name: sum(1 for row in rows if row["outcome"] == name) for name in ["win", "loss", "draw"]},
        "avg_early_neutral_prod": sum(row["diagnostics"]["early_neutral_prod"] for row in rows) / max(1, len(rows)),
        "avg_enemy_early_neutral_prod": sum(row["diagnostics"]["enemy_early_neutral_prod"] for row in rows) / max(1, len(rows)),
        "avg_total_lost_high_prod": sum(row["diagnostics"]["total_lost_high_prod"] for row in rows) / max(1, len(rows)),
        "rows": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"failure_analysis_{args.variant}_{stamp}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"json: {path}")


if __name__ == "__main__":
    main()
