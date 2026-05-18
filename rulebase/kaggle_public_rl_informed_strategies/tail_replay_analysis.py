"""Replay diagnostics for third-party tail-capture variants."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

os.environ.setdefault("KAGGLE_ENVS_LOG_LEVEL", "ERROR")

from kaggle_environments import make

ROOT = Path("/data2/solo/Orbit-Wars")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulebase.kaggle_public_rl_informed_strategies.multiplayer_eval import make_named_agent
from rulebase.kaggle_public_rl_informed_strategies.rl_informed_agent import RLInformedPublicRuleAgent
from rulebase.kaggle_public_rl_informed_strategies.state import parse_observation
from rulebase.kaggle_public_rl_informed_strategies.strategy_config import ABLATION_SUITES

OUT_DIR = ROOT / "rulebase/kaggle_public_rl_informed_strategies/experiments"


class TailDebugAgent(RLInformedPublicRuleAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tail_events: list[dict] = []
        self._debug_step = 0

    def act(self, obs):
        self._debug_step = int(obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0))
        return super().act(obs)

    def _try_single_attack(self, source, target, local, under_attack, exhausted_planet_ids, moves):
        plan = self._third_party_tail_capture_plan(source, target, local)
        before = len(moves)
        ok = super()._try_single_attack(source, target, local, under_attack, exhausted_planet_ids, moves)
        if ok and plan is not None and len(moves) > before:
            sent = int(moves[-1][2])
            self.tail_events.append(
                {
                    "step": int(local.step),
                    "source": int(source.id),
                    "target": int(target.id),
                    "sent": sent,
                    "source_ships": int(source.ships),
                    "source_after": int(source.ships - sent),
                    "target_owner": int(target.owner),
                    "target_ships": int(target.ships),
                    "target_production": float(target.production),
                    "arrival": int(plan["arrival"]),
                    "delay": int(plan["delay"]),
                    "enemy_arrival": int(plan["enemy_arrival"]),
                    "enemy_post_capture": int(plan["enemy_post_capture"]),
                    "direct_need": int(plan["direct_need"]),
                    "savings": int(plan["direct_need"] - sent),
                    "post_capture_ships": int(plan.get("post_capture_ships", 0)),
                    "score": float(plan["score"]),
                }
            )
        return ok


def make_debug_agent(params: dict) -> tuple[TailDebugAgent, object]:
    instance = TailDebugAgent(**params)

    def agent(obs, configuration=None):
        return instance.act(obs)

    return instance, agent


def owners_by_step(env, seat: int) -> list[dict[int, int]]:
    rows = []
    for step_idx in range(1, len(env.steps)):
        obs = env.steps[step_idx][seat].observation
        local = parse_observation(obs)
        rows.append({p.id: p.owner for p in local.planets})
    return rows


def production_by_planet(env, seat: int) -> dict[int, float]:
    if len(env.steps) <= 1:
        return {}
    local = parse_observation(env.steps[1][seat].observation)
    return {p.id: p.production for p in local.planets}


def final_stats(env, seat: int) -> dict:
    local = parse_observation(env.steps[-1][seat].observation)
    player = local.player
    mine = [p for p in local.planets if p.owner == player]
    return {
        "planets": len(mine),
        "production": sum(p.production for p in mine),
        "ships": sum(p.ships for p in mine) + sum(f.ships for f in local.fleets if f.owner == player),
    }


def annotate_tail_events(env, seat: int, events: list[dict]) -> list[dict]:
    owner_rows = owners_by_step(env, seat)
    prod_by_id = production_by_planet(env, seat)
    if not owner_rows:
        return events
    player = parse_observation(env.steps[1][seat].observation).player
    annotated = []
    for event in events:
        target = event["target"]
        launch = event["step"]
        capture_step = None
        lost_step = None
        for idx, owners in enumerate(owner_rows, start=1):
            if idx < launch:
                continue
            if capture_step is None and owners.get(target) == player:
                capture_step = idx
            elif capture_step is not None and owners.get(target) != player:
                lost_step = idx
                break
        row = dict(event)
        row["target_prod"] = prod_by_id.get(target, row["target_production"])
        row["captured"] = capture_step is not None
        row["capture_step"] = capture_step
        row["lost_step"] = lost_step
        row["held_ticks"] = None if capture_step is None else (len(owner_rows) - capture_step if lost_step is None else lost_step - capture_step)
        row["lost_within_20"] = bool(capture_step is not None and lost_step is not None and lost_step - capture_step <= 20)
        annotated.append(row)
    return annotated


def run_variant_game(seed: int, seat: int, variant: str, opponents: list[str]) -> dict:
    params = None
    for suite in ABLATION_SUITES.values():
        if variant in suite:
            params = dict(suite[variant])
            break
    if params is None:
        raise ValueError(f"Unknown variant: {variant}")
    debug_instance, debug_agent = make_debug_agent(params)
    agents = [make_named_agent(opponents[i % len(opponents)]) for i in range(3)]
    agents.insert(seat, debug_agent)
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run(agents)
    rewards = [env.steps[-1][idx].reward for idx in range(4)]
    reward = rewards[seat]
    win = reward == max(rewards) and reward > 0
    loss = reward < max(rewards)
    return {
        "variant": variant,
        "seed": seed,
        "seat": seat,
        "reward": reward,
        "rewards": rewards,
        "win": win,
        "loss": loss,
        "steps": len(env.steps),
        "final": final_stats(env, seat),
        "tail_events": annotate_tail_events(env, seat, debug_instance.tail_events),
    }


def load_rows(csv_path: Path, variants: list[str]) -> dict[tuple[str, int, int], dict]:
    rows = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["variant"] not in variants:
                continue
            seat = int(row["seat"][1:])
            rows[(row["variant"], int(row["seed"]), seat)] = row
    return rows


def outcome(row: dict | None) -> str:
    if row is None:
        return "missing"
    if row.get("win") in (True, "True"):
        return "win"
    if row.get("loss") in (True, "True"):
        return "loss"
    return "draw"


def summarize(events: list[dict]) -> dict:
    if not events:
        return {
            "events": 0,
            "captured": 0,
            "lost_within_20": 0,
            "avg_savings": None,
            "avg_delay": None,
            "avg_source_after": None,
        }
    return {
        "events": len(events),
        "captured": sum(1 for e in events if e["captured"]),
        "lost_within_20": sum(1 for e in events if e["lost_within_20"]),
        "avg_savings": mean(e["savings"] for e in events),
        "avg_delay": mean(e["delay"] for e in events),
        "avg_source_after": mean(e["source_after"] for e in events),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--variant", default="tail_gated_keep1_saves12")
    parser.add_argument("--opponents", nargs=3, default=["regular", "recapture_s45_e180_w50_b40_regular", "regular"])
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    rows = load_rows(args.csv, ["regular", args.variant])
    keys = sorted({(seed, seat) for variant, seed, seat in rows if variant == args.variant})
    buckets = {
        "variant_win_regular_loss": [],
        "variant_loss_regular_win": [],
        "both_win": [],
        "both_loss": [],
    }
    for seed, seat in keys:
        regular = rows.get(("regular", seed, seat))
        variant = rows.get((args.variant, seed, seat))
        pair = {"seed": seed, "seat": seat, "regular": outcome(regular), "variant": outcome(variant)}
        if pair["variant"] == "win" and pair["regular"] == "loss":
            buckets["variant_win_regular_loss"].append(pair)
        elif pair["variant"] == "loss" and pair["regular"] == "win":
            buckets["variant_loss_regular_win"].append(pair)
        elif pair["variant"] == "win" and pair["regular"] == "win":
            buckets["both_win"].append(pair)
        elif pair["variant"] == "loss" and pair["regular"] == "loss":
            buckets["both_loss"].append(pair)

    selected = []
    for name in ("variant_win_regular_loss", "variant_loss_regular_win", "both_win", "both_loss"):
        selected.extend((name, item) for item in buckets[name][: args.limit])

    games = []
    for bucket, pair in selected:
        game = run_variant_game(pair["seed"], pair["seat"], args.variant, args.opponents)
        game["bucket"] = bucket
        game["regular_outcome"] = pair["regular"]
        games.append(game)

    events_by_bucket = {}
    for bucket in buckets:
        events = [event for game in games if game["bucket"] == bucket for event in game["tail_events"]]
        events_by_bucket[bucket] = summarize(events)

    summary = {
        "variant": args.variant,
        "csv": str(args.csv),
        "bucket_counts": {name: len(items) for name, items in buckets.items()},
        "sampled_games": len(games),
        "events_by_bucket": events_by_bucket,
        "games": games,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"tail_replay_analysis_{args.variant}_{stamp}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "games"}, ensure_ascii=False, indent=2))
    print(f"json: {path}")


if __name__ == "__main__":
    main()
