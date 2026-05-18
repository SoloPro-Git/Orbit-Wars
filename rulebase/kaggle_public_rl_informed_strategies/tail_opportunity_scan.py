"""Scan regular games for third-party tail-capture opportunities."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from rulebase.kaggle_public_rl_informed_strategies.strategy_config import CHAMPION_OPPONENT_VARIANTS, REGULAR_CONFIG

OUT_DIR = ROOT / "rulebase/kaggle_public_rl_informed_strategies/experiments"


SCAN_PARAMS = {
    **REGULAR_CONFIG.to_agent_kwargs(),
    "enable_third_party_tail_capture": True,
    "enable_third_party_tail_hold_filter": True,
    "third_party_tail_min_active_players": 3,
    "third_party_tail_min_production": 3.0,
    "third_party_tail_max_enemy_arrival": 75,
    "third_party_tail_min_delay": 1,
    "third_party_tail_max_delay": 18,
    "third_party_tail_margin": 1,
    "third_party_tail_min_send": 1,
    "third_party_tail_max_ships": 28,
    "third_party_tail_source_min_after": 8,
    "third_party_tail_min_savings": 0,
    "third_party_tail_min_savings_ratio": 0.0,
    "third_party_tail_min_post_capture_ships": 0,
    "third_party_tail_neutral_max_arrival": 999,
    "third_party_tail_neutral_min_post_capture_ships": 0,
    "third_party_tail_neutral_min_enemy_post_capture": 0,
    "third_party_tail_roi_multiplier": 1.0,
    "third_party_tail_min_net_value": 0.0,
    "third_party_tail_hold_enemy_radius": 45.0,
    "third_party_tail_hold_enemy_max_arrival": 40,
    "third_party_tail_hold_margin": 3,
}


def make_regular_agent():
    params = CHAMPION_OPPONENT_VARIANTS["regular"]
    instance = RLInformedPublicRuleAgent(**dict(params))

    def agent(obs, configuration=None):
        return instance.act(obs)

    return agent


def result_owner_at(env, seat: int, planet_id: int, from_step: int, owner: int) -> dict:
    capture_step = None
    lost_step = None
    final_owner = None
    for idx in range(max(1, from_step), len(env.steps)):
        local = parse_observation(env.steps[idx][seat].observation)
        by_id = {p.id: p.owner for p in local.planets}
        final_owner = by_id.get(planet_id)
        if capture_step is None and final_owner == owner:
            capture_step = idx
        elif capture_step is not None and final_owner != owner:
            lost_step = idx
            break
    return {
        "captured_by_us_later": capture_step is not None,
        "captured_by_us_step": capture_step,
        "lost_after_capture_step": lost_step,
        "final_owner": final_owner,
    }


def nearest_owned_distance(local, target) -> float | None:
    if not local.mine:
        return None
    from rulebase.kaggle_public_rl_informed_strategies.geometry import distance

    return min(distance(source, target) for source in local.mine)


def fleet_first_capture_target(scanner: RLInformedPublicRuleAgent, fleet, local):
    best = None
    for target in local.planets:
        if target.owner == fleet.owner:
            continue
        arrival = scanner._fleet_arrival_to_target(fleet, target, local)
        if arrival is None:
            continue
        if best is None or arrival < best[0]:
            best = (arrival, target)
    if best is None:
        return None

    arrival, target = best
    target_defense = int(target.ships)
    if target.owner != -1:
        target_defense += int(target.production * arrival)
    if fleet.ships <= target_defense:
        return None
    return target, int(arrival), int(fleet.ships - target_defense), int(target_defense + 1)


def append_event(
    events: list[dict],
    seed: int,
    seat: int,
    step: int,
    source,
    target,
    plan: dict,
    direct_need: int,
    enemy_arrival: int,
    enemy_post_capture: int,
    result: dict,
    mode: str,
    fleet_id: int | None = None,
    nearest_distance: float | None = None,
) -> None:
    events.append(
        {
            "seed": seed,
            "seat": seat,
            "step": step,
            "mode": mode,
            "fleet_id": fleet_id,
            "source": source.id,
            "target": target.id,
            "target_owner": target.owner,
            "target_ships": target.ships,
            "target_prod": target.production,
            "source_ships": source.ships,
            "source_after": source.ships - int(plan["ships"]),
            "ships": int(plan["ships"]),
            "arrival": int(plan["arrival"]),
            "delay": int(plan["delay"]),
            "enemy_arrival": int(enemy_arrival),
            "enemy_post_capture": int(enemy_post_capture),
            "direct_need": int(direct_need),
            "savings": int(direct_need - plan["ships"]),
            "savings_ratio": (float(direct_need - plan["ships"]) / max(1.0, float(direct_need))),
            "post_capture_ships": int(plan.get("post_capture_ships", 0)),
            "nearest_owned_distance": nearest_distance,
            **result,
        }
    )


def scan_game(seed: int, seat: int, opponents: list[str], max_events: int) -> dict:
    agents = [make_named_agent(opponents[i % len(opponents)]) for i in range(3)]
    agents.insert(seat, make_regular_agent())
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run(agents)

    scanner = RLInformedPublicRuleAgent(**SCAN_PARAMS)
    events = []
    seen = set()
    for step in range(1, len(env.steps) - 1):
        obs = env.steps[step][seat].observation
        local = parse_observation(obs)
        scanner.act(obs)
        for source in local.mine:
            for target in local.targets:
                plan = scanner._third_party_tail_capture_plan(source, target, local)
                if plan is None:
                    continue
                key = (target.id, int(plan["enemy_arrival"]), int(plan["arrival"]))
                if key in seen:
                    continue
                seen.add(key)
                result = result_owner_at(env, seat, target.id, step, local.player)
                append_event(
                    events,
                    seed,
                    seat,
                    step,
                    source,
                    target,
                    plan,
                    int(plan["direct_need"]),
                    int(plan["enemy_arrival"]),
                    int(plan["enemy_post_capture"]),
                    result,
                    "target_scan",
                    nearest_distance=nearest_owned_distance(local, target),
                )
                if len(events) >= max_events:
                    break
            if len(events) >= max_events:
                break
        if len(events) >= max_events:
            break

        for fleet in local.fleets:
            if fleet.owner in (-1, local.player) or fleet.ships <= 0:
                continue
            hit = fleet_first_capture_target(scanner, fleet, local)
            if hit is None:
                continue
            target, enemy_arrival, enemy_post_capture, direct_need = hit
            if target.owner == local.player:
                continue
            for source in local.mine:
                plan = scanner._tail_capture_ships_for_enemy_capture(
                    source,
                    target,
                    local,
                    enemy_arrival,
                    enemy_post_capture,
                )
                if plan is None:
                    continue
                key = ("fleet", fleet.id, source.id, target.id, enemy_arrival, int(plan["arrival"]))
                if key in seen:
                    continue
                seen.add(key)
                result = result_owner_at(env, seat, target.id, step, local.player)
                append_event(
                    events,
                    seed,
                    seat,
                    step,
                    source,
                    target,
                    plan,
                    direct_need,
                    enemy_arrival,
                    enemy_post_capture,
                    result,
                    "fleet_scan",
                    int(fleet.id),
                    nearest_distance=nearest_owned_distance(local, target),
                )
                if len(events) >= max_events:
                    break
            if len(events) >= max_events:
                break

    rewards = [env.steps[-1][idx].reward for idx in range(4)]
    return {"seed": seed, "seat": seat, "reward": rewards[seat], "rewards": rewards, "events": events}


def scan_task(task: tuple[int, int, list[str], int]) -> dict:
    return scan_game(*task)


def bucket(events: list[dict], predicate) -> dict:
    rows = [e for e in events if predicate(e)]
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "avg_prod": mean(e["target_prod"] for e in rows),
        "avg_arrival": mean(e["arrival"] for e in rows),
        "avg_savings": mean(e["savings"] for e in rows),
        "avg_savings_ratio": mean(e["savings_ratio"] for e in rows),
        "avg_post_capture": mean(e["post_capture_ships"] for e in rows),
        "neutral": sum(1 for e in rows if e["target_owner"] == -1),
        "eventually_captured_by_us": sum(1 for e in rows if e["captured_by_us_later"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-events-per-game", type=int, default=12)
    parser.add_argument("--opponents", nargs=3, default=["regular", "recapture_s45_e180_w50_b40_regular", "regular"])
    args = parser.parse_args()

    tasks = [
        (9000 + seed_idx, seat, args.opponents, args.max_events_per_game)
        for seed_idx in range(args.seeds)
        for seat in range(4)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(scan_task, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[{idx}/{len(futures)}] seed={row['seed']} seat={row['seat']} events={len(row['events'])}", flush=True)

    events = [event for row in rows for event in row["events"]]
    summary = {
        "games": len(rows),
        "events": len(events),
        "all": bucket(events, lambda e: True),
        "good_shape": bucket(
            events,
            lambda e: e["target_prod"] >= 4.0
            and e["arrival"] <= 14
            and e["post_capture_ships"] >= 6
            and e["savings"] >= 10
            and e["savings_ratio"] >= 0.4,
        ),
        "bad_neutral_slow": bucket(events, lambda e: e["target_owner"] == -1 and e["arrival"] > 14),
        "bad_low_garrison": bucket(events, lambda e: e["post_capture_ships"] < 4),
        "high_savings_enemy_owned": bucket(
            events,
            lambda e: e["target_owner"] != -1
            and e["target_prod"] >= 4.0
            and e["savings"] >= 12
            and e["post_capture_ships"] >= 4,
        ),
        "top_events": sorted(
            events,
            key=lambda e: (
                e["target_prod"],
                e["savings"],
                e["post_capture_ships"],
                -e["arrival"],
            ),
            reverse=True,
        )[:30],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"tail_opportunity_scan_{stamp}.json"
    path.write_text(json.dumps({"summary": summary, "games": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "top_events"}, ensure_ascii=False, indent=2))
    print(f"json: {path}")


if __name__ == "__main__":
    main()
