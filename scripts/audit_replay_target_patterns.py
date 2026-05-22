from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.expert.action_labeling import infer_target_planet_id


def dist(a: list[Any], b: list[Any]) -> float:
    return math.hypot(float(a[2]) - float(b[2]), float(a[3]) - float(b[3]))


def by_id(planets: list[list[Any]]) -> dict[int, list[Any]]:
    return {int(p[0]): p for p in planets}


def fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    ratio = max(0.0, min(1.0, math.log(max(ships, 1.0)) / math.log(1000.0)))
    return 1.0 + 5.0 * (ratio**1.5)


def load_frames(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    frames: list[dict[str, Any]] = []
    for step_row in data.get("steps", []):
        if not step_row:
            continue
        obs = step_row[0].get("observation")
        if obs:
            frames.append(obs)
    return data, frames


def moving_planets(frames: list[dict[str, Any]]) -> set[int]:
    if not frames:
        return set()
    first = by_id(frames[0].get("planets", []))
    last = by_id(frames[min(len(frames) - 1, 60)].get("planets", []))
    out = set()
    for pid, p0 in first.items():
        p1 = last.get(pid)
        if p1 is not None and dist(p0, p1) > 0.05:
            out.add(pid)
    return out


def prod(obs: dict[str, Any], player: int) -> float:
    return sum(float(p[6]) for p in obs.get("planets", []) if int(p[1]) == player)


def ships(obs: dict[str, Any], player: int) -> float:
    return sum(float(p[5]) for p in obs.get("planets", []) if int(p[1]) == player) + sum(
        float(f[6]) for f in obs.get("fleets", []) if int(f[1]) == player
    )


def planet_count(obs: dict[str, Any], player: int) -> int:
    return sum(1 for p in obs.get("planets", []) if int(p[1]) == player)


def nearest_frame(frames: list[dict[str, Any]], step: int) -> dict[str, Any]:
    return min(frames, key=lambda obs: abs(int(obs.get("step", 0)) - step))


@dataclass
class ActionAudit:
    step: int
    player: int
    target_owner: int
    target_prod: float
    target_ships: float
    eta: float
    moving_target: bool
    low_connector: bool
    own_front_neighbor: bool
    enemy_front_neighbor: bool


def target_flags(obs: dict[str, Any], player: int, target: list[Any]) -> tuple[bool, bool, bool]:
    own_front = False
    enemy_front = False
    high_anchor = False
    for planet in obs.get("planets", []):
        if int(planet[0]) == int(target[0]):
            continue
        d = dist(planet, target)
        owner = int(planet[1])
        if owner == player and d <= 42.0:
            own_front = True
        if owner not in (-1, player) and d <= 42.0:
            enemy_front = True
        if owner != -1 and float(planet[6]) >= 3.0 and d <= 55.0:
            high_anchor = True
    low_connector = float(target[6]) <= 2.0 and own_front and (enemy_front or high_anchor)
    return low_connector, own_front, enemy_front


def build_actions(data: dict[str, Any], frames: list[dict[str, Any]], player: int, max_step: int) -> list[ActionAudit]:
    moving = moving_planets(frames)
    out: list[ActionAudit] = []
    for step_row in data.get("steps", []):
        if player >= len(step_row):
            continue
        agent = step_row[player]
        obs = agent.get("observation") or {}
        step = int(obs.get("step", 0))
        if step > max_step:
            continue
        actions = agent.get("action") or []
        planets = by_id(obs.get("planets", []))
        for raw in actions:
            if len(raw) < 3:
                continue
            source = planets.get(int(raw[0]))
            target_id = infer_target_planet_id(obs, int(raw[0]), float(raw[1]), int(raw[2]))
            target = planets.get(target_id) if target_id is not None else None
            if source is None or target is None:
                continue
            eta = dist(source, target) / max(fleet_speed(float(raw[2])), 1e-6)
            low_connector, own_front, enemy_front = target_flags(obs, player, target)
            out.append(
                ActionAudit(
                    step=step,
                    player=player,
                    target_owner=int(target[1]),
                    target_prod=float(target[6]),
                    target_ships=float(target[5]),
                    eta=eta,
                    moving_target=int(target[0]) in moving,
                    low_connector=low_connector,
                    own_front_neighbor=own_front,
                    enemy_front_neighbor=enemy_front,
                )
            )
    return out


def summarize_actions(actions: list[ActionAudit], player: int) -> dict[str, Any]:
    targeted = [a for a in actions if a.target_owner != player]
    low = [a for a in targeted if a.target_prod <= 2.0]
    low_connector = [a for a in targeted if a.low_connector]
    moving_short = [a for a in targeted if a.moving_target and a.eta <= 18.0]
    enemy = [a for a in targeted if a.target_owner not in (-1, player)]
    high = [a for a in targeted if a.target_prod >= 3.0]
    return {
        "launches": len(actions),
        "targets": len(targeted),
        "enemy_targets": len(enemy),
        "high_prod_targets": len(high),
        "low_prod_targets": len(low),
        "low_connector_targets": len(low_connector),
        "moving_short_targets": len(moving_short),
        "avg_eta": round(mean(a.eta for a in targeted), 2) if targeted else None,
        "low_connector_rate": round(len(low_connector) / len(targeted), 3) if targeted else 0.0,
        "moving_short_rate": round(len(moving_short) / len(targeted), 3) if targeted else 0.0,
    }


def replay_roles(data: dict[str, Any]) -> tuple[list[str], int | None, int]:
    names = [str(n) for n in data.get("info", {}).get("TeamNames", [])]
    rewards = [float(x) for x in data.get("rewards", [])]
    winner = max(range(len(names)), key=lambda i: rewards[i] if i < len(rewards) else -999.0)
    solo = next((i for i, name in enumerate(names) if name.lower() == "solo"), None)
    return names, solo, winner


def summarize_replay(path: Path, max_step: int) -> dict[str, Any]:
    data, frames = load_frames(path)
    names, solo, winner = replay_roles(data)
    focal_players: list[tuple[str, int]] = [("winner", winner)]
    if solo is not None:
        focal_players.append(("solo", solo))
    if any("vadasz" in name.lower() for name in names):
        focal_players.extend(("vadasz", i) for i, name in enumerate(names) if "vadasz" in name.lower())

    frame50 = nearest_frame(frames, 50) if frames else {}
    frame80 = nearest_frame(frames, 80) if frames else {}
    rows = []
    seen = set()
    for role, player in focal_players:
        if player in seen:
            continue
        seen.add(player)
        actions = build_actions(data, frames, player, max_step)
        summary = summarize_actions(actions, player)
        rows.append(
            {
                "path": str(path),
                "players": len(names),
                "role": role,
                "player": player,
                "name": names[player] if player < len(names) else str(player),
                "winner": winner,
                "result": "win" if player == winner else "loss",
                "p50_prod": prod(frame50, player) if frame50 else None,
                "p80_prod": prod(frame80, player) if frame80 else None,
                "p80_planets": planet_count(frame80, player) if frame80 else None,
                "p80_ships": round(ships(frame80, player), 1) if frame80 else None,
                **summary,
            }
        )
    return {"rows": rows}


def aggregate(rows: list[dict[str, Any]], role: str | None = None, players: int | None = None) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if (role is None or row["role"] == role) and (players is None or row["players"] == players)
    ]
    if not selected:
        return {"count": 0}
    totals = Counter()
    weighted_rates = defaultdict(float)
    for row in selected:
        for key in ["launches", "targets", "enemy_targets", "high_prod_targets", "low_prod_targets", "low_connector_targets", "moving_short_targets"]:
            totals[key] += int(row[key])
        for key in ["p50_prod", "p80_prod", "p80_planets", "p80_ships"]:
            if row[key] is not None:
                weighted_rates[key] += float(row[key])
    targets = max(1, totals["targets"])
    return {
        "count": len(selected),
        "launches": totals["launches"],
        "targets": totals["targets"],
        "enemy_target_rate": round(totals["enemy_targets"] / targets, 3),
        "high_prod_target_rate": round(totals["high_prod_targets"] / targets, 3),
        "low_prod_target_rate": round(totals["low_prod_targets"] / targets, 3),
        "low_connector_rate": round(totals["low_connector_targets"] / targets, 3),
        "moving_short_rate": round(totals["moving_short_targets"] / targets, 3),
        "avg_p50_prod": round(weighted_rates["p50_prod"] / len(selected), 2),
        "avg_p80_prod": round(weighted_rates["p80_prod"] / len(selected), 2),
        "avg_p80_planets": round(weighted_rates["p80_planets"] / len(selected), 2),
        "avg_p80_ships": round(weighted_rates["p80_ships"] / len(selected), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-step", type=int, default=120)
    parser.add_argument("--top", type=int, default=16)
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)

    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            rows.extend(summarize_replay(path, args.max_step)["rows"])
        except Exception as exc:  # noqa: BLE001 - audit should keep going.
            print(f"WARN {path}: {exc}", file=sys.stderr)

    print("AGGREGATE")
    for players in [2, 4, None]:
        for role in ["winner", "solo", "vadasz"]:
            agg = aggregate(rows, role=role, players=players)
            if agg["count"]:
                label = f"{role} players={players or 'all'}"
                print(label, json.dumps(agg, ensure_ascii=False, sort_keys=True))

    print("\nTOP LOW-CONNECTOR USERS")
    ranked = sorted(rows, key=lambda r: (r["low_connector_rate"], r["low_connector_targets"]), reverse=True)
    for row in ranked[: args.top]:
        print(
            f"{row['role']:6s} {row['players']}p {row['name'][:24]:24s} "
            f"lc={row['low_connector_targets']}/{row['targets']} rate={row['low_connector_rate']} "
            f"move_short={row['moving_short_targets']} p80={row['p80_planets']}/{row['p80_prod']} "
            f"{row['path']}"
        )


if __name__ == "__main__":
    main()
