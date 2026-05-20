"""Fast simulator diagnostics for Vadasz-inspired ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path("/data2/solo/Orbit-Wars")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulebase.kaggle_public_rl_informed_strategies.rl_informed_agent import RLInformedPublicRuleAgent
from rulebase.kaggle_public_rl_informed_strategies.state import LocalObs, Planet, parse_observation
from rulebase.kaggle_public_rl_informed_strategies.strategy_config import ABLATION_SUITES, CHAMPION_OPPONENT_VARIANTS
from training2.fast_orbit_wars import make_fast_orbit_wars

OUT_DIR = ROOT / "rulebase/kaggle_public_rl_informed_strategies/experiments"


def _phase(step: int) -> str:
    if step <= 50:
        return "p0_50"
    if step <= 80:
        return "p51_80"
    if step <= 120:
        return "p81_120"
    return "p121_plus"


def _owner_stats(local: LocalObs, player: int) -> dict[str, float]:
    mine = [p for p in local.planets if p.owner == player]
    enemy = [p for p in local.planets if p.owner not in (-1, player)]
    my_fleets = [f for f in local.fleets if f.owner == player]
    enemy_fleets = [f for f in local.fleets if f.owner not in (-1, player)]
    my_ships = sum(p.ships for p in mine) + sum(f.ships for f in my_fleets)
    enemy_ships = sum(p.ships for p in enemy) + sum(f.ships for f in enemy_fleets)
    return {
        "planets": float(len(mine)),
        "prod": float(sum(p.production for p in mine)),
        "ships": float(my_ships),
        "enemy_planets": float(len(enemy)),
        "enemy_prod": float(sum(p.production for p in enemy)),
        "enemy_ships": float(enemy_ships),
        "planet_diff": float(len(mine) - len(enemy)),
        "prod_diff": float(sum(p.production for p in mine) - sum(p.production for p in enemy)),
        "ship_diff": float(my_ships - enemy_ships),
    }


def _empty_diag() -> dict[str, object]:
    return {
        "launches": defaultdict(int),
        "launch_ships": defaultdict(int),
        "attack_launches": defaultdict(int),
        "enemy_attack_launches": defaultdict(int),
        "enemy_highprod_attack_launches": defaultdict(int),
        "neutral_highprod_attack_launches": defaultdict(int),
        "first_enemy_highprod_attack_step": None,
        "highprod_captures": 0,
        "highprod_capture_activated_5": 0,
        "highprod_capture_activated_10": 0,
        "highprod_capture_activated_20": 0,
        "highprod_capture_supported_10": 0,
        "lost_highprod": 0,
        "checkpoints": {},
        "capture_seed_moves": 0,
        "capture_seed_turns": 0,
        "enemy_hp_pressure_positive": 0,
        "enemy_hp_pressure_score_sum": 0.0,
        "contested_stoploss_blocks": 0,
        "contested_stoploss_penalty_positive": 0,
        "final_step": 0,
    }


@dataclass(slots=True)
class InstrumentedAgent(RLInformedPublicRuleAgent):
    diag: dict[str, object] = field(default_factory=_empty_diag)
    previous_owners_for_diag: dict[int, int] = field(default_factory=dict)
    highprod_capture_steps: dict[int, int] = field(default_factory=dict)
    highprod_activation_delay: dict[int, int] = field(default_factory=dict)
    highprod_support_delay: dict[int, int] = field(default_factory=dict)
    current_step_for_diag: int = 0
    current_player_for_diag: int = -1
    reinf_len_before_diag: int = 0

    def act(self, obs) -> list[list[float | int]]:
        local = parse_observation(obs)
        self.current_step_for_diag = local.step
        self.current_player_for_diag = local.player
        self.reinf_len_before_diag = len(self.reinforcement_trajectories)
        self._record_state(local)

        moves = RLInformedPublicRuleAgent.act(self, obs)

        phase = _phase(local.step)
        self.diag["launches"][phase] += len(moves)
        self.diag["launch_ships"][phase] += sum(int(move[2]) for move in moves)
        self.diag["final_step"] = local.step
        self._record_recent_capture_support(local)
        return moves

    def _record_state(self, local: LocalObs) -> None:
        stats = _owner_stats(local, local.player)
        checkpoints = self.diag["checkpoints"]
        for step in (50, 80, 120):
            if local.step >= step and step not in checkpoints:
                checkpoints[step] = stats

        by_id = {p.id: p for p in local.planets}
        if self.previous_owners_for_diag:
            for pid, planet in by_id.items():
                previous = self.previous_owners_for_diag.get(pid)
                if previous is None or previous == planet.owner:
                    continue
                if previous != local.player and planet.owner == local.player and planet.production >= 3.0:
                    self.diag["highprod_captures"] += 1
                    self.highprod_capture_steps[pid] = local.step
                if previous == local.player and planet.owner != local.player and planet.production >= 3.0:
                    self.diag["lost_highprod"] += 1
        self.previous_owners_for_diag = {p.id: p.owner for p in local.planets}

    def _record_recent_capture_support(self, local: LocalObs) -> None:
        for row in self.reinforcement_trajectories[self.reinf_len_before_diag :]:
            target = row["target"]
            captured_step = self.highprod_capture_steps.get(target.id)
            if captured_step is None or target.id in self.highprod_support_delay:
                continue
            delay = local.step - captured_step
            if 0 <= delay <= 10:
                self.highprod_support_delay[target.id] = delay
                self.diag["highprod_capture_supported_10"] += 1

    def _track_attack(self, source: Planet, target: Planet, angle: float, ships: int, arrive_tick: int) -> None:
        phase = _phase(self.current_step_for_diag)
        self.diag["attack_launches"][phase] += 1
        if target.owner not in (-1, self.current_player_for_diag):
            self.diag["enemy_attack_launches"][phase] += 1
            if target.production >= 3.0:
                self.diag["enemy_highprod_attack_launches"][phase] += 1
                if self.diag["first_enemy_highprod_attack_step"] is None:
                    self.diag["first_enemy_highprod_attack_step"] = self.current_step_for_diag
        elif target.owner == -1 and target.production >= 3.0:
            self.diag["neutral_highprod_attack_launches"][phase] += 1

        captured_step = self.highprod_capture_steps.get(source.id)
        if captured_step is not None and source.id not in self.highprod_activation_delay:
            delay = self.current_step_for_diag - captured_step
            if delay >= 0:
                self.highprod_activation_delay[source.id] = delay
                if delay <= 5:
                    self.diag["highprod_capture_activated_5"] += 1
                if delay <= 10:
                    self.diag["highprod_capture_activated_10"] += 1
                if delay <= 20:
                    self.diag["highprod_capture_activated_20"] += 1

        RLInformedPublicRuleAgent._track_attack(self, source, target, angle, ships, arrive_tick)

    def _append_high_prod_capture_seed(self, local, under_attack, exhausted_planet_ids, moves, attack_count_before) -> None:
        before = len(moves)
        RLInformedPublicRuleAgent._append_high_prod_capture_seed(
            self,
            local,
            under_attack,
            exhausted_planet_ids,
            moves,
            attack_count_before,
        )
        delta = len(moves) - before
        if delta > 0:
            self.diag["capture_seed_moves"] += delta
            self.diag["capture_seed_turns"] += 1

    def _enemy_high_prod_pressure_score(self, source: Planet, target: Planet, local: LocalObs) -> float:
        score = RLInformedPublicRuleAgent._enemy_high_prod_pressure_score(self, source, target, local)
        if score > 0.0:
            self.diag["enemy_hp_pressure_positive"] += 1
            self.diag["enemy_hp_pressure_score_sum"] += score
        return score

    def _contested_stop_loss_blocks(self, source: Planet, target: Planet, total_ships: int, local: LocalObs) -> bool:
        blocked = RLInformedPublicRuleAgent._contested_stop_loss_blocks(self, source, target, total_ships, local)
        if blocked:
            self.diag["contested_stoploss_blocks"] += 1
        return blocked

    def _contested_stop_loss_penalty(self, target: Planet, local: LocalObs) -> float:
        penalty = RLInformedPublicRuleAgent._contested_stop_loss_penalty(self, target, local)
        if penalty > 0.0:
            self.diag["contested_stoploss_penalty_positive"] += 1
        return penalty


def make_agent(params: dict, *, instrument: bool):
    instance = InstrumentedAgent(**params) if instrument else RLInformedPublicRuleAgent(**params)

    def agent(obs, configuration=None):
        return instance.act(obs)

    return agent, instance


def flatten_diag(diag: dict[str, object]) -> dict[str, float | int | str | None]:
    out: dict[str, float | int | str | None] = {}
    for phase in ("p0_50", "p51_80", "p81_120", "p121_plus"):
        out[f"launches_{phase}"] = int(diag["launches"][phase])
        out[f"launch_ships_{phase}"] = int(diag["launch_ships"][phase])
        out[f"attack_launches_{phase}"] = int(diag["attack_launches"][phase])
        out[f"enemy_attack_launches_{phase}"] = int(diag["enemy_attack_launches"][phase])
        out[f"enemy_highprod_attack_launches_{phase}"] = int(diag["enemy_highprod_attack_launches"][phase])
        out[f"neutral_highprod_attack_launches_{phase}"] = int(diag["neutral_highprod_attack_launches"][phase])

    checkpoints = diag["checkpoints"]
    for step in (50, 80, 120):
        stats = checkpoints.get(step, {})
        for key in ("prod_diff", "ship_diff", "planet_diff", "prod", "enemy_prod"):
            out[f"t{step}_{key}"] = float(stats.get(key, 0.0)) if stats else None

    for key in (
        "first_enemy_highprod_attack_step",
        "highprod_captures",
        "highprod_capture_activated_5",
        "highprod_capture_activated_10",
        "highprod_capture_activated_20",
        "highprod_capture_supported_10",
        "lost_highprod",
        "capture_seed_moves",
        "capture_seed_turns",
        "enemy_hp_pressure_positive",
        "contested_stoploss_blocks",
        "contested_stoploss_penalty_positive",
        "final_step",
    ):
        out[key] = diag[key]
    out["enemy_hp_pressure_score_sum"] = float(diag["enemy_hp_pressure_score_sum"])
    return out


def run_one(task: dict) -> dict:
    variant_agent, variant_instance = make_agent(dict(task["params"]), instrument=True)
    opponent_agent, _ = make_agent(dict(CHAMPION_OPPONENT_VARIANTS["regular"]), instrument=False)
    env = make_fast_orbit_wars({"seed": int(task["seed"]), "episodeSteps": 500}, keep_history=False, use_numba=bool(task["use_numba"]))
    if task["seat"] == "variant_p0":
        env.run([variant_agent, opponent_agent])
        variant_reward = env.steps[-1][0]["reward"]
        opponent_reward = env.steps[-1][1]["reward"]
    else:
        env.run([opponent_agent, variant_agent])
        opponent_reward = env.steps[-1][0]["reward"]
        variant_reward = env.steps[-1][1]["reward"]

    if variant_reward > opponent_reward:
        outcome = "win"
    elif variant_reward < opponent_reward:
        outcome = "loss"
    else:
        outcome = "draw"

    return {
        "variant": task["variant"],
        "seed": task["seed"],
        "seat": task["seat"],
        "outcome": outcome,
        "variant_reward": variant_reward,
        "opponent_reward": opponent_reward,
        **flatten_diag(variant_instance.diag),
    }


def build_tasks(variant_names: list[str], games_per_seat: int, use_numba: bool) -> list[dict]:
    all_variants: dict[str, dict] = {}
    for suite in ABLATION_SUITES.values():
        for name, params in suite.items():
            all_variants.setdefault(name, params)
    tasks = []
    for name in variant_names:
        params = all_variants[name]
        for idx in range(games_per_seat):
            tasks.append({"variant": name, "params": params, "seed": 42 + idx, "seat": "variant_p0", "use_numba": use_numba})
            tasks.append({"variant": name, "params": params, "seed": 1042 + idx, "seat": "variant_p1", "use_numba": use_numba})
    return tasks


def summarize(rows: list[dict]) -> list[dict]:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in {"variant", "seed", "seat", "outcome"} and isinstance(value, (int, float, type(None)))
    ]
    summaries = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        summary = {
            "variant": variant,
            "games": len(group),
            "wins": sum(row["outcome"] == "win" for row in group),
            "losses": sum(row["outcome"] == "loss" for row in group),
            "draws": sum(row["outcome"] == "draw" for row in group),
        }
        summary["win_rate"] = summary["wins"] / max(1, len(group))
        for key in numeric_keys:
            values = [float(row[key]) for row in group if row[key] is not None]
            if values:
                summary[f"avg_{key}"] = mean(values)
        summaries.append(summary)
    return summaries


def write_markdown(path: Path, summaries: list[dict]) -> None:
    interesting = [
        "win_rate",
        "avg_t80_prod_diff",
        "avg_t120_prod_diff",
        "avg_launches_p0_50",
        "avg_launches_p51_80",
        "avg_enemy_highprod_attack_launches_p0_50",
        "avg_enemy_highprod_attack_launches_p51_80",
        "avg_first_enemy_highprod_attack_step",
        "avg_highprod_captures",
        "avg_highprod_capture_activated_10",
        "avg_highprod_capture_supported_10",
        "avg_lost_highprod",
        "avg_capture_seed_moves",
        "avg_enemy_hp_pressure_positive",
        "avg_contested_stoploss_blocks",
    ]
    lines = ["# Vadasz Ablation Diagnostics", ""]
    lines.append("| variant | " + " | ".join(interesting) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(interesting)) + "|")
    for row in sorted(summaries, key=lambda item: item["win_rate"], reverse=True):
        cells = [row["variant"]]
        for key in interesting:
            value = row.get(key)
            if value is None:
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "regular",
            "p2_stoploss_prod2_w45_f3_p35_hold8",
            "p2_seed_neutral_prod4_s0_45_eta28_lag10_m12",
            "p2_enemy_hp_prod3_s25_110_b16_pw7_ship020_eta030",
        ],
    )
    parser.add_argument("--games-per-seat", type=int, default=40)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--no-numba", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args.variants, args.games_per_seat, not args.no_numba)
    print(
        f"Running diagnostics: {len(tasks)} games, variants={args.variants}, "
        f"workers={args.workers}, games_per_seat={args.games_per_seat}, numba={not args.no_numba}",
        flush=True,
    )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[{idx}/{len(tasks)}] {row['variant']} {row['seat']} seed={row['seed']} -> {row['outcome']}", flush=True)

    summaries = summarize(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"vadasz_ablation_diagnostics_{stamp}.csv"
    json_path = OUT_DIR / f"vadasz_ablation_diagnostics_{stamp}.json"
    md_path = OUT_DIR / f"vadasz_ablation_diagnostics_{stamp}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"rows": rows, "summaries": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, summaries)

    print("\n==== DIAGNOSTIC SUMMARY ====")
    for row in sorted(summaries, key=lambda item: item["win_rate"], reverse=True):
        print(
            f"{row['variant']}: {row['wins']}-{row['losses']}-{row['draws']} "
            f"wr={row['win_rate']:.3f} t80_prod={row.get('avg_t80_prod_diff', 0):.2f} "
            f"t120_prod={row.get('avg_t120_prod_diff', 0):.2f} "
            f"launch50={row.get('avg_launches_p0_50', 0):.2f} "
            f"enemy_hp_0_80={row.get('avg_enemy_highprod_attack_launches_p0_50', 0) + row.get('avg_enemy_highprod_attack_launches_p51_80', 0):.2f}"
        )
    print(f"csv:  {csv_path}")
    print(f"json: {json_path}")
    print(f"md:   {md_path}")


if __name__ == "__main__":
    main()
