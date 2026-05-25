from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.expert.action_labeling import infer_target_planet_id
from training2 import make_fast_orbit_wars
from training2.rulebase_bridge import make_rulebase_agent

from tinyPPO.agents import TinyPPOAgent
from tinyPPO.imitation_regular import row_from_regular_action


@dataclass
class RawDecision:
    obs: dict[str, Any]
    player: int
    players: int
    regular_action: list[list]


class LoggingRegularAgent:
    def __init__(self, player: int, players: int, rows: list[RawDecision], keep_noop_prob: float):
        self.player = int(player)
        self.players = int(players)
        self.rows = rows
        self.keep_noop_prob = float(keep_noop_prob)
        self.agent = make_rulebase_agent("regular")

    def __call__(self, obs: dict[str, Any], configuration=None) -> list[list]:
        del configuration
        action = self.agent(obs) or []
        if action or random.random() < self.keep_noop_prob:
            self.rows.append(RawDecision(obs=obs, player=self.player, players=self.players, regular_action=action))
        return action


def _players_list(raw: str) -> list[int]:
    players = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not players or any(p < 2 for p in players):
        raise ValueError(f"invalid players list: {raw!r}")
    return players


def collect_regular_states(
    *,
    players_list: list[int],
    games_per_players: int,
    seed: int,
    episode_steps: int,
    rows_per_game: int,
    keep_noop_prob: float,
    use_numba: bool,
    progress: bool,
) -> list[RawDecision]:
    rows: list[RawDecision] = []
    jobs = [(players, seed + players * 1_000_000 + game) for players in players_list for game in range(games_per_players)]
    iterator = jobs
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(jobs, desc="collect regular states", dynamic_ncols=True)
        except Exception:
            pass
    for players, game_seed in iterator:
        game_rows: list[RawDecision] = []
        env = make_fast_orbit_wars(
            {"episodeSteps": episode_steps, "seed": game_seed},
            keep_history=False,
            use_numba=use_numba,
        )
        agents = [LoggingRegularAgent(pid, players, game_rows, keep_noop_prob) for pid in range(players)]
        env.run(agents)
        if rows_per_game > 0 and len(game_rows) > rows_per_game:
            game_rows = random.sample(game_rows, rows_per_game)
        rows.extend(game_rows)
    return rows


def action_keys(obs: dict[str, Any], action: list[list]) -> tuple[Counter, Counter, Counter]:
    source_keys: Counter = Counter()
    source_target_keys: Counter = Counter()
    source_target_ship_keys: Counter = Counter()
    for move in action:
        if not isinstance(move, list) or len(move) < 3:
            continue
        try:
            source_id = int(move[0])
            angle = float(move[1])
            ships = max(1, int(float(move[2])))
        except (TypeError, ValueError):
            continue
        target_id = infer_target_planet_id(obs, source_id, angle, ships)
        if target_id is None:
            continue
        ship_bucket = int(math.floor(math.log2(max(1, ships))))
        source_keys[source_id] += 1
        source_target_keys[(source_id, int(target_id))] += 1
        source_target_ship_keys[(source_id, int(target_id), ship_bucket)] += 1
    return source_keys, source_target_keys, source_target_ship_keys


def _prf(pred: Counter, true: Counter) -> tuple[float, float, float]:
    tp = sum((pred & true).values())
    pred_n = sum(pred.values())
    true_n = sum(true.values())
    precision = tp / pred_n if pred_n else (1.0 if true_n == 0 else 0.0)
    recall = tp / true_n if true_n else (1.0 if pred_n == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    states = collect_regular_states(
        players_list=_players_list(args.players_list),
        games_per_players=args.games_per_players,
        seed=args.seed,
        episode_steps=args.episode_steps,
        rows_per_game=args.rows_per_game,
        keep_noop_prob=args.keep_noop_prob,
        use_numba=not args.no_numba,
        progress=args.progress,
    )
    if args.max_rows > 0 and len(states) > args.max_rows:
        states = random.sample(states, args.max_rows)
    if not states:
        raise RuntimeError("no regular states collected")

    agent = TinyPPOAgent(
        Path(args.checkpoint),
        device=args.device,
        deterministic=not args.stochastic,
        launch_bias=args.launch_bias + args.aggression,
        ship_bias=args.ship_bias + args.aggression,
        launch_temperature=args.launch_temperature,
    )

    source_p = source_r = source_f1 = 0.0
    target_p = target_r = target_f1 = 0.0
    action_p = action_r = action_f1 = 0.0
    count_abs = 0.0
    pred_count = 0.0
    true_count = 0.0
    labelled_rows = 0
    skipped_rows = 0
    model_errors = 0

    iterator = states
    if args.progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(states, desc="eval regular imitation", dynamic_ncols=True)
        except Exception:
            pass
    for row in iterator:
        bc_row = row_from_regular_action(row.obs, row.player, row.regular_action, players=row.players)
        if bc_row is None:
            skipped_rows += 1
            continue
        labelled_rows += 1
        try:
            model_action = agent(row.obs)
        except Exception:
            model_errors += 1
            model_action = []
        true_source, true_target, true_action = action_keys(row.obs, row.regular_action)
        pred_source, pred_target, pred_action = action_keys(row.obs, model_action)
        sp, sr, sf = _prf(pred_source, true_source)
        tp, tr, tf = _prf(pred_target, true_target)
        ap, ar, af = _prf(pred_action, true_action)
        source_p += sp
        source_r += sr
        source_f1 += sf
        target_p += tp
        target_r += tr
        target_f1 += tf
        action_p += ap
        action_r += ar
        action_f1 += af
        pc = sum(pred_target.values())
        tc = sum(true_target.values())
        pred_count += pc
        true_count += tc
        count_abs += abs(pc - tc)

    denom = max(1, labelled_rows)
    return {
        "states": float(len(states)),
        "labelled_rows": float(labelled_rows),
        "skipped_rows": float(skipped_rows),
        "model_errors": float(model_errors),
        "source_precision": source_p / denom,
        "source_recall": source_r / denom,
        "source_f1": source_f1 / denom,
        "source_target_precision": target_p / denom,
        "source_target_recall": target_r / denom,
        "source_target_f1": target_f1 / denom,
        "action_precision": action_p / denom,
        "action_recall": action_r / denom,
        "action_f1": action_f1 / denom,
        "action_count_mae": count_abs / denom,
        "model_actions_per_state": pred_count / denom,
        "regular_actions_per_state": true_count / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TinyPPO BC checkpoint on held-out regular-policy states.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--players-list", default="2")
    parser.add_argument("--games-per-players", type=int, default=32)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--episode-steps", type=int, default=180)
    parser.add_argument("--rows-per-game", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--keep-noop-prob", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aggression", type=float, default=0.0)
    parser.add_argument("--launch-bias", type=float, default=0.0)
    parser.add_argument("--ship-bias", type=float, default=0.0)
    parser.add_argument("--launch-temperature", type=float, default=1.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
