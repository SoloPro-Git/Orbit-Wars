"""数据提取与指标计算工具。"""

import pandas as pd
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


def total_ships(obs, player_id):
    """计算玩家总飞船数（行星 + 舰队）。"""
    planets = obs["planets"]
    fleets = obs["fleets"]
    planet_ships = sum(p[5] for p in planets if p[1] == player_id)
    fleet_ships = sum(f[6] for f in fleets if f[1] == player_id)
    return planet_ships + fleet_ships


def extract_game_timeseries(env, my_position):
    """提取每步游戏指标，返回 dict 列表。

    包含：行星数、飞船数、各类捕获事件等。
    """
    rows = []
    prev_owners = None

    for step_idx in range(1, len(env.steps)):
        obs = env.steps[step_idx][my_position].observation
        player = obs.player
        planets = obs.planets
        curr_owners = {p[0]: p[1] for p in planets}

        my_neutral_captures = 0
        my_enemy_captures = 0
        enemy_neutral_captures = 0
        enemy_enemy_captures = 0
        lost_planets = 0

        if prev_owners is not None:
            for planet_id, curr_owner in curr_owners.items():
                prev_owner = prev_owners.get(planet_id)
                if prev_owner is None or prev_owner == curr_owner:
                    continue
                if curr_owner == player and prev_owner == -1:
                    my_neutral_captures += 1
                elif curr_owner == player and prev_owner not in (-1, player):
                    my_enemy_captures += 1
                elif prev_owner == player and curr_owner != player:
                    lost_planets += 1
                elif prev_owner == -1 and curr_owner not in (-1, player):
                    enemy_neutral_captures += 1
                elif (
                    prev_owner not in (-1, player)
                    and curr_owner not in (-1, player)
                    and prev_owner != curr_owner
                ):
                    enemy_enemy_captures += 1

        my_planets = [p for p in planets if p[1] == player]
        neutral_planets = [p for p in planets if p[1] == -1]

        ships_by_player = {}
        planet_count_by_player = {}
        for p in planets:
            pid = p[1]
            if pid == -1:
                continue
            ships_by_player[pid] = ships_by_player.get(pid, 0) + p[5]
            planet_count_by_player[pid] = planet_count_by_player.get(pid, 0) + 1

        enemy_players = [pid for pid in ships_by_player if pid != player]
        if enemy_players:
            best_enemy_ships = max(ships_by_player[pid] for pid in enemy_players)
            best_enemy_planets = max(planet_count_by_player[pid] for pid in enemy_players)
        else:
            best_enemy_ships = 0
            best_enemy_planets = 0

        rows.append({
            "step": step_idx,
            "my_planets": len(my_planets),
            "best_enemy_planets": best_enemy_planets,
            "neutral_planets": len(neutral_planets),
            "my_ships": sum(p[5] for p in my_planets),
            "best_enemy_ships": best_enemy_ships,
            "my_neutral_captures": my_neutral_captures,
            "my_enemy_captures": my_enemy_captures,
            "enemy_neutral_captures": enemy_neutral_captures,
            "enemy_enemy_captures": enemy_enemy_captures,
            "lost_planets": lost_planets,
        })

        prev_owners = curr_owners

    return rows


def compute_game_result(env, my_position):
    """计算单局结果（胜/负/排名/飞船差），返回 dict。"""
    final_states = env.steps[-1]
    final_turns = len(env.steps)
    n_players = len(final_states)

    total_ships_list = []
    for player_id in range(n_players):
        obs = final_states[player_id].observation
        total_ships_list.append(total_ships(obs, player_id))

    max_ships = max(total_ships_list)
    winners = [
        pid for pid, ships in enumerate(total_ships_list) if ships == max_ships
    ]
    winner = winners[0] if len(winners) == 1 else -1

    my_total = total_ships_list[my_position]
    best_other = max(
        ships for pid, ships in enumerate(total_ships_list) if pid != my_position
    )

    sorted_ships = sorted(total_ships_list, reverse=True)
    my_rank = sorted_ships.index(my_total) + 1

    result = {
        "my_won": winner == my_position,
        "my_rank": my_rank,
        "winner": winner,
        "my_total_ships": my_total,
        "best_other_ships": best_other,
        "ship_margin_vs_best_other": my_total - best_other,
        "final_turns": final_turns,
    }

    for pid, ships in enumerate(total_ships_list):
        result[f"total_ships_{pid}"] = ships

    return result


def compute_game_ranking(env):
    """计算最终排名，返回排序后的 [(player_id, score)] 列表。"""
    final_obs = env.steps[-1][0].observation
    planets = [Planet(*p) for p in final_obs.planets]
    fleets = final_obs.fleets

    all_players = set()
    for step in env.steps:
        obs = step[0].observation
        for p in obs.planets:
            if p[1] != -1:
                all_players.add(p[1])
        for f in obs.fleets:
            if f[1] != -1:
                all_players.add(f[1])

    scores = {}
    for pid in sorted(all_players):
        planet_ships = sum(p.ships for p in planets if p.owner == pid)
        fleet_ships = sum(f[6] for f in fleets if f[1] == pid)
        scores[pid] = planet_ships + fleet_ships

    return sorted(scores.items(), key=lambda x: x[1], reverse=True), final_obs.player
