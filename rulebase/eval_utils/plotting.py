"""评估可视化函数。"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from .metrics import compute_game_ranking


def print_game_result(env):
    """打印单局游戏结果（排名和分数）。"""
    ranking, player = compute_game_ranking(env)

    scores = dict(ranking)
    my_score = scores.get(player, 0)
    my_rank = [pid for pid, _ in ranking].index(player) + 1

    print("===== Game Result =====")
    print(f"My player ID: {player}")
    print(f"My score: {my_score}")
    print(f"My rank: {my_rank} / {len(ranking)}")
    print("Result:", "WIN" if my_rank == 1 else "LOSE")

    print("\nRanking:")
    for rank, (pid, score) in enumerate(ranking, start=1):
        marker = " <-- me" if pid == player else ""
        print(f"{rank}. Player {pid}: {score}{marker}")


def plot_game_state(env):
    """绘制单局游戏状态图，自动区分 2 人/4 人模式。"""
    steps = []
    my_neutral_capture_steps = []
    my_enemy_capture_steps = []
    enemy_neutral_capture_steps = []
    enemy_enemy_capture_steps = []
    lost_planet_steps = []
    prev_owners = None

    # 收集所有玩家
    players = set()
    for step_idx in range(len(env.steps)):
        obs = env.steps[step_idx][0].observation
        for p in obs.planets:
            if p[1] != -1:
                players.add(p[1])
    players = sorted(players)
    is_multi = len(players) > 2

    # 提取捕获事件
    for step_idx in range(1, len(env.steps)):
        obs = env.steps[step_idx][0].observation
        player = obs.player
        planets = [Planet(*p) for p in obs.planets]
        curr_owners = {p.id: p.owner for p in planets}

        mc = ec = enc = eec = lp = 0
        if prev_owners is not None:
            for pid, curr_owner in curr_owners.items():
                prev_owner = prev_owners.get(pid)
                if prev_owner is None or prev_owner == curr_owner:
                    continue
                if curr_owner == player and prev_owner == -1:
                    mc += 1
                elif curr_owner == player and prev_owner not in (-1, player):
                    ec += 1
                elif prev_owner == player and curr_owner != player:
                    lp += 1
                elif prev_owner == -1 and curr_owner not in (-1, player):
                    enc += 1
                elif (
                    prev_owner not in (-1, player)
                    and curr_owner not in (-1, player)
                    and prev_owner != curr_owner
                ):
                    eec += 1

        my_neutral_capture_steps.append(mc)
        my_enemy_capture_steps.append(ec)
        enemy_neutral_capture_steps.append(enc)
        enemy_enemy_capture_steps.append(eec)
        lost_planet_steps.append(lp)
        prev_owners = curr_owners
        steps.append(step_idx)

    print_game_result(env)

    if not is_multi:
        _plot_2p_game(
            env, steps,
            my_neutral_capture_steps, my_enemy_capture_steps,
            enemy_neutral_capture_steps, lost_planet_steps,
        )
    else:
        _plot_4p_game(
            env, steps, players,
            my_neutral_capture_steps, my_enemy_capture_steps,
            enemy_neutral_capture_steps, enemy_enemy_capture_steps,
            lost_planet_steps,
        )


def _plot_2p_game(
    env, steps,
    my_neutral_captures, my_enemy_captures,
    enemy_neutral_captures, lost_planets,
):
    """2 人模式游戏状态图。"""
    my_planets_list = []
    enemy_planets_list = []
    neutral_planets_list = []
    my_ships_list = []
    enemy_ships_list = []

    for step_idx in range(1, len(env.steps)):
        obs = env.steps[step_idx][0].observation
        player = obs.player
        planets = [Planet(*p) for p in obs.planets]

        my_p = [p for p in planets if p.owner == player]
        enemy_p = [p for p in planets if p.owner not in (-1, player)]
        neutral_p = [p for p in planets if p.owner == -1]

        my_planets_list.append(len(my_p))
        enemy_planets_list.append(len(enemy_p))
        neutral_planets_list.append(len(neutral_p))
        my_ships_list.append(sum(p.ships for p in my_p))
        enemy_ships_list.append(sum(p.ships for p in enemy_p))

    fig, ax1 = plt.subplots(figsize=(12, 5))

    line1, = ax1.plot(steps, my_planets_list, label="My planets")
    line2, = ax1.plot(steps, enemy_planets_list, label="Enemy planets")
    line3, = ax1.plot(steps, neutral_planets_list, label="Neutral planets")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Planet count")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    line4, = ax2.plot(steps, my_ships_list, linestyle="--", label="My ships")
    line5, = ax2.plot(steps, enemy_ships_list, linestyle="--", label="Enemy ships")
    ax2.set_ylabel("Ships")

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))

    bars = []
    bottom = [0] * len(steps)

    configs = [
        (my_neutral_captures, "My neutral captures", "skyblue"),
        (my_enemy_captures, "My enemy captures", "green"),
        (enemy_neutral_captures, "Enemy neutral captures", "orange"),
        (lost_planets, "Lost planets", "red"),
    ]

    for data, label, color in configs:
        bar = ax3.bar(steps, data, bottom=bottom, alpha=0.35, width=1.0, label=label, color=color)
        bars.append(bar)
        bottom = [b + v for b, v in zip(bottom, data)]

    max_capture = max(
        [sum(vs) for vs in zip(my_neutral_captures, my_enemy_captures, enemy_neutral_captures, lost_planets)],
        default=0,
    )
    ax3.set_ylim(0, max_capture + 1)
    ax3.set_ylabel("Capture events")
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

    lines = [line1, line2, line3, line4, line5] + bars
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Game State and Capture Events Over Time")
    plt.show()


def _plot_4p_game(
    env, steps, players,
    my_neutral_captures, my_enemy_captures,
    enemy_neutral_captures, enemy_enemy_captures, lost_planets,
):
    """4 人模式游戏状态图（3 张子图）。"""
    # 图 1：捕获事件
    fig, ax = plt.subplots(figsize=(12, 4))
    bottom = [0] * len(steps)

    configs = [
        (my_neutral_captures, "My neutral captures", "skyblue"),
        (my_enemy_captures, "My enemy captures", "green"),
        (enemy_neutral_captures, "Enemy neutral captures", "orange"),
    ]

    if any(v > 0 for v in enemy_enemy_captures):
        configs.append((enemy_enemy_captures, "Enemy enemy captures", "purple"))

    configs.append((lost_planets, "Lost planets", "red"))

    for data, label, color in configs:
        ax.bar(steps, data, bottom=bottom, color=color, alpha=0.4, width=1.0, label=label)
        bottom = [b + v for b, v in zip(bottom, data)]

    max_capture = max(
        [sum(vs) for vs in zip(my_neutral_captures, my_enemy_captures, enemy_neutral_captures, enemy_enemy_captures, lost_planets)],
        default=0,
    )
    ax.set_ylim(0, max_capture + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Capture Events Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Capture events")
    ax.legend(loc="upper left")
    plt.show()

    # 图 2：行星数
    player_planets = {pid: [] for pid in players}
    neutral_planets_by_step = []

    for step_idx in range(1, len(env.steps)):
        obs = env.steps[step_idx][0].observation
        planets = [Planet(*p) for p in obs.planets]
        neutral_planets_by_step.append(len([p for p in planets if p.owner == -1]))
        for pid in players:
            player_planets[pid].append(len([p for p in planets if p.owner == pid]))

    fig, ax = plt.subplots(figsize=(12, 5))
    for pid in players:
        ax.plot(steps, player_planets[pid], label=f"Player {pid}")
    ax.plot(steps, neutral_planets_by_step, linestyle="--", label="Neutral")
    ax.set_title("Planet Count by Player + Neutral")
    ax.set_xlabel("Step")
    ax.set_ylabel("Planet count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    plt.show()

    # 图 3：飞船数
    player_ships = {pid: [] for pid in players}
    for step_idx in range(1, len(env.steps)):
        obs = env.steps[step_idx][0].observation
        planets = [Planet(*p) for p in obs.planets]
        for pid in players:
            player_ships[pid].append(sum(p.ships for p in planets if p.owner == pid))

    fig, ax = plt.subplots(figsize=(12, 5))
    for pid in players:
        ax.plot(steps, player_ships[pid], linestyle="--", label=f"Player {pid}")
    ax.set_title("Ship Count by Player")
    ax.set_xlabel("Step")
    ax.set_ylabel("Ships")
    ax.legend()
    plt.show()


def plot_average_game_state(timeseries_df):
    """绘制多局平均游戏状态图（按胜/负分组）。"""
    for result_label, flag in [("Winning Games", True), ("Losing Games", False)]:
        df_part = timeseries_df[timeseries_df["my_won"] == flag]
        if len(df_part) == 0:
            continue

        avg_df = df_part.groupby("step", as_index=False).mean(numeric_only=True)

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(avg_df["step"], avg_df["my_planets"], label="My planets")
        ax1.plot(avg_df["step"], avg_df["best_enemy_planets"], label="Best opponent")
        ax1.plot(avg_df["step"], avg_df["neutral_planets"], label="Neutral")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Planet count")

        ax2 = ax1.twinx()
        ax2.plot(avg_df["step"], avg_df["my_ships"], linestyle="--", label="My ships")
        ax2.plot(avg_df["step"], avg_df["best_enemy_ships"], linestyle="--", label="Best opponent ships")
        ax2.set_ylabel("Ships")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.title(f"Average Game State — {result_label} (Best Opponent)")
        plt.grid(True)
        plt.show()


def plot_average_capture_events(timeseries_df, step_range=None):
    """绘制多局平均捕获事件堆叠图。"""
    df = timeseries_df.copy()
    if step_range is not None:
        start, end = step_range
        df = df[(df["step"] >= start) & (df["step"] <= end)]

    for result_label, flag in [("Winning Games", True), ("Losing Games", False)]:
        df_part = df[df["my_won"] == flag]
        if len(df_part) == 0:
            print(f"No {result_label.lower()} found.")
            continue

        avg_df = df_part.groupby("step", as_index=False).mean(numeric_only=True)

        fig, ax = plt.subplots(figsize=(12, 4))
        bottom = np.zeros(len(avg_df))

        cols = [
            ("my_neutral_captures", "My neutral captures"),
            ("my_enemy_captures", "My enemy captures"),
            ("enemy_neutral_captures", "Enemy neutral captures"),
            ("enemy_enemy_captures", "Enemy enemy captures"),
            ("lost_planets", "Lost planets"),
        ]

        for col, label in cols:
            ax.bar(avg_df["step"], avg_df[col], bottom=bottom, alpha=0.4, width=1.0, label=label)
            bottom += avg_df[col].values

        ax.set_title(f"Average Capture Events — {result_label}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Average capture events")
        ax.legend(loc="upper left")
        plt.show()


def summarize_results(results, plot=True):
    """打印汇总统计并可选绘制飞船差柱状图。"""
    df = pd.DataFrame(results)
    n_games = len(df)

    wins = df["my_won"].sum()
    draws = (df["winner"] == -1).sum()
    losses = n_games - wins - draws

    print("games:", n_games)
    print("wins:", wins)
    print("losses:", losses)
    print("draws:", draws)
    print("win rate:", wins / n_games)
    print("avg rank:", df["my_rank"].mean())
    print("avg total ships:", df["my_total_ships"].mean())
    print("avg best other ships:", df["best_other_ships"].mean())
    print("avg ship margin vs best other:", df["ship_margin_vs_best_other"].mean())
    print("avg final turns:", df["final_turns"].mean())

    if plot:
        plt.figure(figsize=(8, 4))
        plt.bar(range(n_games), df["ship_margin_vs_best_other"])
        plt.axhline(0, linestyle="--")
        plt.xlabel("Game")
        plt.ylabel("Ship margin")
        plt.title("Ship Margin vs Best Opponent")
        plt.grid(True)
        plt.show()

    return df
