"""对局运行与批量评估工具。"""

import random

import pandas as pd
from kaggle_environments import make

from .metrics import total_ships, extract_game_timeseries
from .plotting import summarize_results


def evaluate_agents(agents, my_agent, seeds=range(20)):
    """批量评估 agent，返回 (results_df, timeseries_df)。

    agents: agent 列表（2 或 4 人）
    my_agent: 我方 agent（必须在 agents 中）
    seeds: 随机种子范围
    """
    results = []
    timeseries_rows = []
    n_players = len(agents)

    if n_players not in [2, 4]:
        raise ValueError("Number of agents must be 2 or 4")

    for seed in seeds:
        lineup = list(agents)
        random.Random(seed).shuffle(lineup)
        my_position = lineup.index(my_agent)

        env = make("orbit_wars", debug=False, configuration={"seed": seed})
        env.run(lineup)

        final_states = env.steps[-1]
        final_turns = len(env.steps)

        total_ships_list = []
        for player_id in range(n_players):
            obs = final_states[player_id].observation
            total_ships_list.append(total_ships(obs, player_id))

        max_ships = max(total_ships_list)
        winners = [pid for pid, ships in enumerate(total_ships_list) if ships == max_ships]
        winner = winners[0] if len(winners) == 1 else -1

        my_total = total_ships_list[my_position]
        best_other = max(
            ships for pid, ships in enumerate(total_ships_list) if pid != my_position
        )

        sorted_ships = sorted(total_ships_list, reverse=True)
        my_rank = sorted_ships.index(my_total) + 1
        my_won = winner == my_position

        result = {
            "seed": seed,
            "my_position": my_position,
            "my_won": my_won,
            "my_rank": my_rank,
            "winner": winner,
            "my_total_ships": my_total,
            "best_other_ships": best_other,
            "ship_margin_vs_best_other": my_total - best_other,
            "final_turns": final_turns,
        }
        for player_id, ships in enumerate(total_ships_list):
            result[f"total_ships_{player_id}"] = ships
        results.append(result)

        game_ts = extract_game_timeseries(env, my_position)
        for row in game_ts:
            row["seed"] = seed
            row["my_won"] = my_won
            row["my_rank"] = my_rank
        timeseries_rows.extend(game_ts)

    return pd.DataFrame(results), pd.DataFrame(timeseries_rows)


def evaluate_against_baseline(
    variant,
    base,
    seeds=range(10),
    variant_label=None,
    extra_info=None,
    return_outputs=False,
):
    """对比评估 variant 与 base agent。

    在 2p 和 4p 模式下，以 variant 对抗 base，汇总胜负统计。
    """
    if variant_label is None:
        variant_label = variant.__name__
    if extra_info is None:
        extra_info = {}

    rows = []
    outputs = {}

    for mode in ["2p", "4p"]:
        print(f"\n Running {mode}: {variant_label}")

        if mode == "2p":
            eval_agents = [variant, base]
        else:
            eval_agents = [variant, base, base, base]

        results, ts = evaluate_agents(eval_agents, my_agent=variant, seeds=seeds)
        df = summarize_results(results, plot=False)

        rows.append({
            "mode": mode,
            "base_agent": base.__name__,
            "agent": variant_label,
            "win_rate": df["my_won"].mean(),
            "avg_rank": df["my_rank"].mean(),
            "avg_ships": df["my_total_ships"].mean(),
            "ship_margin": df["ship_margin_vs_best_other"].mean(),
            **extra_info,
        })

        outputs[mode] = {
            "results": results,
            "summary_df": df,
            "timeseries": ts,
            "agents": eval_agents,
        }

    if return_outputs:
        return pd.DataFrame(rows), outputs
    return rows


def run_evaluations(agents_dict, seeds=range(10)):
    """运行多 agent 评估（2p + 4p），返回 eval_outputs。"""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    eval_outputs = {"2p": {}, "4p": {}}

    for name, agent in tqdm(agents_dict.items(), desc="Running ablation"):
        results, ts = evaluate_agents(
            agents=[agent, "random"],
            my_agent=agent,
            seeds=seeds,
        )
        eval_outputs["2p"][name] = {"results": results, "timeseries": ts}

        results, ts = evaluate_agents(
            agents=[agent, "random", "random", "random"],
            my_agent=agent,
            seeds=seeds,
        )
        eval_outputs["4p"][name] = {"results": results, "timeseries": ts}

    return eval_outputs


def make_ablation_summary(eval_outputs):
    """从 eval_outputs 生成消融摘要 DataFrame。"""
    rows = []
    for name, output in eval_outputs.items():
        df = output["results"]
        rows.append({
            "agent": name,
            "games": len(df),
            "win_rate": df["my_won"].mean(),
            "avg_rank": df["my_rank"].mean(),
            "avg_total_ships": df["my_total_ships"].mean(),
            "avg_best_other_ships": df["best_other_ships"].mean(),
            "avg_margin": df["ship_margin_vs_best_other"].mean(),
            "avg_final_turns": df["final_turns"].mean(),
        })

    return pd.DataFrame(rows).sort_values("avg_margin", ascending=False)
