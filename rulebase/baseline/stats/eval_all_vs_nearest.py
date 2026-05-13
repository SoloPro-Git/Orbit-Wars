import csv
import json
import math
from datetime import datetime
from pathlib import Path
import importlib.util

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

ROOT = Path('/data2/solo/Orbit-Wars')
AGENT_PATH = ROOT / 'rulebase/baseline/orbit-wars-baseline-all-strategies-agent.py'
OUT_DIR = ROOT / 'rulebase/baseline/stats'


def load_agent_fn(path: Path):
    spec = importlib.util.spec_from_file_location('baseline_all_agent_mod', str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.agent


def nearest_planet_agent(obs, configuration=None):
    moves = []
    player = obs.get('player', 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get('planets', []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]
    if not targets:
        return moves

    for mine in my_planets:
        nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])
    return moves


def run_match(agent_a, agent_b, num_games=40, seed_base=42):
    rows = []
    a_wins = b_wins = draws = 0
    for i in range(num_games):
        seed = seed_base + i
        env = make('orbit_wars', configuration={'seed': seed}, debug=True)
        env.run([agent_a, agent_b])
        final = env.steps[-1]
        r0 = final[0].reward
        r1 = final[1].reward
        if r0 > r1:
            outcome = 'A_WIN'
            a_wins += 1
        elif r0 < r1:
            outcome = 'B_WIN'
            b_wins += 1
        else:
            outcome = 'DRAW'
            draws += 1
        rows.append({'seed': seed, 'reward_a': r0, 'reward_b': r1, 'outcome': outcome})
        print(f'game {i+1}/{num_games} seed={seed} -> {outcome} ({r0} vs {r1})')

    return rows, {'a_wins': a_wins, 'b_wins': b_wins, 'draws': draws, 'games': num_games}


def main():
    agent_all = load_agent_fn(AGENT_PATH)

    # Seat 1: all-strategy as player0
    rows_p0, sum_p0 = run_match(agent_all, nearest_planet_agent, num_games=40, seed_base=42)

    # Seat 2: all-strategy as player1 (swap order)
    rows_p1_raw, sum_p1_raw = run_match(nearest_planet_agent, agent_all, num_games=40, seed_base=1042)

    # Normalize seat-2 to "all strategy" perspective
    rows_p1 = []
    all_wins_p1 = nearest_wins_p1 = draws_p1 = 0
    for r in rows_p1_raw:
        if r['outcome'] == 'A_WIN':
            # A is nearest, so all-strategy loses
            outcome_all = 'LOSE'
            nearest_wins_p1 += 1
        elif r['outcome'] == 'B_WIN':
            outcome_all = 'WIN'
            all_wins_p1 += 1
        else:
            outcome_all = 'DRAW'
            draws_p1 += 1

        rows_p1.append({
            'seed': r['seed'],
            'reward_nearest_p0': r['reward_a'],
            'reward_all_p1': r['reward_b'],
            'all_outcome': outcome_all,
        })

    all_wins_total = sum_p0['a_wins'] + all_wins_p1
    nearest_wins_total = sum_p0['b_wins'] + nearest_wins_p1
    draws_total = sum_p0['draws'] + draws_p1
    total_games = sum_p0['games'] + len(rows_p1)

    summary = {
        'agent': 'orbit-wars-baseline-all-strategies-agent.py',
        'opponent': 'nearest_planet_agent',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'seat_breakdown': {
            'all_as_player0': {
                'games': sum_p0['games'],
                'wins': sum_p0['a_wins'],
                'losses': sum_p0['b_wins'],
                'draws': sum_p0['draws'],
                'win_rate': sum_p0['a_wins'] / sum_p0['games'],
            },
            'all_as_player1': {
                'games': len(rows_p1),
                'wins': all_wins_p1,
                'losses': nearest_wins_p1,
                'draws': draws_p1,
                'win_rate': all_wins_p1 / len(rows_p1),
            },
        },
        'overall': {
            'games': total_games,
            'wins': all_wins_total,
            'losses': nearest_wins_total,
            'draws': draws_total,
            'win_rate': all_wins_total / total_games,
            'non_loss_rate': (all_wins_total + draws_total) / total_games,
        },
    }

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = OUT_DIR / f'all_vs_nearest_{stamp}.json'
    csv_path = OUT_DIR / f'all_vs_nearest_{stamp}.csv'

    with json_path.open('w', encoding='utf-8') as f:
        json.dump(
            {
                'summary': summary,
                'all_as_player0_games': rows_p0,
                'all_as_player1_games': rows_p1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['seat', 'seed', 'all_outcome'])
        writer.writeheader()
        for r in rows_p0:
            writer.writerow({'seat': 'all_p0', 'seed': r['seed'], 'all_outcome': 'WIN' if r['outcome'] == 'A_WIN' else ('LOSE' if r['outcome'] == 'B_WIN' else 'DRAW')})
        for r in rows_p1:
            writer.writerow({'seat': 'all_p1', 'seed': r['seed'], 'all_outcome': r['all_outcome']})

    print('\n==== SUMMARY ====')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'json: {json_path}')
    print(f'csv:  {csv_path}')


if __name__ == '__main__':
    main()
