import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

from kaggle_environments import make

ROOT = Path("/data2/solo/Orbit-Wars")
OUT_DIR = ROOT / "rulebase/kaggle_public_strategies/stats"
BASELINE_PATH = ROOT / "rulebase/baseline/orbit-wars-baseline-all-strategies-agent.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulebase.kaggle_public_strategies.public_rule_agent import PublicRuleAgent


def load_function(path: Path, function_name: str):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def make_public_agent():
    instance = PublicRuleAgent()

    def _agent(obs, configuration=None):
        return instance.act(obs)

    return _agent


def run_game(seed: int, player0, player1) -> dict:
    env = make("orbit_wars", configuration={"seed": seed}, debug=True)
    env.run([player0, player1])
    final = env.steps[-1]
    p0_reward = final[0].reward
    p1_reward = final[1].reward
    if p0_reward > p1_reward:
        winner = "p0"
    elif p1_reward > p0_reward:
        winner = "p1"
    else:
        winner = "draw"
    return {
        "seed": seed,
        "p0_reward": p0_reward,
        "p1_reward": p1_reward,
        "winner": winner,
        "steps": len(env.steps),
    }


def main():
    baseline_agent = load_function(BASELINE_PATH, "agent")
    rows = []

    public_wins = 0
    baseline_wins = 0
    draws = 0

    for i in range(10):
        seed = 42 + i
        result = run_game(seed, make_public_agent(), baseline_agent)
        public_outcome = "win" if result["winner"] == "p0" else "loss" if result["winner"] == "p1" else "draw"
        rows.append({"seat": "public_p0", "public_outcome": public_outcome, **result})
        print(f"public P0 game {i + 1}/10 seed={seed}: {public_outcome} ({result['p0_reward']} vs {result['p1_reward']})")
        if public_outcome == "win":
            public_wins += 1
        elif public_outcome == "loss":
            baseline_wins += 1
        else:
            draws += 1

    for i in range(10):
        seed = 1042 + i
        result = run_game(seed, baseline_agent, make_public_agent())
        public_outcome = "win" if result["winner"] == "p1" else "loss" if result["winner"] == "p0" else "draw"
        rows.append({"seat": "public_p1", "public_outcome": public_outcome, **result})
        print(f"public P1 game {i + 1}/10 seed={seed}: {public_outcome} ({result['p0_reward']} vs {result['p1_reward']})")
        if public_outcome == "win":
            public_wins += 1
        elif public_outcome == "loss":
            baseline_wins += 1
        else:
            draws += 1

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "games": len(rows),
        "public_agent": "rulebase.kaggle_public_strategies.PublicRuleAgent",
        "baseline_agent": str(BASELINE_PATH),
        "public_wins": public_wins,
        "baseline_wins": baseline_wins,
        "draws": draws,
        "public_win_rate": public_wins / len(rows),
        "public_non_loss_rate": (public_wins + draws) / len(rows),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"public_vs_baseline_{stamp}.json"
    csv_path = OUT_DIR / f"public_vs_baseline_{stamp}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "games": rows}, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")


if __name__ == "__main__":
    main()
