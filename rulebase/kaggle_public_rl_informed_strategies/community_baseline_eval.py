"""Evaluate variants against the community structured baseline notebook with the fast simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path("/data2/solo/Orbit-Wars")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rulebase.kaggle_public_rl_informed_strategies.multiplayer_eval import make_named_agent
from rulebase.kaggle_public_rl_informed_strategies.rl_informed_agent import RLInformedPublicRuleAgent
from rulebase.kaggle_public_rl_informed_strategies.strategy_config import ABLATION_SUITES
from training2.fast_orbit_wars import make_fast_orbit_wars

OUT_DIR = ROOT / "rulebase/kaggle_public_rl_informed_strategies/experiments"
DEFAULT_NOTEBOOK = ROOT / "kaggle_ipynb/orbit-wars-structured-baseline.ipynb"


def extract_submission_source(notebook_path: Path) -> str:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        lines = source.splitlines()
        if not lines:
            continue
        first = lines[0].strip()
        if first == "%%writefile submission.py" or first == "%%writefile -a submission.py":
            chunks.append("\n".join(lines[1:]))
    if not chunks:
        raise ValueError(f"No submission.py writefile cells found in {notebook_path}")
    return "\n\n".join(chunks) + "\n"


def load_notebook_agent(notebook_path: str):
    source = extract_submission_source(Path(notebook_path))
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    path = Path(tempfile.gettempdir()) / f"orbit_structured_baseline_{digest}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"orbit_structured_baseline_{digest}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extracted notebook module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "agent"):
        raise RuntimeError(f"Extracted notebook module has no agent(): {path}")
    return module.agent


def make_variant_agent(params: dict):
    instance = RLInformedPublicRuleAgent(**dict(params))

    def agent(obs, configuration=None):
        return instance.act(obs)

    return agent


def make_opponent(name: str, notebook_path: str):
    if name == "structured_baseline_notebook":
        return load_notebook_agent(notebook_path)
    return make_named_agent(name)


def run_one(
    seed: int,
    variant_seat: int,
    variant_name: str,
    params: dict,
    opponents: list[str],
    notebook_path: str,
    use_numba: bool,
) -> dict:
    agents = [make_opponent(opponents[i % len(opponents)], notebook_path) for i in range(3)]
    agents.insert(variant_seat, make_variant_agent(params))
    env = make_fast_orbit_wars({"seed": seed}, keep_history=False, use_numba=use_numba)
    env.run(agents)
    rewards = [env.steps[-1][idx]["reward"] for idx in range(4)]
    variant_reward = rewards[variant_seat]
    rank = 1 + sum(reward > variant_reward for reward in rewards)
    win = variant_reward == max(rewards) and variant_reward > 0
    loss = variant_reward < max(rewards)
    return {
        "variant": variant_name,
        "seed": seed,
        "seat": f"p{variant_seat}",
        "opponents": ",".join(opponents),
        "variant_reward": variant_reward,
        "rewards": ";".join(str(r) for r in rewards),
        "rank": rank,
        "win": bool(win),
        "loss": bool(loss),
        "steps": env.steps[-1][0]["observation"].get("step", None),
    }


def run_task(task: dict) -> dict:
    return run_one(
        seed=int(task["seed"]),
        variant_seat=int(task["variant_seat"]),
        variant_name=str(task["variant"]),
        params=dict(task["params"]),
        opponents=list(task["opponents"]),
        notebook_path=str(task["notebook_path"]),
        use_numba=bool(task["use_numba"]),
    )


def build_tasks(
    variants: dict[str, dict],
    games_per_seat: int,
    opponents: list[str],
    notebook_path: Path,
    use_numba: bool,
) -> list[dict]:
    tasks = []
    for name, params in variants.items():
        for seat in range(4):
            for i in range(games_per_seat):
                tasks.append(
                    {
                        "variant": name,
                        "params": params,
                        "seed": 12000 + i,
                        "variant_seat": seat,
                        "opponents": opponents,
                        "notebook_path": str(notebook_path),
                        "use_numba": use_numba,
                    }
                )
    return tasks


def summarize_variant(name: str, params: dict, rows: list[dict]) -> dict:
    wins = sum(1 for row in rows if row["win"])
    losses = sum(1 for row in rows if row["loss"])
    ties = len(rows) - wins - losses
    avg_rank = sum(float(row["rank"]) for row in rows) / len(rows)
    return {
        "variant": name,
        "params": params,
        "games": len(rows),
        "wins": wins,
        "losses": losses,
        "draws_or_ties": ties,
        "win_rate": wins / len(rows),
        "non_loss_rate": (wins + ties) / len(rows),
        "avg_rank": avg_rank,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(ABLATION_SUITES), required=True)
    parser.add_argument(
        "--opponents",
        nargs=3,
        default=["structured_baseline_notebook", "structured_baseline_notebook", "structured_baseline_notebook"],
    )
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--games-per-seat", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--no-numba", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = ABLATION_SUITES[args.suite]
    use_numba = not args.no_numba
    tasks = build_tasks(variants, args.games_per_seat, args.opponents, args.notebook, use_numba)
    all_rows = []
    print(
        f"Running FAST community baseline suite={args.suite}, opponents={args.opponents}: "
        f"{len(tasks)} games, {len(variants)} variants, workers={args.workers}, numba={use_numba}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            all_rows.append(row)
            print(
                f"[{idx}/{len(tasks)}] {row['variant']} {row['seat']} seed={row['seed']} "
                f"reward={row['variant_reward']} rank={row['rank']}",
                flush=True,
            )

    rows_by_variant = {name: [] for name in variants}
    for row in all_rows:
        rows_by_variant[row["variant"]].append(row)
    summaries = [summarize_variant(name, params, rows_by_variant[name]) for name, params in variants.items()]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    opponent_tag = hashlib.sha1(",".join(args.opponents).encode("utf-8")).hexdigest()[:8]
    json_path = OUT_DIR / f"{args.suite}_fast_community_baseline_4p_{stamp}_{opponent_tag}.json"
    csv_path = OUT_DIR / f"{args.suite}_fast_community_baseline_4p_{stamp}_{opponent_tag}.csv"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "environment": "fast_orbit_wars",
        "use_numba": use_numba,
        "suite": args.suite,
        "opponents": args.opponents,
        "notebook": str(args.notebook),
        "games_per_seat": args.games_per_seat,
        "workers": args.workers,
        "summaries": summaries,
        "games": all_rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n==== FAST COMMUNITY BASELINE 4P SUMMARY ====")
    for row in sorted(summaries, key=lambda r: (r["win_rate"], -r["avg_rank"]), reverse=True):
        print(
            f"{row['variant']}: {row['wins']}-{row['losses']}-{row['draws_or_ties']} "
            f"win_rate={row['win_rate']:.2f} non_loss={row['non_loss_rate']:.2f} avg_rank={row['avg_rank']:.2f}"
        )
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")


if __name__ == "__main__":
    main()
