from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if "_e" in stem:
        try:
            return int(stem.rsplit("_e", 1)[1]), path.name
        except ValueError:
            pass
    return int(path.stat().st_mtime), path.name


def _known_outputs(out_dir: Path) -> set[str]:
    return {path.stem.replace("phase1_gate_", "") for path in out_dir.glob("phase1_gate_*.json")}


def _gate_name(path: Path) -> str:
    return f"phase1_gate_{path.stem}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 same-state imitation gates as new BC checkpoints appear.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pattern", default="regular_bc_ray_e*.pt")
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--players-list", default="2")
    parser.add_argument("--seeds", default="992000,993000,994000")
    parser.add_argument("--games-per-players", type=int, default=8)
    parser.add_argument("--rows-per-game", type=int, default=12)
    parser.add_argument("--episode-steps", type=int, default=180)
    parser.add_argument("--target-pair-weight", type=float, default=1.0)
    parser.add_argument("--target-mask-mode", default="all_planets")
    parser.add_argument("--baseline-checkpoint", default="")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    print(json.dumps({"event": "phase1_gate_watch_start", "run_dir": str(run_dir), "pattern": args.pattern}), flush=True)
    while True:
        completed = _known_outputs(run_dir)
        checkpoints = sorted(run_dir.glob(args.pattern), key=_checkpoint_sort_key)
        for checkpoint in checkpoints:
            if checkpoint.stem in completed:
                continue
            out_path = run_dir / _gate_name(checkpoint)
            cmd = [
                sys.executable,
                "-m",
                "tinyPPO.phase1_gate",
                "--checkpoint",
                str(checkpoint),
                "--players-list",
                args.players_list,
                "--seeds",
                args.seeds,
                "--games-per-players",
                str(args.games_per_players),
                "--rows-per-game",
                str(args.rows_per_game),
                "--episode-steps",
                str(args.episode_steps),
                "--device",
                args.device,
                "--target-mask-mode",
                args.target_mask_mode,
                "--target-pair-weight",
                str(args.target_pair_weight),
                "--diagnose-policy",
                "--out",
                str(out_path),
            ]
            if args.baseline_checkpoint:
                cmd.extend(["--baseline-checkpoint", args.baseline_checkpoint])
            print(json.dumps({"event": "phase1_gate_start", "checkpoint": str(checkpoint), "out": str(out_path)}), flush=True)
            completed_process = subprocess.run(cmd, check=False)
            if completed_process.returncode != 0 and not out_path.exists():
                failure = {
                    "pass_phase1": False,
                    "reasons": [f"phase1_gate command failed with returncode {completed_process.returncode}"],
                    "candidate": {"checkpoint": str(checkpoint), "summary": None, "seeds": []},
                    "baseline": {"checkpoint": args.baseline_checkpoint, "summary": None, "seeds": []},
                    "config": {
                        "players_list": args.players_list,
                        "seeds": [int(seed) for seed in args.seeds.split(",") if seed.strip()],
                        "games_per_players": args.games_per_players,
                        "rows_per_game": args.rows_per_game,
                        "episode_steps": args.episode_steps,
                        "target_mask_mode": args.target_mask_mode,
                        "target_pair_weight": args.target_pair_weight,
                    },
                }
                out_path.write_text(json.dumps(failure, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "event": "phase1_gate_done",
                        "checkpoint": str(checkpoint),
                        "out": str(out_path),
                        "returncode": completed_process.returncode,
                    }
                ),
                flush=True,
            )
            completed.add(checkpoint.stem)
            if args.once:
                return
        if args.once:
            return
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
