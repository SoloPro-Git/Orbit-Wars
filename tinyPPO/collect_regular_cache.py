from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import ray

from tinyPPO.imitation_regular_ray import collect_regular_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect regular-vs-regular BC rows to a pickle cache without training.")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--out", required=True)
    parser.add_argument("--players-list", default="2")
    parser.add_argument("--games-per-players", type=int, default=1000)
    parser.add_argument("--collect-games-per-task", type=int, default=1)
    parser.add_argument("--rows-per-game", type=int, default=16)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--keep-noop-prob", type=float, default=0.15)
    parser.add_argument(
        "--row-target-mask-mode",
        choices=["candidate", "safe", "all_planets"],
        default="candidate",
        help="Target mask stored in newly collected BC rows.",
    )
    parser.add_argument("--seed", type=int, default=260526)
    parser.add_argument("--partial-cache-interval", type=int, default=0)
    parser.add_argument("--partial-cache-keep", type=int, default=2)
    parser.add_argument("--no-numba", action="store_true")
    args = parser.parse_args()

    ray.init(
        address=args.ray_address,
        ignore_reinit_error=True,
        runtime_env={
            "excludes": [
                "swanlog/**",
                "wandb/**",
                "tinyPPO/data/*.pkl",
                "tinyPPO/runs/**/*.pt",
                "tinyPPO/runs/**/*.pkl",
                "tinyPPO/runs/**/train.log",
            ]
        },
    )
    out = Path(args.out)
    rows, metrics = collect_regular_dataset(args, out)
    metrics["cache_path"] = str(out)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump({"rows": rows, "metrics": metrics, "args": vars(args)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(out)
        metrics["cache_saved"] = 1.0
    print(json.dumps({"event": "regular_cache_saved", "metrics": metrics}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
