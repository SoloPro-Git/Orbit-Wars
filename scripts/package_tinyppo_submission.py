"""Build a Kaggle submission zip for the tinyPPO agent."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TINY_PPO_FILES = [
    "__init__.py",
    "agents.py",
    "features.py",
    "model.py",
]

MAIN_PY = '''"""Kaggle submission entry point for the tinyPPO stochastic agent."""

from __future__ import annotations

import os
from pathlib import Path

try:
    import torch

    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
except Exception:
    torch = None

from tinyPPO.agents import TinyPPOAgent


def _checkpoint_path():
    candidates = [
        Path("tinyppo_checkpoint.pt"),
        Path("/kaggle_simulations/agent/tinyppo_checkpoint.pt"),
    ]
    if "__file__" in globals():
        candidates.insert(0, Path(__file__).resolve().with_name("tinyppo_checkpoint.pt"))
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


_AGENT = TinyPPOAgent(
    _checkpoint_path(),
    device="cpu",
    deterministic=False,
)


def agent(obs, configuration=None):
    try:
        return _AGENT(obs, configuration)
    except Exception:
        return []
'''


def _cpu_checkpoint(src: Path, dst: Path) -> None:
    payload = torch.load(src, map_location="cpu", weights_only=True)
    state_dict = {key: value.detach().cpu() for key, value in payload["state_dict"].items()}
    torch.save(
        {
            "state_dict": state_dict,
            "model": payload.get("model", {}),
            "update": payload.get("update", -1),
            "metrics": payload.get("metrics", {}),
        },
        dst,
    )


def build_submission(checkpoint: Path, output: Path, build_dir: Path | None = None) -> None:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    with tempfile.TemporaryDirectory(prefix="tinyppo_submission_") as tmp:
        root = Path(tmp) if build_dir is None else build_dir.resolve()
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

        (root / "main.py").write_text(MAIN_PY, encoding="utf-8")
        package_dst = root / "tinyPPO"
        package_dst.mkdir()
        for name in TINY_PPO_FILES:
            shutil.copy2(ROOT / "tinyPPO" / name, package_dst / name)
        _cpu_checkpoint(checkpoint, root / "tinyppo_checkpoint.pt")

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-dir", default="")
    args = parser.parse_args()

    build_submission(
        Path(args.checkpoint),
        Path(args.output),
        Path(args.build_dir) if args.build_dir else None,
    )
    print({"output": args.output, "checkpoint": args.checkpoint, "stochastic": True})


if __name__ == "__main__":
    main()
