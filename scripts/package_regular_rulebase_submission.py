"""Build a slim Kaggle submission tarball for the regular rulebase agent."""

from __future__ import annotations

import argparse
import py_compile
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "rulebase/kaggle_public_rl_informed_strategies"
DEFAULT_OUTPUT = (
    ROOT
    / ".kaggle/orbit_wars_regular_p4lowhome_active4_p2trickle_s30_p4midborder_s40_tail_m2_max14_net7_overpay4_slim_20260519.tar.gz"
)
MODULES = [
    "__init__.py",
    "agent.py",
    "defense.py",
    "geometry.py",
    "public_rule_agent.py",
    "rl_informed_agent.py",
    "rl_signals.py",
    "scoring.py",
    "ship_requirements.py",
    "state.py",
    "strategy_config.py",
]
WRAPPER_MAIN = '''"""Kaggle submission entry point for the regular rulebase agent."""
from rulebase.kaggle_public_rl_informed_strategies.agent import agent
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build-dir", type=Path, default=None)
    return parser.parse_args()


def remove_pycache(path: Path) -> None:
    for cache_dir in path.rglob("__pycache__"):
        shutil.rmtree(cache_dir)


def build_submission(output: Path, build_dir: Path | None) -> None:
    output = output.resolve()
    build_dir = (
        build_dir.resolve()
        if build_dir is not None
        else output.parent / f"build_{output.name.removesuffix('.tar.gz')}"
    )

    if build_dir.exists():
        shutil.rmtree(build_dir)
    package_dst = build_dir / "rulebase/kaggle_public_rl_informed_strategies"
    package_dst.mkdir(parents=True)
    (build_dir / "rulebase/__init__.py").write_text("", encoding="utf-8")
    (build_dir / "main.py").write_text(WRAPPER_MAIN, encoding="utf-8")

    for name in MODULES:
        shutil.copy2(PACKAGE / name, package_dst / name)

    py_compile.compile(str(build_dir / "main.py"), doraise=True)
    for path in package_dst.glob("*.py"):
        py_compile.compile(str(path), doraise=True)
    remove_pycache(build_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        tar.add(build_dir / "main.py", arcname="main.py")
        tar.add(build_dir / "rulebase", arcname="rulebase")

    print(output)


def main() -> None:
    args = parse_args()
    build_submission(args.output, args.build_dir)


if __name__ == "__main__":
    main()
