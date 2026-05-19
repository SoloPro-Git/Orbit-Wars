"""Copy local files to Ray worker nodes through Ray object transfer."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import ray


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXCLUDED_PARTS = {".ray_temp", "__pycache__", "checkpoints"}
DEFAULT_EXCLUDED_SUFFIXES = {".log", ".pyc"}


@ray.remote
class AssetWriter:
    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def stat(self, rel_path: str) -> dict[str, Any]:
        path = self.root / rel_path
        if not path.exists():
            return {"exists": False, "size": 0}
        return {"exists": True, "size": path.stat().st_size}

    def begin(self, rel_path: str) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def append(self, rel_path: str, chunk: bytes) -> None:
        path = self.root / rel_path
        with path.open("ab") as f:
            f.write(chunk)

    def finish(self, rel_path: str, expected_size: int, expected_sha256: str | None) -> dict[str, Any]:
        path = self.root / rel_path
        size = path.stat().st_size
        digest = None
        if expected_sha256:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        ok = size == expected_size and (expected_sha256 is None or digest == expected_sha256)
        return {"path": str(path), "size": size, "sha256": digest, "ok": ok}


def _is_excluded(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in DEFAULT_EXCLUDED_PARTS for part in rel.parts):
        return True
    return path.suffix in DEFAULT_EXCLUDED_SUFFIXES


def _iter_files(path: Path, root: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [p for p in sorted(path.rglob("*")) if p.is_file() and not _is_excluded(p, root)]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--resource", default="")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    ray.init(address=args.address, ignore_reinit_error=True)
    options: dict[str, Any] = {}
    if args.resource:
        options["resources"] = {args.resource: 0.001}
    writer = AssetWriter.options(**options).remote(args.root)

    root = Path(args.root)
    chunk_size = max(1, args.chunk_mb) * 1024 * 1024
    local_files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        local_files.extend(_iter_files(path, root))

    total_bytes = sum(p.stat().st_size for p in local_files)
    copied_bytes = 0
    for index, path in enumerate(local_files, 1):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        remote_stat = ray.get(writer.stat.remote(rel))
        if remote_stat.get("exists") and int(remote_stat.get("size", -1)) == size and not args.checksum:
            copied_bytes += size
            print(f"[skip] {index}/{len(local_files)} {rel} size={size}", flush=True)
            continue

        print(f"[copy] {index}/{len(local_files)} {rel} size={size}", flush=True)
        ray.get(writer.begin.remote(rel))
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                ray.get(writer.append.remote(rel, chunk))
                copied_bytes += len(chunk)
        digest = _sha256(path) if args.checksum else None
        result = ray.get(writer.finish.remote(rel, size, digest))
        if not result["ok"]:
            raise RuntimeError(f"sync failed for {rel}: {result}")
        print(f"[done] {rel} copied={copied_bytes}/{total_bytes}", flush=True)


if __name__ == "__main__":
    main()
