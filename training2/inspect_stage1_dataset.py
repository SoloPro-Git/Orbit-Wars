"""Inspect stage1 JSONL data for candidate/target health."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def _iter_jsonl(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--limit", type=int, default=20000)
    args = parser.parse_args()

    path = Path(args.path)
    target_counts: collections.Counter[int] = collections.Counter()
    candidate_counts: collections.Counter[int] = collections.Counter()
    proposal_label_rows = 0
    rows = 0
    files = _iter_jsonl(path)
    for file_path in files:
        with file_path.open() as f:
            for line in f:
                if args.limit and rows >= args.limit:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                target_counts[int(row.get("target", -1))] += 1
                candidate_counts[int(sum(row.get("candidate_mask", [])))] += 1
                proposal_label_rows += int("proposal_valid" in row)
        if args.limit and rows >= args.limit:
            break

    print(
        json.dumps(
            {
                "path": str(path),
                "files": len(files),
                "rows": rows,
                "target_counts": dict(sorted(target_counts.items())),
                "target_top": target_counts.most_common(10),
                "candidate_counts": dict(sorted(candidate_counts.items())),
                "candidate_top": candidate_counts.most_common(10),
                "proposal_label_rows": proposal_label_rows,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
