from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def _read_pid(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def _iter_events(log_path: Path) -> list[dict]:
    events: list[dict] = []
    if not log_path.exists():
        return events
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate(pid: int, grace_seconds: float) -> None:
    if not _process_alive(pid):
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + max(0.0, grace_seconds)
    while time.time() < deadline:
        if not _process_alive(pid):
            return
        time.sleep(1.0)
    if _process_alive(pid):
        os.kill(pid, signal.SIGKILL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop a BC run after a bad online eval checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pid-file", default="train.pid")
    parser.add_argument("--log-file", default="train.log")
    parser.add_argument("--min-epoch", type=int, default=1100)
    parser.add_argument("--min-nonloss", type=float, default=0.08)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--grace-seconds", type=float, default=30.0)
    parser.add_argument("--decision-out", default="stop_bad_online_eval_decision.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pid_path = run_dir / args.pid_file
    log_path = run_dir / args.log_file
    out_path = run_dir / args.decision_out
    print(
        json.dumps(
            {
                "event": "stop_bad_online_eval_start",
                "run_dir": str(run_dir),
                "min_epoch": args.min_epoch,
                "min_nonloss": args.min_nonloss,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    seen: set[tuple[int, int]] = set()
    while True:
        try:
            pid = _read_pid(pid_path)
        except Exception as exc:
            print(json.dumps({"event": "missing_pid", "error": str(exc)}, ensure_ascii=True), flush=True)
            time.sleep(max(1.0, args.poll_seconds))
            continue
        if not _process_alive(pid):
            decision = {"event": "process_exited", "pid": pid}
            out_path.write_text(json.dumps(decision, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(decision, ensure_ascii=True), flush=True)
            return

        events = _iter_events(log_path)
        for index, event in enumerate(events):
            if event.get("event") != "async_eval_done":
                continue
            epoch = int(event.get("epoch", -1))
            key = (epoch, index)
            if key in seen:
                continue
            seen.add(key)
            metrics = event.get("eval_vs_regular") or {}
            nonloss = float(metrics.get("nonloss", -1.0))
            decision = {
                "event": "online_eval_seen",
                "epoch": epoch,
                "nonloss": nonloss,
                "metrics": metrics,
                "threshold": args.min_nonloss,
            }
            print(json.dumps(decision, ensure_ascii=True), flush=True)
            if epoch >= args.min_epoch:
                if nonloss < args.min_nonloss:
                    decision.update({"action": "terminate", "pid": pid})
                    out_path.write_text(json.dumps(decision, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps(decision, ensure_ascii=True), flush=True)
                    _terminate(pid, args.grace_seconds)
                    return
                decision.update({"action": "continue", "pid": pid})
                out_path.write_text(json.dumps(decision, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
