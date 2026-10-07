"""Run a fresh benchmark process with a deadline and an OOM/error receipt."""

import argparse
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main() -> None:
    """Retain a bounded attempt, including failures the runner cannot save."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "command": command,
        "timeout_s": args.timeout,
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "wrapper_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    start = time.perf_counter()
    timed_out = False
    with (args.output / "stdout.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    log_path = args.output / "stdout.log"
    with log_path.open("rb") as log:
        log.seek(max(0, log_path.stat().st_size - 16384))
        tail = log.read().decode(errors="replace")
    receipt.update(
        finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        elapsed_s=time.perf_counter() - start,
        returncode=code,
        status="timeout"
        if timed_out
        else "complete"
        if code == 0
        else "error",
        oom_reported="out of memory" in tail.lower(),
        output_tail=tail if code else None,
    )
    (args.output / "attempt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
