"""Run and audit the patched official CacheBlend ten-sample example."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from probekv.io import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cacheblend", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    root = Path(args.cacheblend).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [sys.executable, "example/blend.py"],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.timeout_seconds,
    )
    log = process.stdout
    (output / "cb0_patched.log").write_text(log, encoding="utf-8")
    cached = log.count("Cached generation:")
    full = log.count("Normal generation:")
    forbidden = (
        "CUDA error",
        "out of memory",
        "illegal memory access",
        "KeyError",
        "Traceback (most recent call last)",
    )
    payload = {
        "gate": "CB0-patched",
        "passed": (
            process.returncode == 0
            and cached == 10
            and full == 10
            and not any(value.lower() in log.lower() for value in forbidden)
        ),
        "returncode": process.returncode,
        "cached_outputs": cached,
        "full_outputs": full,
        "forbidden_errors": [
            value for value in forbidden if value.lower() in log.lower()
        ],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "original_cb0_status": "failed_and_preserved_in_stage1",
        "paper_evidence": False,
        "evidence_class": "server_pilot",
    }
    atomic_write_json(output / "cb0_patched_gate.json", payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
