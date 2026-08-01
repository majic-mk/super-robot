"""Validate the A800-produced capability and correctness audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.v6_runtime_qualification import evaluate_runtime_qualification


def _json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-lock", default="configs/a800_server_lock.json")
    parser.add_argument("--job-manifest", required=True)
    parser.add_argument("--runtime-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_runtime_qualification(
        _json(args.server_lock),
        _json(args.job_manifest),
        _json(args.runtime_audit),
    )
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gpu_runtime_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
