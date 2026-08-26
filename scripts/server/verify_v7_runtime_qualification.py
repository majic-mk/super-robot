"""Validate one model's real A800 v7 runtime audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file
from probekv.v7_runtime_qualification import evaluate_v7_runtime_qualification


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-lock", default="configs/a800_server_lock_v7.json")
    parser.add_argument("--job-manifest", required=True)
    parser.add_argument("--runtime-audit", required=True)
    parser.add_argument("--prefix-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = Path(args.job_manifest).resolve()
    runtime = Path(args.runtime_audit).resolve()
    prefix = Path(args.prefix_audit).resolve()
    result = evaluate_v7_runtime_qualification(
        _json(Path(args.server_lock).resolve()),
        _json(manifest),
        _json(runtime),
        _json(prefix),
        job_manifest_sha256=sha256_file(manifest),
        runtime_audit_sha256=sha256_file(runtime),
        prefix_audit_sha256=sha256_file(prefix),
    )
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gpu_runtime_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
