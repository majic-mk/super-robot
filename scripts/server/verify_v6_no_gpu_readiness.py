"""Hard gate for the immutable handoff from a CPU instance to an A800."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file
from probekv.v6_server_readiness import (
    collect_no_gpu_host,
    evaluate_no_gpu_readiness,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--server-lock", default="configs/a800_server_lock.json")
    parser.add_argument("--config", default="configs/v6_a800_microbench.json")
    parser.add_argument("--contract", default="configs/experiment_contract.yaml")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--job-manifest", required=True)
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    data_root = Path(args.data_root).resolve()
    server_lock = Path(args.server_lock).resolve()
    config = Path(args.config).resolve()
    contract = Path(args.contract).resolve()
    jobs = Path(args.jobs).resolve()
    host = collect_no_gpu_host(repo, data_root)
    result = evaluate_no_gpu_readiness(
        _json(server_lock),
        _json(Path(args.job_manifest).resolve()),
        host,
        _json(Path(args.model_audit).resolve()),
        _json(Path(args.patch_audit).resolve()),
        expected_code_commit=args.expected_commit,
        actual_hashes={
            "jobs_sha256": sha256_file(jobs),
            "config_sha256": sha256_file(config),
            "contract_sha256": sha256_file(contract),
            "server_lock_sha256": sha256_file(server_lock),
        },
    )
    result["host"] = host
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["artifact_preparation_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
