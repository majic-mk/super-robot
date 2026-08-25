from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file
from probekv.v6_server_readiness import (
    collect_no_gpu_host,
    evaluate_dual_model_no_gpu_readiness,
)


def _json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--server-lock", default="configs/a800_server_lock.json")
    parser.add_argument("--contract", default="configs/experiment_contract.yaml")
    parser.add_argument("--config", default="configs/v6_a800_microbench.json")
    parser.add_argument("--storage-audit", required=True)
    parser.add_argument("--runtime-source-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    for model in ("mistral", "qwen"):
        parser.add_argument("--%s-model-audit" % model, required=True)
        parser.add_argument("--%s-jobs" % model, required=True)
        parser.add_argument("--%s-manifest" % model, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    config = Path(args.config).resolve()
    contract = Path(args.contract).resolve()
    server_lock = Path(args.server_lock).resolve()
    manifests = {
        key: _json(getattr(args, "%s_manifest" % key))
        for key in ("mistral", "qwen")
    }
    audits = {
        key: _json(getattr(args, "%s_model_audit" % key))
        for key in ("mistral", "qwen")
    }
    hashes = {
        key: {
            "jobs_sha256": sha256_file(Path(getattr(args, "%s_jobs" % key))),
            "config_sha256": sha256_file(config),
            "contract_sha256": sha256_file(contract),
            "server_lock_sha256": sha256_file(server_lock),
        }
        for key in ("mistral", "qwen")
    }
    result = evaluate_dual_model_no_gpu_readiness(
        _json(server_lock), manifests,
        collect_no_gpu_host(repo, Path(args.data_root).resolve()),
        _json(args.storage_audit), audits, _json(args.patch_audit),
        _json(args.runtime_source_audit),
        expected_code_commit=args.expected_commit,
        actual_hashes_by_model=hashes,
    )
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gpu_rental_ready_for_runtime_qualification"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
