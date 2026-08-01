"""Freeze the non-paper v6 A800 correctness/microbenchmark job matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from probekv.cacheblend_patch import (
    combined_patch_sha256,
    load_patch_manifest,
    patch_files_for_mode,
)
from probekv.io import sha256_file, write_json, write_jsonl
from probekv.v6_a800_jobs import (
    build_v6_a800_job_manifest,
    build_v6_a800_jobs,
)


def _git(command: list[str], repo: Path) -> str:
    return subprocess.check_output(
        ["git", *command], cwd=str(repo), text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v6_a800_microbench.json"
    )
    parser.add_argument(
        "--contract", default="configs/experiment_contract.yaml"
    )
    parser.add_argument(
        "--server-lock", default="configs/a800_server_lock.json"
    )
    parser.add_argument(
        "--patch-manifest", default="patches/cacheblend/manifest.json"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).resolve()
    contract_path = Path(args.contract).resolve()
    server_lock_path = Path(args.server_lock).resolve()
    patch_manifest_path = Path(args.patch_manifest).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("protocol_version") != 6 or raw.get("paper_evidence"):
        raise ValueError("v6 A800 bring-up matrix must remain non-paper")
    lock = json.loads(server_lock_path.read_text(encoding="utf-8"))
    patch_manifest = load_patch_manifest(patch_manifest_path)
    patch_mode = str(lock["stack"]["cacheblend_patch_mode"])
    patch_paths = patch_files_for_mode(patch_manifest_path, patch_mode)
    jobs = build_v6_a800_jobs(raw)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs_path = output / "jobs.jsonl"
    write_jsonl(jobs_path, [job.to_row() for job in jobs])
    git_status = _git(["status", "--porcelain"], repo)
    summary = build_v6_a800_job_manifest(
        jobs,
        jobs_sha256=sha256_file(jobs_path),
        code_commit=_git(["rev-parse", "HEAD"], repo),
        git_clean=not bool(git_status),
        config_sha256=sha256_file(config_path),
        contract_sha256=sha256_file(contract_path),
        server_lock_sha256=sha256_file(server_lock_path),
        model_id=str(lock["model"]["model_id"]),
        model_revision=str(lock["model"]["revision"]),
        runtime_backend=str(lock["runtime"]["backend"]),
        runtime_implementation_status=str(
            lock["runtime"]["implementation_status"]
        ),
        cacheblend_commit=str(patch_manifest["base_commit"]),
        cacheblend_patch_mode=patch_mode,
        cacheblend_patch_sha256=combined_patch_sha256(patch_paths),
    )
    write_json(output / "manifest.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
