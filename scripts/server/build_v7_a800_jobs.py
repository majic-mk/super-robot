"""Freeze the non-paper v7 A800 single-Artifact qualification matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from probekv.cacheblend_patch import combined_patch_sha256, patch_files_for_mode
from probekv.io import sha256_file, write_json, write_jsonl
from probekv.v7_a800_jobs import build_v7_a800_job_manifest, build_v7_a800_jobs


def _git(command, repo: Path) -> str:
    return subprocess.check_output(["git", *command], cwd=str(repo), text=True).strip()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v7_a800_microbench.json")
    parser.add_argument("--contract", default="configs/experiment_contract.yaml")
    parser.add_argument("--server-lock", default="configs/a800_server_lock_v7.json")
    parser.add_argument("--patch-manifest", default="patches/cacheblend/manifest.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-key", choices=("mistral", "qwen"), required=True)
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).resolve()
    contract_path = Path(args.contract).resolve()
    lock_path = Path(args.server_lock).resolve()
    manifest_path = Path(args.patch_manifest).resolve()
    lock = _json(lock_path)
    model = lock["models"][args.model_key]
    model_audit = _json(Path(args.model_audit).resolve())
    patch_audit = _json(Path(args.patch_audit).resolve())
    jobs = build_v7_a800_jobs(_json(config_path))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs_path = output / ("jobs_%s.jsonl" % args.model_key)
    write_jsonl(jobs_path, [job.to_row() for job in jobs])
    patch_mode = lock["stack"]["cacheblend_patch_mode"]
    patch_paths = patch_files_for_mode(manifest_path, patch_mode)
    result = build_v7_a800_job_manifest(
        jobs,
        jobs_sha256=sha256_file(jobs_path),
        code_commit=_git(["rev-parse", "HEAD"], repo),
        git_clean=not bool(_git(["status", "--porcelain"], repo)),
        config_sha256=sha256_file(config_path),
        contract_sha256=sha256_file(contract_path),
        server_lock_sha256=sha256_file(lock_path),
        model_id=model["model_id"],
        model_revision=model["revision"],
        tokenizer_hash=model_audit["tokenizer_hash"],
        adapter_name=model["adapter_name"],
        runtime_backend=lock["runtime"]["backend"],
        cacheblend_commit=lock["stack"]["cacheblend_commit"],
        cacheblend_patch_mode=patch_mode,
        cacheblend_patch_sha256=combined_patch_sha256(patch_paths),
        cacheblend_tree=patch_audit["cacheblend_tree"],
    )
    write_json(output / ("manifest_%s.json" % args.model_key), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
