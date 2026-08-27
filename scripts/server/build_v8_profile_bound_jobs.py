from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from probekv.io import sha256_file, write_json, write_jsonl
from probekv.v8_a800_jobs import (
    build_v8_a800_jobs,
    build_v8_profile_bound_qualification_manifest,
)
from probekv.v8_profile import validate_frozen_selector_profile


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--lock", default="configs/a800_server_lock_v8.json")
    parser.add_argument("--model-key", choices=("mistral", "qwen"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(repo), text=True
    ).strip():
        raise ValueError("Profile-bound jobs require the exact clean checkout")
    profile = load(Path(args.profile).resolve())
    audit = load(Path(args.model_audit).resolve())
    patch = load(Path(args.patch_audit).resolve())
    lock = load(Path(args.lock).resolve())
    model = lock["models"][args.model_key]
    if audit.get("complete") is not True:
        raise ValueError("model audit is incomplete")
    if patch.get("patch_mode") != lock["stack"]["cacheblend_patch_mode"]:
        raise ValueError("CacheBlend patch audit used another mode")
    validate_frozen_selector_profile(
        profile,
        model_key=args.model_key,
        code_commit=code_commit,
        model_revision=model["revision"],
        tokenizer_hash=audit["tokenizer_hash"],
        cacheblend_patch_sha256=patch["cacheblend_patch_sha256"],
    )
    jobs = build_v8_a800_jobs()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs_path = output / ("jobs_%s.jsonl" % args.model_key)
    write_jsonl(jobs_path, [item.to_row() for item in jobs])
    manifest = build_v8_profile_bound_qualification_manifest(
        jobs,
        profile=profile,
        code_commit=code_commit,
        model_id=model["model_id"],
        model_revision=model["revision"],
        tokenizer_hash=audit["tokenizer_hash"],
        adapter_name=model["adapter_name"],
        cacheblend_patch_sha256=patch["cacheblend_patch_sha256"],
        cacheblend_tree=patch["cacheblend_tree"],
        jobs_sha256=sha256_file(jobs_path),
    )
    write_json(output / ("manifest_%s.json" % args.model_key), manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
