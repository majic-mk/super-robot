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
    parser.add_argument("--runtime-cost-profile", required=True)
    parser.add_argument("--profile-freeze-contract", required=True)
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
    runtime_cost_profile = load(Path(args.runtime_cost_profile).resolve())
    profile_freeze_contract = load(Path(args.profile_freeze_contract).resolve())
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
    policy_slug = profile["selection_execution_policy"]
    jobs_path = output / ("jobs_%s_%s.jsonl" % (args.model_key, policy_slug))
    write_jsonl(jobs_path, [item.to_row() for item in jobs])
    manifest = build_v8_profile_bound_qualification_manifest(
        jobs,
        profile=profile,
        runtime_cost_profile=runtime_cost_profile,
        profile_freeze_contract=profile_freeze_contract,
        code_commit=code_commit,
        model_id=model["model_id"],
        model_revision=model["revision"],
        tokenizer_hash=audit["tokenizer_hash"],
        adapter_name=model["adapter_name"],
        cacheblend_patch_sha256=patch["cacheblend_patch_sha256"],
        cacheblend_tree=patch["cacheblend_tree"],
        jobs_sha256=sha256_file(jobs_path),
    )
    write_json(output / ("manifest_%s_%s.json" % (args.model_key, policy_slug)), manifest)
    canary_path = output / ("canary_jobs_%s_%s.jsonl" % (args.model_key, policy_slug))
    write_jsonl(canary_path, [item.to_row() for item in jobs[:5]])
    write_json(
        output / ("canary_manifest_%s_%s.json" % (args.model_key, policy_slug)),
        {
            "schema_version": 5,
            "protocol_version": 8,
            "stage": "v8_profile_bound_five_job_canary",
            "paper_evidence": False,
            "locked_test_accessed": False,
            "code_commit": code_commit,
            "selector_profile_sha256": profile["profile_sha256"],
            "qualification_runtime_cost_profile_sha256": runtime_cost_profile[
                "runtime_cost_profile_sha256"
            ],
            "parent_qualification_job_digest": manifest["job_digest"],
            "canary_jobs": 5,
            "canary_jobs_sha256": sha256_file(canary_path),
            "may_unlock_h1": False,
        },
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
