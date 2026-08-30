"""Bind a passing schema-v6 sentinel to the next immutable Profile run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file
from probekv.v8_schema6_jobs import build_schema6_runtime_profile_manifest


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=(
        "causal_commit_wait", "immediate_staggered_closed_loop"
    ))
    parser.add_argument("--sentinel-gate", required=True)
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--profile-freeze-contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    code_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=str(repo), text=True
    ).strip()
    if subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=str(repo), text=True
    ).strip():
        raise ValueError("schema-v6 Profile handoff requires a clean checkout")
    sentinel_path = Path(args.sentinel_gate).resolve()
    sentinel = load(sentinel_path)
    model = load(Path(args.model_audit).resolve())
    patch = load(Path(args.patch_audit).resolve())
    contract_path = Path(args.profile_freeze_contract).resolve()
    contract = load(contract_path)
    if (
        sentinel.get("code_commit") != code_commit
        or sentinel.get("schema_v6_runtime_contract_passed") is not True
        or sentinel.get("mistral_correctness_sentinel_passed") is not True
    ):
        raise ValueError("Profile handoff requires a passing sentinel at current SHA")
    if model.get("complete") is not True:
        raise ValueError("Mistral model audit is incomplete")
    if (
        contract.get("protocol_version") != 8
        or contract.get("profile_freeze_contract_frozen") is not True
    ):
        raise ValueError("Profile-freeze contract is not frozen")
    manifest = build_schema6_runtime_profile_manifest(
        policy=args.policy,
        code_commit=code_commit,
        model_revision=str(model["revision"]),
        tokenizer_hash=str(model["tokenizer_hash"]),
        cacheblend_patch_sha256=str(patch["cacheblend_patch_sha256"]),
        cacheblend_tree=str(patch["cacheblend_tree"]),
        sentinel_gate_sha256=sha256_file(sentinel_path),
        profile_freeze_contract_sha256=str(
            contract["profile_freeze_contract_sha256"]
        ),
    )
    atomic_write_json(Path(args.output).resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
