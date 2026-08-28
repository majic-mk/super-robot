"""Build the immutable no-GPU handoff for the schema-v6 Mistral sentinel."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file
from probekv.model_adapters import MISTRAL_SCHEMA6_SPEC
from probekv.v8_schema6_jobs import build_schema6_sentinel_manifest


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default="configs/v8_schema6_a800_sentinel.json"
    )
    parser.add_argument(
        "--contract", default="configs/experiment_contract_v8_schema6.yaml"
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(repo), text=True
    ).strip()
    if status:
        raise RuntimeError("schema-v6 handoff requires a clean Git worktree")
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
    ).strip()
    model_audit = load(Path(args.model_audit).resolve())
    patch_audit = load(Path(args.patch_audit).resolve())
    if (
        model_audit.get("complete") is not True
        or model_audit.get("model_id") != MISTRAL_SCHEMA6_SPEC.model_id
        or model_audit.get("revision") != MISTRAL_SCHEMA6_SPEC.revision
    ):
        raise ValueError("Mistral model audit is incomplete or incompatible")
    if patch_audit.get("patch_mode") != "probekv_v8_schema6_joint_cfo":
        raise ValueError("schema-v6 handoff requires the schema-v6 CacheBlend patch")
    manifest = build_schema6_sentinel_manifest(
        code_commit=code_commit,
        model_revision=MISTRAL_SCHEMA6_SPEC.revision,
        tokenizer_hash=str(model_audit["tokenizer_hash"]),
        cacheblend_patch_sha256=str(patch_audit["cacheblend_patch_sha256"]),
        cacheblend_tree=str(patch_audit["cacheblend_tree"]),
        config_sha256=sha256_file((repo / args.config).resolve()),
        contract_sha256=sha256_file((repo / args.contract).resolve()),
    )
    atomic_write_json(Path(args.output).resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
