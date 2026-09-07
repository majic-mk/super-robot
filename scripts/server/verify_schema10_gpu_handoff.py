"""Fail-closed preflight for a future schema10 GPU sentinel.

This command only checks immutable files and the manifest shape. It does not
load a model or start CUDA; the native factory remains the final hardware gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_storage import file_digest


REQUIRED_BINDING = ("code_commit", "patch_sha256", "config_sha256", "model_signature",
                    "model_revision", "tokenizer_hash", "global_byte_budget")


def verify(manifest_path: Path, *, repository: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    if manifest.get("manifest_sha256") != digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        errors.append("manifest_sha256_mismatch")
    if (manifest.get("protocol_version"), manifest.get("schema_version"),
            manifest.get("max_integrated_concurrency")) != (8, 10, 1):
        errors.append("wrong_protocol_or_concurrency")
    binding = manifest.get("binding", {})
    for field in REQUIRED_BINDING:
        value = binding.get(field)
        if not value or (field != "global_byte_budget" and not isinstance(value, str)):
            errors.append("missing_binding:" + field)
    if binding.get("code_commit"):
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
        if head != binding["code_commit"]:
            errors.append("checkout_sha_mismatch")
    if manifest.get("locked_test_accessed") is not False or manifest.get("paper_evidence") is not False:
        errors.append("paper_or_locked_test_flag")
    runtime = manifest.get("native_runtime")
    if not isinstance(runtime, dict):
        errors.append("native_runtime_attachment_missing")
    else:
        audit = runtime.get("model_audit_path")
        if not audit or not Path(audit).is_file():
            errors.append("model_audit_missing")
        elif file_digest(Path(audit)) != runtime.get("model_audit_sha256"):
            errors.append("model_audit_sha_mismatch")
        source = runtime.get("source_provenance", {})
        if source.get("model_revision") != binding.get("model_revision"):
            errors.append("model_revision_mismatch")
        if source.get("tokenizer_hash") != binding.get("tokenizer_hash"):
            errors.append("tokenizer_hash_mismatch")
        if runtime.get("cost_table_sha256"):
            errors.append("cost_table_already_bound_before_measurement")
        if runtime.get("sentinel_evidence_paths"):
            errors.append("correctness_evidence_must_be_generated_on_actual_gpu")
    return {"manifest": str(manifest_path), "code_commit": binding.get("code_commit"),
            "ready_for_gpu_sentinel": not errors, "errors": errors,
            "hardware_and_runtime_checked": False, "gpu_started": False,
            "paper_evidence": False, "locked_test_accessed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = verify(args.manifest.resolve(), repository=args.repository.resolve())
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0 if result["ready_for_gpu_sentinel"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
