from __future__ import annotations

from typing import Mapping, Sequence


def evaluate_schema9_runtime_qualification(
    *,
    runtime_audit: Mapping[str, object],
    code_commit: str,
    model_id: str,
    model_revision: str,
    cacheblend_patch_sha256: str,
    selection_depth_profile_sha256: str,
    variant_admission_profile_sha256: str,
    repair_policy_profile_sha256: str,
    runtime_cost_profile_sha256: str,
    manifest_sha256: str,
) -> Mapping[str, object]:
    if (runtime_audit.get("protocol_version"), runtime_audit.get("schema_version")) != (8, 9):
        raise ValueError("only a schema9 audit can unlock schema9")
    shas = (
        cacheblend_patch_sha256,
        selection_depth_profile_sha256,
        variant_admission_profile_sha256,
        repair_policy_profile_sha256,
        runtime_cost_profile_sha256,
        manifest_sha256,
    )
    if any(len(value) != 64 for value in shas):
        raise ValueError("schema9 qualification provenance SHA is incomplete")
    planned = int(runtime_audit.get("planned", -1))
    completed = int(runtime_audit.get("completed", -1))
    failed = int(runtime_audit.get("failed", -1))
    checks = {
        "job_matrix_complete": planned == completed == 140 and failed == 0,
        "real_cuda_timing": runtime_audit.get("cuda_event_timing") is True
        and runtime_audit.get("fake_timing") is not True,
        "absolute_residual_admission_verified": runtime_audit.get(
            "absolute_residual_admission_verified"
        )
        is True,
        "dense_exact_materialization_verified": runtime_audit.get(
            "dense_exact_materialization_verified"
        )
        is True,
        "partial_repair_promotion_forbidden": runtime_audit.get(
            "partial_repair_promotion_forbidden"
        )
        is True,
        "r1_dense_equivalence": runtime_audit.get("r1_dense_equivalence") is True,
        "source_digest_unchanged": runtime_audit.get("source_digest_unchanged") is True,
    }
    passed = all(checks.values())
    return {
        "protocol_version": 8,
        "schema_version": 9,
        "gate_schema_version": 9,
        "stage": "v8_schema9_runtime_qualification",
        "code_commit": code_commit,
        "model_id": model_id,
        "model_revision": model_revision,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "selection_depth_profile_sha256": selection_depth_profile_sha256,
        "variant_admission_profile_sha256": variant_admission_profile_sha256,
        "repair_policy_profile_sha256": repair_policy_profile_sha256,
        "runtime_cost_profile_sha256": runtime_cost_profile_sha256,
        "manifest_sha256": manifest_sha256,
        "checks": checks,
        "gpu_runtime_qualified": passed,
        "h1_h2_execution_allowed": passed,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "failures": [name for name, value in checks.items() if not value],
    }


def validate_schema9_h1_gate(
    gate: Mapping[str, object],
    *,
    expected_code_commit: str,
    expected_model_id: str,
    expected_profile_shas: Sequence[str],
) -> None:
    if (gate.get("protocol_version"), gate.get("schema_version")) != (8, 9):
        raise ValueError("schema8 or older Gate cannot unlock schema9 H1")
    if gate.get("gate_schema_version") != 9:
        raise ValueError("schema9 H1 requires Gate schema9")
    if gate.get("code_commit") != expected_code_commit:
        raise ValueError("schema9 Gate code SHA differs")
    if gate.get("model_id") != expected_model_id:
        raise ValueError("schema9 Gate model differs")
    actual = (
        gate.get("selection_depth_profile_sha256"),
        gate.get("variant_admission_profile_sha256"),
        gate.get("repair_policy_profile_sha256"),
        gate.get("runtime_cost_profile_sha256"),
    )
    if tuple(expected_profile_shas) != actual:
        raise ValueError("schema9 Gate Profile SHAs differ")
    if gate.get("gpu_runtime_qualified") is not True or gate.get(
        "h1_h2_execution_allowed"
    ) is not True:
        raise ValueError("schema9 GPU qualification has not passed")
