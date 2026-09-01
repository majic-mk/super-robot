from __future__ import annotations

from typing import Mapping, Sequence


def evaluate_schema10_runtime_qualification(
    *,
    runtime_audit: Mapping[str, object],
    code_commit: str,
    model_id: str,
    model_revision: str,
    cacheblend_patch_sha256: str,
    selection_depth_profile_sha256: str,
    variant_admission_profile_sha256: str,
    preparation_policy_profile_sha256: str,
    repair_policy_profile_sha256: str,
    runtime_cost_profile_sha256: str,
    manifest_sha256: str,
) -> Mapping[str, object]:
    if (runtime_audit.get("protocol_version"), runtime_audit.get("schema_version")) != (8, 10):
        raise ValueError("only a schema10 audit can unlock schema10")
    shas = (
        cacheblend_patch_sha256,
        selection_depth_profile_sha256,
        variant_admission_profile_sha256,
        preparation_policy_profile_sha256,
        repair_policy_profile_sha256,
        runtime_cost_profile_sha256,
        manifest_sha256,
    )
    if any(len(value) != 64 for value in shas):
        raise ValueError("schema10 qualification provenance SHA is incomplete")
    expected_audit_provenance = {
        "code_commit": code_commit,
        "model_id": model_id,
        "model_revision": model_revision,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "selection_depth_profile_sha256": selection_depth_profile_sha256,
        "variant_admission_profile_sha256": variant_admission_profile_sha256,
        "preparation_policy_profile_sha256": preparation_policy_profile_sha256,
        "repair_policy_profile_sha256": repair_policy_profile_sha256,
        "runtime_cost_profile_sha256": runtime_cost_profile_sha256,
        "manifest_sha256": manifest_sha256,
    }
    checks = {
        "audit_provenance_match": all(
            runtime_audit.get(name) == expected
            for name, expected in expected_audit_provenance.items()
        ),
        "profiles_frozen": runtime_audit.get("profiles_frozen") is True,
        "gpu_identity_recorded": bool(runtime_audit.get("gpu_uuid"))
        and len(str(runtime_audit.get("runtime_environment_hash", ""))) == 64,
        "job_matrix_complete": int(runtime_audit.get("planned", -1)) == 140
        and int(runtime_audit.get("completed", -1)) == 140
        and int(runtime_audit.get("failed", -1)) == 0,
        "real_cuda_timing": runtime_audit.get("cuda_event_timing") is True
        and runtime_audit.get("fake_timing") is not True,
        "dense_exact_materialization_verified": runtime_audit.get("dense_exact_materialization_verified") is True,
        "bounded_probation_verified": runtime_audit.get("bounded_probation_verified") is True,
        "exploration_novelty_separation_verified": runtime_audit.get("exploration_novelty_separation_verified") is True,
        "gate1_counterfactual_verified": runtime_audit.get("gate1_counterfactual_verified") is True,
        "atomic_preparation_reservation_verified": runtime_audit.get("atomic_preparation_reservation_verified") is True,
        "final_commit_admission_verified": runtime_audit.get("final_commit_admission_verified") is True,
        "r1_dense_equivalence": runtime_audit.get("r1_dense_equivalence") is True,
    }
    passed = all(checks.values())
    return {
        "protocol_version": 8,
        "schema_version": 10,
        "gate_schema_version": 10,
        "stage": "v8_schema10_runtime_qualification",
        "code_commit": code_commit,
        "model_id": model_id,
        "model_revision": model_revision,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "selection_depth_profile_sha256": selection_depth_profile_sha256,
        "variant_admission_profile_sha256": variant_admission_profile_sha256,
        "preparation_policy_profile_sha256": preparation_policy_profile_sha256,
        "repair_policy_profile_sha256": repair_policy_profile_sha256,
        "runtime_cost_profile_sha256": runtime_cost_profile_sha256,
        "manifest_sha256": manifest_sha256,
        "gpu_uuid": runtime_audit.get("gpu_uuid", ""),
        "runtime_environment_hash": runtime_audit.get(
            "runtime_environment_hash", ""
        ),
        "checks": checks,
        "gpu_runtime_qualified": passed,
        "h1_h2_execution_allowed": passed,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "failures": [name for name, value in checks.items() if not value],
    }


def validate_schema10_h1_gate(
    gate: Mapping[str, object],
    *,
    expected_code_commit: str,
    expected_model_id: str,
    expected_model_revision: str,
    expected_cacheblend_patch_sha256: str,
    expected_manifest_sha256: str,
    expected_profile_shas: Sequence[str],
) -> None:
    if (gate.get("protocol_version"), gate.get("schema_version"), gate.get("gate_schema_version")) != (8, 10, 10):
        raise ValueError("schema9 or older Gate cannot unlock schema10 H1")
    if (
        gate.get("code_commit") != expected_code_commit
        or gate.get("model_id") != expected_model_id
        or gate.get("model_revision") != expected_model_revision
        or gate.get("cacheblend_patch_sha256")
        != expected_cacheblend_patch_sha256
        or gate.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("schema10 Gate provenance differs")
    actual = (
        gate.get("selection_depth_profile_sha256"),
        gate.get("variant_admission_profile_sha256"),
        gate.get("preparation_policy_profile_sha256"),
        gate.get("repair_policy_profile_sha256"),
        gate.get("runtime_cost_profile_sha256"),
    )
    if tuple(expected_profile_shas) != actual:
        raise ValueError("schema10 Gate Profile SHAs differ")
    if gate.get("gpu_runtime_qualified") is not True or gate.get("h1_h2_execution_allowed") is not True:
        raise ValueError("schema10 GPU qualification has not passed")
