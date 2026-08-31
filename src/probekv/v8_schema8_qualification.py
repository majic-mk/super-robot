from __future__ import annotations

from typing import Any, Mapping


def evaluate_schema8_runtime_qualification(
    *,
    binding: Mapping[str, Any],
    runtime_audit: Mapping[str, Any],
    expected_jobs: int = 140,
) -> Mapping[str, Any]:
    """Build a schema-isolated Gate only from real, frozen GPU evidence."""

    required_binding = (
        "code_commit",
        "model_id",
        "model_revision",
        "cacheblend_patch_sha256",
        "cacheblend_tree",
        "job_manifest_sha256",
        "selection_depth_profile_sha256",
        "repair_policy_profile_sha256",
        "selected_repair_policy",
        "runtime_cost_profile_sha256",
        "gpu_uuid",
    )
    failures = [key for key in required_binding if not binding.get(key)]
    if (runtime_audit.get("protocol_version"), runtime_audit.get("schema_version")) != (8, 8):
        failures.append("runtime audit is not schema-v8")
    checks = {
        "planned": runtime_audit.get("planned") == expected_jobs,
        "completed": runtime_audit.get("completed") == expected_jobs,
        "failed": runtime_audit.get("failed") == 0,
        "cuda_event_timing": runtime_audit.get("cuda_event_timing") is True,
        "native_prefix_cache_qualified": runtime_audit.get(
            "native_prefix_cache_qualified"
        ) is True,
        "dense_d1_d2_barrier_verified": runtime_audit.get(
            "dense_d1_d2_barrier_verified"
        ) is True,
        "gate1_optimistic_marginal_verified": runtime_audit.get(
            "gate1_optimistic_marginal_verified"
        ) is True,
        "final_commit_joint_timeline_verified": runtime_audit.get(
            "final_commit_joint_timeline_verified"
        ) is True,
        "cpu_ssd_lru_verified": runtime_audit.get("cpu_ssd_lru_verified") is True,
        "backing_migration_verified": runtime_audit.get(
            "backing_migration_verified"
        ) is True,
        "repair_ratio_scope_verified": runtime_audit.get(
            "repair_ratio_scope_verified"
        ) is True,
        "r1_dense_equivalence": runtime_audit.get("r1_dense_equivalence") is True,
        "source_digest_unchanged": runtime_audit.get(
            "source_digest_unchanged"
        ) is True,
        "fake_timing": runtime_audit.get("fake_timing") is False,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    qualified = not failures
    return {
        "protocol_version": 8,
        "schema_version": 8,
        "gate_schema_version": 8,
        "stage": "v8_schema8_a800_runtime_qualification",
        **{key: binding.get(key) for key in required_binding},
        "planned": runtime_audit.get("planned", 0),
        "completed": runtime_audit.get("completed", 0),
        "failed": runtime_audit.get("failed", 0),
        "gpu_runtime_qualified": qualified,
        "h1_h2_execution_allowed": qualified,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "failures": failures,
    }


def validate_schema8_h1_gate(
    gate: Mapping[str, Any], *, expected_binding: Mapping[str, Any]
) -> None:
    if (
        gate.get("protocol_version"),
        gate.get("schema_version"),
        gate.get("gate_schema_version"),
    ) != (8, 8, 8):
        raise ValueError("only a schema-v8 Gate can unlock schema-v8 H1")
    for key, expected in expected_binding.items():
        if gate.get(key) != expected:
            raise ValueError("schema-v8 Gate binding differs: %s" % key)
    if gate.get("gpu_runtime_qualified") is not True:
        raise ValueError("schema-v8 GPU runtime is not qualified")
    if gate.get("h1_h2_execution_allowed") is not True:
        raise ValueError("schema-v8 H1/H2 remains locked")
    if gate.get("paper_evidence") is not False:
        raise ValueError("qualification Gate must remain non-paper evidence")
