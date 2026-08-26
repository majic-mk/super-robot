from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .v6_runtime_qualification import evaluate_runtime_qualification


def evaluate_v7_runtime_qualification(
    lock: Mapping[str, Any],
    job_manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
    prefix_audit: Mapping[str, Any],
    *,
    job_manifest_sha256: str,
    runtime_audit_sha256: str,
    prefix_audit_sha256: str,
) -> Dict[str, Any]:
    """Build the only gate allowed to authorize v7 H1/H2."""
    base = evaluate_runtime_qualification(
        lock,
        job_manifest,
        audit,
        prefix_audit,
        job_manifest_sha256=job_manifest_sha256,
        runtime_audit_sha256=runtime_audit_sha256,
        prefix_audit_sha256=prefix_audit_sha256,
    )
    failures: List[str] = list(base["failures"])
    if lock.get("protocol_version") != 7:
        failures.append("v7 qualification requires a protocol-v7 server lock")
    if job_manifest.get("protocol_version") != 7 or job_manifest.get("schema_version") != 3:
        failures.append("v7 qualification requires a schema-v3 job manifest")
    if audit.get("protocol_version") != 7:
        failures.append("runtime audit did not execute protocol v7")
    if audit.get("stage") != "v7_a800_runtime_qualification":
        failures.append("runtime audit has the wrong v7 stage")
    if audit.get("locked_test_accessed") is not False:
        failures.append("runtime qualification accessed or omitted locked-test state")
    if job_manifest.get("paper_evidence") is not False or job_manifest.get(
        "locked_test_accessed"
    ) is not False:
        failures.append("v7 job manifest changed evidence boundaries")
    if job_manifest.get("git_clean") is not True:
        failures.append("v7 job manifest was built from a dirty worktree")
    if job_manifest.get("runtime", {}).get("artifact_policy") != "single_canonical_lossless":
        failures.append("job manifest does not freeze the single-Artifact policy")
    if audit.get("single_artifact_policy_verified") is not True:
        failures.append("runtime did not verify one Artifact per Source Variant")
    if int(audit.get("max_artifacts_per_source_variant_observed", -1)) != 1:
        failures.append("runtime observed a Source with other than one Artifact")
    correctness = audit.get("correctness", {})
    if correctness.get("artifact_digests_unchanged") is not True:
        failures.append("canonical Artifact digest changed")
    if audit.get("all_job_artifact_digests_unchanged") is not True:
        failures.append("one or more qualification jobs did not verify Artifact digests")
    if audit.get("repair_rounding_policy") != "ceil":
        failures.append("runtime did not use conservative ceil repair rounding")
    runtime = lock.get("runtime", {})
    alignment = int(audit.get("alignment_quantum", -1))
    block_size = int(audit.get("runtime_vllm_block_size", -1))
    if alignment != int(runtime.get("alignment_quantum", 16)):
        failures.append("runtime alignment quantum differs from the v7 contract")
    if block_size != int(runtime.get("runtime_vllm_block_size", 16)):
        failures.append("runtime vLLM block size differs from the frozen experiment")
    alignment_matches = alignment == block_size == 16
    qualified = not failures
    return {
        **base,
        "schema_version": 3,
        "protocol_version": 7,
        "stage": "v7_a800_runtime_qualification",
        "artifact_policy": "single_canonical_lossless",
        "single_artifact_policy_verified": audit.get(
            "single_artifact_policy_verified"
        ) is True,
        "artifact_digests_unchanged": correctness.get(
            "artifact_digests_unchanged"
        ) is True,
        "all_job_artifact_digests_unchanged": audit.get(
            "all_job_artifact_digests_unchanged"
        ) is True,
        "repair_rounding_policy": audit.get("repair_rounding_policy"),
        "alignment_quantum": alignment,
        "runtime_vllm_block_size": block_size,
        "alignment_quantum_matches_runtime": alignment_matches,
        "experiment_contract_compatible": alignment_matches,
        "gpu_runtime_qualified": qualified,
        "h1_h2_execution_allowed": qualified,
        "failures": failures,
    }


def validate_v7_h1_gate(
    gate: Mapping[str, Any],
    *,
    code_commit: str,
    model_id: str,
    model_revision: str,
    adapter_name: str,
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
) -> None:
    failures = []
    if gate.get("schema_version") != 3 or gate.get("protocol_version") != 7:
        failures.append("only schema-v3 protocol-v7 gates can unlock v7 H1")
    if gate.get("stage") != "v7_a800_runtime_qualification":
        failures.append("qualification gate has the wrong stage")
    if gate.get("paper_evidence") is not False or gate.get("locked_test_accessed") is not False:
        failures.append("qualification gate changed evidence boundaries")
    expected = {
        "code_commit": code_commit,
        "model_id": model_id,
        "model_revision": model_revision,
        "adapter_name": adapter_name,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "cacheblend_tree": cacheblend_tree,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            failures.append("qualification gate binding differs: %s" % key)
    if gate.get("artifact_policy") != "single_canonical_lossless":
        failures.append("qualification gate used another Artifact policy")
    if gate.get("single_artifact_policy_verified") is not True:
        failures.append("single-Artifact policy was not GPU-verified")
    if gate.get("artifact_digests_unchanged") is not True:
        failures.append("Artifact digest invariance was not verified")
    if gate.get("all_job_artifact_digests_unchanged") is not True:
        failures.append("Artifact digest invariance was not verified for every job")
    if gate.get("repair_rounding_policy") != "ceil":
        failures.append("qualification used another repair rounding policy")
    for key in (
        "native_prefix_cache_qualified",
        "gpu_runtime_qualified",
        "h1_h2_execution_allowed",
        "experiment_contract_compatible",
        "cuda_event_timing",
    ):
        if gate.get(key) is not True:
            failures.append("qualification gate did not pass %s" % key)
    if int(gate.get("qualified_jobs_planned", -1)) != 140:
        failures.append("qualification did not plan 140 jobs")
    if int(gate.get("qualified_jobs_completed", -1)) != 140:
        failures.append("qualification did not complete 140 jobs")
    if int(gate.get("qualified_jobs_failed", -1)) != 0:
        failures.append("qualification contains failed jobs")
    if gate.get("failures"):
        failures.append("qualification gate contains recorded failures")
    if failures:
        raise RuntimeError("; ".join(failures))


def build_v7_joint_gate(
    *,
    code_commit: str,
    mistral_gate: Mapping[str, Any],
    qwen_gate: Mapping[str, Any],
    mistral_h1_sentinel: Mapping[str, Any],
    qwen_h1_sentinel: Mapping[str, Any],
) -> Dict[str, Any]:
    failures = []
    for name, gate in (("mistral", mistral_gate), ("qwen", qwen_gate)):
        if gate.get("code_commit") != code_commit:
            failures.append("%s gate used another commit" % name)
        if gate.get("gpu_runtime_qualified") is not True:
            failures.append("%s runtime is not qualified" % name)
    for name, sentinel in (
        ("mistral", mistral_h1_sentinel),
        ("qwen", qwen_h1_sentinel),
    ):
        if sentinel.get("protocol_version") != 7:
            failures.append("%s H1 sentinel used another protocol" % name)
        if sentinel.get("passed") is not True:
            failures.append("%s H1 sentinel failed" % name)
        if int(sentinel.get("appended_rows_this_run", -1)) != 36:
            failures.append("%s H1 sentinel did not produce 36 rows" % name)
        if sentinel.get("r1_dense_equivalence_passed") is not True:
            failures.append("%s H1 sentinel failed r=1 equivalence" % name)
    ready = not failures
    return {
        "schema_version": 3,
        "protocol_version": 7,
        "stage": "v7_dual_model_ready_for_full_h1",
        "code_commit": code_commit,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "mistral_runtime_qualified": mistral_gate.get("gpu_runtime_qualified") is True,
        "mistral_h1_sentinel_passed": mistral_h1_sentinel.get("passed") is True,
        "qwen_runtime_qualified": qwen_gate.get("gpu_runtime_qualified") is True,
        "qwen_h1_sentinel_passed": qwen_h1_sentinel.get("passed") is True,
        "ready_for_full_h1_pilot": ready,
        "full_h1_started": False,
        "failures": failures,
    }
