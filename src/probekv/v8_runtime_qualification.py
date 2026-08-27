from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .v8_profile import (
    validate_frozen_selector_profile,
    validate_profile_freeze_contract,
    validate_runtime_cost_profile,
)


def evaluate_v8_runtime_qualification(
    lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
    profile: Mapping[str, Any],
    runtime_cost_profile: Mapping[str, Any],
    profile_freeze_contract: Mapping[str, Any],
    audit: Mapping[str, Any],
    prefix_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a schema-v5 Gate from independently hashed GPU evidence."""

    failures: List[str] = []
    if lock.get("protocol_version") != 8 or lock.get("schema_version") != 5:
        failures.append("qualification lock is not v8 schema-v5")
    if manifest.get("protocol_version") != 8 or manifest.get("schema_version") != 5:
        failures.append("qualification manifest is not v8 schema-v5")
    model = manifest.get("model", {})
    model_key = next(
        (key for key, item in lock.get("models", {}).items()
         if item.get("model_id") == model.get("model_id")),
        None,
    )
    try:
        validate_frozen_selector_profile(
            profile,
            model_key=model_key,
            policy=manifest.get("selection_execution_policy"),
            code_commit=manifest.get("code_commit"),
            model_revision=model.get("revision"),
            tokenizer_hash=model.get("tokenizer_hash"),
            cacheblend_patch_sha256=manifest.get("cacheblend", {}).get("patch_sha256"),
        )
        if profile.get("schema_version") != 5:
            raise ValueError("schema-v4 Selector Profile is read-only and cannot unlock schema-v5")
        validate_profile_freeze_contract(profile_freeze_contract)
        validate_runtime_cost_profile(
            runtime_cost_profile,
            model_key=model_key,
            policy=manifest.get("selection_execution_policy"),
            code_commit=manifest.get("code_commit"),
            cacheblend_patch_sha256=manifest.get("cacheblend", {}).get("patch_sha256"),
        )
    except ValueError as error:
        failures.append(str(error))

    evidence_hashes = {
        "selector_profile_sha256": profile.get("profile_sha256"),
        "profile_freeze_contract_sha256": profile_freeze_contract.get(
            "profile_freeze_contract_sha256"
        ),
        "qualification_runtime_cost_profile_sha256": runtime_cost_profile.get(
            "runtime_cost_profile_sha256"
        ),
    }
    for key, expected in evidence_hashes.items():
        if not expected or manifest.get(key) != expected:
            failures.append("qualification manifest binding differs: %s" % key)
        if audit.get(key) != expected:
            failures.append("runtime audit binding differs: %s" % key)
    if manifest.get("profile_freeze_runtime_cost_profile_sha256") != profile.get(
        "profile_freeze_runtime_cost_profile_sha256"
    ):
        failures.append("qualification manifest lost the Profile-freeze runtime binding")

    if audit.get("protocol_version") != 8 or audit.get("schema_version") != 5:
        failures.append("runtime audit is not v8 schema-v5")
    for item, name in ((audit, "runtime audit"), (manifest, "qualification manifest")):
        if item.get("paper_evidence") is not False or item.get("locked_test_accessed") is not False:
            failures.append("%s crossed the evidence boundary" % name)
    if audit.get("cuda_event_timing") is not True or audit.get("fake_timing") is True:
        failures.append("qualification lacks real CUDA Event timing")
    if manifest.get("cacheblend", {}).get("patch_mode") != "probekv_v8_training_free_residual_k":
        failures.append("qualification manifest used another CacheBlend mode")
    for key, message in {
        "single_artifact_policy_verified": "single-Artifact policy was not verified",
        "selection_state_k_only_verified": "K-only SelectionState path was not verified",
        "selection_state_separate_backing_verified": (
            "SelectionState backing was not independent from full KV"
        ),
    }.items():
        if audit.get(key) is not True:
            failures.append(message)
    if not 0 <= int(audit.get("selection_scratch_peak_bytes", -1)) <= int(
        audit.get("selection_scratch_capacity_bytes", -2)
    ):
        failures.append("SelectionState GPU scratch exceeded its frozen capacity")
    if audit.get("fixed_repair_ratio") != 0.15:
        failures.append("qualification used another online repair ratio")
    for name in lock.get("runtime", {}).get("required_capabilities", ()):
        if audit.get("capabilities", {}).get(name) is not True:
            failures.append("runtime capability missing: %s" % name)

    jobs = audit.get("jobs", {})
    if (jobs.get("planned"), jobs.get("completed"), jobs.get("failed")) != (140, 140, 0):
        failures.append("qualification did not complete 140/140 jobs")
    for key in (
        "r1_dense_token_ids_equal",
        "source_digest_unchanged",
        "artifact_digest_unchanged",
        "absolute_union_mask_verified",
        "completed_depth_hook_verified",
    ):
        if audit.get("correctness", {}).get(key) is not True:
            failures.append("runtime correctness failed: %s" % key)
    for key in (
        "request_attributed_full_kv_bytes_transferred_for_selection",
        "request_attributed_nonwinner_full_kv_bytes_transferred",
        "request_attributed_full_kv_prefetch_before_source_freeze",
    ):
        if int(audit.get("selection_transfer", {}).get(key, -1)) != 0:
            failures.append("selection transferred forbidden full KV: %s" % key)

    if prefix_audit.get("native_prefix_cache_qualified") is not True:
        failures.append("native Prefix Cache sentinel did not pass")
    bindings = {
        "code_commit": manifest.get("code_commit"),
        "model_id": model.get("model_id"),
        "model_revision": model.get("revision"),
        "adapter_name": model.get("adapter_name"),
        "tokenizer_hash": model.get("tokenizer_hash"),
        "cacheblend_patch_sha256": manifest.get("cacheblend", {}).get("patch_sha256"),
        "cacheblend_tree": manifest.get("cacheblend", {}).get("tree"),
        "selector_profile_sha256": manifest.get("selector_profile_sha256"),
        "profile_freeze_contract_sha256": manifest.get("profile_freeze_contract_sha256"),
        "profile_freeze_runtime_cost_profile_sha256": manifest.get(
            "profile_freeze_runtime_cost_profile_sha256"
        ),
        "qualification_runtime_cost_profile_sha256": manifest.get(
            "qualification_runtime_cost_profile_sha256"
        ),
        "hardware_compatibility_signature": manifest.get(
            "hardware_compatibility_signature"
        ),
        "job_digest": manifest.get("job_digest"),
    }
    for key, expected in bindings.items():
        if audit.get(key) != expected:
            failures.append("runtime audit binding differs: %s" % key)
    locked_model = lock.get("models", {}).get(model_key, {}) if model_key else {}
    if not locked_model:
        failures.append("manifest model is absent from the v8 server lock")
    elif (locked_model.get("revision") != bindings["model_revision"]
          or locked_model.get("adapter_name") != bindings["adapter_name"]):
        failures.append("manifest model binding differs from the v8 server lock")
    for key, expected in (
        ("code_commit", bindings["code_commit"]),
        ("model_revision", bindings["model_revision"]),
        ("cacheblend_patch_sha256", bindings["cacheblend_patch_sha256"]),
    ):
        if prefix_audit.get(key) != expected:
            failures.append("Prefix Cache audit binding differs: %s" % key)

    alignment_matches = audit.get("runtime_vllm_block_size") == lock.get(
        "runtime", {}
    ).get("alignment_quantum")
    if not alignment_matches:
        failures.append("runtime block size differs from the frozen experiment contract")
    if audit.get("gpu_uuid") != manifest.get("qualification_gpu_uuid"):
        failures.append("qualification ran on another GPU than its RuntimeCostProfile")
    if audit.get("hardware_compatibility_signature") != manifest.get(
        "hardware_compatibility_signature"
    ):
        failures.append("qualification hardware signature differs from its RuntimeCostProfile")

    qualified = not failures
    return {
        "schema_version": 5,
        "protocol_version": 8,
        "stage": "v8_a800_profile_bound_runtime_qualification",
        "paper_evidence": False,
        "locked_test_accessed": False,
        **bindings,
        "gpu_uuid": audit.get("gpu_uuid"),
        "selector_profile_frozen": profile.get("selector_profile_frozen") is True,
        "native_prefix_cache_qualified": prefix_audit.get("native_prefix_cache_qualified") is True,
        "experiment_contract_compatible": alignment_matches,
        "gpu_runtime_qualified": qualified,
        "h1_h2_execution_allowed": qualified,
        "qualified_jobs_planned": jobs.get("planned"),
        "qualified_jobs_completed": jobs.get("completed"),
        "qualified_jobs_failed": jobs.get("failed"),
        "failures": failures,
    }


def validate_v8_h1_gate(
    gate: Mapping[str, Any],
    *,
    code_commit: str,
    model_id: str,
    model_revision: str,
    adapter_name: str,
    tokenizer_hash: str,
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
    selector_profile_sha256: str,
    profile_freeze_contract_sha256: str,
    profile_freeze_runtime_cost_profile_sha256: str,
    qualification_runtime_cost_profile_sha256: str,
    job_digest: str,
) -> None:
    failures = []
    if gate.get("protocol_version") != 8 or gate.get("schema_version") != 5:
        failures.append("only v8 schema-v5 Gate may unlock v8 H1")
    expected = {
        "code_commit": code_commit,
        "model_id": model_id,
        "model_revision": model_revision,
        "adapter_name": adapter_name,
        "tokenizer_hash": tokenizer_hash,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "cacheblend_tree": cacheblend_tree,
        "selector_profile_sha256": selector_profile_sha256,
        "profile_freeze_contract_sha256": profile_freeze_contract_sha256,
        "profile_freeze_runtime_cost_profile_sha256": profile_freeze_runtime_cost_profile_sha256,
        "qualification_runtime_cost_profile_sha256": qualification_runtime_cost_profile_sha256,
        "job_digest": job_digest,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            failures.append("qualification Gate binding differs: %s" % key)
    for key in (
        "selector_profile_frozen",
        "native_prefix_cache_qualified",
        "gpu_runtime_qualified",
        "h1_h2_execution_allowed",
    ):
        if gate.get(key) is not True:
            failures.append("qualification Gate did not pass %s" % key)
    if gate.get("paper_evidence") is not False or gate.get("locked_test_accessed") is not False:
        failures.append("qualification Gate crossed the evidence boundary")
    if gate.get("failures"):
        failures.append("qualification Gate contains recorded failures")
    if failures:
        raise RuntimeError("; ".join(failures))
