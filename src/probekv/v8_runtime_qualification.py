from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .v8_profile import validate_frozen_selector_profile


def evaluate_v8_runtime_qualification(
    lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
    profile: Mapping[str, Any],
    audit: Mapping[str, Any],
    prefix_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    failures: List[str] = []
    if lock.get("protocol_version") != 8 or lock.get("schema_version") != 4:
        failures.append("qualification lock is not v8 schema-v4")
    if manifest.get("protocol_version") != 8 or manifest.get("schema_version") != 4:
        failures.append("qualification manifest is not v8 schema-v4")
    manifest_model = manifest.get("model", {})
    locked_model_key = next(
        (
            key
            for key, item in lock.get("models", {}).items()
            if item.get("model_id") == manifest_model.get("model_id")
        ),
        None,
    )
    try:
        validate_frozen_selector_profile(
            profile,
            model_key=locked_model_key,
            policy=manifest.get("selection_execution_policy"),
            code_commit=manifest.get("code_commit"),
            model_revision=manifest_model.get("revision"),
            tokenizer_hash=manifest_model.get("tokenizer_hash"),
            cacheblend_patch_sha256=manifest.get("cacheblend", {}).get("patch_sha256"),
        )
    except ValueError as error:
        failures.append(str(error))
    if manifest.get("profile_sha256") != profile.get("profile_sha256"):
        failures.append("qualification manifest used another Profile")
    if audit.get("profile_sha256") != profile.get("profile_sha256"):
        failures.append("runtime audit used another Profile")
    if audit.get("cuda_event_timing") is not True or audit.get("fake_timing") is True:
        failures.append("qualification lacks real CUDA Event timing")
    if audit.get("protocol_version") != 8 or audit.get("schema_version") != 4:
        failures.append("runtime audit is not v8 schema-v4")
    if audit.get("paper_evidence") is not False or audit.get("locked_test_accessed") is not False:
        failures.append("runtime audit crossed the evidence boundary")
    if manifest.get("paper_evidence") is not False or manifest.get("locked_test_accessed") is not False:
        failures.append("qualification manifest crossed the evidence boundary")
    if manifest.get("cacheblend", {}).get("patch_mode") != "probekv_v8_training_free_residual_k":
        failures.append("qualification manifest used another CacheBlend mode")
    if audit.get("single_artifact_policy_verified") is not True:
        failures.append("single-Artifact policy was not verified")
    if audit.get("selection_state_k_only_verified") is not True:
        failures.append("K-only SelectionState path was not verified")
    if audit.get("selection_state_separate_backing_verified") is not True:
        failures.append("SelectionState backing was not independent from full KV")
    if not 0 <= int(audit.get("selection_scratch_peak_bytes", -1)) <= int(
        audit.get("selection_scratch_capacity_bytes", -2)
    ):
        failures.append("SelectionState GPU scratch exceeded its frozen capacity")
    if audit.get("fixed_repair_ratio") != 0.15:
        failures.append("qualification used another online repair ratio")
    required_capabilities = tuple(lock.get("runtime", {}).get("required_capabilities", ()))
    capabilities = audit.get("capabilities", {})
    for name in required_capabilities:
        if capabilities.get(name) is not True:
            failures.append("runtime capability missing: %s" % name)
    jobs = audit.get("jobs", {})
    if (jobs.get("planned"), jobs.get("completed"), jobs.get("failed")) != (140, 140, 0):
        failures.append("qualification did not complete 140/140 jobs")
    correctness = audit.get("correctness", {})
    for key in (
        "r1_dense_token_ids_equal",
        "source_digest_unchanged",
        "artifact_digest_unchanged",
        "absolute_union_mask_verified",
        "completed_depth_hook_verified",
    ):
        if correctness.get(key) is not True:
            failures.append("runtime correctness failed: %s" % key)
    selection = audit.get("selection_transfer", {})
    for key in (
        "request_attributed_full_kv_bytes_transferred_for_selection",
        "request_attributed_nonwinner_full_kv_bytes_transferred",
        "request_attributed_full_kv_prefetch_before_source_freeze",
    ):
        if int(selection.get(key, -1)) != 0:
            failures.append("selection transferred forbidden full KV: %s" % key)
    if prefix_audit.get("native_prefix_cache_qualified") is not True:
        failures.append("native Prefix Cache sentinel did not pass")
    bindings = {
        "code_commit": manifest.get("code_commit"),
        "model_id": manifest.get("model", {}).get("model_id"),
        "model_revision": manifest.get("model", {}).get("revision"),
        "adapter_name": manifest.get("model", {}).get("adapter_name"),
        "tokenizer_hash": manifest.get("model", {}).get("tokenizer_hash"),
        "cacheblend_patch_sha256": manifest.get("cacheblend", {}).get("patch_sha256"),
        "cacheblend_tree": manifest.get("cacheblend", {}).get("tree"),
        "profile_sha256": manifest.get("profile_sha256"),
        "job_digest": manifest.get("job_digest"),
    }
    for key, value in bindings.items():
        observed = audit.get(key)
        if observed != value:
            failures.append("runtime audit binding differs: %s" % key)
    locked_model = next(
        (
            item
            for item in lock.get("models", {}).values()
            if item.get("model_id") == bindings["model_id"]
        ),
        None,
    )
    if locked_model is None:
        failures.append("manifest model is absent from the v8 server lock")
    elif (
        locked_model.get("revision") != bindings["model_revision"]
        or locked_model.get("adapter_name") != bindings["adapter_name"]
    ):
        failures.append("manifest model binding differs from the v8 server lock")
    if prefix_audit.get("code_commit") != bindings["code_commit"]:
        failures.append("Prefix Cache audit used another code commit")
    if prefix_audit.get("model_revision") != bindings["model_revision"]:
        failures.append("Prefix Cache audit used another model revision")
    if prefix_audit.get("cacheblend_patch_sha256") != bindings["cacheblend_patch_sha256"]:
        failures.append("Prefix Cache audit used another CacheBlend patch")
    alignment_matches = (
        audit.get("runtime_vllm_block_size")
        == lock.get("runtime", {}).get("alignment_quantum")
    )
    if not alignment_matches:
        failures.append("runtime block size differs from the frozen alignment contract")
    qualified = not failures
    return {
        "schema_version": 4,
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
    profile_sha256: str,
    job_digest: str,
) -> None:
    failures = []
    if gate.get("protocol_version") != 8 or gate.get("schema_version") != 4:
        failures.append("only v8 schema-v4 Gate may unlock v8 H1")
    expected = {
        "code_commit": code_commit,
        "model_id": model_id,
        "model_revision": model_revision,
        "adapter_name": adapter_name,
        "tokenizer_hash": tokenizer_hash,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "cacheblend_tree": cacheblend_tree,
        "profile_sha256": profile_sha256,
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
