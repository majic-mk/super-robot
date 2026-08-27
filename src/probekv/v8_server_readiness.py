from __future__ import annotations

from typing import Any, Dict, List, Mapping


def evaluate_v8_no_gpu_readiness(
    lock: Mapping[str, Any],
    preprofile_manifests: Mapping[str, Mapping[str, Any]],
    model_audits: Mapping[str, Mapping[str, Any]],
    patch_audit: Mapping[str, Any],
    *,
    expected_code_commit: str,
    actual_code_commit: str,
    git_clean: bool,
    storage_ready: bool,
    runtime_source_ready: bool,
) -> Dict[str, Any]:
    failures: List[str] = []
    if lock.get("protocol_version") != 8 or lock.get("schema_version") != 5:
        failures.append("server lock is not protocol-v8 schema-v5")
    if actual_code_commit != expected_code_commit:
        failures.append("checkout differs from the frozen v8 commit")
    if not git_clean:
        failures.append("worktree is not clean")
    if not storage_ready:
        failures.append("130GB storage contract is not ready")
    if not runtime_source_ready:
        failures.append("v8 runtime source audit failed")
    if patch_audit.get("patch_mode") != "probekv_v8_training_free_residual_k":
        failures.append("CacheBlend patch audit used another mode")
    if patch_audit.get("cacheblend_patch_sha256") is None or patch_audit.get("cacheblend_tree") is None:
        failures.append("CacheBlend patch audit lacks immutable patch/tree identity")
    per_profile = {}
    policies = {
        "causal_wait": "causal_commit_wait",
        "immediate_staggered": "immediate_staggered_closed_loop",
    }
    contract_hashes = set()
    partition_hashes = set()
    for model_key in ("mistral", "qwen"):
        audit = model_audits.get(model_key, {})
        for policy_key, expected_policy in policies.items():
            key = "%s_%s" % (model_key, policy_key)
            manifest = preprofile_manifests.get(key, {})
            model_failures = []
            if manifest.get("protocol_version") != 8 or manifest.get("schema_version") != 5:
                model_failures.append("pre-profile manifest is not v8 schema-v5")
            if manifest.get("stage") != "v8_no_gpu_profile_preparation":
                model_failures.append("pre-profile manifest has the wrong stage")
            if manifest.get("code_commit") != expected_code_commit:
                model_failures.append("pre-profile manifest used another commit")
            if manifest.get("selection_execution_policy") != expected_policy:
                model_failures.append("pre-profile manifest used another A/C policy")
            if manifest.get("profile_bound_qualification_manifest_generated") is not False:
                model_failures.append("final qualification was generated before Profile freeze")
            contract_hash = manifest.get("profile_freeze_contract_sha256")
            partition_hash = manifest.get("development_partition_sha256")
            if not contract_hash or not partition_hash:
                model_failures.append("pre-profile manifest lacks frozen contract/partition")
            else:
                contract_hashes.add(str(contract_hash))
                partition_hashes.add(str(partition_hash))
            if audit.get("complete") is not True:
                model_failures.append("model audit is incomplete")
            expected_model = lock.get("models", {}).get(model_key, {})
            if manifest.get("model", {}).get("model_id") != expected_model.get("model_id"):
                model_failures.append("manifest model differs from lock")
            if manifest.get("model", {}).get("revision") != expected_model.get("revision"):
                model_failures.append("manifest revision differs from lock")
            if manifest.get("model", {}).get("adapter_name") != expected_model.get("adapter_name"):
                model_failures.append("manifest adapter differs from lock")
            if manifest.get("model", {}).get("tokenizer_hash") != audit.get("tokenizer_hash"):
                model_failures.append("manifest tokenizer differs from model audit")
            if manifest.get("cacheblend", {}).get("patch_sha256") != patch_audit.get("cacheblend_patch_sha256"):
                model_failures.append("manifest CacheBlend patch differs from audit")
            if manifest.get("cacheblend", {}).get("tree") != patch_audit.get("cacheblend_tree"):
                model_failures.append("manifest CacheBlend tree differs from audit")
            per_profile[key] = {"ready": not model_failures, "failures": model_failures}
            failures.extend("%s: %s" % (key, value) for value in model_failures)
    if len(contract_hashes) != 1:
        failures.append("the four profiles do not share one Profile-freeze contract")
    if len(partition_hashes) != 1:
        failures.append("the four profiles do not share one development partition")
    ready = not failures
    return {
        "schema_version": 5,
        "protocol_version": 8,
        "stage": "v8_dual_model_no_gpu_profile_preparation",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "expected_code_commit": expected_code_commit,
        "v8_local_implementation_complete": ready,
        "v8_schema5_local_implementation_complete": ready,
        "artifact_preparation_ready": ready,
        "v8_no_gpu_profile_preparation_ready": ready,
        "gpu_rental_ready_for_profile_freeze": ready,
        "selector_profile_frozen": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "mistral_runtime_source_ready": all(
            per_profile[key]["ready"]
            for key in ("mistral_causal_wait", "mistral_immediate_staggered")
        ),
        "qwen_runtime_source_ready": all(
            per_profile[key]["ready"]
            for key in ("qwen_causal_wait", "qwen_immediate_staggered")
        ),
        "per_profile": per_profile,
        "failures": failures,
    }
