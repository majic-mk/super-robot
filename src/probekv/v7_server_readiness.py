from __future__ import annotations

from typing import Any, Dict, List, Mapping


def _version_matches(expected: Any, observed: Any, *, local_suffix: bool = False) -> bool:
    observed_text = str(observed)
    if local_suffix:
        observed_text = observed_text.split("+", 1)[0]
    return observed_text == str(expected)


def evaluate_v7_no_gpu_readiness(
    lock: Mapping[str, Any],
    job_manifests: Mapping[str, Mapping[str, Any]],
    host: Mapping[str, Any],
    storage: Mapping[str, Any],
    model_audits: Mapping[str, Mapping[str, Any]],
    patch_audit: Mapping[str, Any],
    runtime_source_audit: Mapping[str, Any],
    *,
    expected_code_commit: str,
    actual_hashes_by_model: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    failures: List[str] = []
    if lock.get("protocol_version") != 7 or lock.get("schema_version") != 3:
        failures.append("server lock is not the v7 schema-v3 lock")
    if host.get("git_commit") != expected_code_commit:
        failures.append("server checkout differs from the frozen v7 commit")
    # collect_no_gpu_host records porcelain output rather than a derived bool.
    if host.get("git_status"):
        failures.append("server worktree is not clean")
    platform = lock.get("platform", {})
    if not str(host.get("python", "")).startswith(
        str(platform.get("python_major_minor", "")) + "."
    ):
        failures.append("server Python differs from the v7 lock")
    if int(host.get("cpu_count", 0)) < int(platform.get("minimum_cpu_count", 0)):
        failures.append("server CPU count is below the v7 lock")
    if float(host.get("host_memory_gib", 0.0)) < float(
        platform.get("minimum_host_memory_gib", 0.0)
    ):
        failures.append("server memory is below the v7 lock")
    stack = lock.get("stack", {})
    if not _version_matches(stack.get("pytorch_cuda"), host.get("nvcc_cuda")):
        failures.append("nvcc CUDA differs from the v7 lock")
    packages = host.get("packages", {})
    for package, lock_key, allow_local_suffix in (
        ("torch", "pytorch", True),
        ("xformers", "xformers", False),
        ("vllm", "vllm", False),
        ("numpy", "numpy", False),
        ("transformers", "transformers", False),
        ("tokenizers", "tokenizers", False),
        ("huggingface-hub", "huggingface-hub", False),
        ("ray", "ray", False),
        ("cmake", "cmake", False),
        ("ninja", "ninja", False),
    ):
        if not _version_matches(
            stack.get(lock_key), packages.get(package),
            local_suffix=allow_local_suffix,
        ):
            failures.append("%s differs from the v7 lock" % package)
    if storage.get("storage_ready") is not True:
        failures.extend("storage: %s" % value for value in storage.get("failures", ()))
    if runtime_source_audit.get("runtime_source_ready") is not True:
        failures.extend(
            "runtime-source: %s" % value
            for value in runtime_source_audit.get("failures", ("audit failed",))
        )
    if runtime_source_audit.get("patch_mode") != "probekv_v7_single_artifact_runtime":
        failures.append("runtime source audit used another patch mode")
    expected_patch_mode = lock.get("stack", {}).get("cacheblend_patch_mode")
    if patch_audit.get("patch_mode") != expected_patch_mode:
        failures.append("patch audit used another CacheBlend mode")
    if patch_audit.get("cacheblend_commit") != stack.get("cacheblend_commit"):
        failures.append("patch audit used another CacheBlend base commit")
    per_model = {}
    for key in ("mistral", "qwen"):
        manifest = job_manifests.get(key, {})
        audit = model_audits.get(key, {})
        hashes = actual_hashes_by_model.get(key, {})
        model_failures = []
        if manifest.get("protocol_version") != 7 or manifest.get("schema_version") != 3:
            model_failures.append("job manifest is not protocol-v7 schema-v3")
        if manifest.get("jobs") != 140 or manifest.get("paper_evidence") is not False:
            model_failures.append("job manifest is not the 140-job non-paper matrix")
        if manifest.get("code_commit") != expected_code_commit:
            model_failures.append("manifest used another code commit")
        cacheblend = manifest.get("cacheblend", {})
        if cacheblend.get("patch_mode") != expected_patch_mode:
            model_failures.append("manifest used another CacheBlend patch mode")
        if cacheblend.get("patch_sha256") != patch_audit.get(
            "cacheblend_patch_sha256"
        ):
            model_failures.append("manifest and patch audit hashes differ")
        if cacheblend.get("tree") != patch_audit.get("cacheblend_tree"):
            model_failures.append("manifest and patched CacheBlend trees differ")
        model = lock.get("models", {}).get(key, {})
        if manifest.get("model", {}).get("model_id") != model.get("model_id"):
            model_failures.append("manifest model differs from lock")
        if manifest.get("model", {}).get("revision") != model.get("revision"):
            model_failures.append("manifest revision differs from lock")
        if audit.get("complete") is not True:
            model_failures.append("model audit is incomplete")
        if audit.get("model_id") != model.get("model_id"):
            model_failures.append("model audit used another model")
        if audit.get("revision") != model.get("revision"):
            model_failures.append("model audit used another revision")
        for label, expected in (
            ("jobs_sha256", manifest.get("jobs_sha256")),
            ("config_sha256", manifest.get("config_sha256")),
            ("contract_sha256", manifest.get("contract_sha256")),
            ("server_lock_sha256", manifest.get("server_lock_sha256")),
        ):
            if not expected or hashes.get(label) != expected:
                model_failures.append("%s binding differs" % label)
        per_model[key] = {"ready": not model_failures, "failures": model_failures}
        failures.extend("%s: %s" % (key, value) for value in model_failures)
    ready = not failures
    return {
        "schema_version": 3,
        "protocol_version": 7,
        "stage": "v7_dual_model_no_gpu_readiness",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "expected_code_commit": expected_code_commit,
        "artifact_preparation_ready": ready,
        "mistral_runtime_source_ready": ready and per_model.get("mistral", {}).get("ready") is True,
        "qwen_runtime_source_ready": ready and per_model.get("qwen", {}).get("ready") is True,
        "gpu_rental_ready_for_runtime_qualification": ready,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "per_model": per_model,
        "failures": failures,
    }
