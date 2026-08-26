from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

from .native_prefix_cache import evaluate_native_prefix_cache_audit


def evaluate_runtime_qualification(
    lock: Mapping[str, Any],
    job_manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
    prefix_audit: Mapping[str, Any] | None = None,
    *,
    job_manifest_sha256: str = "",
    runtime_audit_sha256: str = "",
    prefix_audit_sha256: str = "",
) -> Dict[str, Any]:
    """Build the schema-v2 gate that is the only valid H1/H2 authority."""

    failures: List[str] = []
    runtime = lock.get("runtime", {})
    required = tuple(runtime.get("required_capabilities", ()))
    model = job_manifest.get("model", {})
    cacheblend = job_manifest.get("cacheblend", {})
    if audit.get("paper_evidence") is not False:
        failures.append("runtime qualification must remain non-paper")
    if audit.get("runtime_backend") != runtime.get("backend"):
        failures.append("qualified runtime backend does not match the lock")
    if audit.get("concrete_engine_hook") is not True:
        failures.append("a concrete vLLM/CacheBlend engine hook was not exercised")
    capabilities = audit.get("capabilities", {})
    for capability in required:
        if capabilities.get(capability) is not True:
            failures.append("missing runtime capability: %s" % capability)
    if audit.get("code_commit") != job_manifest.get("code_commit"):
        failures.append("runtime audit used another ProbeKV commit")
    if audit.get("job_digest") != job_manifest.get("job_digest"):
        failures.append("runtime audit used another A800 job matrix")
    if audit.get("model_id") != model.get("model_id"):
        failures.append("runtime audit used another model")
    if audit.get("model_revision") != model.get("revision"):
        failures.append("runtime audit used another model revision")
    if audit.get("adapter_name") != model.get("adapter_name"):
        failures.append("runtime audit used another model adapter")
    if audit.get("cacheblend_patch_sha256") != cacheblend.get("patch_sha256"):
        failures.append("runtime audit used another CacheBlend patchset")
    provenance = audit.get("runtime_provenance", {})
    expected_tree = cacheblend.get("tree")
    if not expected_tree or provenance.get("cacheblend_tree") != expected_tree:
        failures.append("imported vLLM code is not the frozen CacheBlend tree")
    stack = lock.get("stack", {})
    for observed_key, expected_key in (
        ("torch", "pytorch"),
        ("vllm", "vllm"),
        ("xformers", "xformers"),
        ("torch_cuda", "pytorch_cuda"),
    ):
        observed = str(provenance.get(observed_key, "")).split("+", 1)[0]
        if observed != str(stack.get(expected_key, "")):
            failures.append("runtime %s differs from the server lock" % observed_key)
    gpu = lock.get("gpu", {})
    if not re.match(
        str(gpu.get("name_regex", r"^NVIDIA A800.*80GB$")),
        str(provenance.get("gpu_name", "")),
    ):
        failures.append("runtime GPU name differs from the server lock")
    capability = ".".join(
        str(value) for value in provenance.get("compute_capability", ())
    )
    if capability != str(gpu.get("compute_capability", "8.0")):
        failures.append("runtime compute capability differs from the server lock")
    correctness = audit.get("correctness", {})
    if correctness.get("r1_dense_token_ids_equal") is not True:
        failures.append("r=1 token IDs do not equal dense recomputation")
    try:
        logit_l2 = float(
            correctness.get("max_teacher_forced_logit_relative_l2", float("inf"))
        )
    except (TypeError, ValueError):
        logit_l2 = float("inf")
    if logit_l2 > 1e-4:
        failures.append("r=1 teacher-forced logit relative-L2 exceeds 1e-4")
    if correctness.get("canonical_source_digests_unchanged") is not True:
        failures.append("canonical Source digest changed")
    if correctness.get("absolute_union_mask_verified") is not True:
        failures.append("absolute union repair mask was not verified")
    jobs = audit.get("jobs", {})
    if int(jobs.get("planned", -1)) != int(job_manifest.get("jobs", -2)):
        failures.append("runtime audit planned-job count is inconsistent")
    if int(jobs.get("completed", -1)) != int(job_manifest.get("jobs", -2)):
        failures.append("not all frozen A800 jobs completed")
    if int(jobs.get("failed", -1)) != 0:
        failures.append("one or more frozen A800 jobs failed")
    if capabilities.get("cuda_event_timing") is not True:
        failures.append("GPU qualification did not use CUDA Event timing")

    expected_layers = {
        "mistral_cacheblend_llama_v041": 32,
        "qwen2_5_vllm041": 28,
    }.get(
        str(model.get("adapter_name", "")),
        int(prefix_audit.get("model_num_layers", 0)) if prefix_audit else 0,
    )
    validated_prefix = evaluate_native_prefix_cache_audit(
        prefix_audit or {}, expected_layers=expected_layers
    )
    if not prefix_audit:
        failures.append("schema-v2 qualification requires a Prefix Cache audit")
    else:
        failures.extend(
            "native prefix: %s" % item for item in validated_prefix["failures"]
        )
        bindings = (
            ("code_commit", job_manifest.get("code_commit")),
            ("model_id", model.get("model_id")),
            ("model_revision", model.get("revision")),
            ("adapter_name", model.get("adapter_name")),
            ("cacheblend_patch_sha256", cacheblend.get("patch_sha256")),
            ("cacheblend_tree", cacheblend.get("tree")),
            ("gpu_uuid", audit.get("gpu_uuid")),
        )
        for key, expected in bindings:
            if prefix_audit.get(key) != expected:
                failures.append("native Prefix Cache audit binding differs: %s" % key)

    for label, value in (
        ("job_manifest_sha256", job_manifest_sha256),
        ("runtime_audit_sha256", runtime_audit_sha256),
        ("prefix_audit_sha256", prefix_audit_sha256),
    ):
        if not value:
            failures.append("schema-v2 qualification is missing %s" % label)

    qualified = not failures
    return {
        "schema_version": 2,
        "stage": "v6_a800_runtime_qualification",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "code_commit": job_manifest.get("code_commit"),
        "model_id": model.get("model_id"),
        "model_revision": model.get("revision"),
        "adapter_name": model.get("adapter_name"),
        "cacheblend_patch_sha256": cacheblend.get("patch_sha256"),
        "cacheblend_tree": cacheblend.get("tree"),
        "job_manifest_sha256": job_manifest_sha256,
        "runtime_audit_sha256": runtime_audit_sha256,
        "native_prefix_cache_audit_sha256": prefix_audit_sha256,
        "gpu_uuid": audit.get("gpu_uuid"),
        "qualified_jobs_planned": int(jobs.get("planned", -1)),
        "qualified_jobs_completed": int(jobs.get("completed", -1)),
        "qualified_jobs_failed": int(jobs.get("failed", -1)),
        "cuda_event_timing": capabilities.get("cuda_event_timing") is True,
        "native_prefix_cache_qualified": bool(prefix_audit) and validated_prefix["passed"],
        "gpu_runtime_qualified": qualified,
        "h1_h2_execution_allowed": qualified,
        "failures": failures,
    }
