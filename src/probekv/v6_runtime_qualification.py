from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping


def evaluate_runtime_qualification(
    lock: Mapping[str, Any],
    job_manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reject contract-only or partial runs before H1/H2 is launched."""

    failures: List[str] = []
    runtime = lock.get("runtime", {})
    required = tuple(runtime.get("required_capabilities", ()))
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
    if audit.get("model_revision") != job_manifest.get("model", {}).get("revision"):
        failures.append("runtime audit used another model revision")
    if audit.get("cacheblend_patch_sha256") != job_manifest.get("cacheblend", {}).get("patch_sha256"):
        failures.append("runtime audit used another CacheBlend patchset")
    provenance = audit.get("runtime_provenance", {})
    expected_tree = job_manifest.get("cacheblend", {}).get("tree")
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
    if float(correctness.get("max_teacher_forced_logit_relative_l2", float("inf"))) > 1e-4:
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
    qualified = not failures
    return {
        "schema_version": 1,
        "stage": "v6_a800_runtime_qualification",
        "paper_evidence": False,
        "gpu_runtime_qualified": qualified,
        "h1_h2_execution_allowed": qualified,
        "failures": failures,
    }
