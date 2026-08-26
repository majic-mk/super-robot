from __future__ import annotations

from typing import Any, Mapping


class H1QualificationError(ValueError):
    """Raised before model import/loading when the GPU gate is not authoritative."""


def validate_h1_qualification_gate(
    gate: Mapping[str, Any],
    *,
    code_commit: str,
    model_id: str,
    model_revision: str,
    adapter_name: str,
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
    gpu_uuid: str,
) -> None:
    failures = []
    if int(gate.get("schema_version", 0)) != 2:
        failures.append("only schema-v2 qualification gates can unlock H1")
    if gate.get("stage") != "v6_a800_runtime_qualification":
        failures.append("qualification gate has the wrong stage")
    if gate.get("paper_evidence") is not False:
        failures.append("qualification gate must remain non-paper")
    if gate.get("locked_test_accessed") is not False:
        failures.append("qualification gate accessed a locked test")
    bindings = (
        ("code_commit", code_commit),
        ("model_id", model_id),
        ("model_revision", model_revision),
        ("adapter_name", adapter_name),
        ("cacheblend_patch_sha256", cacheblend_patch_sha256),
        ("cacheblend_tree", cacheblend_tree),
        ("gpu_uuid", gpu_uuid),
    )
    for key, expected in bindings:
        if gate.get(key) != expected:
            failures.append("qualification gate binding differs: %s" % key)
    for key in (
        "job_manifest_sha256",
        "runtime_audit_sha256",
        "native_prefix_cache_audit_sha256",
    ):
        if not str(gate.get(key, "")):
            failures.append("qualification gate is missing %s" % key)
    if int(gate.get("qualified_jobs_planned", -1)) != 140:
        failures.append("qualification gate did not plan exactly 140 jobs")
    if int(gate.get("qualified_jobs_completed", -1)) != 140:
        failures.append("qualification gate did not complete all 140 jobs")
    if int(gate.get("qualified_jobs_failed", -1)) != 0:
        failures.append("qualification gate contains a failed job")
    if gate.get("cuda_event_timing") is not True:
        failures.append("qualification gate contains fake or missing GPU timing")
    if gate.get("native_prefix_cache_qualified") is not True:
        failures.append("native Prefix Cache sentinel is not qualified")
    if gate.get("gpu_runtime_qualified") is not True:
        failures.append("GPU runtime is not qualified")
    if gate.get("h1_h2_execution_allowed") is not True:
        failures.append("qualification gate does not authorize H1/H2")
    if gate.get("failures") not in ([], ()): 
        failures.append("qualification gate contains recorded failures")
    if failures:
        raise H1QualificationError("; ".join(failures))

