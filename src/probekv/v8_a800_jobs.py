from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Mapping, Sequence, Tuple

from .v6_a800_jobs import V6A800Job, build_v6_a800_jobs
from .v7_a800_jobs import v7_a800_job_digest
from .v8_profile import validate_frozen_selector_profile


def build_v8_a800_jobs() -> Tuple[V6A800Job, ...]:
    raw = {
        "segment_count_samples_are_not_runtime_caps": True,
        "correctness_segments": [1, 2, 5, 10],
        "correctness_variants": [1, 4, 16],
        "profile_segments": [1, 5, 10, 15],
        "compare_k": [1, 2, 4, 8, 16],
        "probe_layers": [1, 2, 3, 4, 5, 6, 7, 8],
        "repair_ratios": [0.0, 0.05, 0.10, 0.16, 0.20, 0.30, 0.50, 0.75, 1.0],
        "warmups": 20,
        "repeats": 100,
    }
    jobs = build_v6_a800_jobs(raw)
    return tuple(replace(item, job_id=item.job_id.replace("v6-", "v8-", 1)) for item in jobs)


def build_v8_preprofile_manifest(
    *,
    code_commit: str,
    model_id: str,
    model_revision: str,
    tokenizer_hash: str,
    adapter_name: str,
    selection_execution_policy: str,
    checkpoint_depths: Sequence[int],
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
) -> Dict[str, Any]:
    if not all(
        str(value).strip()
        for value in (
            code_commit,
            model_id,
            model_revision,
            tokenizer_hash,
            adapter_name,
            cacheblend_patch_sha256,
            cacheblend_tree,
        )
    ):
        raise ValueError("v8 pre-profile provenance is incomplete")
    if selection_execution_policy not in {
        "causal_commit_wait", "immediate_staggered_closed_loop"
    }:
        raise ValueError("unsupported v8 execution policy")
    if not checkpoint_depths or min(checkpoint_depths) < 1:
        raise ValueError("v8 online checkpoints must start after d=0")
    return {
        "schema_version": 4,
        "protocol_version": 8,
        "stage": "v8_no_gpu_profile_preparation",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "code_commit": code_commit,
        "model": {
            "model_id": model_id,
            "revision": model_revision,
            "tokenizer_hash": tokenizer_hash,
            "adapter_name": adapter_name,
        },
        "selection_execution_policy": selection_execution_policy,
        "checkpoint_depths": list(checkpoint_depths),
        "negative_control_depths": [0],
        "compare_k": [1, 2, 4, 8, 16],
        "selection_state_tiers": ["pinned_cpu", "ssd"],
        "gpu_comparison_storage": "ephemeral_scratch",
        "development_datasets": ["musique", "2wikimultihopqa", "hotpotqa"],
        "profile_bound_qualification_manifest_generated": False,
        "cacheblend": {
            "patch_mode": "probekv_v8_training_free_residual_k",
            "patch_sha256": cacheblend_patch_sha256,
            "tree": cacheblend_tree,
        },
        "next_required_gate": "v8-A800-profile-freeze",
    }


def build_v8_profile_bound_qualification_manifest(
    jobs: Sequence[V6A800Job],
    *,
    profile: Mapping[str, Any],
    code_commit: str,
    model_id: str,
    model_revision: str,
    tokenizer_hash: str,
    adapter_name: str,
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
    jobs_sha256: str = "",
) -> Dict[str, Any]:
    if len(jobs) != 140 or any(job.paper_evidence for job in jobs):
        raise ValueError("v8 requires 140 frozen non-paper qualification jobs")
    validate_frozen_selector_profile(
        profile,
        code_commit=code_commit,
        model_revision=model_revision,
        tokenizer_hash=tokenizer_hash,
        cacheblend_patch_sha256=cacheblend_patch_sha256,
    )
    if not jobs_sha256:
        raise ValueError("Profile-bound qualification requires the JSONL SHA256")
    for key, value in (
        ("code_commit", code_commit),
        ("model_revision", model_revision),
        ("tokenizer_hash", tokenizer_hash),
        ("cacheblend_patch_sha256", cacheblend_patch_sha256),
    ):
        if profile.get(key) != value:
            raise ValueError("profile binding differs: %s" % key)
    return {
        "schema_version": 4,
        "protocol_version": 8,
        "stage": "v8_profile_bound_runtime_qualification_manifest",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "jobs": 140,
        "job_digest": v7_a800_job_digest(jobs),
        "jobs_sha256": jobs_sha256,
        "code_commit": code_commit,
        "profile_sha256": profile["profile_sha256"],
        "selection_execution_policy": profile["selection_execution_policy"],
        "model": {
            "model_id": model_id,
            "revision": model_revision,
            "tokenizer_hash": tokenizer_hash,
            "adapter_name": adapter_name,
        },
        "cacheblend": {
            "patch_mode": "probekv_v8_training_free_residual_k",
            "patch_sha256": cacheblend_patch_sha256,
            "tree": cacheblend_tree,
        },
        "runtime": {
            "training_free_selector": True,
            "fixed_repair_ratio": 0.15,
            "single_artifact_policy": True,
            "fake_timing_allowed": False,
        },
    }
