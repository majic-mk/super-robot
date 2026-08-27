from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Dict, Mapping, Sequence, Tuple

from .v6_a800_jobs import V6A800Job, build_v6_a800_jobs
from .v7_a800_jobs import v7_a800_job_digest
from .v8_profile import (
    validate_frozen_selector_profile,
    validate_profile_freeze_contract,
    validate_runtime_cost_profile,
)


V8_H1_RATIO_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0)


def v8_h1_ratio_grid_sha256() -> str:
    return hashlib.sha256(
        json.dumps(V8_H1_RATIO_GRID, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_v8_a800_jobs() -> Tuple[V6A800Job, ...]:
    raw = {
        "segment_count_samples_are_not_runtime_caps": True,
        "correctness_segments": [1, 2, 5, 10],
        "correctness_variants": [1, 4, 16],
        "profile_segments": [1, 5, 10, 15],
        "compare_k": [1, 2, 4, 8, 16],
        "probe_layers": [1, 2, 3, 4, 5, 6, 7, 8],
        "repair_ratios": list(V8_H1_RATIO_GRID),
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
    profile_freeze_contract_sha256: str = "",
    development_partition_sha256: str = "",
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
            profile_freeze_contract_sha256,
            development_partition_sha256,
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
        "schema_version": 5,
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
        "comparison_destination": "gpu_ephemeral_scratch",
        "selection_scratch_capacity_mib": 256,
        "development_datasets": ["musique", "2wikimultihopqa", "hotpotqa"],
        "development_partition_sha256": development_partition_sha256,
        "profile_freeze_contract_sha256": profile_freeze_contract_sha256,
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
    runtime_cost_profile: Mapping[str, Any],
    profile_freeze_contract: Mapping[str, Any],
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
    validate_profile_freeze_contract(profile_freeze_contract)
    validate_runtime_cost_profile(
        runtime_cost_profile,
        model_key=str(profile.get("model_key")),
        policy=str(profile.get("selection_execution_policy")),
        code_commit=code_commit,
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
    if (
        profile.get("profile_freeze_contract_sha256")
        != profile_freeze_contract.get("profile_freeze_contract_sha256")
    ):
        raise ValueError("Selector Profile used another Profile-freeze contract")
    return {
        "schema_version": 5,
        "protocol_version": 8,
        "stage": "v8_profile_bound_runtime_qualification_manifest",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "jobs": 140,
        "job_digest": v7_a800_job_digest(jobs),
        "jobs_sha256": jobs_sha256,
        "code_commit": code_commit,
        "selector_profile_sha256": profile["profile_sha256"],
        "profile_freeze_runtime_cost_profile_sha256": profile[
            "profile_freeze_runtime_cost_profile_sha256"
        ],
        "qualification_runtime_cost_profile_sha256": runtime_cost_profile[
            "runtime_cost_profile_sha256"
        ],
        "profile_freeze_contract_sha256": profile_freeze_contract[
            "profile_freeze_contract_sha256"
        ],
        "qualification_gpu_uuid": runtime_cost_profile["gpu_uuid"],
        "hardware_compatibility_signature": runtime_cost_profile[
            "hardware_compatibility_signature"
        ],
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
            "h1_ratio_grid": list(V8_H1_RATIO_GRID),
            "h1_ratio_grid_sha256": v8_h1_ratio_grid_sha256(),
            "single_artifact_policy": True,
            "fake_timing_allowed": False,
        },
    }


def build_v8_qualification_matrix_index(
    manifests: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Audit the four independent 140-job Model x A/C manifests."""
    expected = {
        "mistral_causal_wait": "causal_commit_wait",
        "mistral_immediate_staggered": "immediate_staggered_closed_loop",
        "qwen_causal_wait": "causal_commit_wait",
        "qwen_immediate_staggered": "immediate_staggered_closed_loop",
    }
    if set(manifests) != set(expected):
        raise ValueError("v8 qualification requires exactly four Model x A/C manifests")
    code_commits = set()
    profile_hashes = set()
    rows = {}
    for key, policy in expected.items():
        manifest = manifests[key]
        if manifest.get("protocol_version") != 8 or manifest.get("schema_version") != 5:
            raise ValueError("qualification matrix contains a non-schema-v5 manifest")
        if manifest.get("jobs") != 140 or manifest.get("paper_evidence") is not False:
            raise ValueError("each Model x A/C manifest must contain 140 non-paper jobs")
        if manifest.get("selection_execution_policy") != policy:
            raise ValueError("qualification policy binding differs: %s" % key)
        model_id = str(manifest.get("model", {}).get("model_id", "")).lower()
        if (key.startswith("mistral_") and "mistral" not in model_id) or (
            key.startswith("qwen_") and "qwen" not in model_id
        ):
            raise ValueError("qualification model binding differs: %s" % key)
        code_commits.add(str(manifest.get("code_commit", "")))
        profile_hashes.add(str(manifest.get("selector_profile_sha256", "")))
        rows[key] = {
            "model_id": manifest["model"]["model_id"],
            "policy": policy,
            "jobs": 140,
            "selector_profile_sha256": manifest["selector_profile_sha256"],
            "qualification_runtime_cost_profile_sha256": manifest[
                "qualification_runtime_cost_profile_sha256"
            ],
        }
    if len(code_commits) != 1 or "" in code_commits:
        raise ValueError("four qualification manifests must bind one code commit")
    if len(profile_hashes) != 4 or "" in profile_hashes:
        raise ValueError("each Model x A/C qualification needs an independent Profile")
    return {
        "schema_version": 5,
        "protocol_version": 8,
        "stage": "v8_four_profile_qualification_matrix",
        "code_commit": next(iter(code_commits)),
        "profiles": rows,
        "qualification_manifests": 4,
        "jobs_per_model_policy": 140,
        "jobs_total": 560,
        "paper_evidence": False,
        "locked_test_accessed": False,
    }
