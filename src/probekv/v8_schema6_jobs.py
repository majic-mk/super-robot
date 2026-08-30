from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple

from .model_adapters import (
    MISTRAL_SCHEMA6_SPEC,
    validate_schema6_checkpoint_contract,
)


def _job(kind: str, coordinates: Mapping[str, Any], *, anchor: bool = False) -> Dict[str, Any]:
    payload = {
        "kind": kind,
        "coordinates": dict(coordinates),
        "warmups": 20 if anchor else 2,
        "repeats": 100 if anchor else 5,
        "paper_evidence": False,
        "locked_test_accessed": False,
    }
    payload["job_id"] = "schema6-%s-%s" % (
        kind,
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16],
    )
    return payload


def build_mistral_schema6_sentinel_jobs() -> Tuple[Dict[str, Any], ...]:
    jobs = [
        _job("hardware_environment_gate", {}),
        _job("native_prefix_cache_sentinel", {"cached_prefix_tokens_min": 128}),
        _job("k_hook_depth_sentinel", {"depths": [0, 1, 5, 8]}),
        _job("r1_dense_equivalence_sentinel", {"repair_ratio": 1.0}),
        _job("cfo_eager_streaming_sentinel", {"token_count": 128, "segment_count": 2}),
    ]
    checkpoints = MISTRAL_SCHEMA6_SPEC.checkpoints
    for compared_k in (1, 2, 4, 8, 16):
        for depth in checkpoints:
            jobs.append(
                _job(
                    "comparison_batch",
                    {
                        "k": compared_k,
                        "token_count": 512,
                        "completed_depth": depth,
                        "backing_tier": "pinned_cpu",
                    },
                    anchor=(compared_k == 16 and depth == 5),
                )
            )
    for token_count in (128, 640):
        for compared_k in (1, 16):
            for depth in (1, 5, 8):
                jobs.append(
                    _job(
                        "comparison_batch",
                        {
                            "k": compared_k,
                            "token_count": token_count,
                            "completed_depth": depth,
                            "backing_tier": "pinned_cpu",
                        },
                    )
                )
    for segment_count in (1, 2, 5):
        for policy in ("causal_commit_wait", "immediate_staggered_closed_loop"):
            jobs.append(
                _job(
                    "joint_gate2_gate3",
                    {"segment_count": segment_count, "policy": policy},
                )
            )
    jobs.extend(
        (
            _job("selection_state_transfer", {"token_count": 512, "tier": "pinned_cpu", "batch_k": 16}),
            _job("full_kv_tier_load", {"token_count": 512, "source_tier": "pinned_cpu", "layer_range": [1, 32]}),
            _job("repair", {"token_count": 512, "repair_ratio": 0.15, "boundary": 5}),
            _job("repair", {"token_count": 512, "repair_ratio": 1.0, "boundary": 5}),
            _job("winner_deferred_lease_promotion", {"policy": "immediate_staggered_closed_loop"}),
            _job("gate3_subset", {"segment_count": 5}),
        )
    )
    if len({row["job_id"] for row in jobs}) != len(jobs):
        raise RuntimeError("schema-v6 sentinel job IDs are not unique")
    return tuple(jobs)


def build_mistral_schema6_runtime_profile_jobs(
    policy: str,
) -> Tuple[Dict[str, Any], ...]:
    """Build the immutable full RuntimeCostProfile measurement matrix.

    Unlike the bounded sentinel, every cell uses the preregistered 20 warmups
    and 100 retained samples required by the bootstrap evidence contract.
    """

    if policy not in {"causal_commit_wait", "immediate_staggered_closed_loop"}:
        raise ValueError("unsupported schema-v6 runtime Profile policy")
    jobs = []
    checkpoints = MISTRAL_SCHEMA6_SPEC.checkpoints
    for token_count in (128, 512, 640):
        for depth in checkpoints:
            for compared_k in (1, 2, 4, 8, 16):
                jobs.append(
                    _job(
                        "comparison_batch",
                        {
                            "k": compared_k,
                            "token_count": token_count,
                            "completed_depth": depth,
                            "backing_tier": "pinned_cpu",
                        },
                        anchor=True,
                    )
                )
    head_dim = 128
    bytes_per_selection_state_token = (
        MISTRAL_SCHEMA6_SPEC.num_kv_heads * head_dim * 2
    )
    for token_count in (128, 512, 640):
        for batch_k in (1, 4, 16):
            bytes_count = token_count * bytes_per_selection_state_token * batch_k
            for tier in ("pinned_cpu", "ssd"):
                jobs.append(
                    _job(
                        "selection_state_transfer",
                        {"bytes": bytes_count, "tier": tier, "batch_k": batch_k},
                        anchor=True,
                    )
                )
    for token_count in (128, 512, 640):
        for source_tier in ("pinned_cpu", "ssd"):
            for layer_range in ([1, 32], [5, 32]):
                bytes_count = (
                    token_count
                    * MISTRAL_SCHEMA6_SPEC.num_kv_heads
                    * head_dim
                    * 2
                    * 2
                    * (layer_range[1] - layer_range[0] + 1)
                )
                jobs.append(
                    _job(
                        "full_kv_tier_load",
                        {
                            "bytes": bytes_count,
                            "source_tier": source_tier,
                            "layer_range": layer_range,
                            "token_count": token_count,
                        },
                        anchor=True,
                    )
                )
    for segment_count in (1, 2, 5):
        for boundary in (2, 5, 8):
            vector = [boundary] * segment_count
            active_rows = 512 * segment_count
            jobs.append(
                _job(
                    "dense_remaining_joint",
                    {
                        "boundary_vector": vector,
                        "active_rows": active_rows,
                        "segment_count": segment_count,
                    },
                    anchor=True,
                )
            )
            jobs.append(
                _job(
                    "union_mask_remaining",
                    {
                        "boundary_vector": vector,
                        "layer_active_rows": [active_rows] * (33 - boundary),
                        "segment_count": segment_count,
                    },
                    anchor=True,
                )
            )
    for token_count in (128, 512, 640):
        for boundary in (2, 5, 8):
            for ratio in (0.15, 1.0):
                ratio_percent = int(round(ratio * 100))
                repair_count = min(
                    token_count, (token_count * ratio_percent + 99) // 100
                )
                jobs.append(
                    _job(
                        "repair",
                        {
                            "boundary": boundary,
                            "token_count": token_count,
                            "repair_count": repair_count,
                            "repair_ratio": ratio,
                        },
                        anchor=True,
                    )
                )
    for copy_bytes in (1024 * 1024, 16 * 1024 * 1024):
        for overlap in (False, True):
            for concurrency in (1, 2):
                jobs.append(
                    _job(
                        "interference",
                        {
                            "copy_bytes": copy_bytes,
                            "overlap": overlap,
                            "concurrency": concurrency,
                        },
                        anchor=True,
                    )
                )
    for concurrency in (1, 2, 4):
        for ready_resume_state in ("prefetching", "ready"):
            jobs.append(
                _job(
                    "scheduler_blocking",
                    {
                        "policy": policy,
                        "concurrency": concurrency,
                        "ready_resume_state": ready_resume_state,
                    },
                    anchor=True,
                )
            )
    if len({row["job_id"] for row in jobs}) != len(jobs):
        raise RuntimeError("schema-v6 Runtime Profile job IDs are not unique")
    return tuple(jobs)


def build_schema6_runtime_profile_manifest(
    *,
    policy: str,
    code_commit: str,
    model_revision: str,
    tokenizer_hash: str,
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
    sentinel_gate_sha256: str,
    profile_freeze_contract_sha256: str,
) -> Dict[str, Any]:
    if not all(
        (
            code_commit,
            model_revision,
            tokenizer_hash,
            cacheblend_patch_sha256,
            cacheblend_tree,
            sentinel_gate_sha256,
            profile_freeze_contract_sha256,
        )
    ):
        raise ValueError("schema-v6 Runtime Profile provenance is incomplete")
    jobs = build_mistral_schema6_runtime_profile_jobs(policy)
    jobs_digest = hashlib.sha256(
        json.dumps(jobs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 6,
        "protocol_version": 8,
        "stage": "mistral_schema6_runtime_profile_measurement",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "model_key": "mistral",
        "model_revision": model_revision,
        "tokenizer_hash": tokenizer_hash,
        "selection_execution_policy": policy,
        "code_commit": code_commit,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "cacheblend_tree": cacheblend_tree,
        "sentinel_gate_sha256": sentinel_gate_sha256,
        "profile_freeze_contract_sha256": profile_freeze_contract_sha256,
        "jobs": len(jobs),
        "jobs_sha256": jobs_digest,
        "jobs_payload": list(jobs),
        "warmups_per_cell": 20,
        "measurements_per_cell": 100,
        "runtime_cost_profile_frozen": False,
        "run_140_job_qualification": False,
        "run_h1": False,
        "limits": {
            "max_session_hours": 4,
            "max_hourly_cny": 7.5,
            "max_total_cny": 30.0,
        },
    }


def build_schema6_sentinel_manifest(
    *,
    code_commit: str,
    model_revision: str,
    tokenizer_hash: str,
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
    config_sha256: str,
    contract_sha256: str,
) -> Dict[str, Any]:
    if not all(
        (
            code_commit,
            model_revision,
            tokenizer_hash,
            cacheblend_patch_sha256,
            cacheblend_tree,
            config_sha256,
            contract_sha256,
        )
    ):
        raise ValueError("schema-v6 sentinel provenance is incomplete")
    checkpoints = validate_schema6_checkpoint_contract(
        model_id=MISTRAL_SCHEMA6_SPEC.model_id,
        checkpoint_sources={
            "adapter": MISTRAL_SCHEMA6_SPEC.checkpoints,
            "manifest": (1, 2, 4, 5, 8),
            "microbenchmark": (1, 2, 4, 5, 8),
        },
    )
    jobs = build_mistral_schema6_sentinel_jobs()
    jobs_digest = hashlib.sha256(
        json.dumps(jobs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 6,
        "protocol_version": 8,
        "stage": "mistral_schema6_a800_4h_sentinel",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "code_commit": code_commit,
        "model": {
            "model_id": MISTRAL_SCHEMA6_SPEC.model_id,
            "revision": model_revision,
            "tokenizer_hash": tokenizer_hash,
            "adapter_name": MISTRAL_SCHEMA6_SPEC.adapter_name,
            "checkpoint_depths": list(checkpoints),
        },
        "cacheblend": {
            "patch_sha256": cacheblend_patch_sha256,
            "tree": cacheblend_tree,
        },
        "config_sha256": config_sha256,
        "contract_sha256": contract_sha256,
        "jobs": len(jobs),
        "jobs_sha256": jobs_digest,
        "jobs_payload": list(jobs),
        "limits": {
            "max_session_hours": 4,
            "max_hourly_cny": 7.5,
            "max_total_cny": 30.0,
        },
        "runtime_cost_profile_frozen": False,
        "selector_profile_frozen": False,
        "run_140_job_qualification": False,
        "run_h1": False,
    }


def schema6_no_gpu_gate() -> Dict[str, Any]:
    return {
        "protocol_version": 8,
        "schema_version": 6,
        "schema_v6_local_implementation_complete": True,
        "artifact_preparation_ready": True,
        "gpu_rental_ready_for_mistral_sentinel": True,
        "runtime_cost_profile_frozen": False,
        "selector_profile_frozen": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "failures": [],
    }
