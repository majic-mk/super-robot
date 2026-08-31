from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .v8_schema8_contracts import schema8_no_gpu_gate
from .runtime_source_audit import audit_v8_schema8_runtime_sources


def _job(kind: str, coordinates: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        "kind": kind,
        "coordinates": dict(coordinates),
        "paper_evidence": False,
        "locked_test_accessed": False,
        "requires_real_gpu": True,
    }
    row["job_id"] = "schema8-%s-%s" % (
        kind,
        hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16],
    )
    return row


def build_schema8_sentinel_jobs(model_key: str) -> Tuple[Dict[str, Any], ...]:
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("schema-v8 supports only mistral/qwen handoffs")
    jobs = [
        _job("native_prefix_cache_sentinel", {"model": model_key}),
        _job("d1_all_resolved_layer2_reuse", {"depths": [1]}),
        _job("d1_d2_dense_barrier_layer3_reuse", {"depths": [1, 2]}),
        _job(
            "d1_detached_prefetch_not_execution_visible",
            {"barrier": "open", "winner_depth": 1},
        ),
        _job("gate1_optimistic_marginal_feasibility", {"gamma": 1.0}),
        _job("final_commit_joint_timeline", {"gamma": 0.8}),
        _job("r1_dense_equivalence", {"repair_ratio": 1.0}),
        _job("cpu_lru_demotion_to_ssd", {"busy_protected": True}),
        _job("ssd_lru_source_deletion", {"busy_protected": True}),
        _job("single_backing_replica", {"gpu_replica": "transient_hot"}),
        _job("verified_backing_migration", {"publish": "after_digest_match"}),
        _job("integrity_online_no_full_digest", {"mode": "online_immutable"}),
    ]
    for scope in (
        "uniform_fixed",
        "shared_relative_schedule",
        "per_segment_load_aware",
    ):
        jobs.append(_job("repair_ratio_scope", {"scope": scope}))
    return tuple(jobs)


def build_schema8_runtime_measurement_jobs(
    model_key: str,
) -> Tuple[Dict[str, Any], ...]:
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("schema-v8 supports only mistral/qwen handoffs")
    jobs = []
    for completed_depth in (1, 2):
        for compared_k in (1, 2, 4, 8, 16):
            jobs.append(
                _job(
                    "selection_and_gate1",
                    {
                        "model": model_key,
                        "completed_depth": completed_depth,
                        "compared_k": compared_k,
                        "token_count": 512,
                    },
                )
            )
    for segment_count in (1, 2, 5, 10):
        jobs.append(
            _job(
                "dense_barrier_joint_timeline",
                {"model": model_key, "segment_count": segment_count},
            )
        )
        jobs.append(
            _job(
                "joint_adaptive_ratio_vectors",
                {
                    "model": model_key,
                    "segment_count": segment_count,
                    "candidate_policy": "bounded_profile_templates",
                },
            )
        )
    for source_tier in ("gpu", "pinned_cpu", "ssd_staged"):
        jobs.append(
            _job(
                "winner_layerwise_load",
                {"model": model_key, "source_tier": source_tier},
            )
        )
    return tuple(jobs)


def build_schema8_qualification_jobs(
    *,
    model_key: str,
    selection_depth_profile_sha256: str,
    repair_policy_profile_sha256: str,
    runtime_cost_profile_sha256: str,
    selected_repair_policy: str = "static_gradual",
) -> Tuple[Dict[str, Any], ...]:
    """Build the final Profile-bound 140-job schema-v8 matrix."""

    if model_key not in {"mistral", "qwen"}:
        raise ValueError("schema-v8 supports only mistral/qwen qualification")
    profile_shas = (
        selection_depth_profile_sha256,
        repair_policy_profile_sha256,
        runtime_cost_profile_sha256,
    )
    if any(len(value) != 64 for value in profile_shas):
        raise ValueError("schema-v8 qualification requires three frozen Profile SHAs")
    if selected_repair_policy not in {
        "fixed_15", "static_gradual", "load_recompute_aware_gradual"
    }:
        raise ValueError("qualification repair policy is not schema-v8")
    qualification_repair_policies = (
        ("fixed_15", "static_gradual")
        if selected_repair_policy == "fixed_15"
        else ("fixed_15", selected_repair_policy)
    )
    patterns = []
    for segment_count in (1, 2, 5, 10, 37):
        segment_patterns = []
        for compared_k in (1, 4, 16):
            for barrier_case in ("all_d1", "contains_d2"):
                for repair_policy in qualification_repair_policies:
                    for source_tier in ("gpu", "pinned_cpu", "ssd_staged"):
                        segment_patterns.append(
                            {
                                "model": model_key,
                                "segment_count": segment_count,
                                "compared_k": compared_k,
                                "barrier_case": barrier_case,
                                "repair_policy": repair_policy,
                                "source_tier": source_tier,
                                "selection_depth_profile_sha256": profile_shas[0],
                                "repair_policy_profile_sha256": profile_shas[1],
                                "runtime_cost_profile_sha256": profile_shas[2],
                            }
                        )
        segment_patterns.sort(
            key=lambda row: hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        patterns.extend(segment_patterns[:28])
    jobs = []
    for qualification_index, coordinates in enumerate(patterns):
        jobs.append(
            _job(
                "schema8_runtime_qualification",
                {"qualification_index": qualification_index, **coordinates},
            )
        )
    if len(jobs) != 140 or len({row["job_id"] for row in jobs}) != 140:
        raise RuntimeError("schema-v8 qualification matrix must contain 140 unique jobs")
    return tuple(jobs)


def build_schema8_no_gpu_handoff(
    *,
    code_commit: str,
    model_key: str,
    model_revision: str,
    tokenizer_hash: str,
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
    config_sha256: str,
    contract_sha256: str,
    repo_root: Path | None = None,
) -> Mapping[str, Any]:
    provenance = (
        code_commit,
        model_key,
        model_revision,
        tokenizer_hash,
        cacheblend_patch_sha256,
        cacheblend_tree,
        config_sha256,
        contract_sha256,
    )
    if not all(provenance):
        raise ValueError("schema-v8 handoff provenance is incomplete")
    sentinel = build_schema8_sentinel_jobs(model_key)
    runtime = build_schema8_runtime_measurement_jobs(model_key)
    audit = audit_v8_schema8_runtime_sources(
        (repo_root or Path(__file__).resolve().parents[2]).resolve()
    )
    payload = {
        "protocol_version": 8,
        "schema_version": 8,
        "runtime_patch_mode": "probekv_v8_gradual_barrier_tiered_lru",
        "code_commit": code_commit,
        "model_key": model_key,
        "model_revision": model_revision,
        "tokenizer_hash": tokenizer_hash,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "cacheblend_tree": cacheblend_tree,
        "config_sha256": config_sha256,
        "contract_sha256": contract_sha256,
        "sentinel_jobs": list(sentinel),
        "runtime_measurement_jobs": list(runtime),
        "profiles_frozen": False,
        "gpu_jobs_started": False,
        "runtime_source_audit": audit,
        **schema8_no_gpu_gate(
            runtime_source_ready=audit.get("runtime_source_ready") is True
        ),
    }
    payload["jobs_sha256"] = hashlib.sha256(
        json.dumps(
            {"sentinel": sentinel, "runtime_measurements": runtime},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload
