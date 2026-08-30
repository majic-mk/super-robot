from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple

from .model_adapters import MISTRAL_SCHEMA6_SPEC, QWEN_SCHEMA6_SPEC
from .v8_schema7_contracts import schema7_no_gpu_gate


def _job(kind: str, coordinates: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        "kind": kind,
        "coordinates": dict(coordinates),
        "paper_evidence": False,
        "locked_test_accessed": False,
        "requires_real_gpu": kind not in {
            "development_manifest_audit", "profile_generator_audit"
        },
    }
    row["job_id"] = "schema7-%s-%s" % (
        kind,
        hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16],
    )
    return row


def build_schema7_sentinel_jobs(model_key: str) -> Tuple[Dict[str, Any], ...]:
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("schema-v7 supports only mistral/qwen handoffs")
    spec = MISTRAL_SCHEMA6_SPEC if model_key == "mistral" else QWEN_SCHEMA6_SPEC
    jobs = [
        _job("native_prefix_cache_sentinel", {"model": model_key}),
        _job("repair_check_off_by_one", {"depths": list(spec.checkpoints)}),
        _job("r1_dense_equivalence", {"repair_ratio": 1.0}),
        _job("integrity_qualification_full", {"destination_digest": True}),
        _job("integrity_online_no_full_digest", {"mode": "online_immutable"}),
        _job("cfo_eager_streaming", {"duplicate_occurrences": True}),
        _job("gds_capability_and_staged_fallback", {}),
        _job("final_commit_partial_subset", {"policies": ["A", "C"]}),
    ]
    for policy in (
        "d1_only", "d1_d2_rescue", "legacy_multicheckpoint",
        "deep_full_candidate_oracle",
    ):
        jobs.append(_job("source_depth_policy", {"policy": policy}))
    for metric in ("winner_k_only", "winner_v_only", "winner_kv_normalized"):
        for repair_policy in (
            "fixed_15", "static_gradual", "load_recompute_aware_gradual",
        ):
            jobs.append(
                _job(
                    "winner_repair_policy",
                    {"metric": metric, "repair_policy": repair_policy},
                )
            )
    for tier in ("gpu", "pinned_cpu", "ssd_staged", "ssd_gds_optional"):
        jobs.append(_job("layerwise_transfer", {"tier": tier}))
    return tuple(jobs)


def build_schema7_development_jobs(model_key: str) -> Tuple[Dict[str, Any], ...]:
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("schema-v7 supports only mistral/qwen handoffs")
    jobs = []
    for dataset in ("musique", "2wikimultihopqa", "hotpotqa"):
        for depth_policy in (
            "d1_only", "d1_d2_rescue", "legacy_multicheckpoint",
            "deep_full_candidate_oracle",
        ):
            jobs.append(
                _job(
                    "depth_profile_development",
                    {"dataset": dataset, "partition": "profile_freeze", "policy": depth_policy},
                )
            )
        for metric in ("winner_k_only", "winner_v_only", "winner_kv_normalized"):
            for policy in ("fixed_15", "static_gradual", "load_recompute_aware_gradual"):
                jobs.append(
                    _job(
                        "repair_profile_development",
                        {
                            "dataset": dataset,
                            "partition": "profile_freeze",
                            "metric": metric,
                            "policy": policy,
                            "oracle_trace": True,
                        },
                    )
                )
    return tuple(jobs)


def build_schema7_runtime_measurement_jobs(
    model_key: str,
) -> Tuple[Dict[str, Any], ...]:
    """Unfrozen measurement manifest; every job still requires real GPU timing."""
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("schema-v7 supports only mistral/qwen handoffs")
    spec = MISTRAL_SCHEMA6_SPEC if model_key == "mistral" else QWEN_SCHEMA6_SPEC
    jobs = []
    for token_count in (128, 512, 640):
        for depth in spec.checkpoints:
            for compared_k in (1, 2, 4, 8, 16):
                jobs.append(_job("comparison_batch", {
                    "token_count": token_count, "completed_depth": depth,
                    "k": compared_k, "backing_tier": "pinned_cpu",
                }))
    for tier in ("pinned_cpu", "ssd_staged", "ssd_gds_optional"):
        jobs.append(_job("selection_state_transfer", {"tier": tier, "batch_k": 16}))
        jobs.append(_job("full_kv_tier_load", {"tier": tier, "layerwise": True}))
    for metric in ("winner_k_only", "winner_v_only", "winner_kv_normalized"):
        for ratio in (0.10, 0.12, 0.15, 1.0):
            jobs.append(_job("repair", {"metric": metric, "ratio": ratio}))
    for segment_count in (1, 2, 5, 10):
        jobs.append(_job("dense_remaining_joint", {"segment_count": segment_count}))
        jobs.append(_job("union_mask_remaining", {"segment_count": segment_count}))
    for policy in ("causal_commit_wait", "immediate_staggered_closed_loop"):
        jobs.append(_job("scheduler_blocking", {"policy": policy}))
        jobs.append(_job("interference", {"policy": policy, "overlap": True}))
    return tuple(jobs)


def build_schema7_no_gpu_handoff(
    *,
    code_commit: str,
    model_key: str,
    model_revision: str,
    tokenizer_hash: str,
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
    config_sha256: str,
    contract_sha256: str,
) -> Mapping[str, Any]:
    if not all(
        (
            code_commit, model_key, model_revision, tokenizer_hash,
            cacheblend_patch_sha256, cacheblend_tree, config_sha256,
            contract_sha256,
        )
    ):
        raise ValueError("schema-v7 handoff provenance is incomplete")
    sentinel = build_schema7_sentinel_jobs(model_key)
    development = build_schema7_development_jobs(model_key)
    runtime_measurements = build_schema7_runtime_measurement_jobs(model_key)
    payload = {
        "protocol_version": 8,
        "schema_version": 7,
        "runtime_patch_mode": "probekv_v8_winner_gradual_streaming",
        "code_commit": code_commit,
        "model_key": model_key,
        "model_revision": model_revision,
        "tokenizer_hash": tokenizer_hash,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "cacheblend_tree": cacheblend_tree,
        "config_sha256": config_sha256,
        "contract_sha256": contract_sha256,
        "sentinel_jobs": list(sentinel),
        "development_jobs": list(development),
        "runtime_measurement_jobs": list(runtime_measurements),
        "profile_generation_order": [
            "selection_depth", "repair_policy", "runtime_cost"
        ],
        "profiles_frozen": False,
        "gpu_jobs_started": False,
        **schema7_no_gpu_gate(),
    }
    payload["jobs_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "sentinel": sentinel,
                "development": development,
                "runtime_measurements": runtime_measurements,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload
