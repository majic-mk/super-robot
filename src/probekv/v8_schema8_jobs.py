from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Tuple

from .v8_schema8_contracts import schema8_no_gpu_gate


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
        _job("gate1_same_origin_critical_path", {"gamma": 1.0}),
        _job("final_commit_joint_timeline", {"gamma": 0.8}),
        _job("r1_dense_equivalence", {"repair_ratio": 1.0}),
        _job("cpu_lru_demotion_to_ssd", {"busy_protected": True}),
        _job("ssd_lru_source_deletion", {"busy_protected": True}),
        _job("single_backing_replica", {"gpu_replica": "transient_hot"}),
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
    for source_tier in ("gpu", "pinned_cpu", "ssd_staged"):
        jobs.append(
            _job(
                "winner_layerwise_load",
                {"model": model_key, "source_tier": source_tier},
            )
        )
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
        **schema8_no_gpu_gate(),
    }
    payload["jobs_sha256"] = hashlib.sha256(
        json.dumps(
            {"sentinel": sentinel, "runtime_measurements": runtime},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload
