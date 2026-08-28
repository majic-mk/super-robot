from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Sequence


SCHEMA_VERSION = 6
PROTOCOL_VERSION = 8
RUNTIME_PROFILE_CATEGORIES = (
    "comparison_batch",
    "selection_state_transfer",
    "full_kv_tier_load",
    "dense_remaining_joint",
    "repair",
    "union_mask_remaining",
    "interference",
    "scheduler_blocking",
)

_REQUIRED_AXES = {
    "comparison_batch": {"k", "token_count", "completed_depth", "backing_tier"},
    "selection_state_transfer": {"bytes", "tier", "batch_k"},
    "full_kv_tier_load": {"bytes", "source_tier", "layer_range"},
    "dense_remaining_joint": {"boundary_vector", "active_rows", "segment_count"},
    "repair": {"boundary", "token_count", "repair_count"},
    "union_mask_remaining": {"boundary_vector", "layer_active_rows"},
    "interference": {"copy_bytes", "overlap", "concurrency"},
    "scheduler_blocking": {"policy", "concurrency", "ready_resume_state"},
}


def _sha(payload: Mapping[str, Any], excluded: Sequence[str]) -> str:
    body = {key: value for key, value in payload.items() if key not in set(excluded)}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def bootstrap_p95_ucb(
    measurements_ms: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260726,
) -> Dict[str, float | int | str]:
    import numpy as np

    values = np.asarray(tuple(float(value) for value in measurements_ms), dtype=np.float64)
    if values.size < 2 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("bootstrap requires at least two finite non-negative measurements")
    if resamples <= 0:
        raise ValueError("bootstrap resample count must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.choice(values, size=values.size, replace=True)
        estimates[index] = np.quantile(sample, 0.95, method="linear")
    return {
        "sample_count": int(values.size),
        "p95_ms": float(np.quantile(values, 0.95, method="linear")),
        "p95_one_sided_95_ucb_ms": float(
            np.quantile(estimates, 0.95, method="linear")
        ),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "bootstrap_prng": "PCG64",
        "quantile_method": "linear",
        "outlier_policy": "none",
        "warmups_included": False,
    }


def make_measurement_cell(
    category: str,
    *,
    axes: Mapping[str, Any],
    measurements_ms: Sequence[float],
    warmups: int,
    resamples: int = 10_000,
    seed: int = 20260726,
) -> Dict[str, Any]:
    if category not in _REQUIRED_AXES:
        raise ValueError("unknown RuntimeCostProfile category")
    if set(axes) != _REQUIRED_AXES[category]:
        raise ValueError("RuntimeCostProfile category axes differ: %s" % category)
    if warmups < 0:
        raise ValueError("warmup count must be non-negative")
    return {
        "category": category,
        "axes": dict(axes),
        "warmups": warmups,
        "statistics": bootstrap_p95_ucb(
            measurements_ms, resamples=resamples, seed=seed
        ),
    }


def build_schema6_runtime_cost_profile(
    *,
    model_key: str,
    policy: str,
    code_commit: str,
    cacheblend_patch_sha256: str,
    gpu_uuid: str,
    hardware_compatibility_signature: str,
    measurement_cells: Sequence[Mapping[str, Any]],
    measurement_sha256: str,
    frozen: bool,
) -> Dict[str, Any]:
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("unsupported RuntimeCostProfile model")
    if policy not in {"causal_commit_wait", "immediate_staggered_closed_loop"}:
        raise ValueError("unsupported RuntimeCostProfile policy")
    if not all(
        (code_commit, cacheblend_patch_sha256, gpu_uuid, hardware_compatibility_signature, measurement_sha256)
    ):
        raise ValueError("RuntimeCostProfile provenance is incomplete")
    categories = [str(cell.get("category", "")) for cell in measurement_cells]
    if any(category not in _REQUIRED_AXES for category in categories):
        raise ValueError("RuntimeCostProfile contains an unknown category")
    if frozen and set(categories) != set(RUNTIME_PROFILE_CATEGORIES):
        raise ValueError("frozen schema-v6 RuntimeCostProfile requires all categories")
    for cell in measurement_cells:
        category = str(cell.get("category", ""))
        if set(cell.get("axes", {})) != _REQUIRED_AXES[category]:
            raise ValueError("RuntimeCostProfile cell axes differ: %s" % category)
        statistics = cell.get("statistics", {})
        for key in ("p95_ms", "p95_one_sided_95_ucb_ms", "sample_count"):
            if key not in statistics or float(statistics[key]) < 0:
                raise ValueError("RuntimeCostProfile cell statistics are incomplete")
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "stage": "v8_schema6_runtime_cost_profile",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "model_key": model_key,
        "selection_execution_policy": policy,
        "code_commit": code_commit,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "gpu_uuid": gpu_uuid,
        "hardware_compatibility_signature": hardware_compatibility_signature,
        "measurement_cells": [dict(cell) for cell in measurement_cells],
        "measurement_sha256": measurement_sha256,
        "cuda_event_timing": True,
        "fake_timing": False,
        "runtime_cost_profile_frozen": bool(frozen),
    }
    payload["runtime_cost_profile_sha256"] = _sha(
        payload, ("runtime_cost_profile_sha256",)
    )
    return payload


def validate_schema6_runtime_cost_profile(
    profile: Mapping[str, Any], *, require_frozen: bool
) -> None:
    failures = []
    if profile.get("protocol_version") != 8 or profile.get("schema_version") != 6:
        failures.append("RuntimeCostProfile is not v8 schema-v6")
    if profile.get("cuda_event_timing") is not True or profile.get("fake_timing") is not False:
        failures.append("RuntimeCostProfile requires real CUDA timing")
    if require_frozen and profile.get("runtime_cost_profile_frozen") is not True:
        failures.append("RuntimeCostProfile is not frozen")
    if profile.get("runtime_cost_profile_sha256") != _sha(
        profile, ("runtime_cost_profile_sha256",)
    ):
        failures.append("RuntimeCostProfile digest is invalid")
    categories = {str(cell.get("category", "")) for cell in profile.get("measurement_cells", ())}
    if require_frozen and categories != set(RUNTIME_PROFILE_CATEGORIES):
        failures.append("frozen RuntimeCostProfile lacks required categories")
    for cell in profile.get("measurement_cells", ()):
        category = str(cell.get("category", ""))
        if category not in _REQUIRED_AXES or set(cell.get("axes", {})) != _REQUIRED_AXES.get(category, set()):
            failures.append("RuntimeCostProfile cell axes are invalid: %s" % category)
    if failures:
        raise ValueError("; ".join(failures))
