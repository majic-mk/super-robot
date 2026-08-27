from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Sequence


V8_PROFILE_DATASETS = ("musique", "2wikimultihopqa", "hotpotqa")


def selector_profile_sha256(profile: Mapping[str, Any]) -> str:
    """Return the content digest used by a frozen v8 Selector Profile.

    The digest covers the complete immutable payload.  The two envelope fields
    added after freezing are deliberately excluded, so readers can recompute
    the same digest instead of trusting a copied ``profile_sha256`` value.
    """

    payload = {
        key: value
        for key, value in profile.items()
        if key not in {"profile_sha256", "selector_profile_frozen"}
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_selector_profile(
    profile: Mapping[str, Any],
    *,
    model_key: str | None = None,
    policy: str | None = None,
    code_commit: str | None = None,
    model_revision: str | None = None,
    tokenizer_hash: str | None = None,
    cacheblend_patch_sha256: str | None = None,
) -> None:
    failures = []
    if profile.get("protocol_version") != 8 or profile.get("schema_version") != 4:
        failures.append("Selector Profile is not v8 schema-v4")
    if profile.get("selector_profile_frozen") is not True:
        failures.append("Selector Profile is not frozen")
    if profile.get("training_free") is not True or profile.get("probability_calibration_used") is not False:
        failures.append("Selector Profile is not training-free")
    observed_digest = profile.get("profile_sha256")
    if not observed_digest or observed_digest != selector_profile_sha256(profile):
        failures.append("Selector Profile content digest is invalid")
    expected = {
        "model_key": model_key,
        "selection_execution_policy": policy,
        "code_commit": code_commit,
        "model_revision": model_revision,
        "tokenizer_hash": tokenizer_hash,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
    }
    for key, value in expected.items():
        if value is not None and profile.get(key) != value:
            failures.append("Selector Profile binding differs: %s" % key)
    if failures:
        raise ValueError("; ".join(failures))


def selector_profile_candidates(model_key: str, policy: str) -> Sequence[Dict[str, Any]]:
    checkpoints = {
        "mistral": (1, 2, 4, 5, 8),
        "qwen": (1, 2, 4, 5, 7),
    }
    if model_key not in checkpoints:
        raise ValueError("unsupported v8 profile model")
    if policy not in {"causal_commit_wait", "immediate_staggered_closed_loop"}:
        raise ValueError("unsupported v8 selection execution policy")
    result = []
    for maximum in checkpoints[model_key]:
        if maximum < 1:
            continue
        depths = tuple(value for value in checkpoints[model_key] if value <= maximum)
        for eta in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for eta_strong in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                if eta_strong < eta:
                    continue
                for tolerance in (0.0, 0.05, 0.10, 0.20):
                    result.append(
                        {
                            "model_key": model_key,
                            "selection_execution_policy": policy,
                            "checkpoint_depths": depths,
                            "max_completed_depth": maximum,
                            "eta": eta,
                            "eta_strong": eta_strong,
                            "residual_band_relative_tolerance": tolerance,
                            "residual_band_numeric_slack": 1e-6,
                            "fixed_repair_ratio": 0.15,
                        }
                    )
    return tuple(result)


def freeze_selector_profile(
    *,
    model_key: str,
    policy: str,
    rows: Sequence[Mapping[str, Any]],
    code_commit: str,
    model_revision: str,
    tokenizer_hash: str,
    cacheblend_patch_sha256: str,
    microbenchmark_sha256: str,
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("profile freeze requires measured development rows")
    required_identity = (
        code_commit,
        model_revision,
        tokenizer_hash,
        cacheblend_patch_sha256,
        microbenchmark_sha256,
    )
    if not all(str(value).strip() for value in required_identity):
        raise ValueError("profile provenance is incomplete")
    observed_datasets = {str(row.get("dataset")) for row in rows}
    if observed_datasets != set(V8_PROFILE_DATASETS):
        raise ValueError("profile freeze must pool the three frozen RAG datasets")
    candidates = {json.dumps(item, sort_keys=True): item for item in selector_profile_candidates(model_key, policy)}
    aggregates: Dict[str, Dict[str, float]] = {}
    datasets_by_candidate: Dict[str, set[str]] = {}
    for row in rows:
        if row.get("paper_evidence") is not False or row.get("locked_test_accessed") is not False:
            raise ValueError("profile rows crossed the evidence boundary")
        if row.get("cuda_event_timing") is not True or row.get("fake_timing") is True:
            raise ValueError("profile freeze requires real CUDA timing")
        key = json.dumps(dict(row.get("candidate_profile", {})), sort_keys=True)
        if key not in candidates:
            raise ValueError("profile row used an undeclared hyperparameter candidate")
        aggregate = aggregates.setdefault(
            key,
            {"rows": 0.0, "regret": 0.0, "depth": 0.0, "overhead": 0.0, "invalid": 0.0},
        )
        aggregate["rows"] += 1
        aggregate["regret"] += float(row["normalized_oracle_regret"])
        aggregate["depth"] += float(row["completed_depth"])
        aggregate["overhead"] = max(
            aggregate["overhead"], float(row["selection_overhead_fraction"])
        )
        aggregate["invalid"] += int(bool(row.get("invalid_lock", False)))
        datasets_by_candidate.setdefault(key, set()).add(str(row["dataset"]))
    feasible = []
    for key, aggregate in aggregates.items():
        if (
            aggregate["invalid"]
            or aggregate["overhead"] > 0.05 + 1e-12
            or datasets_by_candidate.get(key) != set(V8_PROFILE_DATASETS)
        ):
            continue
        count = aggregate["rows"]
        feasible.append(
            (
                aggregate["regret"] / count,
                aggregate["depth"] / count,
                candidates[key]["max_completed_depth"],
                candidates[key]["eta"],
                candidates[key]["eta_strong"],
                candidates[key]["residual_band_relative_tolerance"],
                key,
            )
        )
    if not feasible:
        raise RuntimeError("no measured v8 selector profile satisfies the frozen constraints")
    selected_key = min(feasible)[-1]
    selected = candidates[selected_key]
    payload = {
        "schema_version": 4,
        "protocol_version": 8,
        "stage": "v8_selector_profile_freeze",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "model_key": model_key,
        "selection_execution_policy": policy,
        "development_datasets": list(V8_PROFILE_DATASETS),
        "probability_calibration_used": False,
        "training_free": True,
        "selected_profile": selected,
        "code_commit": code_commit,
        "model_revision": model_revision,
        "tokenizer_hash": tokenizer_hash,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "microbenchmark_sha256": microbenchmark_sha256,
    }
    payload["profile_sha256"] = selector_profile_sha256(payload)
    payload["selector_profile_frozen"] = True
    return payload
