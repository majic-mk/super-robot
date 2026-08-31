from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence


V8_PROFILE_DATASETS = ("musique", "2wikimultihopqa", "hotpotqa")
V8_PROFILE_SCHEMA_VERSION = 5
V8_LEGACY_PROFILE_SCHEMA_VERSION = 4
REGRET_NORMALIZATION_FLOOR = 2 ** -7
REGRET_NUMERIC_SLACK = 1e-6


def _content_sha256(payload: Mapping[str, Any], excluded: Sequence[str]) -> str:
    canonical = {key: value for key, value in payload.items() if key not in set(excluded)}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selector_profile_sha256(profile: Mapping[str, Any]) -> str:
    """Return the content digest used by a frozen v8 Selector Profile.

    The digest covers the complete immutable payload.  The two envelope fields
    added after freezing are deliberately excluded, so readers can recompute
    the same digest instead of trusting a copied ``profile_sha256`` value.
    """

    return _content_sha256(profile, ("profile_sha256", "selector_profile_frozen"))


def runtime_cost_profile_sha256(profile: Mapping[str, Any]) -> str:
    return _content_sha256(
        profile,
        ("runtime_cost_profile_sha256", "runtime_cost_profile_frozen"),
    )


def profile_freeze_contract_sha256(contract: Mapping[str, Any]) -> str:
    return _content_sha256(
        contract,
        ("profile_freeze_contract_sha256", "profile_freeze_contract_frozen"),
    )


def build_profile_freeze_contract(
    *,
    code_commit: str,
    development_partition_sha256: str,
) -> Dict[str, Any]:
    if not code_commit or not development_partition_sha256:
        raise ValueError("Profile-freeze contract provenance is incomplete")
    payload: Dict[str, Any] = {
        "schema_version": 5,
        "protocol_version": 8,
        "stage": "v8_profile_freeze_precommit",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "code_commit": code_commit,
        "development_partition_sha256": development_partition_sha256,
        "profile_datasets": list(V8_PROFILE_DATASETS),
        "thresholds": {
            "state_availability_min": 0.99,
            "selection_coverage_min": 0.80,
            "early_resolution_rate_at_depth5_min": 0.80,
            "wrong_early_lock_rate_max": 0.05,
            "mean_stable_normalized_oracle_regret_max": 0.10,
            "predicted_selection_budget_fraction_max": 0.05,
            "actual_selection_critical_path_p95_fraction_max": 0.05,
            "selection_budget_realized_overrun_rate_max": 0.05,
            "budget_admission_violation_count_max": 0,
            "illegal_lock_count_max": 0,
        },
        "early_depth_cap_completed_depth": 5,
        "metric_denominators": {
            "state_availability": "requests_with_at_least_one_correctness_eligible_source",
            "selection_coverage": "requests_with_at_least_one_correctness_eligible_source",
            "early_resolution_rate_at_depth5": "successful_legal_source_locks",
            "multi_source_early_exit_rate_at_depth5": (
                "requests_with_at_least_two_correctness_eligible_and_available_states"
            ),
            "wrong_early_lock_rate": "legal_early_source_locks",
        },
        "regret": {
            "oracle": "minimum_lmax_residual_over_full_reference_source_set",
            "normalization_floor": REGRET_NORMALIZATION_FLOOR,
            "normalization_floor_semantics": (
                "fixed_data_independent_bf16_source_state_scale_not_J_resolution"
            ),
            "numeric_slack": REGRET_NUMERIC_SLACK,
            "hard_metric": "mean_stable_normalized_oracle_regret",
            "reported_quantiles": [0.5, 0.9, 0.95],
            "absolute_regret_reported": True,
        },
        "profile_tie_break_order": [
            "mean_lock_completed_depth",
            "selection_critical_path_p95_fraction",
            "max_completed_depth",
            "mean_stable_normalized_oracle_regret",
            "profile_id",
        ],
        "pre_lmax_same_checkpoint_economic_fallback": False,
        "lmax_residual_band_economic_selection": True,
        "multi_source_early_exit_rate_hard_gate": False,
    }
    payload["profile_freeze_contract_sha256"] = profile_freeze_contract_sha256(payload)
    payload["profile_freeze_contract_frozen"] = True
    return payload


def validate_profile_freeze_contract(contract: Mapping[str, Any]) -> None:
    failures = []
    if contract.get("protocol_version") != 8 or contract.get("schema_version") != 5:
        failures.append("Profile-freeze contract is not v8 schema-v5")
    if contract.get("profile_freeze_contract_frozen") is not True:
        failures.append("Profile-freeze contract is not frozen")
    observed = contract.get("profile_freeze_contract_sha256")
    if not observed or observed != profile_freeze_contract_sha256(contract):
        failures.append("Profile-freeze contract digest is invalid")
    if contract.get("paper_evidence") is not False or contract.get("locked_test_accessed") is not False:
        failures.append("Profile-freeze contract crossed the evidence boundary")
    if failures:
        raise ValueError("; ".join(failures))


def build_runtime_cost_profile(
    *,
    model_key: str,
    policy: str,
    code_commit: str,
    cacheblend_patch_sha256: str,
    gpu_uuid: str,
    hardware_compatibility_signature: str,
    comparison_batch_upper_ms: Mapping[int, float],
    measurement_sha256: str,
) -> Dict[str, Any]:
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("unsupported RuntimeCostProfile model")
    if policy not in {"causal_commit_wait", "immediate_staggered_closed_loop"}:
        raise ValueError("unsupported RuntimeCostProfile policy")
    if set(comparison_batch_upper_ms) != {1, 2, 4, 8, 16}:
        raise ValueError("RuntimeCostProfile requires K=1/2/4/8/16")
    if any(float(value) < 0 for value in comparison_batch_upper_ms.values()):
        raise ValueError("RuntimeCostProfile timings must be non-negative")
    if not all(
        (code_commit, cacheblend_patch_sha256, gpu_uuid, hardware_compatibility_signature, measurement_sha256)
    ):
        raise ValueError("RuntimeCostProfile provenance is incomplete")
    payload: Dict[str, Any] = {
        "schema_version": 5,
        "protocol_version": 8,
        "stage": "v8_runtime_cost_profile",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "model_key": model_key,
        "selection_execution_policy": policy,
        "code_commit": code_commit,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "gpu_uuid": gpu_uuid,
        "hardware_compatibility_signature": hardware_compatibility_signature,
        "comparison_batch_upper_ms": {
            str(key): float(comparison_batch_upper_ms[key])
            for key in (1, 2, 4, 8, 16)
        },
        "measurement_sha256": measurement_sha256,
        "cuda_event_timing": True,
        "fake_timing": False,
    }
    payload["runtime_cost_profile_sha256"] = runtime_cost_profile_sha256(payload)
    payload["runtime_cost_profile_frozen"] = True
    return payload


def validate_runtime_cost_profile(
    profile: Mapping[str, Any],
    *,
    model_key: str | None = None,
    policy: str | None = None,
    code_commit: str | None = None,
    cacheblend_patch_sha256: str | None = None,
) -> None:
    failures = []
    if profile.get("protocol_version") != 8 or profile.get("schema_version") != 5:
        failures.append("RuntimeCostProfile is not v8 schema-v5")
    if profile.get("runtime_cost_profile_frozen") is not True:
        failures.append("RuntimeCostProfile is not frozen")
    observed = profile.get("runtime_cost_profile_sha256")
    if not observed or observed != runtime_cost_profile_sha256(profile):
        failures.append("RuntimeCostProfile digest is invalid")
    if profile.get("cuda_event_timing") is not True or profile.get("fake_timing") is not False:
        failures.append("RuntimeCostProfile requires real CUDA timing")
    if set(profile.get("comparison_batch_upper_ms", {})) != {"1", "2", "4", "8", "16"}:
        failures.append("RuntimeCostProfile lacks the frozen K curve")
    expected = {
        "model_key": model_key,
        "selection_execution_policy": policy,
        "code_commit": code_commit,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
    }
    for key, value in expected.items():
        if value is not None and profile.get(key) != value:
            failures.append("RuntimeCostProfile binding differs: %s" % key)
    if failures:
        raise ValueError("; ".join(failures))


def stable_residual_regret(chosen: float, oracle: float) -> Dict[str, float]:
    if min(chosen, oracle) < 0 or chosen + 1e-12 < oracle:
        raise ValueError("Residual regret requires non-negative chosen >= oracle")
    absolute = max(0.0, float(chosen) - float(oracle))
    effective = max(0.0, absolute - REGRET_NUMERIC_SLACK)
    return {
        "absolute_oracle_regret": absolute,
        "stable_normalized_oracle_regret": (
            effective / max(float(oracle), REGRET_NORMALIZATION_FLOOR)
        ),
    }


def evaluate_runtime_profile_compatibility(
    profile: Mapping[str, Any],
    *,
    actual_gpu_uuid: str,
    actual_hardware_compatibility_signature: str,
    code_commit: str,
    cacheblend_patch_sha256: str,
) -> Dict[str, Any]:
    """Decide whether R_profile can also serve as R_qual on another A800."""
    failures = []
    try:
        validate_runtime_cost_profile(
            profile,
            code_commit=code_commit,
            cacheblend_patch_sha256=cacheblend_patch_sha256,
        )
    except ValueError as error:
        failures.append(str(error))
    if profile.get("hardware_compatibility_signature") != actual_hardware_compatibility_signature:
        failures.append("qualification hardware compatibility signature differs")
    if profile.get("gpu_uuid") != actual_gpu_uuid:
        failures.append("a new qualification GPU requires a new RuntimeCostProfile")
    return {
        "schema_version": 5,
        "protocol_version": 8,
        "stage": "v8_runtime_profile_compatibility",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "runtime_cost_profile_sha256": profile.get("runtime_cost_profile_sha256"),
        "profile_gpu_uuid": profile.get("gpu_uuid"),
        "actual_gpu_uuid": actual_gpu_uuid,
        "same_physical_gpu": profile.get("gpu_uuid") == actual_gpu_uuid,
        "hardware_compatibility_signature": actual_hardware_compatibility_signature,
        "runtime_profile_reusable_for_qualification": not failures,
        "must_measure_new_qualification_runtime_profile": bool(failures),
        "failures": failures,
    }


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
    schema = profile.get("schema_version")
    if profile.get("protocol_version") != 8 or schema not in {4, 5}:
        failures.append("Selector Profile is not a supported v8 schema")
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
    if schema == 5:
        for key in (
            "profile_freeze_contract_sha256",
            "profile_freeze_runtime_cost_profile_sha256",
            "development_partition_sha256",
        ):
            if not profile.get(key):
                failures.append("schema-v5 Selector Profile lacks %s" % key)
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
    profile_freeze_contract: Mapping[str, Any] | None = None,
    runtime_cost_profile: Mapping[str, Any] | None = None,
    development_partition_sha256: str = "",
    schema_version: int = 5,
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
    if schema_version not in {4, 5}:
        raise ValueError("unsupported Selector Profile schema")
    if schema_version == 5:
        if profile_freeze_contract is None or runtime_cost_profile is None:
            raise ValueError("schema-v5 Profile requires frozen contract and RuntimeCostProfile")
        validate_profile_freeze_contract(profile_freeze_contract)
        validate_runtime_cost_profile(
            runtime_cost_profile,
            model_key=model_key,
            policy=policy,
            code_commit=code_commit,
            cacheblend_patch_sha256=cacheblend_patch_sha256,
        )
        if not development_partition_sha256:
            raise ValueError("schema-v5 Profile requires a development partition SHA")
        if (
            profile_freeze_contract.get("development_partition_sha256")
            != development_partition_sha256
        ):
            raise ValueError("Profile-freeze contract used another development partition")
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
            {
                "rows": 0.0,
                "regret": 0.0,
                "depth": 0.0,
                "overhead_values": [],
                "invalid": 0.0,
                "wrong_early": 0.0,
                "early_locks": 0.0,
                "locks": 0.0,
                "state_available": 0.0,
                "correctness_eligible": 0.0,
                "selection_eligible": 0.0,
                "multisource_eligible": 0.0,
                "multisource_early_locks": 0.0,
                "budget_admission_violations": 0.0,
                "budget_overruns": 0.0,
            },
        )
        aggregate["rows"] += 1
        aggregate["regret"] += float(
            row.get("stable_normalized_oracle_regret", row.get("normalized_oracle_regret", 0.0))
        )
        aggregate["depth"] += float(row["completed_depth"])
        aggregate["overhead_values"].append(float(row.get(
            "selection_critical_path_fraction",
            row["selection_overhead_fraction"],
        )))
        aggregate["invalid"] += int(bool(row.get("illegal_lock", row.get("invalid_lock", False))))
        aggregate["wrong_early"] += int(bool(row.get("wrong_early_lock", False)))
        correctness = bool(row.get("correctness_eligible", True))
        state_available = bool(row.get("selection_state_available", True))
        locked = correctness and bool(row.get("legal_source_lock", True))
        aggregate["locks"] += int(locked)
        aggregate["early_locks"] += int(locked and int(row["completed_depth"]) <= 5)
        aggregate["correctness_eligible"] += int(correctness)
        aggregate["state_available"] += int(correctness and state_available)
        aggregate["selection_eligible"] += int(correctness)
        multisource = int(row.get("correctness_eligible_k", 2)) >= 2
        aggregate["multisource_eligible"] += int(correctness and multisource)
        aggregate["multisource_early_locks"] += int(
            locked
            and multisource
            and bool(row.get("current_state_ranking_performed", True))
            and int(row["completed_depth"]) <= 5
        )
        aggregate["budget_admission_violations"] += int(
            row.get("budget_admission_violation_count", 0)
        )
        aggregate["budget_overruns"] += int(
            bool(row.get("selection_budget_realized_overrun", False))
        )
        datasets_by_candidate.setdefault(key, set()).add(str(row["dataset"]))
    feasible = []
    for key, aggregate in aggregates.items():
        count = aggregate["rows"]
        overheads = sorted(aggregate["overhead_values"])
        p95_index = max(0, int(math.ceil(0.95 * len(overheads))) - 1)
        overhead_p95 = overheads[p95_index]
        state_availability = aggregate["state_available"] / max(
            aggregate["correctness_eligible"], 1.0
        )
        coverage = aggregate["locks"] / max(aggregate["selection_eligible"], 1.0)
        early_rate = aggregate["early_locks"] / max(aggregate["locks"], 1.0)
        wrong_rate = aggregate["wrong_early"] / max(aggregate["early_locks"], 1.0)
        overrun_rate = aggregate["budget_overruns"] / count
        mean_regret = aggregate["regret"] / count
        if (
            aggregate["invalid"]
            or aggregate["budget_admission_violations"]
            or state_availability < 0.99 - 1e-12
            or coverage < 0.80 - 1e-12
            or early_rate < 0.80 - 1e-12
            or wrong_rate > 0.05 + 1e-12
            or mean_regret > 0.10 + 1e-12
            or overhead_p95 > 0.05 + 1e-12
            or overrun_rate > 0.05 + 1e-12
            or datasets_by_candidate.get(key) != set(V8_PROFILE_DATASETS)
        ):
            continue
        feasible.append(
            (
                aggregate["depth"] / count,
                overhead_p95,
                candidates[key]["max_completed_depth"],
                mean_regret,
                key,
            )
        )
    if not feasible:
        raise RuntimeError("no measured v8 selector profile satisfies the frozen constraints")
    selected_key = min(feasible)[-1]
    selected = candidates[selected_key]
    selected_aggregate = aggregates[selected_key]
    selected_count = selected_aggregate["rows"]
    selected_overheads = sorted(selected_aggregate["overhead_values"])
    selected_p95 = selected_overheads[
        max(0, int(math.ceil(0.95 * len(selected_overheads))) - 1)
    ]
    selected_metrics = {
        "state_availability": selected_aggregate["state_available"] / max(
            selected_aggregate["correctness_eligible"], 1.0
        ),
        "selection_coverage": selected_aggregate["locks"] / max(
            selected_aggregate["correctness_eligible"], 1.0
        ),
        "early_resolution_rate_at_completed_depth5": (
            selected_aggregate["early_locks"] / max(selected_aggregate["locks"], 1.0)
        ),
        "multi_source_early_exit_rate_at_completed_depth5_report_only": (
            selected_aggregate["multisource_early_locks"]
            / max(selected_aggregate["multisource_eligible"], 1.0)
        ),
        "wrong_early_lock_rate": selected_aggregate["wrong_early"] / max(
            selected_aggregate["early_locks"], 1.0
        ),
        "mean_stable_normalized_oracle_regret": (
            selected_aggregate["regret"] / selected_count
        ),
        "selection_critical_path_p95_fraction": selected_p95,
        "selection_budget_realized_overrun_rate": (
            selected_aggregate["budget_overruns"] / selected_count
        ),
        "budget_admission_violation_count": int(
            selected_aggregate["budget_admission_violations"]
        ),
        "illegal_lock_count": int(selected_aggregate["invalid"]),
    }
    payload = {
        "schema_version": schema_version,
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
        "selected_profile_metrics": selected_metrics,
        "code_commit": code_commit,
        "model_revision": model_revision,
        "tokenizer_hash": tokenizer_hash,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "microbenchmark_sha256": microbenchmark_sha256,
    }
    if schema_version == 5:
        payload.update(
            {
                "profile_freeze_contract_sha256": profile_freeze_contract[
                    "profile_freeze_contract_sha256"
                ],
                "profile_freeze_runtime_cost_profile_sha256": runtime_cost_profile[
                    "runtime_cost_profile_sha256"
                ],
                "development_partition_sha256": development_partition_sha256,
                "regret_normalization_floor": REGRET_NORMALIZATION_FLOOR,
                "regret_normalization_floor_semantics": (
                    "fixed_data_independent_bf16_source_state_scale_not_J_resolution"
                ),
            }
        )
    payload["profile_sha256"] = selector_profile_sha256(payload)
    payload["selector_profile_frozen"] = True
    return payload
