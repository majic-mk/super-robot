"""Fail-closed evidence boundaries for schema10 research measurements.

Real tensors and CUDA events do not make a simulated decision an online one.
Historical artifacts remain readable; only revision-2 evidence may unlock a
new Profile. Unknown measurements are None, never invented successes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from statistics import mean
from typing import Any, Mapping, Sequence


EVIDENCE_CONTRACT_VERSION = 2


class EvidenceOrigin(str, Enum):
    MEASURED = "measured"
    DERIVED = "derived"
    DIAGNOSTIC = "diagnostic"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MetricEvidence:
    value: float | bool | None
    origin: EvidenceOrigin
    input_event_ids: tuple[str, ...] = ()
    formula_version: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", EvidenceOrigin(self.origin))
        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError("evidence must be finite")
        if self.origin is EvidenceOrigin.UNAVAILABLE and self.value is not None:
            raise ValueError("unavailable evidence must be null")
        if self.origin in {EvidenceOrigin.MEASURED, EvidenceOrigin.DERIVED}:
            if self.value is None or not self.input_event_ids:
                raise ValueError("measured/derived evidence requires input events")
        if self.origin is EvidenceOrigin.DERIVED and not self.formula_version:
            raise ValueError("derived evidence requires a formula version")


@dataclass(frozen=True)
class RequestTimingEvidence:
    request_id: str
    arrival_ns: int
    first_token_ns: int
    completion_ns: int
    dense_reference_ttft_ms: float
    # Non-overlapping attribution is enforced by taking the union, not a sum.
    selection_intervals_ns: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id or not 0 <= self.arrival_ns <= self.first_token_ns <= self.completion_ns:
            raise ValueError("request timing endpoints are not ordered")
        if not math.isfinite(self.dense_reference_ttft_ms) or self.dense_reference_ttft_ms <= 0:
            raise ValueError("a matched dense TTFT reference is required")
        for start, end in self.selection_intervals_ns:
            if not self.arrival_ns <= start <= end <= self.first_token_ns:
                raise ValueError("selection interval is outside the TTFT endpoint")

    @property
    def ttft_ms(self) -> float:
        return (self.first_token_ns - self.arrival_ns) / 1e6

    @property
    def selection_active_ms(self) -> float:
        merged: list[list[int]] = []
        for start, end in sorted(self.selection_intervals_ns):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return sum(end - start for start, end in merged) / 1e6

    @property
    def selection_dense_fraction(self) -> float:
        # Descriptive active-time ratio, NOT a measurement of causal slowdown.
        return self.selection_active_ms / self.dense_reference_ttft_ms


def summarize_repair_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_request_answer_f1_drop_max: float | None = None,
) -> Mapping[str, Any]:
    """Separate actual repair correctness from QA and dense fallback.

    A caller must supply a pre-registered quality contract before a violation
    count is defined. Neither digest equality nor fixed15-vs-itself qualifies QA.
    """
    if per_request_answer_f1_drop_max is not None and (
        not math.isfinite(per_request_answer_f1_drop_max)
        or not 0 <= per_request_answer_f1_drop_max <= 1
    ):
        raise ValueError("invalid pre-registered quality threshold")
    keys = [(str(r["case_id"]), r.get("source_id"), float(r["repair_ratio"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate repair evidence row")
    actual = [r for r in rows if r.get("source_id") is not None]
    fixed = [r for r in actual if float(r["repair_ratio"]) == 0.15]
    endpoints = [r for r in actual if float(r["repair_ratio"]) == 1.0]
    fixed_cases = {str(r["case_id"]) for r in fixed}
    endpoint_cases = {str(r["case_id"]) for r in endpoints}
    integrity = sum(any(r.get(name) is not True for name in (
        "source_digest_unchanged", "artifact_digest_unchanged", "absolute_union_mask_verified"
    )) for r in fixed)
    endpoint_failures = sum(
        r.get("token_ids_equal_full") is not True
        or not math.isfinite(float(r.get("logit_relative_l2", float("inf"))))
        or float(r.get("logit_relative_l2", float("inf"))) > 1e-4
        for r in endpoints
    )
    drops = [float(r["answer_f1_drop"]) for r in fixed]
    if any(not math.isfinite(x) for x in drops):
        raise ValueError("answer-quality observations must be finite")
    by_case: dict[str, list[float]] = {}
    for r in fixed:
        by_case.setdefault(str(r["case_id"]), []).append(float(r["answer_f1_drop"]))
    quality_violations = None
    if per_request_answer_f1_drop_max is not None and by_case:
        quality_violations = sum(max(x) > per_request_answer_f1_drop_max for x in by_case.values())
    return {
        "integrity_violations": integrity,
        "actual_reuse_request_units": len(fixed_cases),
        "dense_fallback_rows": len(rows) - len(actual),
        "r1_measured_rows": len(endpoints),
        "r1_failures": endpoint_failures,
        "r1_coverage_complete": bool(fixed_cases) and fixed_cases == endpoint_cases
        and {(r["case_id"], r["source_id"]) for r in fixed}
        == {(r["case_id"], r["source_id"]) for r in endpoints},
        "observed_quality_violations": quality_violations,
        "quality_contract_present": per_request_answer_f1_drop_max is not None,
        "mean_answer_f1_drop_vs_dense": mean(drops) if drops else None,
        "quality_tail_rate_1pct_certified": False,
    }


def validate_disjoint_case_groups(
    fit_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]],
) -> None:
    if not fit_rows or not validation_rows:
        raise ValueError("fit and validation partitions must be non-empty")
    for rows in (fit_rows, validation_rows):
        if any(not r.get("case_id") or not r.get("content_group_id") for r in rows):
            raise ValueError("partition evidence requires case and content-group IDs")
    if {r["case_id"] for r in fit_rows} & {r["case_id"] for r in validation_rows}:
        raise ValueError("fit/validation case leakage")
    if {r["content_group_id"] for r in fit_rows} & {r["content_group_id"] for r in validation_rows}:
        raise ValueError("fit/validation content-group leakage")


def summarize_measured_coverage(
    rows_by_capacity: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    """Aggregate independent, actually executed capacity traces, not K16 slices.

    Compatibility may be a residual proxy and is labeled as such. The useful
    coverage additionally requires measured QA and positive paired TTFT saving.
    Creation at epoch t is after that request; it is unavailable at epoch t.
    """
    curves = []
    expected_requests = None
    for capacity, rows in sorted(rows_by_capacity.items()):
        if not 1 <= capacity <= 16 or not rows:
            raise ValueError("coverage requires nonempty capacity-specific traces")
        ids = [str(r["request_id"]) for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate request in capacity trace")
        if expected_requests is not None and set(ids) != expected_requests:
            raise ValueError("capacity curves must evaluate the same requests")
        expected_requests = set(ids)
        compatible = selected = committed = useful = 0
        for row in rows:
            if row.get("execution_kind") != "online" or row.get("capacity") != capacity:
                raise ValueError("coverage requires its own online capacity run")
            epoch = int(row["request_epoch"])
            visible = dict(row["visible_variant_creation_epochs"])
            if any(int(created) >= epoch for created in visible.values()):
                raise ValueError("operational coverage has future Variant visibility")
            eligible_ids = set(row["compatible_variant_ids"])
            selected_ids = set(row["selected_variant_ids"])
            committed_ids = set(row["committed_variant_ids"])
            if not eligible_ids <= set(visible) or not selected_ids <= eligible_ids or not committed_ids <= selected_ids:
                raise ValueError("coverage selection/commit identity chain is invalid")
            compatible += bool(eligible_ids)
            selected += bool(selected_ids)
            committed += bool(committed_ids)
            if committed_ids:
                if not isinstance(row.get("quality_passed"), bool):
                    raise ValueError("useful coverage requires actual QA outcome")
                dense = float(row["matched_dense_ttft_ms"])
                actual = float(row["actual_ttft_ms"])
                if not all(math.isfinite(x) and x > 0 for x in (dense, actual)):
                    raise ValueError("useful coverage requires measured TTFT")
                useful += row["quality_passed"] and dense > actual
        n = len(rows)
        curves.append({
            "k": capacity, "requests": n, "evidence_kind": "online_event_derived",
            "residual_compatible_coverage": (compatible / n if all(
                r.get("residual_compatibility_observed", True) for r in rows) else None),
            "selected_coverage": selected / n, "commit_coverage": committed / n,
            "qualified_positive_saving_coverage": useful / n,
        })
    if not curves:
        raise ValueError("capacity traces are missing")
    return curves


def validate_paired_gate1_executions(
    enabled: Mapping[str, Any], bypassed: Mapping[str, Any],
) -> Mapping[str, float]:
    """A dense-vs-forced-reuse experiment is NOT a Gate1 ablation."""
    for row in (enabled, bypassed):
        if row.get("execution_kind") != "online_policy" or row.get("forced_source") is not False:
            raise ValueError("Gate1 A/B requires production policies, not forced reuse")
        dense_abstention = (row.get("final_commit_not_applicable_reason") == "no_frozen_sources"
                            and row.get("selected_source_variant_ids") == []
                            and bool(row.get("selection_events"))
                            and all(not e.get("decision", {}).get("selected_source_variant_id")
                                    for e in row["selection_events"]))
        if row.get("final_commit_executed") is not True and not dense_abstention:
            raise ValueError("Gate1 bypass must retain actual FinalCommit")
    for key in ("request_id", "initial_pool_snapshot_sha256", "code_commit", "model_signature"):
        if not enabled.get(key) or enabled.get(key) != bypassed.get(key):
            raise ValueError("paired Gate1 provenance differs: " + key)
    configs = [dict(row["dispatch_config"]) for row in (enabled, bypassed)]
    if configs[0].pop("gate1_mode", None) != "explicit_barrier" or configs[1].pop("gate1_mode", None) != "fused_advisory":
        raise ValueError("paired execution must toggle only Gate1 mode")
    if configs[0] != configs[1]:
        raise ValueError("paired execution changed another dispatch parameter")
    times = [float(row["request_ttft_ms"]) for row in (enabled, bypassed)]
    if any(not math.isfinite(x) or x <= 0 for x in times):
        raise ValueError("paired execution requires request TTFT measurements")
    return {"ttft_delta_ms_without_gate1": times[1] - times[0]}


def require_profile_freeze_evidence(payload: Mapping[str, Any]) -> None:
    """Reject old proxy-only aggregates before any frozen files are created."""
    if payload.get("evidence_contract_version") != EVIDENCE_CONTRACT_VERSION:
        raise ValueError("Profile requires evidence contract v2; historical proxy aggregates are read-only")
    if payload.get("evidence_scope") != "production_policy_validation":
        raise ValueError("diagnostic measurements cannot freeze a production Profile")
    required = (
        "production_selector_replay", "request_ttft_accounting",
        "causal_operational_coverage", "task_quality_validation",
        "actual_final_commit_consistency", "same_dispatch_gate1_paired_ab",
    )
    reports = payload.get("evidence_reports", {})
    for name in required:
        report = reports.get(name, {})
        if report.get("status") != "passed" or report.get("origin") not in {"measured", "derived"}:
            raise ValueError("missing validated evidence: " + name)
        digest = report.get("input_events_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("evidence input event digest required: " + name)
        if report.get("request_units", 0) < 1 or report.get("code_commit") != payload.get("code_commit"):
            raise ValueError("evidence request count/code binding differs: " + name)
    partitions = payload.get("selection_validation_partitions", {})
    validate_disjoint_case_groups(partitions.get("fit", ()), partitions.get("validation", ()))
    quality = payload.get("selected_repair_policy", {})
    if quality.get("quality_contract_sha256") is None or quality.get("observed_development_violations") is None:
        raise ValueError("task quality cannot be inferred from integrity or fixed15 self-comparison")
    consistency = payload.get("final_consistency", {})
    if consistency.get("final_commit_gamma_violations") != 0 or consistency.get("runtime_cost_consistent") is not True:
        raise ValueError("actual final-commit consistency has not passed")
