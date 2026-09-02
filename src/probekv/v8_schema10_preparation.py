from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence, Tuple

from .v8_schema10_contracts import Gate1Mode
from .v8_schema10_profile import PreparationPolicyProfile


@dataclass(frozen=True)
class PreparationCostObservation:
    request_id: str
    dense_reference_total_ms: float
    gate1_passed: bool
    additional_winner_full_kv_bytes_without_gate1: int
    additional_visible_copy_ms_without_gate1: float
    additional_pinned_staging_ms_without_gate1: float
    additional_copy_interference_ms_without_gate1: float
    additional_hbm_reservation_byte_ms_without_gate1: float
    additional_wasted_preparation_ms_without_gate1: float
    ttft_delta_ms_without_gate1: float
    counterfactual_path_economically_invalid: bool
    counterfactual_final_commit_admitted: bool
    correctness_violation_without_gate1: bool = False
    final_gamma_violation_without_gate1: bool = False

    def __post_init__(self) -> None:
        if not self.request_id or self.dense_reference_total_ms <= 0:
            raise ValueError("counterfactual observation requires request/dense time")
        if min(
            self.additional_winner_full_kv_bytes_without_gate1,
            self.additional_visible_copy_ms_without_gate1,
            self.additional_pinned_staging_ms_without_gate1,
            self.additional_copy_interference_ms_without_gate1,
            self.additional_hbm_reservation_byte_ms_without_gate1,
            self.additional_wasted_preparation_ms_without_gate1,
        ) < 0:
            raise ValueError("counterfactual preparation costs must be non-negative")


@dataclass(frozen=True)
class Gate1PairedABObservation:
    request_id: str
    dataset: str
    dense_reference_total_ms: float
    shadow_additional_overhead_ms: float
    realized_additional_overhead_ms: float
    gate1_enabled_wall_ms: float
    gate1_bypassed_wall_ms: float
    additional_transferred_bytes: int
    final_commit_match: bool
    correctness_match: bool

    def __post_init__(self) -> None:
        if not self.request_id or not self.dataset or self.dense_reference_total_ms <= 0:
            raise ValueError("paired Gate1 A/B observation is incomplete")
        if min(
            self.shadow_additional_overhead_ms,
            self.realized_additional_overhead_ms,
            self.gate1_enabled_wall_ms,
            self.gate1_bypassed_wall_ms,
            self.additional_transferred_bytes,
        ) < 0:
            raise ValueError("paired Gate1 A/B costs must be non-negative")


@dataclass(frozen=True)
class Gate1CounterfactualSummary:
    observations: int
    gate1_pass_rate: float
    gate1_exclusive_rejection_rate: float
    invalid_path_catch_rate: float
    mean_overhead_fraction: float
    p95_overhead_fraction: float
    additional_transferred_bytes_fraction: float
    correctness_or_gamma_violations: int
    uncaught_invalid_paths: int
    paired_observations: int
    paired_mean_absolute_error_fraction: float
    paired_p95_absolute_error_fraction: float
    paired_final_commit_or_correctness_mismatches: int
    recommended_gate1_mode: Gate1Mode
    reasons: Tuple[str, ...]


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("quantile requires observations")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def evaluate_gate1_counterfactual(
    observations: Sequence[PreparationCostObservation],
    *,
    total_winner_full_kv_bytes_with_gate1: int,
    profile: PreparationPolicyProfile,
    paired_observations: Sequence[Gate1PairedABObservation] = (),
) -> Gate1CounterfactualSummary:
    """Pure audit: it performs no transfer, lease, reservation, or state mutation."""
    if not observations:
        raise ValueError("Gate1 counterfactual requires development observations")
    if total_winner_full_kv_bytes_with_gate1 < 0:
        raise ValueError("winner bytes must be non-negative")
    overhead_fractions = [
        row.additional_wasted_preparation_ms_without_gate1
        / row.dense_reference_total_ms
        for row in observations
    ]
    mean_overhead = mean(overhead_fractions)
    p95_overhead = _linear_quantile(overhead_fractions, 0.95)
    additional_bytes = sum(
        row.additional_winner_full_kv_bytes_without_gate1 for row in observations
    )
    byte_fraction = additional_bytes / max(1, total_winner_full_kv_bytes_with_gate1)
    violations = sum(
        row.correctness_violation_without_gate1
        or row.final_gamma_violation_without_gate1
        for row in observations
    )
    invalid_rows = [row for row in observations if row.counterfactual_path_economically_invalid]
    uncaught = sum(row.counterfactual_final_commit_admitted for row in invalid_rows)
    caught = len(invalid_rows) - uncaught
    reasons = []
    if violations:
        reasons.append("correctness_or_final_gamma_violation")
    if uncaught:
        reasons.append("final_commit_missed_invalid_path")
    if mean_overhead > profile.mean_overhead_limit_fraction:
        reasons.append("mean_preparation_overhead_exceeded")
    if p95_overhead > profile.p95_overhead_limit_fraction:
        reasons.append("p95_preparation_overhead_exceeded")
    if byte_fraction > profile.transferred_bytes_limit_fraction:
        reasons.append("additional_transferred_bytes_exceeded")
    paired_errors = [
        abs(row.shadow_additional_overhead_ms - row.realized_additional_overhead_ms)
        / row.dense_reference_total_ms
        for row in paired_observations
    ]
    paired_mean_error = mean(paired_errors) if paired_errors else float("inf")
    paired_p95_error = (
        _linear_quantile(paired_errors, 0.95) if paired_errors else float("inf")
    )
    paired_mismatches = sum(
        not row.final_commit_match or not row.correctness_match
        for row in paired_observations
    )
    if len(paired_observations) < 18:
        reasons.append("paired_gate1_ab_coverage_incomplete")
    if paired_mean_error > profile.paired_mean_error_limit_fraction:
        reasons.append("paired_gate1_mean_estimation_error_exceeded")
    if paired_p95_error > profile.paired_p95_error_limit_fraction:
        reasons.append("paired_gate1_p95_estimation_error_exceeded")
    if paired_mismatches:
        reasons.append("paired_gate1_outcome_mismatch")
    recommended = (
        Gate1Mode.FUSED_ADVISORY if not reasons else Gate1Mode.EXPLICIT_BARRIER
    )
    return Gate1CounterfactualSummary(
        observations=len(observations),
        gate1_pass_rate=sum(row.gate1_passed for row in observations)
        / len(observations),
        gate1_exclusive_rejection_rate=sum(
            (not row.gate1_passed)
            and (not row.counterfactual_path_economically_invalid)
            and row.counterfactual_final_commit_admitted
            for row in observations
        )
        / len(observations),
        invalid_path_catch_rate=(caught / len(invalid_rows) if invalid_rows else 1.0),
        mean_overhead_fraction=mean_overhead,
        p95_overhead_fraction=p95_overhead,
        additional_transferred_bytes_fraction=byte_fraction,
        correctness_or_gamma_violations=violations,
        uncaught_invalid_paths=uncaught,
        paired_observations=len(paired_observations),
        paired_mean_absolute_error_fraction=paired_mean_error,
        paired_p95_absolute_error_fraction=paired_p95_error,
        paired_final_commit_or_correctness_mismatches=paired_mismatches,
        recommended_gate1_mode=recommended,
        reasons=tuple(reasons),
    )


def assert_preparation_contract(
    *,
    atomic_reservation_acquired: bool,
    final_commit_admitted: bool,
    selective_reuse_started: bool,
) -> None:
    if selective_reuse_started and not atomic_reservation_acquired:
        raise RuntimeError("selective reuse cannot bypass AtomicPreparationReservation")
    if selective_reuse_started and not final_commit_admitted:
        raise RuntimeError("selective reuse cannot bypass FinalCommitAdmission")
