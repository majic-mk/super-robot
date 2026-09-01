from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence

from .v8_schema10_contracts import VariantMaterializationReasonV10


@dataclass(frozen=True)
class MaterializationOutcome:
    variant_id: str
    reason: VariantMaterializationReasonV10
    context_novelty_claimed: bool
    deep_full_candidate_oracle_novel: bool | None
    compared_within_32: bool
    selected_within_32: bool
    final_commit_within_32: bool
    positive_saved_ms_within_32: bool
    write_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.variant_id or self.write_bytes < 0:
            raise ValueError("invalid materialization outcome")
        if self.context_novelty_claimed and self.deep_full_candidate_oracle_novel is None:
            raise ValueError("novelty claim requires a full-candidate oracle")


def materialization_quality_metrics(
    rows: Sequence[MaterializationOutcome],
) -> Mapping[str, float | int | None]:
    novelty_claims = [row for row in rows if row.context_novelty_claimed]
    exploration = [
        row
        for row in rows
        if row.reason
        is VariantMaterializationReasonV10.BUDGET_TRUNCATED_EXPLORATION
    ]
    useful = [
        row
        for row in rows
        if row.final_commit_within_32 and row.positive_saved_ms_within_32
    ]
    return {
        "materializations": len(rows),
        "novelty_precision": (
            sum(row.deep_full_candidate_oracle_novel is True for row in novelty_claims)
            / len(novelty_claims)
            if novelty_claims
            else None
        ),
        "exploration_yield_at_32": (
            sum(row.compared_within_32 or row.selected_within_32 for row in exploration)
            / len(exploration)
            if exploration
            else None
        ),
        "useful_materialization_precision_at_32": (
            len(useful) / len(rows) if rows else None
        ),
        "write_amplification_bytes": sum(row.write_bytes for row in rows),
    }


@dataclass(frozen=True)
class VariantGrowthPoint:
    request_epoch: int
    variant_count: int
    selected_variant_rank: int | None
    selected_variant_age: int | None
    cpu_resident_bytes: int
    ssd_resident_bytes: int
    probation_count: int
    verified_count: int
    expired_count: int
    evicted_count: int
    materialization_write_bytes: int = 0
    replacements_this_request: int = 0
    miss_to_reuse_conversion: bool = False
    marginal_reuse_admission_improvement_ms: float = 0.0
    first_selection_delay_requests: int | None = None
    ttft_ms: float = 0.0
    steady_state: bool = False

    def __post_init__(self) -> None:
        if self.request_epoch < 0 or not 1 <= self.variant_count <= 16:
            raise ValueError("growth point is outside the schema10 Variant bounds")
        if min(
            self.cpu_resident_bytes,
            self.ssd_resident_bytes,
            self.probation_count,
            self.verified_count,
            self.expired_count,
            self.evicted_count,
            self.materialization_write_bytes,
            self.replacements_this_request,
            self.marginal_reuse_admission_improvement_ms,
            self.ttft_ms,
        ) < 0:
            raise ValueError("growth counters must be non-negative")
        if (
            self.first_selection_delay_requests is not None
            and self.first_selection_delay_requests < 0
        ):
            raise ValueError("first-selection delay must be non-negative")


def summarize_variant_growth(rows: Sequence[VariantGrowthPoint]) -> Mapping[str, float | int]:
    if not rows:
        raise ValueError("Variant growth summary requires a trace")
    first_selection_delays = [
        row.first_selection_delay_requests
        for row in rows
        if row.first_selection_delay_requests is not None
    ]
    warmup_ttft = [row.ttft_ms for row in rows if not row.steady_state]
    steady_ttft = [row.ttft_ms for row in rows if row.steady_state]
    return {
        "requests": len(rows),
        "mean_variant_count": mean(row.variant_count for row in rows),
        "saturation_probability_k16": sum(row.variant_count == 16 for row in rows)
        / len(rows),
        "final_variant_count": rows[-1].variant_count,
        "peak_cpu_resident_bytes": max(row.cpu_resident_bytes for row in rows),
        "peak_ssd_resident_bytes": max(row.ssd_resident_bytes for row in rows),
        "final_probation_count": rows[-1].probation_count,
        "final_verified_count": rows[-1].verified_count,
        "final_expired_count": rows[-1].expired_count,
        "cumulative_evicted_count": rows[-1].evicted_count,
        "materialization_write_amplification_bytes": sum(
            row.materialization_write_bytes for row in rows
        ),
        "replacement_count": sum(row.replacements_this_request for row in rows),
        "replacement_frequency": sum(
            row.replacements_this_request > 0 for row in rows
        )
        / len(rows),
        "miss_to_reuse_conversion_rate": sum(
            row.miss_to_reuse_conversion for row in rows
        )
        / len(rows),
        "mean_marginal_reuse_admission_improvement_ms": mean(
            row.marginal_reuse_admission_improvement_ms for row in rows
        ),
        "mean_first_selection_delay_requests": (
            mean(first_selection_delays) if first_selection_delays else 0.0
        ),
        "warmup_mean_ttft_ms": mean(warmup_ttft) if warmup_ttft else 0.0,
        "steady_state_mean_ttft_ms": mean(steady_ttft) if steady_ttft else 0.0,
    }
