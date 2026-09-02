from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Mapping, Sequence, Tuple

from .v8_schema10_contracts import VariantMaterializationReasonV10


@dataclass(frozen=True)
class MaterializationOutcome:
    variant_id: str
    reason: VariantMaterializationReasonV10
    no_compatible_stored_variant_claimed: bool
    deep_full_candidate_oracle_no_compatible: bool | None
    compared_within_32: bool
    selected_within_32: bool
    final_commit_within_32: bool
    positive_saved_ms_within_32: bool
    write_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.variant_id or self.write_bytes < 0:
            raise ValueError("invalid materialization outcome")
        if (
            self.no_compatible_stored_variant_claimed
            and self.deep_full_candidate_oracle_no_compatible is None
        ):
            raise ValueError("novelty claim requires a full-candidate oracle")

    @property
    def context_novelty_claimed(self) -> bool:
        """Compatibility alias for historical, pre-Profile schema10 rows."""
        return self.no_compatible_stored_variant_claimed

    @property
    def deep_full_candidate_oracle_novel(self) -> bool | None:
        """Historical reader alias; new evidence only claims stored-pool mismatch."""
        return self.deep_full_candidate_oracle_no_compatible


def materialization_quality_metrics(
    rows: Sequence[MaterializationOutcome],
) -> Mapping[str, float | int | None]:
    novelty_claims = [
        row for row in rows if row.no_compatible_stored_variant_claimed
    ]
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
        "stored_pool_mismatch_precision": (
            sum(
                row.deep_full_candidate_oracle_no_compatible is True
                for row in novelty_claims
            )
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


def one_sided_clopper_pearson_upper(
    violations: int,
    request_units: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Exact one-sided binomial upper confidence bound without SciPy."""
    if request_units < 1 or not 0 <= violations <= request_units:
        raise ValueError("invalid binomial certification counts")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    if violations == request_units:
        return 1.0
    alpha = 1.0 - confidence

    def cdf(probability: float) -> float:
        return sum(
            math.comb(request_units, index)
            * probability**index
            * (1.0 - probability) ** (request_units - index)
            for index in range(violations + 1)
        )

    lower, upper = 0.0, 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        # P(X <= observed) decreases with p.  The upper limit solves CDF=alpha.
        if cdf(midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


@dataclass(frozen=True)
class CoverageVariantObservation:
    variant_id: str
    creation_epoch: int
    metadata_rank: int
    residual_score: float
    absolute_compatible: bool
    final_commit_admitted: bool
    realized_saved_ms: float
    verified_at_creation: bool = True

    def __post_init__(self) -> None:
        if not self.variant_id or self.creation_epoch < 0 or self.metadata_rank < 0:
            raise ValueError("invalid coverage Variant observation")
        if not math.isfinite(self.residual_score) or not math.isfinite(
            self.realized_saved_ms
        ):
            raise ValueError("coverage scores must be finite")


@dataclass(frozen=True)
class CoverageTraceRequest:
    request_id: str
    request_epoch: int
    content_id: str
    compared_k_budget: int
    variants: Tuple[CoverageVariantObservation, ...]
    materialized_variant_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.request_id
            or not self.content_id
            or self.request_epoch < 0
            or self.compared_k_budget < 0
        ):
            raise ValueError("invalid coverage trace request")
        ids = tuple(row.variant_id for row in self.variants)
        if len(ids) != len(set(ids)):
            raise ValueError("coverage request repeats a Variant")
        if self.materialized_variant_id is not None and self.materialized_variant_id not in ids:
            raise ValueError("materialized Variant lacks trace provenance")


@dataclass(frozen=True)
class CoverageCurvePoint:
    k: int
    requests: int
    compatible_coverage: float
    selected_coverage: float
    commit_coverage: float
    resident_variant_peak: int
    oracle: bool


def _catalog(
    rows: Sequence[CoverageTraceRequest],
) -> Dict[Tuple[str, str], CoverageVariantObservation]:
    values: Dict[Tuple[str, str], CoverageVariantObservation] = {}
    for request in rows:
        for variant in request.variants:
            key = (request.content_id, variant.variant_id)
            existing = values.get(key)
            if existing is None or variant.creation_epoch < existing.creation_epoch:
                values[key] = variant
    return values


def replay_coverage_curve(
    rows: Sequence[CoverageTraceRequest],
    *,
    k_values: Sequence[int] = (1, 2, 4, 8, 16),
    oracle: bool = False,
) -> Tuple[CoverageCurvePoint, ...]:
    """Replay Variant visibility causally, or report an explicit future-pool oracle.

    Operational replay never exposes ``creation_epoch >= request_epoch``.  The
    oracle deliberately sees the complete future catalog and is marked in the
    returned points so it cannot be mixed with the online curve.
    """
    ordered = tuple(sorted(rows, key=lambda row: (row.request_epoch, row.request_id)))
    if not ordered or tuple(sorted(set(k_values))) != tuple(k_values) or any(
        not 1 <= int(value) <= 16 for value in k_values
    ):
        raise ValueError("coverage replay requires rows and sorted K in [1,16]")
    catalog = _catalog(ordered)
    points = []
    for capacity in k_values:
        pools: Dict[str, Dict[str, Dict[str, int | bool]]] = {}
        known_inserted: set[Tuple[str, str]] = set()
        compatible = selected = committed = peak = 0
        for request in ordered:
            pool = pools.setdefault(request.content_id, {})
            if oracle:
                # Oracle rows carry request-specific counterfactual scores for
                # every eventual Variant. Never reuse another request's J_s.
                visible = list(request.variants)
                visible.sort(key=lambda row: (row.residual_score, row.variant_id))
                visible = visible[:capacity]
            else:
                # Creation at epoch t happens after request t and first becomes
                # visible at the next request.  Evicted rows never reappear.
                new_rows = sorted(
                    (
                        value for (content, variant_id), value in catalog.items()
                        if content == request.content_id
                        and value.creation_epoch < request.request_epoch
                        and (content, variant_id) not in known_inserted
                    ),
                    key=lambda row: (row.creation_epoch, row.variant_id),
                )
                for value in new_rows:
                    pool[value.variant_id] = {
                        "last_use": value.creation_epoch,
                        "registered": value.creation_epoch,
                        "comparisons": 2 if value.verified_at_creation else 0,
                        "lookup_opportunities": 0,
                        "verified": bool(value.verified_at_creation),
                    }
                    while len(pool) > capacity:
                        probation = sorted(
                            (
                                variant_id for variant_id, state in pool.items()
                                if not bool(state["verified"])
                                and int(state["lookup_opportunities"]) < 2
                            ),
                            key=lambda variant_id: (
                                int(pool[variant_id]["registered"]), variant_id
                            ),
                            reverse=True,
                        )
                        protected = set(probation[:2])
                        victims = [
                            variant_id for variant_id in pool
                            if variant_id not in protected
                            and variant_id != value.variant_id
                        ]
                        if not victims:
                            del pool[value.variant_id]
                            break
                        victim = min(
                            victims,
                            key=lambda variant_id: (
                                int(pool[variant_id]["last_use"]), variant_id
                            ),
                        )
                        del pool[victim]
                    known_inserted.add((request.content_id, value.variant_id))
                by_id = {row.variant_id: row for row in request.variants}
                visible = [by_id[variant_id] for variant_id in pool if variant_id in by_id]
                visible.sort(key=lambda row: (row.metadata_rank, row.variant_id))
                visible = visible[: min(len(visible), request.compared_k_budget)]
            peak = max(peak, len(visible) if oracle else len(pool))
            compatible_rows = [row for row in visible if row.absolute_compatible]
            if compatible_rows:
                compatible += 1
                winner = min(
                    compatible_rows,
                    key=lambda row: (row.residual_score, row.metadata_rank, row.variant_id),
                )
                selected += 1
                if winner.final_commit_admitted and winner.realized_saved_ms > 0:
                    committed += 1
                if not oracle and winner.variant_id in pool:
                    pool[winner.variant_id]["last_use"] = request.request_epoch
            if not oracle:
                compared_ids = {row.variant_id for row in visible}
                for variant_id, state in pool.items():
                    if variant_id in compared_ids:
                        state["comparisons"] = int(state["comparisons"]) + 1
                        if int(state["comparisons"]) >= 2:
                            state["verified"] = True
                    if not bool(state["verified"]):
                        state["lookup_opportunities"] = (
                            int(state["lookup_opportunities"]) + 1
                        )
        denominator = len(ordered)
        points.append(
            CoverageCurvePoint(
                int(capacity), denominator,
                compatible / denominator,
                selected / denominator,
                committed / denominator,
                peak,
                oracle,
            )
        )
    return tuple(points)


def coverage_curve_summary(
    operational: Sequence[CoverageCurvePoint],
    oracle: Sequence[CoverageCurvePoint],
    *,
    saturation_gap: float = 0.01,
) -> Mapping[str, object]:
    if tuple(row.k for row in operational) != tuple(row.k for row in oracle):
        raise ValueError("Oracle and operational coverage K grids differ")
    if any(row.oracle for row in operational) or any(not row.oracle for row in oracle):
        raise ValueError("Oracle and operational curves are mislabeled")
    if not 0 <= saturation_gap <= 1:
        raise ValueError("coverage saturation gap is invalid")

    def near_saturation(points: Sequence[CoverageCurvePoint]) -> int:
        target = points[-1].commit_coverage - saturation_gap
        return next(row.k for row in points if row.commit_coverage >= target)

    def payload(points: Sequence[CoverageCurvePoint]) -> list[Mapping[str, object]]:
        result = []
        previous = None
        for row in points:
            gain = None if previous is None else row.commit_coverage - previous
            result.append({**row.__dict__, "marginal_commit_gain": gain})
            previous = row.commit_coverage
        return result

    return {
        "operational": payload(operational),
        "oracle": payload(oracle),
        "operational_near_saturation_k": near_saturation(operational),
        "oracle_near_saturation_k": near_saturation(oracle),
        "saturation_gap": saturation_gap,
    }
