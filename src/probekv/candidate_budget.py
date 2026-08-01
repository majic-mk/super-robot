from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .v6_contracts import VariantComparisonAudit


@dataclass(frozen=True)
class VariantComparisonCandidate:
    segment_id: str
    source_id: str
    metadata_score: float
    predicted_saved_ms: float
    comparison_upper_ms: float
    eligible: bool = True

    def __post_init__(self) -> None:
        if not self.segment_id or not self.source_id:
            raise ValueError("comparison candidate identifiers are required")
        if self.comparison_upper_ms < 0:
            raise ValueError("comparison time upper bound must be non-negative")
        if not all(
            math.isfinite(value)
            for value in (
                self.metadata_score,
                self.predicted_saved_ms,
                self.comparison_upper_ms,
            )
        ):
            raise ValueError("comparison candidate values must be finite")

    @property
    def benefit_density(self) -> float:
        if self.comparison_upper_ms <= 1e-12:
            return float("inf") if self.predicted_saved_ms > 0 else 0.0
        return max(0.0, self.predicted_saved_ms) / self.comparison_upper_ms


@dataclass(frozen=True)
class RequestComparisonAllocation:
    audits: Tuple[VariantComparisonAudit, ...]
    budget_limit_ms: float
    budget_available_ms: float
    budget_used_ms: float
    probe_ms: float
    metadata_ms: float
    full_reference_ms: float

    def __post_init__(self) -> None:
        if min(
            self.budget_limit_ms,
            self.budget_available_ms,
            self.budget_used_ms,
            self.probe_ms,
            self.metadata_ms,
            self.full_reference_ms,
        ) < 0:
            raise ValueError("comparison allocation timings must be non-negative")
        if self.budget_used_ms > self.budget_available_ms + 1e-9:
            raise ValueError("comparison allocation exceeds request budget")

    def compared_by_segment(self) -> Mapping[str, Tuple[str, ...]]:
        return {
            audit.segment_id: audit.compared_source_ids
            for audit in self.audits
        }


def allocate_variant_comparisons(
    candidates: Sequence[VariantComparisonCandidate],
    *,
    full_reference_ms: float,
    probe_ms: float,
    metadata_ms: float,
    budget_fraction: float = 0.05,
    max_per_segment: int = 16,
    segment_ids: Sequence[str] = (),
    stored_count_by_segment: Optional[Mapping[str, int]] = None,
) -> RequestComparisonAllocation:
    """Allocate current-state comparisons without enumerating Source products.

    All summaries are compared when they fit the request-level budget. Under
    pressure, the metadata-best candidate for high-value segments is allocated
    first, followed by remaining variants in benefit-density order. A segment
    that receives no comparison must abstain; metadata alone never selects a
    Source in the ProbeKV main policy.
    """

    if full_reference_ms <= 0:
        raise ValueError("full reference time must be positive")
    if min(probe_ms, metadata_ms) < 0:
        raise ValueError("probe and metadata times must be non-negative")
    if not all(
        math.isfinite(value)
        for value in (full_reference_ms, probe_ms, metadata_ms, budget_fraction)
    ):
        raise ValueError("comparison budget values must be finite")
    if not 0 < budget_fraction <= 1:
        raise ValueError("comparison budget fraction must be in (0, 1]")
    if max_per_segment < 1:
        raise ValueError("max_per_segment must be positive")
    pairs = [(candidate.segment_id, candidate.source_id) for candidate in candidates]
    if len(pairs) != len(set(pairs)):
        raise ValueError("comparison candidates must be unique per segment")

    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("segment inventory must contain unique IDs")
    grouped: Dict[str, list] = {str(segment_id): [] for segment_id in segment_ids}
    for candidate in candidates:
        grouped.setdefault(candidate.segment_id, []).append(candidate)
    stored_counts = dict(stored_count_by_segment or {})
    if set(stored_counts) - set(grouped):
        raise ValueError("stored counts reference an unknown segment")
    for segment_id, segment_candidates in grouped.items():
        if len(segment_candidates) > max_per_segment:
            raise ValueError(
                "segment %s exceeds the comparison safety ceiling" % segment_id
            )
        stored = int(stored_counts.get(segment_id, len(segment_candidates)))
        if stored != len(segment_candidates) or not 0 <= stored <= 16:
            raise ValueError("stored variant count is inconsistent")

    budget_limit = budget_fraction * full_reference_ms
    budget_available = max(0.0, budget_limit - probe_ms - metadata_ms)
    eligible_candidates = [
        candidate for candidate in candidates if candidate.eligible
    ]
    total_compare = sum(
        candidate.comparison_upper_ms for candidate in eligible_candidates
    )
    selected = []
    selected_keys = set()
    used = 0.0

    if total_compare <= budget_available + 1e-12:
        selected = list(eligible_candidates)
        selected_keys = {
            (candidate.segment_id, candidate.source_id)
            for candidate in eligible_candidates
        }
        used = total_compare
    else:
        metadata_best = []
        remaining = []
        for segment_id, segment_candidates in grouped.items():
            segment_candidates = [
                candidate
                for candidate in segment_candidates
                if candidate.eligible
            ]
            if not segment_candidates:
                continue
            ordered = sorted(
                segment_candidates,
                key=lambda item: (
                    item.metadata_score,
                    -item.predicted_saved_ms,
                    item.source_id,
                ),
            )
            metadata_best.append(ordered[0])
            remaining.extend(ordered[1:])
        metadata_best.sort(
            key=lambda item: (
                -item.benefit_density,
                item.metadata_score,
                item.segment_id,
                item.source_id,
            )
        )
        remaining.sort(
            key=lambda item: (
                -item.benefit_density,
                item.metadata_score,
                item.segment_id,
                item.source_id,
            )
        )
        for candidate in metadata_best + remaining:
            next_used = used + candidate.comparison_upper_ms
            if next_used > budget_available + 1e-12:
                continue
            key = (candidate.segment_id, candidate.source_id)
            selected.append(candidate)
            selected_keys.add(key)
            used = next_used

    audits = []
    for segment_id in grouped:
        segment_candidates = grouped[segment_id]
        stored = int(stored_counts.get(segment_id, len(segment_candidates)))
        compared = tuple(
            candidate.source_id
            for candidate in selected
            if candidate.segment_id == segment_id
        )
        dropped = tuple(
            candidate.source_id
            for candidate in segment_candidates
            if (segment_id, candidate.source_id) not in selected_keys
        )
        segment_used = sum(
            candidate.comparison_upper_ms
            for candidate in selected
            if candidate.segment_id == segment_id
        )
        audits.append(
            VariantComparisonAudit(
                segment_id=segment_id,
                stored_k=stored,
                eligible_k=sum(
                    candidate.eligible for candidate in segment_candidates
                ),
                compared_source_ids=compared,
                dropped_source_ids=dropped,
                budget_used_ms=segment_used,
                budget_limit_ms=budget_limit,
            )
        )
    return RequestComparisonAllocation(
        audits=tuple(audits),
        budget_limit_ms=budget_limit,
        budget_available_ms=budget_available,
        budget_used_ms=used,
        probe_ms=probe_ms,
        metadata_ms=metadata_ms,
        full_reference_ms=full_reference_ms,
    )
