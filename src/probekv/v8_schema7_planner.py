from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from .v8_schema6_contracts import PlannerSnapshot
from .v8_schema6_planner import (
    FrozenSegmentCandidate,
    JointTimelineEstimator,
    PredictedJointPlannerV6,
    RefinedJointPlannerV6,
)
from .v8_schema7_contracts import FinalCommitDecision


@dataclass(frozen=True)
class PreparationAdmissionDecision:
    disposition_by_segment: Mapping[str, str]
    predicted_request_total_ms: float
    dense_reference_total_ms: float
    joint_future_ms: float
    planner_snapshot: PlannerSnapshot
    inventory_segment_ids: Tuple[str, ...]
    source_variant_by_segment: Mapping[str, str]


class PreparationAdmissionPlanner:
    """Request-level predicted joint plan before any physical preparation."""

    def __init__(self, estimator: JointTimelineEstimator, *, gamma: float = 0.8) -> None:
        self._planner = PredictedJointPlannerV6(estimator, gamma=gamma)

    def plan_incremental(
        self,
        *,
        inventory_segment_ids: Sequence[str],
        frozen_candidates: Sequence[FrozenSegmentCandidate],
        existing_provisional_segment_ids: Sequence[str],
        existing_deferred_segment_ids: Sequence[str],
        predicted_dense_segment_ids: Sequence[str],
        committed_segment_ids: Sequence[str],
        actual_sunk_ms: float,
        dense_reference_total_ms: float,
        selection_closed: bool,
        snapshot: PlannerSnapshot,
        current_snapshot: PlannerSnapshot,
        union_mask_digest: str,
    ) -> PreparationAdmissionDecision:
        legacy = self._planner.plan_incremental(
            inventory_segment_ids=inventory_segment_ids,
            frozen_candidates=frozen_candidates,
            existing_provisional_segment_ids=existing_provisional_segment_ids,
            existing_deferred_segment_ids=existing_deferred_segment_ids,
            predicted_dense_segment_ids=predicted_dense_segment_ids,
            committed_segment_ids=committed_segment_ids,
            actual_sunk_ms=actual_sunk_ms,
            dense_reference_total_ms=dense_reference_total_ms,
            selection_closed=selection_closed,
            snapshot=snapshot,
            current_snapshot=current_snapshot,
            union_mask_digest=union_mask_digest,
        )
        return PreparationAdmissionDecision(
            {key: value.value for key, value in legacy.disposition_by_segment.items()},
            legacy.predicted_request_total_ms,
            legacy.dense_reference_total_ms,
            legacy.joint_future_ms,
            legacy.snapshot,
            legacy.inventory_segment_ids,
            legacy.source_variant_by_segment,
        )


class FinalCommitPlanner:
    """Refined ready-subset admission immediately before selective execution."""

    def __init__(self, estimator: JointTimelineEstimator, *, gamma: float = 0.8) -> None:
        self._planner = RefinedJointPlannerV6(estimator, gamma=gamma)

    def plan_ready_subset(
        self,
        *,
        inventory_segment_ids: Sequence[str],
        eligible_ready_segment_ids: Sequence[str],
        committed_segment_ids: Sequence[str],
        actual_boundary_by_segment: Mapping[str, int],
        actual_sunk_ms: float,
        dense_reference_total_ms: float,
        snapshot: PlannerSnapshot,
        current_snapshot: PlannerSnapshot,
        union_mask_digest: str,
    ) -> FinalCommitDecision:
        legacy = self._planner.plan_subset(
            inventory_segment_ids=inventory_segment_ids,
            eligible_ready_segment_ids=eligible_ready_segment_ids,
            committed_segment_ids=committed_segment_ids,
            actual_boundary_by_segment=actual_boundary_by_segment,
            actual_sunk_ms=actual_sunk_ms,
            dense_reference_total_ms=dense_reference_total_ms,
            snapshot=snapshot,
            current_snapshot=current_snapshot,
            union_mask_digest=union_mask_digest,
        )
        return FinalCommitDecision(
            accepted_ready_segment_ids=legacy.accepted_ready_segment_ids,
            rejected_ready_segment_ids=legacy.rejected_ready_segment_ids,
            untouched_segment_ids=legacy.untouched_segment_ids,
            request_total_ms=legacy.request_total_ms,
            dense_reference_total_ms=legacy.dense_reference_total_ms,
            planner_snapshot=snapshot,
            reason_by_segment=legacy.reason_by_segment,
        )
