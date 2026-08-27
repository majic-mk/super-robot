from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from .v8_contracts import ResidualSelectionDecision, ResidualSelectionState
from .v8_leases import (
    LeasePurpose,
    LeaseRecord,
    ReplicaLeaseRequest,
    V8LeaseManager,
)
from .v8_planner import (
    PredictedJointPlanner,
    PredictedRequestPlan,
    PredictedSegmentOption,
    RefinedJointPlanner,
    RefinedRequestPlan,
    RefinedSegmentMeasurement,
    SegmentPlanState,
    UnifiedCostComponents,
)


@dataclass(frozen=True)
class V8ClosedLoopState:
    request_id: str
    request_generation: int
    selected_source_by_segment: Mapping[str, str]
    logical_lease_ids: Tuple[str, ...]
    physical_lease_ids: Tuple[str, ...]
    predicted_plan: PredictedRequestPlan


class V8RequestOrchestrator:
    """Bind selector output to leases, two planners and final monotonic admission."""

    def __init__(
        self,
        lease_manager: V8LeaseManager,
        predicted_planner: PredictedJointPlanner,
        refined_planner: RefinedJointPlanner,
    ) -> None:
        self.lease_manager = lease_manager
        self.predicted_planner = predicted_planner
        self.refined_planner = refined_planner

    def freeze_and_predict(
        self,
        *,
        request_id: str,
        request_generation: int,
        decisions: Mapping[str, ResidualSelectionDecision],
        options: Sequence[PredictedSegmentOption],
        shared_sunk: UnifiedCostComponents,
        dense_reference_ms: float,
        predicted_remaining_s: float,
        joint_interference_upper_ms: float = 0.0,
    ) -> V8ClosedLoopState:
        option_by_segment = {item.segment_id: item for item in options}
        locked = {
            segment_id: str(decision.selected_source_variant_id)
            for segment_id, decision in decisions.items()
            if decision.state is ResidualSelectionState.LOCKED
        }
        if len(option_by_segment) != len(options):
            raise ValueError("Predicted options must be unique per Segment")
        if set(option_by_segment) != set(locked):
            raise ValueError("only selector-locked Segments may enter Predicted planning")
        for segment_id, source_id in locked.items():
            if option_by_segment[segment_id].source_variant_id != source_id:
                raise RuntimeError("Predicted planner cannot substitute the selected Source")

        logical: list[LeaseRecord] = []
        try:
            for segment_id, source_id in sorted(locked.items()):
                logical.append(
                    self.lease_manager.freeze_and_acquire_logical(
                        request_id=request_id,
                        request_generation=request_generation,
                        segment_id=segment_id,
                        source_variant_id=source_id,
                        predicted_remaining_s=predicted_remaining_s,
                    )
                )
        except Exception:
            for lease in logical:
                self.lease_manager.release(lease.lease_id, reason="logical_batch_rollback")
            raise

        predicted = self.predicted_planner.plan(
            request_id,
            options,
            shared_sunk=shared_sunk,
            dense_reference_ms=dense_reference_ms,
            joint_interference_upper_ms=joint_interference_upper_ms,
        )
        provisional = {
            decision.segment_id
            for decision in predicted.decisions
            if decision.state is SegmentPlanState.PROVISIONAL_REUSE
        }
        physical_requests = []
        for segment_id in sorted(provisional):
            option = option_by_segment[segment_id]
            physical_requests.append(
                ReplicaLeaseRequest(
                    segment_id=segment_id,
                    source_variant_id=option.source_variant_id,
                    artifact_id=option.artifact_id,
                    replica_id=option.replica_id,
                    replica_generation=option.replica_generation,
                    placement_epoch=option.placement_epoch,
                    purpose=LeasePurpose.EXECUTION,
                )
            )
        try:
            physical = self.lease_manager.compare_and_lease_batch(
                request_id=request_id,
                request_generation=request_generation,
                requests=physical_requests,
                predicted_remaining_s=predicted_remaining_s,
                hbm_capacity_bytes=self.predicted_planner.hbm_capacity_bytes,
            ) if physical_requests else ()
        except Exception:
            for lease in logical:
                self.lease_manager.release(lease.lease_id, reason="physical_batch_rollback")
            raise
        return V8ClosedLoopState(
            request_id=request_id,
            request_generation=request_generation,
            selected_source_by_segment=locked,
            logical_lease_ids=tuple(item.lease_id for item in logical),
            physical_lease_ids=tuple(item.lease_id for item in physical),
            predicted_plan=predicted,
        )

    def refine(
        self,
        state: V8ClosedLoopState,
        measurements: Mapping[str, RefinedSegmentMeasurement],
        *,
        actual_shared_sunk_ms: float,
        joint_actual_interference_ms: float = 0.0,
    ) -> RefinedRequestPlan:
        for segment_id, measurement in measurements.items():
            selected = state.selected_source_by_segment.get(segment_id)
            if selected != measurement.source_variant_id:
                raise RuntimeError("Refined Planner cannot reselect a Source")
        return self.refined_planner.plan(
            state.predicted_plan,
            measurements,
            actual_shared_sunk_ms=actual_shared_sunk_ms,
            joint_actual_interference_ms=joint_actual_interference_ms,
        )

    def release_request(self, state: V8ClosedLoopState, *, reason: str) -> None:
        for lease_id in (*state.physical_lease_ids, *state.logical_lease_ids):
            self.lease_manager.release(lease_id, reason=reason)
