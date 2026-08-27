from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple

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


class RuntimeSegmentPhase(str, Enum):
    PROBING = "probing"
    SELECTOR_DECISION_READY = "selector_decision_ready"
    SOURCE_LOCKED_AND_LEASED = "source_locked_and_leased"
    PREDICTED_DENSE = "predicted_dense"
    PROVISIONAL_REUSE = "provisional_reuse"
    PREFETCHING = "prefetching"
    REUSE_READY = "reuse_ready"
    REUSE_COMMIT = "reuse_commit"
    REFINED_DENSE = "refined_dense"
    ABSTAIN_DENSE = "abstain_dense"


class V8JobOutcome(str, Enum):
    COMPLETED_REUSE = "completed_reuse"
    COMPLETED_DENSE = "completed_dense"
    COMPLETED_ABSTAIN = "completed_abstain"
    FAILED = "failed"


def classify_v8_job_outcome(
    *,
    execution_mode: str,
    abstained: bool = False,
    invariant_error: bool = False,
    runtime_error: bool = False,
) -> V8JobOutcome:
    """Dense fallback is a legal result; only real execution errors fail a job."""
    if invariant_error or runtime_error:
        return V8JobOutcome.FAILED
    if abstained:
        return V8JobOutcome.COMPLETED_ABSTAIN
    if execution_mode in {"full_reuse", "partial_reuse"}:
        return V8JobOutcome.COMPLETED_REUSE
    if execution_mode in {"dense", "predicted_dense", "refined_dense"}:
        return V8JobOutcome.COMPLETED_DENSE
    raise ValueError("unknown v8 execution mode")


@dataclass
class RuntimeSegmentRecord:
    segment_id: str
    order: int
    phase: RuntimeSegmentPhase = RuntimeSegmentPhase.PROBING
    decision_depth: Optional[int] = None
    source_variant_id: Optional[str] = None
    actual_boundary: Optional[int] = None
    timeout_before_commit: bool = False


class V8IncrementalCommitController:
    """No-GPU-verifiable A/C state machine around incremental Gate 2/3.

    Physical prefetch is reversible.  REUSE_COMMIT is not.  Policy A delays a
    Segment until every causally downstream Segment has resolved selection;
    policy C may commit as soon as that Segment is ready and Gate 3 passes.
    """

    def __init__(self, ordered_segment_ids: Sequence[str], policy: str) -> None:
        if not ordered_segment_ids or len(set(ordered_segment_ids)) != len(ordered_segment_ids):
            raise ValueError("ordered Segment identities must be non-empty and unique")
        if policy not in {"causal_commit_wait", "immediate_staggered_closed_loop"}:
            raise ValueError("unsupported v8 execution policy")
        self.policy = policy
        self.records: Dict[str, RuntimeSegmentRecord] = {
            segment_id: RuntimeSegmentRecord(segment_id, order)
            for order, segment_id in enumerate(ordered_segment_ids)
        }

    def decision_ready(self, segment_id: str, source_variant_id: str, depth: int) -> None:
        record = self.records[segment_id]
        if record.phase is not RuntimeSegmentPhase.PROBING or depth < 1 or not source_variant_id:
            raise RuntimeError("invalid Selector decision transition")
        record.phase = RuntimeSegmentPhase.SELECTOR_DECISION_READY
        record.source_variant_id = source_variant_id
        record.decision_depth = depth

    def gate1_result(self, segment_id: str, passed: bool, *, at_lmax: bool = False) -> None:
        record = self.records[segment_id]
        if record.phase is not RuntimeSegmentPhase.SELECTOR_DECISION_READY:
            raise RuntimeError("Gate 1 requires a provisional selector decision")
        if passed:
            record.phase = RuntimeSegmentPhase.SOURCE_LOCKED_AND_LEASED
            return
        record.source_variant_id = None
        record.decision_depth = None
        record.phase = (
            RuntimeSegmentPhase.ABSTAIN_DENSE
            if at_lmax
            else RuntimeSegmentPhase.PROBING
        )

    def resolve_abstain(self, segment_id: str) -> None:
        record = self.records[segment_id]
        if record.phase in {RuntimeSegmentPhase.REUSE_COMMIT, RuntimeSegmentPhase.REFINED_DENSE}:
            raise RuntimeError("final Segment state cannot be overwritten")
        record.phase = RuntimeSegmentPhase.ABSTAIN_DENSE
        record.source_variant_id = None

    def gate2_result(self, segment_id: str, passed: bool) -> None:
        record = self.records[segment_id]
        if record.phase is not RuntimeSegmentPhase.SOURCE_LOCKED_AND_LEASED:
            raise RuntimeError("incremental Gate 2 requires a frozen Source")
        record.phase = (
            RuntimeSegmentPhase.PROVISIONAL_REUSE
            if passed
            else RuntimeSegmentPhase.PREDICTED_DENSE
        )

    def start_prefetch(self, segment_id: str) -> None:
        record = self.records[segment_id]
        if record.phase is not RuntimeSegmentPhase.PROVISIONAL_REUSE:
            raise RuntimeError("full-KV prefetch requires Gate 2 admission")
        record.phase = RuntimeSegmentPhase.PREFETCHING

    def mark_ready(self, segment_id: str, actual_boundary: int) -> None:
        record = self.records[segment_id]
        if record.phase is not RuntimeSegmentPhase.PREFETCHING or actual_boundary < 1:
            raise RuntimeError("Source ready requires active prefetch and a valid boundary")
        record.phase = RuntimeSegmentPhase.REUSE_READY
        record.actual_boundary = actual_boundary

    def _selection_resolved(self, record: RuntimeSegmentRecord) -> bool:
        return record.phase not in {
            RuntimeSegmentPhase.PROBING,
            RuntimeSegmentPhase.SELECTOR_DECISION_READY,
        }

    def causal_commit_ready(self, segment_id: str) -> bool:
        record = self.records[segment_id]
        if record.phase is not RuntimeSegmentPhase.REUSE_READY:
            return False
        if self.policy == "immediate_staggered_closed_loop":
            return True
        return all(
            self._selection_resolved(other)
            for other in self.records.values()
            if other.order >= record.order
        )

    def selection_boundary(self, segment_id: str) -> int:
        record = self.records[segment_id]
        if record.decision_depth is None:
            raise RuntimeError("Segment lacks a completed-depth decision")
        if self.policy == "immediate_staggered_closed_loop":
            return record.decision_depth + 1
        downstream_depths = [
            other.decision_depth
            for other in self.records.values()
            if other.order >= record.order and other.decision_depth is not None
        ]
        if not all(
            self._selection_resolved(other)
            for other in self.records.values()
            if other.order >= record.order
        ):
            raise RuntimeError("A-policy boundary requires downstream selection closure")
        return 1 + max(downstream_depths, default=record.decision_depth)

    def gate3_result(self, segment_ids: Sequence[str], passed: bool) -> None:
        if not segment_ids or len(set(segment_ids)) != len(segment_ids):
            raise ValueError("Gate 3 commit set must be non-empty and unique")
        records = [self.records[segment_id] for segment_id in segment_ids]
        if any(not self.causal_commit_ready(item.segment_id) for item in records):
            raise RuntimeError("Gate 3 attempted commit before policy readiness")
        for record in records:
            record.phase = (
                RuntimeSegmentPhase.REUSE_COMMIT
                if passed
                else RuntimeSegmentPhase.REFINED_DENSE
            )

    def source_timeout(self, segment_id: str) -> None:
        record = self.records[segment_id]
        if record.phase is RuntimeSegmentPhase.REUSE_COMMIT:
            raise RuntimeError("Source timeout cannot trigger dense fallback after commit")
        if record.phase not in {
            RuntimeSegmentPhase.PROVISIONAL_REUSE,
            RuntimeSegmentPhase.PREFETCHING,
            RuntimeSegmentPhase.REUSE_READY,
        }:
            raise RuntimeError("Source timeout occurred outside a reuse preparation phase")
        record.timeout_before_commit = True
        record.phase = RuntimeSegmentPhase.REFINED_DENSE


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
