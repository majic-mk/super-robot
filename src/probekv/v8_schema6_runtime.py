from __future__ import annotations

from typing import Dict, Mapping, Sequence

from .v8_leases import LeasePurpose, ReplicaLeaseRequest, V8LeaseManager
from .v8_schema6_contracts import (
    CommitAxisState,
    Gate2AxisState,
    Gate3SubsetDecision,
    PlannerSnapshot,
    PreparationAxisState,
    Schema6SegmentRuntimeState,
    SelectionAxisState,
)
from .v8_schema6_hbm import (
    HBMReservationKind,
    UnifiedHBMReservationManager,
)


class Schema6RequestController:
    """Orthogonal, monotonic request state for A/C incremental execution."""

    def __init__(
        self,
        *,
        request_id: str,
        request_generation: int,
        ordered_segment_ids: Sequence[str],
        policy: str,
        lease_manager: V8LeaseManager,
        hbm_manager: UnifiedHBMReservationManager,
    ) -> None:
        if not request_id or request_generation < 1:
            raise ValueError("request identity is invalid")
        if not ordered_segment_ids or len(set(ordered_segment_ids)) != len(ordered_segment_ids):
            raise ValueError("ordered Segment inventory must be non-empty and unique")
        if policy not in {"causal_commit_wait", "immediate_staggered_closed_loop"}:
            raise ValueError("unsupported schema-v6 execution policy")
        self.request_id = request_id
        self.request_generation = request_generation
        self.policy = policy
        self.lease_manager = lease_manager
        self.hbm_manager = hbm_manager
        self.records: Dict[str, Schema6SegmentRuntimeState] = {
            segment_id: Schema6SegmentRuntimeState(segment_id, order)
            for order, segment_id in enumerate(ordered_segment_ids)
        }

    def decision_ready(self, segment_id: str, source_variant_id: str, completed_depth: int) -> None:
        row = self.records[segment_id]
        if row.selection_state is not SelectionAxisState.PROBING:
            raise RuntimeError("Selector decision is not monotonic")
        if not source_variant_id or completed_depth < 1:
            raise ValueError("Selector decision is incomplete")
        row.selection_state = SelectionAxisState.DECISION_READY
        row.source_variant_id = source_variant_id
        row.decision_completed_depth = completed_depth

    def gate1(
        self,
        segment_id: str,
        *,
        passed: bool,
        at_lmax: bool,
        predicted_remaining_s: float,
    ) -> None:
        row = self.records[segment_id]
        if row.selection_state is not SelectionAxisState.DECISION_READY:
            raise RuntimeError("Gate 1 requires DECISION_READY")
        if not passed:
            row.source_variant_id = None
            row.decision_completed_depth = None
            row.selection_state = (
                SelectionAxisState.ABSTAINED if at_lmax else SelectionAxisState.PROBING
            )
            row.reason = "gate1_lmax_abstain" if at_lmax else "gate1_continue_probe"
            row.validate()
            return
        lease = self.lease_manager.freeze_and_acquire_logical(
            request_id=self.request_id,
            request_generation=self.request_generation,
            segment_id=segment_id,
            source_variant_id=str(row.source_variant_id),
            predicted_remaining_s=predicted_remaining_s,
        )
        row.logical_lease_id = lease.lease_id
        row.selection_state = SelectionAxisState.SOURCE_FROZEN
        row.reason = "gate1_source_local_pass"
        row.validate()

    def abstain(self, segment_id: str, *, reason: str) -> None:
        row = self.records[segment_id]
        if row.selection_state is SelectionAxisState.SOURCE_FROZEN:
            raise RuntimeError("a frozen Source cannot become selector abstention")
        row.selection_state = SelectionAxisState.ABSTAINED
        row.source_variant_id = None
        row.reason = reason
        row.validate()

    def apply_gate2(
        self,
        disposition_by_segment: Mapping[str, str],
        *,
        snapshot: PlannerSnapshot,
        current_snapshot: PlannerSnapshot,
    ) -> None:
        snapshot.assert_current(current_snapshot)
        for segment_id, disposition in disposition_by_segment.items():
            row = self.records[segment_id]
            if row.selection_state is not SelectionAxisState.SOURCE_FROZEN:
                raise RuntimeError("Gate 2 can only update frozen Sources")
            target = Gate2AxisState(disposition)
            if row.gate2_state is Gate2AxisState.PREDICTED_DENSE and target is not Gate2AxisState.PREDICTED_DENSE:
                raise RuntimeError("PREDICTED_DENSE cannot be promoted")
            if row.gate2_state is Gate2AxisState.PROVISIONAL_REUSE and target is Gate2AxisState.DEFERRED:
                raise RuntimeError("admitted Segment cannot return to deferred")
            if target is Gate2AxisState.PROVISIONAL_REUSE and row.gate2_state is Gate2AxisState.DEFERRED:
                self._promote_preparation_if_present(row)
            if target is Gate2AxisState.PREDICTED_DENSE:
                self.release_physical_preparation(
                    segment_id, reason="gate2_closed_predicted_dense"
                )
                self._release_logical_lease(
                    row, reason="gate2_closed_predicted_dense"
                )
            row.gate2_state = target
            row.validate()

    def _promote_preparation_if_present(self, row: Schema6SegmentRuntimeState) -> None:
        if row.physical_lease_id is None:
            return
        lease = self.lease_manager.leases[row.physical_lease_id]
        if lease.purpose is LeasePurpose.SPECULATIVE_PREPARATION:
            self.lease_manager.promote_speculative_to_execution(lease.lease_id)
            self.hbm_manager.promote(
                str(row.hbm_reservation_id),
                expected=HBMReservationKind.WINNER_PREFETCH,
                target=HBMReservationKind.COMMITTED_EXECUTION,
            )

    def begin_winner_prefetch(
        self,
        segment_id: str,
        *,
        artifact_id: str,
        replica_id: str,
        replica_generation: int,
        placement_epoch: int,
        target_hbm_bytes: int,
        predicted_remaining_s: float,
        speculative_resource_admitted: bool = False,
    ) -> None:
        row = self.records[segment_id]
        if row.selection_state is not SelectionAxisState.SOURCE_FROZEN:
            raise RuntimeError("full-KV prefetch before Source freeze is forbidden")
        if row.preparation_state is not PreparationAxisState.NONE:
            raise RuntimeError("winner prefetch already started")
        if row.gate2_state is Gate2AxisState.PROVISIONAL_REUSE:
            purpose = LeasePurpose.EXECUTION
            reservation_kind = HBMReservationKind.COMMITTED_EXECUTION
        elif row.gate2_state is Gate2AxisState.DEFERRED and speculative_resource_admitted:
            if row.speculative_prefetch_disabled:
                raise RuntimeError("speculative prefetch was disabled after realized overrun")
            purpose = LeasePurpose.SPECULATIVE_PREPARATION
            reservation_kind = HBMReservationKind.WINNER_PREFETCH
        else:
            raise RuntimeError("winner prefetch lacks Gate 2/resource admission")
        request = ReplicaLeaseRequest(
            segment_id,
            str(row.source_variant_id),
            artifact_id,
            replica_id,
            replica_generation,
            placement_epoch,
            purpose,
        )
        leases = self.lease_manager.compare_and_lease_batch(
            request_id=self.request_id,
            request_generation=self.request_generation,
            requests=(request,),
            predicted_remaining_s=predicted_remaining_s,
        )
        lease = leases[0]
        try:
            reservation = self.hbm_manager.reserve_batch(
                owner_request_id=self.request_id,
                rows=((segment_id, target_hbm_bytes, reservation_kind),),
            )[0]
        except Exception:
            self.lease_manager.release(lease.lease_id, reason="hbm_reservation_rollback")
            raise
        row.physical_lease_id = lease.lease_id
        row.hbm_reservation_id = reservation.reservation_id
        row.preparation_state = PreparationAxisState.PREFETCHING
        row.validate()

    def mark_winner_ready(self, segment_id: str, *, actual_reuse_boundary: int) -> None:
        row = self.records[segment_id]
        if row.preparation_state is not PreparationAxisState.PREFETCHING:
            raise RuntimeError("winner ready requires an active prefetch")
        if actual_reuse_boundary < 1:
            raise ValueError("actual reuse boundary is 1-based")
        lease = self.lease_manager.leases[str(row.physical_lease_id)]
        if lease.copy_active:
            self.lease_manager.mark_physical_ready(lease.lease_id)
        row.preparation_state = PreparationAxisState.READY
        row.actual_reuse_boundary = actual_reuse_boundary
        row.validate()

    def record_speculative_realized_overrun(self, segment_id: str, overrun_ms: float) -> None:
        if overrun_ms < 0:
            raise ValueError("realized overrun must be non-negative")
        row = self.records[segment_id]
        row.speculative_realized_overrun_ms += overrun_ms
        if overrun_ms > 0:
            row.speculative_prefetch_disabled = True

    def selection_closed_for(self, segment_id: str) -> bool:
        row = self.records[segment_id]
        return all(
            other.selection_state in {SelectionAxisState.SOURCE_FROZEN, SelectionAxisState.ABSTAINED}
            for other in self.records.values()
            if other.order >= row.order
        )

    def gate3_eligible(self, segment_id: str) -> bool:
        row = self.records[segment_id]
        if (
            row.gate2_state is not Gate2AxisState.PROVISIONAL_REUSE
            or row.preparation_state is not PreparationAxisState.READY
            or row.commit_state is not CommitAxisState.UNCOMMITTED
        ):
            return False
        return self.policy == "immediate_staggered_closed_loop" or self.selection_closed_for(segment_id)

    def apply_gate3_subset(
        self,
        decision: Gate3SubsetDecision,
        *,
        current_snapshot: PlannerSnapshot,
    ) -> None:
        decision.planner_snapshot.assert_current(current_snapshot)
        touched = set(decision.accepted_ready_segment_ids) | set(decision.rejected_ready_segment_ids)
        if any(not self.gate3_eligible(segment_id) for segment_id in touched):
            raise RuntimeError("Gate 3 subset contains a policy-ineligible Segment")
        for segment_id in decision.accepted_ready_segment_ids:
            self.records[segment_id].commit_state = CommitAxisState.REUSE_COMMIT
            self.records[segment_id].reason = decision.reason_by_segment.get(segment_id, "")
            self.records[segment_id].validate()
        for segment_id in decision.rejected_ready_segment_ids:
            row = self.records[segment_id]
            row.commit_state = CommitAxisState.REFINED_DENSE
            row.reason = decision.reason_by_segment.get(segment_id, "")
            self.release_physical_preparation(segment_id, reason="gate3_refined_dense")
            self._release_logical_lease(row, reason="gate3_refined_dense")
            row.validate()

    def _release_logical_lease(
        self, row: Schema6SegmentRuntimeState, *, reason: str
    ) -> None:
        if not row.logical_lease_id:
            return
        self.lease_manager.release(row.logical_lease_id, reason=reason)

    def complete_reuse_execution(self, segment_id: str) -> None:
        """Release execution ownership only after the committed reuse finishes.

        IDs and READY state remain as immutable audit evidence; the underlying
        lease and reservation records are released and cannot protect/charge
        resources after request completion.
        """

        row = self.records[segment_id]
        if row.commit_state is not CommitAxisState.REUSE_COMMIT:
            raise RuntimeError("only a committed reuse can complete execution")
        if row.physical_lease_id:
            self.lease_manager.release(
                row.physical_lease_id, reason="reuse_execution_complete"
            )
        if row.hbm_reservation_id:
            self.hbm_manager.release(row.hbm_reservation_id)
        self._release_logical_lease(row, reason="reuse_execution_complete")

    def release_physical_preparation(self, segment_id: str, *, reason: str) -> None:
        row = self.records[segment_id]
        if row.physical_lease_id:
            self.lease_manager.release(row.physical_lease_id, reason=reason)
            row.physical_lease_id = None
        if row.hbm_reservation_id:
            self.hbm_manager.release(row.hbm_reservation_id)
            row.hbm_reservation_id = None
        row.preparation_state = PreparationAxisState.NONE
