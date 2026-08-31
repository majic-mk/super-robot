from __future__ import annotations

from typing import Sequence

from .v8_leases import V8LeaseManager
from .v8_schema6_contracts import (
    Gate2AxisState,
    PlannerSnapshot,
    PreparationAxisState,
    CommitAxisState,
    SelectionAxisState,
)
from .v8_schema6_hbm import UnifiedHBMReservationManager
from .v8_schema6_runtime import Schema6RequestController
from .v8_schema7_contracts import FinalCommitDecision
from .v8_schema8_barrier import close_dense_selection_barrier
from .v8_schema8_contracts import DenseSelectionBarrierDecision
from .v8_schema8_planner import Gate1LocalPlan
from .v8_schema8_selector import Schema8SourceDecision


class Schema8BarrierRequestController(Schema6RequestController):
    """One dense d1/d2 selection barrier with no online A/C branch."""

    def __init__(
        self,
        *,
        request_id: str,
        request_generation: int,
        ordered_segment_ids: Sequence[str],
        lease_manager: V8LeaseManager,
        hbm_manager: UnifiedHBMReservationManager,
    ) -> None:
        super().__init__(
            request_id=request_id,
            request_generation=request_generation,
            ordered_segment_ids=ordered_segment_ids,
            policy="causal_commit_wait",
            lease_manager=lease_manager,
            hbm_manager=hbm_manager,
        )
        self.barrier_decision: DenseSelectionBarrierDecision | None = None
        self.detached_preparation_segment_ids: set[str] = set()

    def apply_selector_decision(
        self,
        segment_id: str,
        decision: Schema8SourceDecision,
        *,
        predicted_remaining_s: float,
    ) -> None:
        if decision.state == "continue_probe":
            return
        if decision.state == "abstained":
            self.abstain(segment_id, reason=decision.reason)
            return
        self.decision_ready(
            segment_id,
            str(decision.selected_source_variant_id),
            decision.completed_depth,
        )
        self.apply_gate1_plan(
            segment_id,
            decision.gate1_plan,
            predicted_remaining_s=predicted_remaining_s,
        )

    def apply_gate1_plan(
        self,
        segment_id: str,
        plan: Gate1LocalPlan,
        *,
        predicted_remaining_s: float,
    ) -> None:
        row = self.records[segment_id]
        if row.source_variant_id != plan.source_variant_id:
            raise RuntimeError("Gate1 plan differs from the selector decision")
        if row.decision_completed_depth != plan.selection_completed_depth:
            raise RuntimeError("Gate1 plan uses another selection depth")
        self.gate1(
            segment_id,
            passed=plan.passed,
            at_lmax=plan.selection_completed_depth == 2,
            predicted_remaining_s=predicted_remaining_s,
        )

    def close_selection_barrier(self) -> DenseSelectionBarrierDecision:
        if self.barrier_decision is not None:
            return self.barrier_decision
        terminal = {
            SelectionAxisState.SOURCE_FROZEN,
            SelectionAxisState.ABSTAINED,
        }
        if any(row.selection_state not in terminal for row in self.records.values()):
            raise RuntimeError("selection barrier cannot close with unresolved Segments")
        depths = {}
        frozen = []
        abstained = []
        for segment_id, row in self.records.items():
            depth = row.decision_completed_depth
            if row.selection_state is SelectionAxisState.ABSTAINED:
                # Gate1 clears the transient decision.  A d=2 terminal abstain
                # still has an auditable barrier resolution at d=2.
                depth = 2 if depth is None else depth
                abstained.append(segment_id)
            else:
                frozen.append(segment_id)
            if depth not in {1, 2}:
                raise RuntimeError("schema-v8 barrier depth must be d=1 or d=2")
            depths[segment_id] = depth
        self.barrier_decision = close_dense_selection_barrier(
            segment_ids=tuple(self.records),
            resolved_completed_depth_by_segment=depths,
            source_frozen_segment_ids=frozen,
            abstained_segment_ids=abstained,
        )
        return self.barrier_decision

    def apply_preparation_admission(
        self,
        segment_ids: Sequence[str],
        *,
        snapshot: PlannerSnapshot,
        current_snapshot: PlannerSnapshot,
    ) -> None:
        barrier = self.close_selection_barrier()
        allowed = set(barrier.reuse_segment_ids)
        requested = set(segment_ids)
        if requested - allowed:
            raise RuntimeError("dense/abstained Segment cannot enter preparation")
        self.apply_gate2(
            {
                segment_id: Gate2AxisState.PROVISIONAL_REUSE.value
                for segment_id in segment_ids
            },
            snapshot=snapshot,
            current_snapshot=current_snapshot,
        )

    def apply_detached_preparation_admission(
        self,
        segment_ids: Sequence[str],
        *,
        snapshot: PlannerSnapshot,
        current_snapshot: PlannerSnapshot,
    ) -> None:
        """Resource-admit d1 winners while the whole request remains dense.

        Source identity is frozen, but preparation remains speculative until
        the d1/d2 selection barrier closes. No execution-visible reuse or
        FinalCommitAdmission is possible through this method.
        """

        if self.barrier_decision is not None:
            raise RuntimeError("detached preparation is only useful before barrier close")
        for segment_id in segment_ids:
            row = self.records[segment_id]
            if (
                row.selection_state is not SelectionAxisState.SOURCE_FROZEN
                or row.decision_completed_depth != 1
            ):
                raise RuntimeError("detached preparation requires a frozen d1 winner")
        self.apply_gate2(
            {
                segment_id: Gate2AxisState.DEFERRED.value
                for segment_id in segment_ids
            },
            snapshot=snapshot,
            current_snapshot=current_snapshot,
        )
        self.detached_preparation_segment_ids.update(segment_ids)

    def begin_winner_prefetch(self, segment_id: str, **kwargs: object) -> None:
        detached = (
            self.barrier_decision is None
            and segment_id in self.detached_preparation_segment_ids
        )
        caller_speculative = bool(kwargs.pop("speculative_resource_admitted", False))
        if self.barrier_decision is None and not detached:
            raise RuntimeError(
                "schema-v8 pre-barrier prefetch requires detached resource admission"
            )
        if caller_speculative and not detached:
            raise RuntimeError("schema-v8 does not expose generic A/C speculation")
        super().begin_winner_prefetch(
            segment_id,
            speculative_resource_admitted=detached,
            **kwargs,
        )

    def gate3_eligible(self, segment_id: str) -> bool:
        row = self.records[segment_id]
        return bool(
            self.barrier_decision is not None
            and row.gate2_state is Gate2AxisState.PROVISIONAL_REUSE
            and row.preparation_state is PreparationAxisState.READY
            and row.commit_state is CommitAxisState.UNCOMMITTED
        )

    def apply_final_commit_admission(
        self,
        decision: FinalCommitDecision,
        *,
        planner_snapshot: PlannerSnapshot,
        current_snapshot: PlannerSnapshot,
    ) -> None:
        if self.barrier_decision is None:
            raise RuntimeError("FinalCommitAdmission requires a closed barrier")
        if decision.planner_snapshot != planner_snapshot:
            raise RuntimeError("FinalCommitAdmission snapshot is stale")
        from .v8_schema6_contracts import Gate3SubsetDecision

        self.apply_gate3_subset(
            Gate3SubsetDecision(
                decision.accepted_ready_segment_ids,
                decision.rejected_ready_segment_ids,
                decision.untouched_segment_ids,
                decision.request_total_ms,
                decision.dense_reference_total_ms,
                planner_snapshot,
                decision.reason_by_segment,
            ),
            current_snapshot=current_snapshot,
        )
