from __future__ import annotations

from typing import Mapping

from .v8_schema6_contracts import Gate3SubsetDecision, PlannerSnapshot
from .v8_schema6_runtime import Schema6RequestController
from .v8_schema7_contracts import FinalCommitDecision


class Schema7RequestController(Schema6RequestController):
    """Schema-v7 naming and invariants over the proven schema-v6 state core.

    The old Gate-3 name is deliberately confined to the compatibility adapter;
    schema-v7 callers see only PreparationAdmission and FinalCommitAdmission.
    """

    def apply_preparation_admission(
        self,
        disposition_by_segment: Mapping[str, str],
        *,
        snapshot: PlannerSnapshot,
        current_snapshot: PlannerSnapshot,
    ) -> None:
        self.apply_gate2(
            disposition_by_segment,
            snapshot=snapshot,
            current_snapshot=current_snapshot,
        )

    def final_commit_eligible(self, segment_id: str) -> bool:
        return self.gate3_eligible(segment_id)

    def apply_final_commit_admission(
        self,
        decision: FinalCommitDecision,
        *,
        planner_snapshot: PlannerSnapshot,
        current_snapshot: PlannerSnapshot,
    ) -> None:
        if decision.planner_snapshot != planner_snapshot:
            raise RuntimeError("FinalCommitAdmission snapshot is stale")
        compatibility = Gate3SubsetDecision(
            accepted_ready_segment_ids=decision.accepted_ready_segment_ids,
            rejected_ready_segment_ids=decision.rejected_ready_segment_ids,
            untouched_segment_ids=decision.untouched_segment_ids,
            request_total_ms=decision.request_total_ms,
            dense_reference_total_ms=decision.dense_reference_total_ms,
            planner_snapshot=planner_snapshot,
            reason_by_segment=decision.reason_by_segment,
        )
        self.apply_gate3_subset(compatibility, current_snapshot=current_snapshot)
