from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from .v7_contracts import SourceVariantIdentity
from .v7_source_pool import StoredSourceVariant
from .v8_schema6_contracts import CommitAxisState
from .v8_schema6_contracts import PlannerSnapshot
from .v8_schema8_runtime import Schema8BarrierRequestController
from .v8_schema10_contracts import (
    Gate1Mode,
    VariantMaterializationDecisionV10,
    VariantMaterializationReasonV10,
    VariantMaterializationStateV10,
)
from .v8_schema10_materialization import (
    VariantMaterializationControllerV10,
    VariantMaterializationRequestV10,
    materialize_exact_dense_variant_v10,
)
from .v8_schema10_pool import Schema10SourcePool
from .v8_schema10_selector import Schema10SourceDecision


class Schema10VariantGrowthRequestController(Schema8BarrierRequestController):
    def __init__(
        self,
        *args: object,
        materialization_controller: VariantMaterializationControllerV10,
        gate1_mode: Gate1Mode,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.materialization_controller = materialization_controller
        self.gate1_mode = Gate1Mode(gate1_mode)
        self.gate1_advisory_failure_by_segment: Dict[str, bool] = {}
        self._materialization_evidence_by_segment: Dict[
            str, Tuple[VariantMaterializationReasonV10, int, int, Optional[float], float]
        ] = {}

    def apply_schema10_selector_decision(
        self,
        segment_id: str,
        decision: Schema10SourceDecision,
        *,
        correctness_eligible_k: int,
        compared_k: int,
        predicted_remaining_s: float,
    ) -> None:
        if decision.state == "continue_probe":
            return
        if decision.state == "abstained":
            if decision.materialization_reason is not None:
                self._materialization_evidence_by_segment[segment_id] = (
                    decision.materialization_reason,
                    correctness_eligible_k,
                    compared_k,
                    decision.best_residual,
                    decision.absolute_threshold,
                )
            self.abstain(segment_id, reason=decision.reason)
            return
        self.decision_ready(
            segment_id,
            str(decision.selected_source_variant_id),
            decision.completed_depth,
        )
        assert decision.gate1_plan is not None
        self.gate1_advisory_failure_by_segment[segment_id] = (
            decision.gate1_was_advisory_failure
        )
        self.gate1(
            segment_id,
            passed=(
                decision.gate1_plan.passed
                or self.gate1_mode is Gate1Mode.FUSED_ADVISORY
            ),
            at_lmax=decision.completed_depth == 2,
            predicted_remaining_s=predicted_remaining_s,
        )
        if decision.gate1_was_advisory_failure:
            self.records[segment_id].reason = "gate1_fused_advisory_source_frozen"

    def apply_atomic_preparation_reservation(
        self,
        segment_ids: Sequence[str],
        *,
        snapshot: PlannerSnapshot,
        current_snapshot: PlannerSnapshot,
    ) -> None:
        """Public schema10 name for the mandatory all-or-nothing preparation step."""
        self.apply_preparation_admission(
            segment_ids,
            snapshot=snapshot,
            current_snapshot=current_snapshot,
        )

    def materialize_after_exact_dense(
        self,
        segment_id: str,
        *,
        request: VariantMaterializationRequestV10,
        pool: Schema10SourcePool,
        identity: SourceVariantIdentity,
        canonical_source_state_digest: str,
        summary_digest: str,
    ) -> Tuple[VariantMaterializationDecisionV10, Optional[StoredSourceVariant]]:
        row = self.records[segment_id]
        if row.commit_state is CommitAxisState.REUSE_COMMIT:
            raise RuntimeError("committed reuse cannot be materialized as dense")
        evidence = self._materialization_evidence_by_segment.get(segment_id)
        if evidence is None:
            raise RuntimeError("missing selector materialization evidence")
        expected = (
            request.reason,
            request.correctness_eligible_k,
            request.compared_k,
            request.best_residual,
            request.absolute_threshold,
        )
        if expected != evidence:
            raise RuntimeError("materialization evidence differs from selector")
        decision = self.materialization_controller.decide_with_pool(
            request,
            pool=pool,
            model_math_signature=identity.model_math_signature,
            reuse_content_key=identity.reuse_content_key,
        )
        if decision.state is VariantMaterializationStateV10.REJECTED:
            return decision, None
        stored = materialize_exact_dense_variant_v10(
            decision=decision,
            pool=pool,
            identity=identity,
            canonical_source_state_digest=canonical_source_state_digest,
            summary_digest=summary_digest,
        )
        return decision, stored
