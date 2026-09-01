from __future__ import annotations

from typing import Dict, Optional, Tuple

from .v7_contracts import SourceVariantIdentity
from .v7_source_pool import StoredSourceVariant, V7SourcePool
from .v8_schema6_contracts import CommitAxisState
from .v8_schema8_runtime import Schema8BarrierRequestController
from .v8_schema9_contracts import (
    VariantMaterializationDecision,
    VariantMaterializationReason,
    VariantMaterializationState,
)
from .v8_schema9_materialization import (
    VariantMaterializationController,
    VariantMaterializationRequest,
    materialize_exact_dense_variant,
)
from .v8_schema9_selector import Schema9SourceDecision


class Schema9AbsoluteAdmissionRequestController(Schema8BarrierRequestController):
    """Connect schema9 selection, dense fallback, and canonical Variant growth."""

    def __init__(self, *args: object, materialization_controller: VariantMaterializationController, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.materialization_controller = materialization_controller
        self._mismatch_evidence_by_segment: Dict[
            str, Tuple[bool, Optional[float], float]
        ] = {}

    def apply_schema9_selector_decision(
        self,
        segment_id: str,
        decision: Schema9SourceDecision,
        *,
        predicted_remaining_s: float,
    ) -> None:
        if decision.state == "continue_probe":
            return
        if decision.state == "abstained":
            if decision.materialization_candidate:
                self._mismatch_evidence_by_segment[segment_id] = (
                    decision.selection_scope_complete,
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
        self.apply_gate1_plan(
            segment_id,
            decision.gate1_plan,
            predicted_remaining_s=predicted_remaining_s,
        )

    def materialize_after_exact_dense(
        self,
        segment_id: str,
        *,
        request: VariantMaterializationRequest,
        pool: V7SourcePool,
        identity: SourceVariantIdentity,
        canonical_source_state_digest: str,
        summary_digest: str,
    ) -> Tuple[VariantMaterializationDecision, Optional[StoredSourceVariant]]:
        row = self.records[segment_id]
        if row.commit_state is CommitAxisState.REUSE_COMMIT:
            raise RuntimeError("committed reuse cannot be materialized as dense")
        if request.reason is VariantMaterializationReason.ABSOLUTE_RESIDUAL_MISMATCH:
            evidence = self._mismatch_evidence_by_segment.get(segment_id)
            if evidence is None:
                raise RuntimeError("missing complete-scope mismatch evidence")
            expected_scope, expected_residual, expected_threshold = evidence
            if (
                request.selection_scope_complete != expected_scope
                or request.best_residual != expected_residual
                or request.absolute_threshold != expected_threshold
            ):
                raise RuntimeError("materialization evidence differs from selector")
        decision = self.materialization_controller.decide_with_pool(
            request,
            pool=pool,
            model_math_signature=identity.model_math_signature,
            reuse_content_key=identity.reuse_content_key,
        )
        if decision.state is VariantMaterializationState.REJECTED:
            return decision, None
        stored = materialize_exact_dense_variant(
            decision=decision,
            pool=pool,
            identity=identity,
            canonical_source_state_digest=canonical_source_state_digest,
            summary_digest=summary_digest,
        )
        return decision, stored
