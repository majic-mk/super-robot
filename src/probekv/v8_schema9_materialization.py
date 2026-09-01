from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .v7_contracts import SourceVariantIdentity
from .v7_source_pool import StoredSourceVariant, V7SourcePool
from .v8_schema9_contracts import (
    DenseKVProvenance,
    VariantMaterializationDecision,
    VariantMaterializationReason,
    VariantMaterializationState,
)
from .v8_schema9_profile import VariantAdmissionProfile


@dataclass(frozen=True)
class VariantMaterializationRequest:
    reason: VariantMaterializationReason
    selection_scope_complete: bool
    best_residual: Optional[float]
    absolute_threshold: Optional[float]
    dense_kv_provenance: DenseKVProvenance
    existing_variant_count: int
    dense_reference_total_ms: float
    estimated_materialization_ms: float
    explicit_exploration_authorized: bool = False


class VariantMaterializationController:
    """Admit exact-dense historical contexts without confusing runtime failures."""

    def __init__(self, profile: VariantAdmissionProfile) -> None:
        self.profile = profile

    def decide(
        self,
        request: VariantMaterializationRequest,
        *,
        replacement_source_variant_id: Optional[str] = None,
    ) -> VariantMaterializationDecision:
        reason = VariantMaterializationReason(request.reason)
        provenance = DenseKVProvenance(request.dense_kv_provenance)
        budget = (
            request.dense_reference_total_ms
            * self.profile.materialization_budget_fraction
        )

        def reject(message: str) -> VariantMaterializationDecision:
            return VariantMaterializationDecision(
                VariantMaterializationState.REJECTED,
                reason,
                request.selection_scope_complete,
                request.best_residual,
                request.absolute_threshold,
                provenance,
                request.existing_variant_count,
                budget,
                request.estimated_materialization_ms,
                replacement_source_variant_id,
                message,
            )

        if provenance is not DenseKVProvenance.DENSE_EXACT:
            return reject("canonical_source_requires_exact_dense_prefill")
        if reason in {
            VariantMaterializationReason.ECONOMIC_REJECTION,
            VariantMaterializationReason.RUNTIME_REJECTION,
            VariantMaterializationReason.BUDGET_TRUNCATED,
        }:
            return reject("runtime_or_budget_failure_is_not_a_new_context")
        if reason is VariantMaterializationReason.ABSOLUTE_RESIDUAL_MISMATCH:
            if self.profile.require_full_candidate_coverage_for_mismatch and not (
                request.selection_scope_complete
            ):
                return reject("absolute_mismatch_requires_full_candidate_coverage")
            if request.best_residual is None or request.absolute_threshold is None:
                return reject("absolute_mismatch_requires_residual_evidence")
            if request.best_residual <= request.absolute_threshold:
                return reject("best_source_is_already_absolute_compatible")
        if reason is VariantMaterializationReason.EXPLICIT_EXPLORATION and not (
            request.explicit_exploration_authorized
        ):
            return reject("exploration_requires_explicit_authorization")
        if request.estimated_materialization_ms > budget:
            return reject("materialization_budget_exceeded")
        if request.existing_variant_count >= self.profile.max_variants_per_content and (
            not replacement_source_variant_id
        ):
            return reject("variant_limit_has_no_safe_replacement")
        return VariantMaterializationDecision(
            VariantMaterializationState.ADMITTED,
            reason,
            request.selection_scope_complete,
            request.best_residual,
            request.absolute_threshold,
            provenance,
            request.existing_variant_count,
            budget,
            request.estimated_materialization_ms,
            replacement_source_variant_id,
            "",
        )

    def decide_with_pool(
        self,
        request: VariantMaterializationRequest,
        *,
        pool: V7SourcePool,
        model_math_signature: str,
        reuse_content_key: str,
    ) -> VariantMaterializationDecision:
        existing = pool.variants_for_content(
            model_math_signature,
            reuse_content_key,
            include_unavailable=True,
        )
        if request.existing_variant_count != len(existing):
            raise RuntimeError("stale Variant materialization snapshot")
        if pool.max_variants_per_content != self.profile.max_variants_per_content:
            raise RuntimeError("Source pool and VariantAdmissionProfile limits differ")
        victim = None
        if len(existing) >= self.profile.max_variants_per_content:
            try:
                victim = pool.plan_variant_replacement(
                    model_math_signature, reuse_content_key
                )
            except MemoryError:
                victim = None
        return self.decide(
            request,
            replacement_source_variant_id=(
                victim.source_variant_id if victim is not None else None
            ),
        )


def materialize_exact_dense_variant(
    *,
    decision: VariantMaterializationDecision,
    pool: V7SourcePool,
    identity: SourceVariantIdentity,
    canonical_source_state_digest: str,
    summary_digest: str,
) -> StoredSourceVariant:
    """Publish an admitted exact-dense Variant through the existing pool.

    Artifact and backing Replica publication remain separate verified steps;
    this function only creates the logical historical Source object.
    """
    if decision.state is not VariantMaterializationState.ADMITTED:
        raise RuntimeError("rejected materialization cannot publish a Variant")
    if decision.dense_kv_provenance is not DenseKVProvenance.DENSE_EXACT:
        raise RuntimeError("only exact dense KV may publish a Variant")
    return pool.register_variant(
        identity,
        canonical_source_state_digest=canonical_source_state_digest,
        summary_digest=summary_digest,
        expected_replacement_source_variant_id=(
            decision.replacement_source_variant_id
        ),
    )
