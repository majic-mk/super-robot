from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .v7_contracts import SourceVariantIdentity
from .v7_source_pool import StoredSourceVariant
from .v8_schema10_contracts import (
    DenseKVProvenance,
    VariantMaterializationDecisionV10,
    VariantMaterializationReasonV10,
    VariantMaterializationStateV10,
)
from .v8_schema10_profile import VariantAdmissionProfileV10
from .v8_schema10_pool import Schema10SourcePool


@dataclass(frozen=True)
class VariantMaterializationRequestV10:
    reason: VariantMaterializationReasonV10
    correctness_eligible_k: int
    compared_k: int
    best_residual: Optional[float]
    absolute_threshold: Optional[float]
    dense_kv_provenance: DenseKVProvenance
    existing_variant_count: int
    dense_reference_total_ms: float
    estimated_materialization_ms: float
    estimated_replacement_ms: float = 0.0
    exploration_materializations_for_content: int = 0
    exploration_authorized: bool = False

    def __post_init__(self) -> None:
        if min(
            self.correctness_eligible_k,
            self.compared_k,
            self.existing_variant_count,
            self.exploration_materializations_for_content,
        ) < 0:
            raise ValueError("Variant materialization counts must be non-negative")
        if self.compared_k > self.correctness_eligible_k:
            raise ValueError("compared_k cannot exceed correctness_eligible_k")
        if self.existing_variant_count > 16:
            raise ValueError("schema10 content cannot exceed 16 Variants")
        if self.dense_reference_total_ms <= 0 or min(
            self.estimated_materialization_ms,
            self.estimated_replacement_ms,
        ) < 0:
            raise ValueError("Variant materialization costs are invalid")

    @property
    def selection_scope_complete(self) -> bool:
        return self.correctness_eligible_k == self.compared_k


class VariantMaterializationControllerV10:
    def __init__(self, profile: VariantAdmissionProfileV10) -> None:
        self.profile = profile

    def decide(
        self,
        request: VariantMaterializationRequestV10,
        *,
        replacement_source_variant_id: Optional[str] = None,
    ) -> VariantMaterializationDecisionV10:
        reason = VariantMaterializationReasonV10(request.reason)
        provenance = DenseKVProvenance(request.dense_kv_provenance)
        complete = request.selection_scope_complete
        budget = request.dense_reference_total_ms * self.profile.materialization_budget_fraction
        replacement_budget = (
            request.dense_reference_total_ms * self.profile.replacement_budget_fraction
        )

        def decision(
            state: VariantMaterializationStateV10,
            *,
            novelty: bool = False,
            rejection: str = "",
        ) -> VariantMaterializationDecisionV10:
            return VariantMaterializationDecisionV10(
                state=state,
                reason=reason,
                selection_scope_complete=complete,
                context_novelty_proven=novelty,
                best_residual=request.best_residual,
                absolute_threshold=request.absolute_threshold,
                dense_kv_provenance=provenance,
                existing_variant_count=request.existing_variant_count,
                materialization_budget_ms=budget,
                estimated_materialization_ms=request.estimated_materialization_ms,
                replacement_source_variant_id=replacement_source_variant_id,
                replacement_budget_ms=replacement_budget,
                estimated_replacement_ms=request.estimated_replacement_ms,
                rejection_reason=rejection,
            )

        def reject(message: str) -> VariantMaterializationDecisionV10:
            return decision(VariantMaterializationStateV10.REJECTED, rejection=message)

        if provenance is not DenseKVProvenance.DENSE_EXACT:
            return reject("canonical_source_requires_exact_dense_prefill")
        if reason in {
            VariantMaterializationReasonV10.ECONOMIC_REJECTION,
            VariantMaterializationReasonV10.RUNTIME_REJECTION,
        }:
            return reject("runtime_or_economic_failure_is_not_a_new_context")
        if request.estimated_materialization_ms > budget:
            return reject("materialization_budget_exceeded")
        if reason is VariantMaterializationReasonV10.CONTENT_MISS:
            if request.correctness_eligible_k != 0:
                return reject("content_miss_has_existing_correctness_candidate")
            if request.existing_variant_count >= self.profile.max_variants_per_content:
                return reject("content_miss_cannot_implicitly_replace_at_variant_limit")
            return decision(VariantMaterializationStateV10.ADMITTED, novelty=True)
        if reason is VariantMaterializationReasonV10.COMPLETE_SCOPE_ABSOLUTE_MISMATCH:
            if not complete:
                return reject("complete_mismatch_requires_full_candidate_coverage")
            if request.best_residual is None or request.absolute_threshold is None:
                return reject("complete_mismatch_requires_residual_evidence")
            if request.best_residual <= request.absolute_threshold:
                return reject("best_source_is_already_absolute_compatible")
            if (
                request.existing_variant_count >= self.profile.max_variants_per_content
                and replacement_source_variant_id is None
            ):
                return reject("variant_limit_has_no_safe_replacement")
            if (
                replacement_source_variant_id is not None
                and request.estimated_replacement_ms > replacement_budget
            ):
                return reject("replacement_budget_exceeded")
            if (
                replacement_source_variant_id is None
                and request.estimated_replacement_ms > 0
            ):
                return reject("replacement_cost_without_replacement")
            return decision(VariantMaterializationStateV10.ADMITTED, novelty=True)
        if reason is VariantMaterializationReasonV10.BUDGET_TRUNCATED_EXPLORATION:
            if complete:
                return reject("exploration_requires_truncated_candidate_scope")
            if not request.exploration_authorized:
                return reject("exploration_requires_explicit_authorization")
            if request.existing_variant_count >= self.profile.max_variants_per_content:
                return reject("exploration_cannot_replace_at_variant_limit")
            if (
                request.exploration_materializations_for_content
                >= self.profile.exploration_quota_per_content
            ):
                return reject("exploration_quota_exhausted")
            return decision(VariantMaterializationStateV10.ADMITTED, novelty=False)
        return reject("unsupported_materialization_reason")

    def decide_with_pool(
        self,
        request: VariantMaterializationRequestV10,
        *,
        pool: Schema10SourcePool,
        model_math_signature: str,
        reuse_content_key: str,
    ) -> VariantMaterializationDecisionV10:
        existing = pool.variants_for_content(
            model_math_signature, reuse_content_key, include_unavailable=True
        )
        if request.existing_variant_count != len(existing):
            raise RuntimeError("stale Variant materialization snapshot")
        if pool.max_variants_per_content != self.profile.max_variants_per_content:
            raise RuntimeError("Source pool and VariantAdmissionProfile limits differ")
        exploration_count = pool.exploration_materialization_count(
            model_math_signature, reuse_content_key
        )
        if request.exploration_materializations_for_content != exploration_count:
            raise RuntimeError("stale exploration quota snapshot")
        victim = None
        if (
            request.reason
            is VariantMaterializationReasonV10.COMPLETE_SCOPE_ABSOLUTE_MISMATCH
            and request.selection_scope_complete
            and len(existing) >= self.profile.max_variants_per_content
        ):
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


def materialize_exact_dense_variant_v10(
    *,
    decision: VariantMaterializationDecisionV10,
    pool: Schema10SourcePool,
    identity: SourceVariantIdentity,
    canonical_source_state_digest: str,
    summary_digest: str,
) -> StoredSourceVariant:
    if decision.state is not VariantMaterializationStateV10.ADMITTED:
        raise RuntimeError("rejected materialization cannot publish a Variant")
    if decision.dense_kv_provenance is not DenseKVProvenance.DENSE_EXACT:
        raise RuntimeError("only exact dense KV may publish a Variant")
    return pool.register_variant(
        identity,
        canonical_source_state_digest=canonical_source_state_digest,
        summary_digest=summary_digest,
        expected_replacement_source_variant_id=decision.replacement_source_variant_id,
        allow_implicit_replacement=False,
        materialization_reason=decision.reason.value,
    )
