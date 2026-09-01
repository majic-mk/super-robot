from __future__ import annotations

from typing import Mapping, Optional

from .contracts import KVLocation
from .global_source_pool import ModelServingMode
from .v7_source_pool import StoredSourceVariant, V7SourcePool
from .v8_schema10_profile import VariantAdmissionProfileV10
from .v8_schema10_contracts import VariantMaterializationReasonV10


class Schema10SourcePool(V7SourcePool):
    """V7 physical pool with schema10's explicit bounded maturity policy."""

    def __init__(
        self,
        *,
        profile: VariantAdmissionProfileV10,
        serving_mode: ModelServingMode = ModelServingMode.SINGLE,
        tier_capacity_bytes: Optional[Mapping[KVLocation, int]] = None,
        prior_saved_ms: float = 1.0,
    ) -> None:
        super().__init__(
            serving_mode=serving_mode,
            max_variants_per_content=profile.max_variants_per_content,
            tier_capacity_bytes=tier_capacity_bytes,
            probation_observations=profile.probation_comparison_observations,
            prior_saved_ms=prior_saved_ms,
            bounded_probation=True,
            max_protected_probation_per_content=(
                profile.max_protected_probation_per_content
            ),
            probation_lookup_opportunities=profile.probation_lookup_opportunities,
        )

    def finish_content_lookup(
        self,
        model_math_signature: str,
        reuse_content_key: str,
    ) -> None:
        """Close one lookup opportunity after its comparisons were recorded."""
        self.record_content_lookup_opportunity(
            model_math_signature, reuse_content_key
        )

    def exploration_materialization_count(
        self,
        model_math_signature: str,
        reuse_content_key: str,
    ) -> int:
        return sum(
            row.materialization_reason
            == VariantMaterializationReasonV10.BUDGET_TRUNCATED_EXPLORATION.value
            for row in self.variants_for_content(
                model_math_signature,
                reuse_content_key,
                include_unavailable=True,
            )
        )

    def plan_variant_replacement(
        self,
        model_math_signature: str,
        reuse_content_key: str,
    ) -> Optional[StoredSourceVariant]:
        """Select the least recently request-used Variant within one Segment.

        Candidate comparison alone does not refresh this epoch. Probation,
        lease, copy and execution protection remain stronger than LRU.
        Cross-content capacity policy remains independent.
        """
        siblings = self.variants_for_content(
            model_math_signature,
            reuse_content_key,
            include_unavailable=True,
        )
        if len(siblings) < self.max_variants_per_content:
            return None
        eligible = tuple(
            row
            for row in siblings
            if not any(replica.busy for replica in row.replicas.values())
            and not self._probation_protected(row)
        )
        if not eligible:
            raise MemoryError("all per-Segment Source Variants are protected")
        return min(
            eligible,
            key=lambda row: (
                row.last_request_use_epoch,
                row.registered_order,
                row.source_variant_id,
            ),
        )
