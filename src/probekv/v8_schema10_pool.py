from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from typing import Iterable, Mapping, Optional

from .contracts import KVLocation
from .global_source_pool import ModelServingMode
from .v7_source_pool import StoredSourceVariant, V7SourcePool
from .v8_schema10_profile import VariantAdmissionProfileV10
from .v8_schema10_contracts import VariantMaterializationReasonV10


@dataclass(frozen=True)
class ContentReplacementTransaction:
    model_math_signature: str
    reuse_content_key: str
    content_generation: str
    victim_source_variant_id: Optional[str]


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
        self.mutation_lock = RLock()
        self.logical_lease_counts: dict[str, int] = {}

    def content_generation(self, model: str, content: str) -> str:
        """Local version: unrelated content changes do not invalidate a plan."""
        rows = self.variants_for_content(model, content, include_unavailable=True)
        state = [(r.source_variant_id, r.state.value, r.last_request_use_epoch,
                  self._probation_protected(r), self.logical_lease_counts.get(r.source_variant_id, 0),
                  [(p.replica_id, p.generation, p.locator.placement_epoch, p.state.value,
                    p.lease_count, p.copy_in_flight, p.execution_in_flight)
                   for p in sorted(r.replicas.values(), key=lambda p: p.replica_id)]) for r in rows]
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

    def replacement_transaction(self, model: str, content: str) -> ContentReplacementTransaction:
        with self.mutation_lock:
            victim = self.plan_variant_replacement(model, content)
            return ContentReplacementTransaction(model, content, self.content_generation(model, content),
                                                 victim.source_variant_id if victim else None)

    def register_variant(self, identity, *, replacement_transaction=None, **kwargs):
        with self.mutation_lock:
            if replacement_transaction is not None:
                plan = replacement_transaction
                if ((plan.model_math_signature, plan.reuse_content_key) !=
                        (identity.model_math_signature, identity.reuse_content_key)
                        or plan.content_generation != self.content_generation(
                            identity.model_math_signature, identity.reuse_content_key)):
                    raise RuntimeError("stale content replacement transaction")
                kwargs["expected_replacement_source_variant_id"] = plan.victim_source_variant_id
                kwargs["allow_implicit_replacement"] = False
            return super().register_variant(identity, **kwargs)

    @contextmanager
    def freeze_source(self, model: str, content: str, source_id: str, *, expected_generation: str):
        """Source freeze + logical lease is one atomic operation; no reselection."""
        with self.mutation_lock:
            if self.content_generation(model, content) != expected_generation:
                raise RuntimeError("Source freeze lost its content snapshot")
            row = self._get(model, content, source_id)
            if not row.runtime_available:
                raise RuntimeError("Source has no healthy backing")
            self.logical_lease_counts[source_id] = self.logical_lease_counts.get(source_id, 0) + 1
            row.last_request_use_epoch = self._tick()
        try:
            yield row
        finally:
            with self.mutation_lock:
                self.logical_lease_counts[source_id] -= 1

    def _evict_variant(self, variant, reason):
        if self.logical_lease_counts.get(variant.source_variant_id, 0):
            raise RuntimeError("logically leased Source cannot be evicted")
        return super()._evict_variant(variant, reason)

    def purge_namespace(self, model_math_signature):
        with self.mutation_lock:
            if any(model == model_math_signature and self.logical_lease_counts.get(row.source_variant_id, 0)
                   for (model, _, _), row in self._variants.items()):
                raise RuntimeError("cannot purge a logically leased namespace")
            return super().purge_namespace(model_math_signature)

    @contextmanager
    def lease_replica(self, model, content, source_id, replica_id):
        from .v7_contracts import ReplicaState
        with self.mutation_lock:
            replica = self._replica(model, content, source_id, replica_id)
            if replica.state not in {ReplicaState.READY, ReplicaState.LEASED}:
                raise RuntimeError("Replica is not available to a new reader")
            replica.lease_count += 1
            self._refresh_busy_state(replica)
        try:
            yield replica
        finally:
            with self.mutation_lock:
                replica.lease_count -= 1
                self._refresh_busy_state(replica)

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
        return self._replacement_victim(siblings)

    def _replacement_victim(self, variants: Iterable[StoredSourceVariant]) -> StoredSourceVariant:
        eligible = tuple(
            row
            for row in variants
            if not any(replica.busy for replica in row.replicas.values())
            and not self._probation_protected(row)
            and not self.logical_lease_counts.get(row.source_variant_id, 0)
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
