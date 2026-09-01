from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple

from .contracts import KVLocation
from .global_source_pool import (
    ModelNamespaceState,
    ModelServingMode,
    SourceValueStats,
)
from .v7_contracts import (
    ArtifactState,
    CanonicalKVArtifact,
    PhysicalReplica,
    ReplicaLocator,
    ReplicaState,
    SourceVariantIdentity,
    SourceVariantState,
)


@dataclass(frozen=True)
class V7PoolEvent:
    sequence: int
    action: str
    model_math_signature: str
    reuse_content_key: str = ""
    source_variant_id: str = ""
    artifact_id: str = ""
    replica_id: str = ""
    tier: Optional[KVLocation] = None
    bytes_released: int = 0
    reason: str = ""


@dataclass
class V7ModelNamespace:
    model_math_signature: str
    state: ModelNamespaceState = ModelNamespaceState.RETIRED
    soft_quota_bytes: Dict[KVLocation, int] = field(default_factory=dict)


@dataclass
class StoredSourceVariant:
    identity: SourceVariantIdentity
    canonical_source_state_digest: str
    summary_digest: str
    registered_order: int
    last_access_order: int
    state: SourceVariantState = SourceVariantState.ACTIVE
    artifact: Optional[CanonicalKVArtifact] = None
    replicas: Dict[str, PhysicalReplica] = field(default_factory=dict)
    stats: SourceValueStats = field(default_factory=SourceValueStats)

    @property
    def source_variant_id(self) -> str:
        return self.identity.source_variant_id

    @property
    def healthy_replicas(self) -> Tuple[PhysicalReplica, ...]:
        return tuple(
            replica
            for replica in self.replicas.values()
            if replica.state in {ReplicaState.READY, ReplicaState.LEASED}
        )

    @property
    def healthy_backing_replicas(self) -> Tuple[PhysicalReplica, ...]:
        return tuple(replica for replica in self.healthy_replicas if replica.is_backing)

    @property
    def runtime_available(self) -> bool:
        return bool(
            self.state is SourceVariantState.ACTIVE
            and self.artifact is not None
            and self.artifact.state is ArtifactState.HEALTHY
            and self.healthy_backing_replicas
        )

    @property
    def resident_bytes(self) -> int:
        return sum(
            replica.size_bytes
            for replica in self.replicas.values()
            if replica.state is not ReplicaState.DELETED
        )


class V7SourcePool:
    """Single-Artifact Source pool with versioned tier Replicas.

    The v6 pool remains untouched for historical protocol compatibility. This
    pool never treats a Summary as a full-KV Artifact and never permits a
    second Artifact under one Source Variant.
    """

    def __init__(
        self,
        *,
        serving_mode: ModelServingMode = ModelServingMode.SINGLE,
        max_variants_per_content: int = 16,
        tier_capacity_bytes: Optional[Mapping[KVLocation, int]] = None,
        probation_observations: int = 2,
        prior_saved_ms: float = 1.0,
    ) -> None:
        if not 1 <= max_variants_per_content <= 16:
            raise ValueError("v7 supports 1-16 Source Variants per content")
        if probation_observations < 0 or prior_saved_ms < 0:
            raise ValueError("invalid value-density parameters")
        self.serving_mode = ModelServingMode(serving_mode)
        self.max_variants_per_content = max_variants_per_content
        self.tier_capacity_bytes = {
            KVLocation(tier): int(capacity)
            for tier, capacity in (tier_capacity_bytes or {}).items()
        }
        if any(capacity < 0 for capacity in self.tier_capacity_bytes.values()):
            raise ValueError("tier capacities must be non-negative")
        self.probation_observations = probation_observations
        self.prior_saved_ms = prior_saved_ms
        self._clock = 0
        self._placement_epoch = 0
        self._replica_generation = 0
        self._namespaces: Dict[str, V7ModelNamespace] = {}
        self._variants: Dict[Tuple[str, str, str], StoredSourceVariant] = {}
        self._events: list[V7PoolEvent] = []

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _emit(self, action: str, model: str, **kwargs: object) -> None:
        self._events.append(
            V7PoolEvent(sequence=self._tick(), action=action, model_math_signature=model, **kwargs)
        )

    @property
    def snapshot_id(self) -> int:
        return self._clock

    @property
    def events(self) -> Tuple[V7PoolEvent, ...]:
        return tuple(self._events)

    def register_namespace(
        self,
        model_math_signature: str,
        soft_quota_bytes: Optional[Mapping[KVLocation, int]] = None,
    ) -> V7ModelNamespace:
        if not model_math_signature:
            raise ValueError("model-math signature is required")
        quotas = {
            KVLocation(tier): int(value)
            for tier, value in (soft_quota_bytes or {}).items()
        }
        if any(value < 0 for value in quotas.values()):
            raise ValueError("model soft quotas must be non-negative")
        namespace = self._namespaces.get(model_math_signature)
        if namespace is None or namespace.state is ModelNamespaceState.DELETED:
            namespace = V7ModelNamespace(model_math_signature, soft_quota_bytes=quotas)
            self._namespaces[model_math_signature] = namespace
        elif soft_quota_bytes is not None:
            namespace.soft_quota_bytes = quotas
        return namespace

    def activate_namespace(self, model_math_signature: str) -> None:
        namespace = self.register_namespace(model_math_signature)
        if self.serving_mode is ModelServingMode.SINGLE:
            active = [
                item
                for item in self._namespaces.values()
                if item.state is ModelNamespaceState.ACTIVE
                and item.model_math_signature != model_math_signature
            ]
            if active:
                raise RuntimeError("single-model switch requires drain and purge")
        namespace.state = ModelNamespaceState.ACTIVE
        self._emit("namespace_activated", model_math_signature)

    def register_variant(
        self,
        identity: SourceVariantIdentity,
        *,
        canonical_source_state_digest: str,
        summary_digest: str,
        expected_replacement_source_variant_id: Optional[str] = None,
    ) -> StoredSourceVariant:
        model = identity.model_math_signature
        namespace = self._namespaces.get(model)
        if namespace is None or namespace.state is not ModelNamespaceState.ACTIVE:
            raise RuntimeError("Source registration requires an active model namespace")
        if not canonical_source_state_digest or not summary_digest:
            raise ValueError("Source and Summary digests are required")
        key = (model, identity.reuse_content_key, identity.source_variant_id)
        existing = self._variants.get(key)
        if existing is not None:
            if (
                existing.canonical_source_state_digest
                != canonical_source_state_digest
                or existing.summary_digest != summary_digest
            ):
                raise ValueError("Source Variant identity collision")
            existing.last_access_order = self._tick()
            return existing
        siblings = self.variants_for_content(model, identity.reuse_content_key, include_unavailable=True)
        if len(siblings) >= self.max_variants_per_content:
            victim = self._lowest_value_variant(siblings)
            if (
                expected_replacement_source_variant_id is not None
                and victim.source_variant_id
                != expected_replacement_source_variant_id
            ):
                raise RuntimeError("stale Source Variant replacement plan")
            self._evict_variant(victim, "variant_limit")
        elif expected_replacement_source_variant_id is not None:
            raise RuntimeError("replacement was requested below the Variant limit")
        order = self._tick()
        stored = StoredSourceVariant(
            identity=identity,
            canonical_source_state_digest=canonical_source_state_digest,
            summary_digest=summary_digest,
            registered_order=order,
            last_access_order=order,
        )
        self._variants[key] = stored
        self._emit(
            "source_variant_registered",
            model,
            reuse_content_key=identity.reuse_content_key,
            source_variant_id=identity.source_variant_id,
        )
        return stored

    def plan_variant_replacement(
        self,
        model_math_signature: str,
        reuse_content_key: str,
    ) -> Optional[StoredSourceVariant]:
        """Return the safe capacity victim without mutating placement state.

        Variant replacement is deliberately separate from CPU/SSD Replica
        demotion.  The former removes one historical context from a content
        bucket; the latter only changes where the same Artifact is stored.
        """
        siblings = self.variants_for_content(
            model_math_signature,
            reuse_content_key,
            include_unavailable=True,
        )
        if len(siblings) < self.max_variants_per_content:
            return None
        return self._lowest_value_variant(siblings)

    def register_artifact(
        self,
        model_math_signature: str,
        reuse_content_key: str,
        source_variant_id: str,
        artifact: CanonicalKVArtifact,
    ) -> None:
        variant = self._get(model_math_signature, reuse_content_key, source_variant_id)
        if artifact.source_variant_id != source_variant_id:
            raise ValueError("Artifact belongs to another Source Variant")
        if artifact.parent_source_state_digest != variant.canonical_source_state_digest:
            raise ValueError("Artifact parent digest differs from canonical Source")
        if variant.artifact is not None:
            raise ValueError("v7 permits exactly one full-KV Artifact per Source")
        variant.artifact = artifact
        self._emit(
            "artifact_registered",
            model_math_signature,
            reuse_content_key=reuse_content_key,
            source_variant_id=source_variant_id,
            artifact_id=artifact.artifact_id,
        )

    def attach_replica(
        self,
        model_math_signature: str,
        reuse_content_key: str,
        source_variant_id: str,
        *,
        tier: KVLocation,
        locator_value: str,
        layout_signature: str,
        bytes_digest: str,
        size_bytes: int,
        derived_from_replica_id: Optional[str] = None,
        is_backing: Optional[bool] = None,
    ) -> PhysicalReplica:
        variant = self._get(model_math_signature, reuse_content_key, source_variant_id)
        artifact = variant.artifact
        if artifact is None or artifact.state is not ArtifactState.HEALTHY:
            raise RuntimeError("Replica requires a healthy canonical Artifact")
        tier = KVLocation(tier)
        if any(
            replica.tier is tier and replica.state is not ReplicaState.DELETED
            for replica in variant.replicas.values()
        ):
            raise ValueError("v7 permits at most one live Replica per tier")
        if derived_from_replica_id is not None:
            parent = variant.replicas.get(derived_from_replica_id)
            if parent is None or parent.state is ReplicaState.DELETED:
                raise ValueError("Replica copy source is unavailable")
            if parent.logical_digest != artifact.artifact_logical_digest:
                raise ValueError("copy source logical digest is corrupt")
        backing = (
            not any(
                replica.is_backing and replica.state is not ReplicaState.DELETED
                for replica in variant.replicas.values()
            )
            if is_backing is None
            else bool(is_backing)
        )
        if backing and any(
            replica.is_backing and replica.state is not ReplicaState.DELETED
            for replica in variant.replicas.values()
        ):
            raise ValueError("v7 permits one live backing Replica per Artifact")
        self._ensure_capacity(tier, size_bytes)
        self._placement_epoch += 1
        self._replica_generation += 1
        replica_id = hashlib.sha256(
            (
                "%s|%s|%d|%d"
                % (artifact.artifact_id, tier.value, self._replica_generation, self._placement_epoch)
            ).encode("utf-8")
        ).hexdigest()
        replica = PhysicalReplica(
            replica_id=replica_id,
            artifact_id=artifact.artifact_id,
            generation=self._replica_generation,
            tier=tier,
            logical_digest=artifact.artifact_logical_digest,
            bytes_digest=bytes_digest,
            size_bytes=size_bytes,
            locator=ReplicaLocator(
                value=locator_value,
                layout_signature=layout_signature,
                placement_epoch=self._placement_epoch,
            ),
            is_backing=backing,
            derived_from_replica_id=derived_from_replica_id,
        )
        variant.replicas[replica_id] = replica
        self._emit(
            "replica_attached",
            model_math_signature,
            reuse_content_key=reuse_content_key,
            source_variant_id=source_variant_id,
            artifact_id=artifact.artifact_id,
            replica_id=replica_id,
            tier=tier,
        )
        return replica

    def record_observation(
        self,
        model_math_signature: str,
        reuse_content_key: str,
        source_variant_id: str,
        *,
        lookup_hit: bool = False,
        compared: bool = False,
        selected: bool = False,
        admitted: bool = False,
        realized_saved_ms: float = 0.0,
    ) -> None:
        """Update the smoothed value-density counters without predicting the future."""
        if realized_saved_ms < 0:
            raise ValueError("realized saved time must be non-negative")
        if admitted and not selected:
            raise ValueError("an admitted Source must have been selected")
        if selected and not compared:
            raise ValueError("a selected Source must have been compared")
        variant = self._get(
            model_math_signature, reuse_content_key, source_variant_id
        )
        variant.stats.lookup_opportunities += 1
        variant.stats.lookup_hits += int(lookup_hit)
        variant.stats.comparisons += int(compared)
        variant.stats.selections += int(selected)
        variant.stats.admissions += int(admitted)
        if admitted:
            variant.stats.realized_saved_ms_sum += realized_saved_ms
        variant.last_access_order = self._tick()

    def relocate_replica(
        self,
        model_math_signature: str,
        reuse_content_key: str,
        source_variant_id: str,
        replica_id: str,
        *,
        locator_value: str,
        layout_signature: Optional[str] = None,
    ) -> PhysicalReplica:
        replica = self._replica(model_math_signature, reuse_content_key, source_variant_id, replica_id)
        if replica.busy:
            raise RuntimeError("busy Replica cannot be relocated")
        self._placement_epoch += 1
        replica.locator = ReplicaLocator(
            value=locator_value,
            layout_signature=layout_signature or replica.locator.layout_signature,
            placement_epoch=self._placement_epoch,
        )
        self._emit(
            "replica_relocated",
            model_math_signature,
            reuse_content_key=reuse_content_key,
            source_variant_id=source_variant_id,
            artifact_id=replica.artifact_id,
            replica_id=replica.replica_id,
            tier=replica.tier,
        )
        return replica

    @contextmanager
    def lease_replica(
        self,
        model_math_signature: str,
        reuse_content_key: str,
        source_variant_id: str,
        replica_id: str,
    ) -> Iterator[PhysicalReplica]:
        replica = self._replica(model_math_signature, reuse_content_key, source_variant_id, replica_id)
        if replica.state is not ReplicaState.READY:
            raise RuntimeError("only ready Replicas may be leased")
        replica.lease_count += 1
        self._refresh_busy_state(replica)
        try:
            yield replica
        finally:
            replica.lease_count -= 1
            self._refresh_busy_state(replica)

    @contextmanager
    def copy_replica(
        self,
        model_math_signature: str,
        reuse_content_key: str,
        source_variant_id: str,
        replica_id: str,
    ) -> Iterator[PhysicalReplica]:
        replica = self._replica(
            model_math_signature, reuse_content_key, source_variant_id, replica_id
        )
        if replica.state not in {ReplicaState.READY, ReplicaState.LEASED}:
            raise RuntimeError("Replica is not available for copy")
        replica.copy_in_flight += 1
        self._refresh_busy_state(replica)
        try:
            yield replica
        finally:
            replica.copy_in_flight -= 1
            self._refresh_busy_state(replica)

    @contextmanager
    def execute_replica(
        self,
        model_math_signature: str,
        reuse_content_key: str,
        source_variant_id: str,
        replica_id: str,
    ) -> Iterator[PhysicalReplica]:
        replica = self._replica(
            model_math_signature, reuse_content_key, source_variant_id, replica_id
        )
        if replica.state not in {ReplicaState.READY, ReplicaState.LEASED}:
            raise RuntimeError("Replica is not available for execution")
        replica.execution_in_flight += 1
        self._refresh_busy_state(replica)
        try:
            yield replica
        finally:
            replica.execution_in_flight -= 1
            self._refresh_busy_state(replica)

    def bind_replica(
        self,
        model_math_signature: str,
        reuse_content_key: str,
        source_variant_id: str,
        replica_id: str,
        *,
        artifact_generation: int,
        replica_generation: int,
        placement_epoch: int,
    ) -> PhysicalReplica:
        variant = self._get(model_math_signature, reuse_content_key, source_variant_id)
        replica = self._replica(model_math_signature, reuse_content_key, source_variant_id, replica_id)
        if variant.artifact is None or variant.artifact.generation != artifact_generation:
            raise RuntimeError("stale Artifact generation")
        if replica.generation != replica_generation:
            raise RuntimeError("stale Replica generation")
        if replica.locator.placement_epoch != placement_epoch:
            raise RuntimeError("stale Replica placement")
        if replica.logical_digest != variant.artifact.artifact_logical_digest:
            raise RuntimeError("Replica logical digest differs from Artifact")
        if replica.state is not ReplicaState.READY:
            raise RuntimeError("Replica is not bindable")
        variant.last_access_order = self._tick()
        return replica

    def variants_for_content(
        self,
        model_math_signature: str,
        reuse_content_key: str,
        *,
        include_unavailable: bool = False,
    ) -> Tuple[StoredSourceVariant, ...]:
        rows = tuple(
            variant
            for (model, content, _), variant in self._variants.items()
            if model == model_math_signature and content == reuse_content_key
        )
        if not include_unavailable:
            rows = tuple(variant for variant in rows if variant.runtime_available)
        return tuple(sorted(rows, key=lambda item: item.registered_order))

    def purge_namespace(self, model_math_signature: str) -> None:
        namespace = self._namespaces.get(model_math_signature)
        if namespace is None:
            return
        variants = [
            variant
            for (model, _, _), variant in self._variants.items()
            if model == model_math_signature
        ]
        if any(replica.busy for variant in variants for replica in variant.replicas.values()):
            raise RuntimeError("cannot purge a namespace with busy Replicas")
        namespace.state = ModelNamespaceState.DRAINING
        namespace.state = ModelNamespaceState.PURGING
        for key in [key for key in self._variants if key[0] == model_math_signature]:
            self._evict_variant(self._variants[key], "namespace_purge")
        namespace.state = ModelNamespaceState.DELETED
        self._emit("namespace_deleted", model_math_signature)

    def switch_single_model(self, model_math_signature: str) -> None:
        """Drain/purge the old namespace before activating a new single model."""
        if self.serving_mode is not ModelServingMode.SINGLE:
            raise RuntimeError("single-model switch is invalid in multi-model mode")
        active = tuple(
            namespace.model_math_signature
            for namespace in self._namespaces.values()
            if namespace.state is ModelNamespaceState.ACTIVE
            and namespace.model_math_signature != model_math_signature
        )
        for old_model in active:
            self.purge_namespace(old_model)
        self.activate_namespace(model_math_signature)

    def _get(self, model: str, content: str, source: str) -> StoredSourceVariant:
        try:
            return self._variants[(model, content, source)]
        except KeyError as error:
            raise KeyError("unknown Source Variant") from error

    @staticmethod
    def _refresh_busy_state(replica: PhysicalReplica) -> None:
        if replica.execution_in_flight:
            replica.state = ReplicaState.EXECUTING
        elif replica.copy_in_flight:
            replica.state = ReplicaState.COPYING
        elif replica.lease_count:
            replica.state = ReplicaState.LEASED
        else:
            replica.state = ReplicaState.READY

    def _replica(self, model: str, content: str, source: str, replica: str) -> PhysicalReplica:
        variant = self._get(model, content, source)
        try:
            result = variant.replicas[replica]
        except KeyError as error:
            raise KeyError("unknown Physical Replica") from error
        if result.state is ReplicaState.DELETED:
            raise RuntimeError("Physical Replica was deleted")
        return result

    def _tier_usage(self, tier: KVLocation) -> int:
        return sum(
            replica.size_bytes
            for variant in self._variants.values()
            for replica in variant.replicas.values()
            if replica.tier is tier and replica.state is not ReplicaState.DELETED
        )

    def _ensure_capacity(self, tier: KVLocation, requested: int) -> None:
        if requested < 0:
            raise ValueError("Replica bytes must be non-negative")
        capacity = self.tier_capacity_bytes.get(tier)
        if capacity is None:
            return
        while self._tier_usage(tier) + requested > capacity:
            candidates = [
                (variant, replica)
                for variant in self._variants.values()
                for replica in variant.replicas.values()
                if replica.tier is tier
                and replica.state is ReplicaState.READY
                and not replica.busy
                and (
                    not replica.is_backing
                    or variant.stats.observations >= self.probation_observations
                )
            ]
            if not candidates:
                raise MemoryError("insufficient unleased Replica capacity")
            variant, replica = min(
                candidates,
                key=lambda pair: (
                    0 if not pair[1].is_backing else 1,
                    pair[0].stats.value_density(
                        pair[0].resident_bytes, self.prior_saved_ms
                    ),
                    pair[0].last_access_order,
                ),
            )
            self._delete_replica(variant, replica, "tier_capacity")

    def _lowest_value_variant(
        self, variants: Iterable[StoredSourceVariant]
    ) -> StoredSourceVariant:
        eligible = [
            variant
            for variant in variants
            if not any(r.busy for r in variant.replicas.values())
            and variant.stats.observations >= self.probation_observations
        ]
        if not eligible:
            raise MemoryError("all Source Variants are busy or in probation")
        return min(
            eligible,
            key=lambda variant: (
                variant.stats.value_density(
                    variant.resident_bytes, self.prior_saved_ms
                ),
                variant.last_access_order,
            ),
        )

    def _delete_replica(
        self, variant: StoredSourceVariant, replica: PhysicalReplica, reason: str
    ) -> None:
        if replica.busy:
            raise RuntimeError("busy Replica cannot be evicted")
        replica.state = ReplicaState.DELETED
        self._emit(
            "replica_evicted",
            variant.identity.model_math_signature,
            reuse_content_key=variant.identity.reuse_content_key,
            source_variant_id=variant.source_variant_id,
            artifact_id=replica.artifact_id,
            replica_id=replica.replica_id,
            tier=replica.tier,
            bytes_released=replica.size_bytes,
            reason=reason,
        )
        if not variant.healthy_replicas:
            variant.state = SourceVariantState.EVICTED

    def _evict_variant(self, variant: StoredSourceVariant, reason: str) -> None:
        if any(replica.busy for replica in variant.replicas.values()):
            raise RuntimeError("busy Source Variant cannot be evicted")
        for replica in tuple(variant.replicas.values()):
            if replica.state is not ReplicaState.DELETED:
                self._delete_replica(variant, replica, reason)
        variant.state = SourceVariantState.EVICTED
        key = (
            variant.identity.model_math_signature,
            variant.identity.reuse_content_key,
            variant.source_variant_id,
        )
        self._variants.pop(key, None)
        self._emit(
            "source_variant_evicted",
            variant.identity.model_math_signature,
            reuse_content_key=variant.identity.reuse_content_key,
            source_variant_id=variant.source_variant_id,
            artifact_id=(variant.artifact.artifact_id if variant.artifact else ""),
            reason=reason,
        )
