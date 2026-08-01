from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple

from .contracts import HistoricalSource, KVLocation
from .model_signature import validate_v6_model_signature


class ModelServingMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


class ModelNamespaceState(str, Enum):
    ACTIVE = "active"
    DRAINING = "draining"
    RETIRED = "retired"
    PURGING = "purging"
    DELETED = "deleted"


class GlobalEvictionPolicy(str, Enum):
    VALUE_DENSITY_V1 = "value_density_v1"
    CACHE_CRAFT_FR = "cache_craft_fr"


@dataclass(frozen=True)
class PoolEvent:
    action: str
    model_signature: str
    content_hash: str = ""
    source_id: str = ""
    location: Optional[KVLocation] = None
    bytes_released: int = 0
    reason: str = ""


@dataclass
class SourceValueStats:
    lookup_hits: int = 0
    lookup_opportunities: int = 0
    comparisons: int = 0
    selections: int = 0
    admissions: int = 0
    realized_saved_ms_sum: float = 0.0
    cache_craft_fr: float = 0.0

    @property
    def observations(self) -> int:
        return self.comparisons

    def value_density(self, resident_bytes: int, prior_saved_ms: float) -> float:
        p_hit = (self.lookup_hits + 1.0) / (self.lookup_opportunities + 2.0)
        p_select = (self.selections + 1.0) / (self.comparisons + 2.0)
        p_admit = (self.admissions + 1.0) / (self.selections + 2.0)
        mean_saved = (
            self.realized_saved_ms_sum + prior_saved_ms
        ) / (self.admissions + 1.0)
        return (
            p_hit * p_select * p_admit * max(0.0, mean_saved)
        ) / float(max(1, resident_bytes))


@dataclass
class PooledSource:
    source: HistoricalSource
    canonical_bytes: int
    registered_order: int
    last_access_order: int
    replicas: Dict[KVLocation, int] = field(default_factory=dict)
    stats: SourceValueStats = field(default_factory=SourceValueStats)
    lease_count: int = 0
    copy_in_flight: int = 0
    execution_in_flight: int = 0

    @property
    def busy(self) -> bool:
        return bool(
            self.lease_count
            or self.copy_in_flight
            or self.execution_in_flight
        )

    @property
    def resident_bytes(self) -> int:
        return self.canonical_bytes + sum(self.replicas.values())


@dataclass
class ModelNamespace:
    model_signature: str
    state: ModelNamespaceState = ModelNamespaceState.RETIRED
    soft_quota_bytes: Dict[KVLocation, int] = field(default_factory=dict)


class GlobalSourcePool:
    """Model-scoped canonical variants under global byte capacities.

    The v6 pool separates logical variants from physical replicas. It protects
    active leases/copies/executions, keeps one variant while a content remains
    retained, and may evict the whole content when the global canonical pool is
    under pressure.
    """

    def __init__(
        self,
        *,
        serving_mode: ModelServingMode = ModelServingMode.SINGLE,
        max_variants_per_content: int = 16,
        min_variants_per_retained_content: int = 1,
        canonical_capacity_bytes: Optional[int] = None,
        tier_capacity_bytes: Optional[Mapping[KVLocation, int]] = None,
        eviction_policy: GlobalEvictionPolicy = (
            GlobalEvictionPolicy.VALUE_DENSITY_V1
        ),
        probation_observations: int = 2,
        prior_saved_ms: float = 1.0,
    ) -> None:
        if not 1 <= min_variants_per_retained_content <= max_variants_per_content:
            raise ValueError("invalid per-content variant bounds")
        if max_variants_per_content > 16:
            raise ValueError("v6 supports at most 16 variants per content")
        if canonical_capacity_bytes is not None and canonical_capacity_bytes < 0:
            raise ValueError("canonical capacity must be non-negative")
        if probation_observations < 0 or prior_saved_ms < 0:
            raise ValueError("invalid value-policy parameters")
        self.serving_mode = ModelServingMode(serving_mode)
        self.max_variants_per_content = max_variants_per_content
        self.min_variants_per_retained_content = (
            min_variants_per_retained_content
        )
        self.canonical_capacity_bytes = canonical_capacity_bytes
        self.tier_capacity_bytes = {
            KVLocation(location): int(size)
            for location, size in (tier_capacity_bytes or {}).items()
        }
        if any(size < 0 for size in self.tier_capacity_bytes.values()):
            raise ValueError("tier capacity must be non-negative")
        self.eviction_policy = GlobalEvictionPolicy(eviction_policy)
        self.probation_observations = probation_observations
        self.prior_saved_ms = prior_saved_ms
        self._clock = 0
        self._namespaces: Dict[str, ModelNamespace] = {}
        self._sources: Dict[Tuple[str, str, str], PooledSource] = {}
        self._events = []

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    @staticmethod
    def _key(
        model_signature: str, content_hash: str, source_id: str
    ) -> Tuple[str, str, str]:
        return model_signature, content_hash, source_id

    def register_namespace(
        self,
        model_signature: str,
        soft_quota_bytes: Optional[Mapping[KVLocation, int]] = None,
    ) -> ModelNamespace:
        if not model_signature:
            raise ValueError("model_signature must be non-empty")
        validate_v6_model_signature(model_signature)
        namespace = self._namespaces.get(model_signature)
        quotas = {
            KVLocation(location): int(size)
            for location, size in (soft_quota_bytes or {}).items()
        }
        if any(size < 0 for size in quotas.values()):
            raise ValueError("model soft quotas must be non-negative")
        if namespace is None or namespace.state is ModelNamespaceState.DELETED:
            namespace = ModelNamespace(model_signature, soft_quota_bytes=quotas)
            self._namespaces[model_signature] = namespace
        elif soft_quota_bytes is not None:
            namespace.soft_quota_bytes = quotas
        return namespace

    def activate_model(self, model_signature: str) -> None:
        namespace = self.register_namespace(model_signature)
        if self.serving_mode is ModelServingMode.SINGLE:
            active = [
                item.model_signature
                for item in self._namespaces.values()
                if item.state is ModelNamespaceState.ACTIVE
                and item.model_signature != model_signature
            ]
            if active:
                raise RuntimeError(
                    "single-model mode already has active model %s" % active[0]
                )
        if namespace.state in {
            ModelNamespaceState.DRAINING,
            ModelNamespaceState.PURGING,
        }:
            raise RuntimeError("cannot activate a draining or purging model")
        namespace.state = ModelNamespaceState.ACTIVE
        self._events.append(PoolEvent("model_active", model_signature))

    def namespace_state(self, model_signature: str) -> ModelNamespaceState:
        return self._namespaces[model_signature].state

    def _model_busy(self, model_signature: str) -> bool:
        return any(
            lifecycle.busy
            for key, lifecycle in self._sources.items()
            if key[0] == model_signature
        )

    def begin_model_unload(self, model_signature: str) -> bool:
        namespace = self._namespaces[model_signature]
        if namespace.state is ModelNamespaceState.DELETED:
            return True
        if namespace.state is ModelNamespaceState.ACTIVE:
            namespace.state = ModelNamespaceState.DRAINING
            self._events.append(PoolEvent("model_draining", model_signature))
        if namespace.state is ModelNamespaceState.DRAINING and not self._model_busy(
            model_signature
        ):
            namespace.state = ModelNamespaceState.RETIRED
            self._events.append(PoolEvent("model_retired", model_signature))
        return namespace.state is ModelNamespaceState.RETIRED

    def purge_model(self, model_signature: str) -> None:
        namespace = self._namespaces[model_signature]
        self.begin_model_unload(model_signature)
        if namespace.state is not ModelNamespaceState.RETIRED:
            raise RuntimeError("model cannot purge before leases and work drain")
        namespace.state = ModelNamespaceState.PURGING
        keys = [key for key in self._sources if key[0] == model_signature]
        for key in keys:
            self._evict_source(key, "model_purge", allow_last=True)
        namespace.state = ModelNamespaceState.DELETED
        self._events.append(PoolEvent("model_deleted", model_signature))

    def switch_single_model(self, model_signature: str) -> bool:
        if self.serving_mode is not ModelServingMode.SINGLE:
            raise RuntimeError("switch_single_model requires single-model mode")
        previous_models = [
            item.model_signature
            for item in self._namespaces.values()
            if item.state is not ModelNamespaceState.DELETED
            and item.model_signature != model_signature
        ]
        for previous in previous_models:
            if not self.begin_model_unload(previous):
                return False
            self.purge_model(previous)
        self.activate_model(model_signature)
        return True

    def _namespace_for_write(self, model_signature: str) -> ModelNamespace:
        namespace = self._namespaces.get(model_signature)
        if namespace is None or namespace.state is not ModelNamespaceState.ACTIVE:
            raise RuntimeError("Source writes require an active model namespace")
        return namespace

    def _bucket_keys(
        self, model_signature: str, content_hash: str
    ) -> Tuple[Tuple[str, str, str], ...]:
        return tuple(
            key for key in self._sources
            if key[0] == model_signature and key[1] == content_hash
        )

    def _canonical_used(self) -> int:
        return sum(item.canonical_bytes for item in self._sources.values())

    def _tier_used(self, location: KVLocation) -> int:
        return sum(
            (
                item.canonical_bytes
                if item.source.kv_location is location
                else 0
            )
            + item.replicas.get(location, 0)
            for item in self._sources.values()
        )

    def _model_tier_used(
        self, model_signature: str, location: KVLocation
    ) -> int:
        return sum(
            (
                item.canonical_bytes
                if item.source.kv_location is location
                else 0
            )
            + item.replicas.get(location, 0)
            for key, item in self._sources.items()
            if key[0] == model_signature
        )

    def _model_over_soft_quota(
        self, model_signature: str, location: KVLocation
    ) -> bool:
        quota = self._namespaces[model_signature].soft_quota_bytes.get(location)
        return quota is not None and (
            self._model_tier_used(model_signature, location) > quota
        )

    def _make_tier_room_for_canonical(
        self, location: KVLocation, required_bytes: int
    ) -> None:
        """Enforce a physical tier cap before adding canonical bytes.

        Optional replicas are discarded first. Canonical eviction then follows
        redundant-variant-before-whole-content order and never touches busy or
        probationary Sources.
        """

        capacity = self.tier_capacity_bytes.get(location)
        if capacity is None:
            return
        while self._tier_used(location) + required_bytes > capacity:
            replicas = [
                (key, item)
                for key, item in self._sources.items()
                if location in item.replicas and not item.busy
            ]
            if replicas:
                key, item = min(
                    replicas,
                    key=lambda pair: (
                        0
                        if self._model_over_soft_quota(pair[0][0], location)
                        else 1,
                        self._eviction_score(pair[1]),
                        pair[1].last_access_order,
                        pair[0],
                    ),
                )
                released = item.replicas.pop(location)
                self._events.append(
                    PoolEvent(
                        "replica_evicted",
                        key[0],
                        key[1],
                        key[2],
                        location,
                        released,
                        "tier_room_for_canonical",
                    )
                )
                continue
            resident = tuple(
                key
                for key, item in self._sources.items()
                if item.source.kv_location is location
            )
            redundant = tuple(
                key
                for key in resident
                if len(self._bucket_keys(key[0], key[1]))
                > self.min_variants_per_retained_content
            )
            eligible_redundant = [
                key
                for key in redundant
                if self._evictable(
                    self._sources[key], respect_probation=True
                )
            ]
            victim = (
                min(
                    eligible_redundant,
                    key=lambda key: (
                        0
                        if self._model_over_soft_quota(key[0], location)
                        else 1,
                        self._eviction_score(self._sources[key]),
                        self._sources[key].last_access_order,
                        key,
                    ),
                )
                if eligible_redundant
                else None
            )
            if victim is not None:
                self._evict_source(
                    victim, "tier_redundant_variant", allow_last=False
                )
                continue
            contents = {}
            for key in resident:
                contents.setdefault((key[0], key[1]), []).append(key)
            eligible = []
            for content_key, keys in contents.items():
                if all(
                    self._evictable(
                        self._sources[key], respect_probation=True
                    )
                    for key in keys
                ):
                    eligible.append(
                        (
                            0
                            if self._model_over_soft_quota(
                                content_key[0], location
                            )
                            else 1,
                            sum(
                                self._eviction_score(self._sources[key])
                                for key in keys
                            ),
                            content_key,
                            tuple(keys),
                        )
                    )
            if not eligible:
                raise MemoryError(
                    "physical Source tier full; all victims are busy or in probation"
                )
            _, _, _, victims = min(
                eligible, key=lambda item: (item[0], item[1], item[2])
            )
            for victim_key in victims:
                self._evict_source(
                    victim_key, "tier_content_eviction", allow_last=True
                )

    def _eviction_score(self, item: PooledSource) -> float:
        if self.eviction_policy is GlobalEvictionPolicy.CACHE_CRAFT_FR:
            return item.stats.cache_craft_fr / float(max(1, item.resident_bytes))
        return item.stats.value_density(item.resident_bytes, self.prior_saved_ms)

    def _evictable(self, item: PooledSource, *, respect_probation: bool) -> bool:
        if item.busy:
            return False
        if respect_probation and (
            item.stats.observations < self.probation_observations
        ):
            return False
        return True

    def _evict_source(
        self,
        key: Tuple[str, str, str],
        reason: str,
        *,
        allow_last: bool,
    ) -> int:
        item = self._sources[key]
        if item.busy:
            raise RuntimeError("cannot evict an active Source")
        bucket_size = len(self._bucket_keys(key[0], key[1]))
        if not allow_last and bucket_size <= self.min_variants_per_retained_content:
            raise RuntimeError("cannot remove the last retained variant")
        released = item.resident_bytes
        del self._sources[key]
        self._events.append(
            PoolEvent(
                action="source_evicted",
                model_signature=key[0],
                content_hash=key[1],
                source_id=key[2],
                bytes_released=released,
                reason=reason,
            )
        )
        return released

    def _lowest_value_source(
        self,
        keys: Tuple[Tuple[str, str, str], ...],
        *,
        respect_probation: bool = True,
    ) -> Optional[Tuple[str, str, str]]:
        candidates = [
            key for key in keys
            if self._evictable(
                self._sources[key], respect_probation=respect_probation
            )
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda key: (
                self._eviction_score(self._sources[key]),
                self._sources[key].last_access_order,
                key,
            ),
        )

    def _make_canonical_room(self, required_bytes: int) -> None:
        capacity = self.canonical_capacity_bytes
        if capacity is None:
            return
        while self._canonical_used() + required_bytes > capacity:
            redundant = tuple(
                key for key in self._sources
                if len(self._bucket_keys(key[0], key[1]))
                > self.min_variants_per_retained_content
            )
            victim = self._lowest_value_source(redundant)
            if victim is not None:
                self._evict_source(victim, "global_redundant_variant", allow_last=False)
                continue
            content_keys = {}
            for key in self._sources:
                content_keys.setdefault((key[0], key[1]), []).append(key)
            eligible_contents = []
            for content_key, keys in content_keys.items():
                if all(
                    self._evictable(
                        self._sources[key], respect_probation=True
                    )
                    for key in keys
                ):
                    score = sum(
                        self._eviction_score(self._sources[key]) for key in keys
                    )
                    eligible_contents.append((score, content_key, tuple(keys)))
            if not eligible_contents:
                raise MemoryError(
                    "canonical pool full; all victims are busy or in probation"
                )
            _, _, victims = min(eligible_contents, key=lambda item: (item[0], item[1]))
            for victim_key in victims:
                self._evict_source(
                    victim_key, "global_content_eviction", allow_last=True
                )

    def register(self, source: HistoricalSource, canonical_bytes: int) -> None:
        source.validate_canonical()
        if canonical_bytes < 0:
            raise ValueError("canonical bytes must be non-negative")
        self._namespace_for_write(source.model_signature)
        key = self._key(
            source.model_signature, source.content_hash, source.source_id
        )
        existing = self._sources.get(key)
        if existing is not None:
            if existing.source != source or existing.canonical_bytes != canonical_bytes:
                raise ValueError("Source identity collision")
            return
        transaction = (
            copy.deepcopy(self._sources),
            list(self._events),
            self._clock,
        )
        bucket = self._bucket_keys(source.model_signature, source.content_hash)
        if any(
            self._sources[item].source.context_id == source.context_id
            for item in bucket
        ):
            raise ValueError("each variant requires an independent context")
        try:
            if len(bucket) >= self.max_variants_per_content:
                victim = self._lowest_value_source(bucket)
                if victim is None:
                    raise MemoryError(
                        "variant cap reached; all existing variants are busy or in probation"
                    )
                self._evict_source(
                    victim, "per_content_variant_cap", allow_last=False
                )
            self._make_tier_room_for_canonical(
                source.kv_location, canonical_bytes
            )
            self._make_canonical_room(canonical_bytes)
            order = self._tick()
            self._sources[key] = PooledSource(
                source=source,
                canonical_bytes=canonical_bytes,
                registered_order=order,
                last_access_order=order,
            )
        except Exception:
            self._sources, self._events, self._clock = transaction
            raise

    def candidates(
        self, model_signature: str, content_hash: str
    ) -> Tuple[HistoricalSource, ...]:
        namespace = self._namespaces.get(model_signature)
        if namespace is None or namespace.state is not ModelNamespaceState.ACTIVE:
            return ()
        keys = self._bucket_keys(model_signature, content_hash)
        return tuple(
            self._sources[key].source
            for key in sorted(
                keys,
                key=lambda value: (
                    self._sources[value].registered_order,
                    value[2],
                ),
            )
        )

    def lifecycle(
        self, model_signature: str, content_hash: str, source_id: str
    ) -> PooledSource:
        return self._sources[self._key(model_signature, content_hash, source_id)]

    def record_lookup(
        self, model_signature: str, content_hash: str, *, opportunity: bool = True
    ) -> None:
        keys = self._bucket_keys(model_signature, content_hash)
        for key in keys:
            item = self._sources[key]
            if opportunity:
                item.stats.lookup_opportunities += 1
            item.stats.lookup_hits += 1
            item.last_access_order = self._tick()

    def record_request_observation(
        self, model_signature: str, requested_content_hashes: Tuple[str, ...]
    ) -> None:
        """Update smoothed hit probability for every retained Source.

        A request is one opportunity for every Source in the active model;
        variants whose exact content occurs in the request receive a hit.
        """

        namespace = self._namespaces.get(model_signature)
        if namespace is None or namespace.state is not ModelNamespaceState.ACTIVE:
            raise RuntimeError("request observation requires an active model")
        requested = set(requested_content_hashes)
        for key, item in self._sources.items():
            if key[0] != model_signature:
                continue
            item.stats.lookup_opportunities += 1
            if key[1] in requested:
                item.stats.lookup_hits += 1
                item.last_access_order = self._tick()

    def record_comparison(
        self, model_signature: str, content_hash: str, source_id: str
    ) -> None:
        item = self.lifecycle(model_signature, content_hash, source_id)
        item.stats.comparisons += 1
        item.last_access_order = self._tick()

    def record_selection(
        self, model_signature: str, content_hash: str, source_id: str
    ) -> None:
        item = self.lifecycle(model_signature, content_hash, source_id)
        item.stats.selections += 1
        item.last_access_order = self._tick()

    def record_admission(
        self,
        model_signature: str,
        content_hash: str,
        source_id: str,
        realized_saved_ms: float,
    ) -> None:
        if realized_saved_ms < 0:
            raise ValueError("realized saving must be non-negative")
        item = self.lifecycle(model_signature, content_hash, source_id)
        item.stats.admissions += 1
        item.stats.realized_saved_ms_sum += realized_saved_ms
        item.last_access_order = self._tick()

    def record_cachecraft_access(
        self, model_signature: str, content_hash: str, source_id: str, cfo: float
    ) -> None:
        if not 0 <= cfo <= 1:
            raise ValueError("Cache-Craft CFO must be in [0, 1]")
        item = self.lifecycle(model_signature, content_hash, source_id)
        item.stats.cache_craft_fr += 1.0 / max(cfo, 1e-6)
        item.last_access_order = self._tick()

    def _activity(
        self,
        model_signature: str,
        content_hash: str,
        source_id: str,
        field_name: str,
        delta: int,
    ) -> HistoricalSource:
        namespace = self._namespaces[model_signature]
        if delta > 0 and namespace.state is not ModelNamespaceState.ACTIVE:
            raise RuntimeError("draining namespace rejects new Source activity")
        item = self.lifecycle(model_signature, content_hash, source_id)
        value = getattr(item, field_name) + delta
        if value < 0:
            raise RuntimeError("cannot finish inactive Source work")
        setattr(item, field_name, value)
        item.last_access_order = self._tick()
        if delta < 0 and namespace.state is ModelNamespaceState.DRAINING:
            self.begin_model_unload(model_signature)
        return item.source

    def lease(self, model_signature: str, content_hash: str, source_id: str) -> HistoricalSource:
        return self._activity(
            model_signature, content_hash, source_id, "lease_count", 1
        )

    def release(self, model_signature: str, content_hash: str, source_id: str) -> None:
        self._activity(
            model_signature, content_hash, source_id, "lease_count", -1
        )

    def begin_copy(self, model_signature: str, content_hash: str, source_id: str) -> None:
        self._activity(
            model_signature, content_hash, source_id, "copy_in_flight", 1
        )

    def end_copy(self, model_signature: str, content_hash: str, source_id: str) -> None:
        self._activity(
            model_signature, content_hash, source_id, "copy_in_flight", -1
        )

    def begin_execution(self, model_signature: str, content_hash: str, source_id: str) -> None:
        self._activity(
            model_signature, content_hash, source_id, "execution_in_flight", 1
        )

    def end_execution(self, model_signature: str, content_hash: str, source_id: str) -> None:
        self._activity(
            model_signature, content_hash, source_id, "execution_in_flight", -1
        )

    def attach_replica(
        self,
        model_signature: str,
        content_hash: str,
        source_id: str,
        location: KVLocation,
        size_bytes: int,
    ) -> Tuple[PoolEvent, ...]:
        if size_bytes < 0:
            raise ValueError("replica bytes must be non-negative")
        location = KVLocation(location)
        target = self.lifecycle(model_signature, content_hash, source_id)
        if target.source.kv_location is location:
            raise ValueError("canonical tier cannot also be attached as a replica")
        previous = target.replicas.get(location, 0)
        capacity = self.tier_capacity_bytes.get(location)
        start = len(self._events)
        required = self._tier_used(location) - previous + size_bytes
        if capacity is not None and required > capacity:
            candidates = [
                (key, item)
                for key, item in self._sources.items()
                if item is not target
                and location in item.replicas
                and not item.busy
            ]
            candidates.sort(
                key=lambda pair: (
                    0
                    if self._namespaces[pair[0][0]].soft_quota_bytes.get(location)
                    is not None
                    and self._model_tier_used(pair[0][0], location)
                    > self._namespaces[pair[0][0]].soft_quota_bytes[location]
                    else 1,
                    self._eviction_score(pair[1]),
                    pair[1].last_access_order,
                    pair[0],
                )
            )
            for key, victim in candidates:
                released = victim.replicas.pop(location)
                required -= released
                self._events.append(
                    PoolEvent(
                        action="replica_evicted",
                        model_signature=key[0],
                        content_hash=key[1],
                        source_id=key[2],
                        location=location,
                        bytes_released=released,
                        reason="global_tier_value_density",
                    )
                )
                if required <= capacity:
                    break
            if required > capacity:
                raise MemoryError("insufficient unleased replica capacity")
        target.replicas[location] = size_bytes
        target.last_access_order = self._tick()
        return tuple(self._events[start:])

    @property
    def events(self) -> Tuple[PoolEvent, ...]:
        return tuple(self._events)

    def audit_snapshot(self) -> Mapping[str, object]:
        return {
            "serving_mode": self.serving_mode.value,
            "eviction_policy": self.eviction_policy.value,
            "canonical_capacity_bytes": self.canonical_capacity_bytes,
            "canonical_used_bytes": self._canonical_used(),
            "tier_capacity_bytes": {
                location.value: size
                for location, size in self.tier_capacity_bytes.items()
            },
            "tier_used_bytes": {
                location.value: self._tier_used(location)
                for location in self.tier_capacity_bytes
            },
            "model_namespaces": {
                signature: {
                    "state": namespace.state.value,
                    "soft_quota_bytes": {
                        location.value: size
                        for location, size in namespace.soft_quota_bytes.items()
                    },
                }
                for signature, namespace in self._namespaces.items()
            },
            "sources": [
                {
                    "model_signature": key[0],
                    "content_hash": key[1],
                    "source_id": key[2],
                    "canonical_location": item.source.kv_location.value,
                    "canonical_bytes": item.canonical_bytes,
                    "replicas": {
                        location.value: size
                        for location, size in item.replicas.items()
                    },
                    "lease_count": item.lease_count,
                    "copy_in_flight": item.copy_in_flight,
                    "execution_in_flight": item.execution_in_flight,
                    "lookup_hits": item.stats.lookup_hits,
                    "lookup_opportunities": item.stats.lookup_opportunities,
                    "comparisons": item.stats.comparisons,
                    "selections": item.stats.selections,
                    "admissions": item.stats.admissions,
                    "value_density": item.stats.value_density(
                        item.resident_bytes, self.prior_saved_ms
                    ),
                    "cache_craft_fr": item.stats.cache_craft_fr,
                }
                for key, item in sorted(self._sources.items())
            ],
            "events": [
                {
                    "action": event.action,
                    "model_signature": event.model_signature,
                    "content_hash": event.content_hash,
                    "source_id": event.source_id,
                    "location": (
                        event.location.value
                        if event.location is not None
                        else None
                    ),
                    "bytes_released": event.bytes_released,
                    "reason": event.reason,
                }
                for event in self._events
            ],
        }

    @property
    def source_count(self) -> int:
        return len(self._sources)
