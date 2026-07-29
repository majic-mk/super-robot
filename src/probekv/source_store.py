from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

from .contracts import HistoricalSource, KVLocation


class SourceEvictionPolicy(str, Enum):
    """Version-retention policy for one exact segment and model."""

    REJECT_WHEN_FULL = "reject_when_full"
    FIFO = "fifo"


class ReplicaEvictionPolicy(str, Enum):
    """Physical KV replica policy; independent of Source selection."""

    REJECT_WHEN_FULL = "reject_when_full"
    LRU = "lru"


@dataclass(frozen=True)
class EvictionEvent:
    source_id: str
    model_signature: str
    content_hash: str
    reason: str
    location: Optional[KVLocation] = None
    bytes_released: int = 0


@dataclass
class SourceLifecycle:
    source: HistoricalSource
    registered_order: int
    last_access_order: int
    lease_count: int = 0
    replicas: Dict[KVLocation, int] = field(default_factory=dict)


class SourceStore:
    """Canonical Source registry plus explicit, selector-neutral lifecycle.

    A repeated segment C may be conditioned by A, B, or E, but S1/S2/S3 must
    each be produced by their own full prefill. Reusing S1 to construct S2
    would carry A's influence and violates this registry's admission rule.

    The default policy preserves the pre-v4 behavior and rejects a fifth
    version.  Online v4 may opt into FIFO version retention.  GPU/CPU replicas
    are managed separately with byte-aware LRU; evicting a replica never turns
    a repaired result into a canonical Source and never changes Source ranking.
    """

    def __init__(
        self,
        online_kmax: int = 4,
        eviction_policy: SourceEvictionPolicy = (
            SourceEvictionPolicy.REJECT_WHEN_FULL
        ),
        replica_eviction_policy: ReplicaEvictionPolicy = (
            ReplicaEvictionPolicy.REJECT_WHEN_FULL
        ),
        tier_capacity_bytes: Optional[Dict[KVLocation, int]] = None,
        fixed_resident: bool = False,
    ) -> None:
        if online_kmax < 1:
            raise ValueError("online_kmax must be positive")
        self.online_kmax = online_kmax
        self.eviction_policy = SourceEvictionPolicy(eviction_policy)
        self.replica_eviction_policy = ReplicaEvictionPolicy(
            replica_eviction_policy
        )
        self.fixed_resident = bool(fixed_resident)
        self._by_key: Dict[
            Tuple[str, str], Dict[str, SourceLifecycle]
        ] = {}
        self._capacity = {
            KVLocation(location): int(value)
            for location, value in (tier_capacity_bytes or {}).items()
        }
        if any(value < 0 for value in self._capacity.values()):
            raise ValueError("tier capacities must be non-negative")
        self._clock = 0
        self._events: List[EvictionEvent] = []

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    @staticmethod
    def _key(source: HistoricalSource) -> Tuple[str, str]:
        return source.model_signature, source.content_hash

    def _bucket(
        self, model_signature: str, content_hash: str
    ) -> Dict[str, SourceLifecycle]:
        return self._by_key.setdefault(
            (model_signature, content_hash), {}
        )

    def _evict_version_if_needed(
        self,
        bucket: Dict[str, SourceLifecycle],
        source: HistoricalSource,
    ) -> None:
        if len(bucket) < self.online_kmax:
            return
        if (
            self.fixed_resident
            or self.eviction_policy
            is SourceEvictionPolicy.REJECT_WHEN_FULL
        ):
            raise ValueError("online Kmax=%d exceeded" % self.online_kmax)
        candidates = [
            lifecycle
            for lifecycle in bucket.values()
            if lifecycle.lease_count == 0
        ]
        if not candidates:
            raise RuntimeError(
                "online Kmax reached and every Source is leased"
            )
        victim = min(
            candidates,
            key=lambda item: (
                item.registered_order,
                item.source.source_id,
            ),
        )
        del bucket[victim.source.source_id]
        self._events.append(
            EvictionEvent(
                victim.source.source_id,
                victim.source.model_signature,
                victim.source.content_hash,
                "version_fifo",
                bytes_released=sum(victim.replicas.values()),
            )
        )

    def register(self, source: HistoricalSource) -> None:
        source.validate_canonical()
        bucket = self._bucket(*self._key(source))
        if source.source_id in bucket:
            if bucket[source.source_id].source != source:
                raise ValueError("source_id collision with different metadata")
            return
        if any(
            lifecycle.source.context_id == source.context_id
            for lifecycle in bucket.values()
        ):
            raise ValueError(
                "each Source version requires an independent context_id"
            )
        self._evict_version_if_needed(bucket, source)
        order = self._tick()
        lifecycle = SourceLifecycle(source, order, order)
        # The immutable source records its initial physical placement.
        if source.kv_handles:
            lifecycle.replicas[source.kv_location] = 0
        bucket[source.source_id] = lifecycle

    def register_many(self, sources: Iterable[HistoricalSource]) -> None:
        for source in sources:
            self.register(source)

    def _resolve_bucket(
        self, content_hash: str, model_signature: Optional[str]
    ) -> Dict[str, SourceLifecycle]:
        if model_signature is not None:
            return self._by_key.get((model_signature, content_hash), {})
        matches = [
            bucket
            for (model, digest), bucket in self._by_key.items()
            if digest == content_hash
        ]
        if len(matches) > 1:
            raise ValueError(
                "model_signature is required for a multi-model segment"
            )
        return matches[0] if matches else {}

    def candidates(
        self, content_hash: str, model_signature: Optional[str] = None
    ) -> List[HistoricalSource]:
        bucket = self._resolve_bucket(content_hash, model_signature)
        return [
            lifecycle.source
            for lifecycle in sorted(
                bucket.values(),
                key=lambda item: (
                    item.registered_order,
                    item.source.source_id,
                ),
            )
        ]

    def get(
        self,
        content_hash: str,
        source_id: str,
        model_signature: Optional[str] = None,
    ) -> HistoricalSource:
        lifecycle = self._resolve_bucket(
            content_hash, model_signature
        )[source_id]
        lifecycle.last_access_order = self._tick()
        return lifecycle.source

    def lease(
        self,
        content_hash: str,
        source_id: str,
        model_signature: Optional[str] = None,
    ) -> HistoricalSource:
        bucket = self._resolve_bucket(content_hash, model_signature)
        lifecycle = bucket[source_id]
        lifecycle.lease_count += 1
        lifecycle.last_access_order = self._tick()
        return lifecycle.source

    def release(
        self,
        content_hash: str,
        source_id: str,
        model_signature: Optional[str] = None,
    ) -> None:
        lifecycle = self._resolve_bucket(
            content_hash, model_signature
        )[source_id]
        if lifecycle.lease_count <= 0:
            raise RuntimeError("cannot release an unleased Source")
        lifecycle.lease_count -= 1
        lifecycle.last_access_order = self._tick()

    def _tier_used(self, location: KVLocation) -> int:
        return sum(
            lifecycle.replicas.get(location, 0)
            for bucket in self._by_key.values()
            for lifecycle in bucket.values()
        )

    def attach_replica(
        self,
        content_hash: str,
        source_id: str,
        location: KVLocation,
        size_bytes: int,
        model_signature: Optional[str] = None,
    ) -> Tuple[EvictionEvent, ...]:
        if size_bytes < 0:
            raise ValueError("replica size must be non-negative")
        location = KVLocation(location)
        lifecycle = self._resolve_bucket(
            content_hash, model_signature
        )[source_id]
        previous = lifecycle.replicas.get(location, 0)
        capacity = self._capacity.get(location)
        required = self._tier_used(location) - previous + size_bytes
        start = len(self._events)
        if capacity is not None and required > capacity:
            if (
                self.fixed_resident
                or self.replica_eviction_policy
                is ReplicaEvictionPolicy.REJECT_WHEN_FULL
            ):
                raise MemoryError(
                    "%s replica capacity exceeded" % location.value
                )
            victims = sorted(
                (
                    other
                    for bucket in self._by_key.values()
                    for other in bucket.values()
                    if other is not lifecycle
                    and other.lease_count == 0
                    and location in other.replicas
                ),
                key=lambda item: (
                    item.last_access_order,
                    item.registered_order,
                    item.source.source_id,
                ),
            )
            releasable = sum(
                victim.replicas[location] for victim in victims
            )
            if required - releasable > capacity:
                raise MemoryError(
                    "insufficient unleased %s replicas to evict"
                    % location.value
                )
            for victim in victims:
                released = victim.replicas.pop(location)
                required -= released
                self._events.append(
                    EvictionEvent(
                        victim.source.source_id,
                        victim.source.model_signature,
                        victim.source.content_hash,
                        "replica_lru",
                        location,
                        released,
                    )
                )
                if required <= capacity:
                    break
        lifecycle.replicas[location] = size_bytes
        lifecycle.last_access_order = self._tick()
        return tuple(self._events[start:])

    def detach_replica(
        self,
        content_hash: str,
        source_id: str,
        location: KVLocation,
        model_signature: Optional[str] = None,
    ) -> int:
        if self.fixed_resident:
            raise RuntimeError("fixed-resident mode forbids replica eviction")
        lifecycle = self._resolve_bucket(
            content_hash, model_signature
        )[source_id]
        if lifecycle.lease_count:
            raise RuntimeError("cannot evict a leased Source replica")
        return lifecycle.replicas.pop(KVLocation(location), 0)

    def lifecycle(
        self,
        content_hash: str,
        source_id: str,
        model_signature: Optional[str] = None,
    ) -> SourceLifecycle:
        return self._resolve_bucket(
            content_hash, model_signature
        )[source_id]

    @property
    def eviction_events(self) -> Tuple[EvictionEvent, ...]:
        return tuple(self._events)
