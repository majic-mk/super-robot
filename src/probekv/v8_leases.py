from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import KVLocation


class ReplicaLifecycle(str, Enum):
    ALLOCATING = "allocating"
    READY = "ready"
    CORRUPT = "corrupt"
    EVICTING = "evicting"
    DELETED = "deleted"


class LeaseLifecycle(str, Enum):
    ACTIVE = "active"
    SUSPECT = "suspect"
    ORPHANED = "orphaned"
    RELEASED = "released"


class LeasePurpose(str, Enum):
    LOGICAL_SOURCE = "logical_source"
    COPY_SOURCE = "copy_source"
    COPY_TARGET = "copy_target"
    EXECUTION = "execution"


@dataclass
class V8ReplicaResource:
    replica_id: str
    source_variant_id: str
    artifact_id: str
    tier: KVLocation
    generation: int
    placement_epoch: int
    size_bytes: int
    is_backing: bool
    lifecycle: ReplicaLifecycle = ReplicaLifecycle.READY
    lease_refcount: int = 0
    copy_in_flight: int = 0
    execution_in_flight: int = 0

    def __post_init__(self) -> None:
        self.tier = KVLocation(self.tier)
        if not all((self.replica_id, self.source_variant_id, self.artifact_id)):
            raise ValueError("complete Replica resource identity is required")
        if min(self.generation, self.placement_epoch) < 1 or self.size_bytes < 0:
            raise ValueError("invalid Replica resource generation or size")

    @property
    def healthy(self) -> bool:
        return self.lifecycle is ReplicaLifecycle.READY

    @property
    def busy(self) -> bool:
        return bool(self.lease_refcount or self.copy_in_flight or self.execution_in_flight)


@dataclass
class LogicalSourceResource:
    source_variant_id: str
    artifact_id: str
    namespace: str
    active: bool = True
    logical_lease_refcount: int = 0
    replicas: Dict[str, V8ReplicaResource] = field(default_factory=dict)

    @property
    def healthy_backing_replica_count(self) -> int:
        return sum(item.healthy and item.is_backing for item in self.replicas.values())


@dataclass
class LeaseRecord:
    lease_id: str
    owner_request_id: str
    owner_request_generation: int
    segment_id: str
    source_variant_id: str
    artifact_id: str
    replica_id: Optional[str]
    purpose: LeasePurpose
    created_at_s: float
    expires_at_s: float
    heartbeat_at_s: float
    state: LeaseLifecycle = LeaseLifecycle.ACTIVE
    suspect_at_s: Optional[float] = None
    release_reason: Optional[str] = None


@dataclass(frozen=True)
class ReplicaLeaseRequest:
    segment_id: str
    source_variant_id: str
    artifact_id: str
    replica_id: str
    replica_generation: int
    placement_epoch: int
    purpose: LeasePurpose

    def __post_init__(self) -> None:
        if self.purpose is LeasePurpose.LOGICAL_SOURCE:
            raise ValueError("physical batch binding requires a physical purpose")


class V8LeaseManager:
    """Request-scoped logical and physical leases with all-or-nothing binding."""

    def __init__(self, *, ttl_floor_s: float = 30.0, orphan_grace_s: float = 5.0) -> None:
        if ttl_floor_s <= 0 or orphan_grace_s < 0:
            raise ValueError("invalid lease recovery timing")
        self.ttl_floor_s = ttl_floor_s
        self.orphan_grace_s = orphan_grace_s
        self.sources: Dict[str, LogicalSourceResource] = {}
        self.leases: Dict[str, LeaseRecord] = {}
        self._frozen_by_segment: Dict[Tuple[str, int, str], str] = {}

    def register_source(
        self, source_variant_id: str, artifact_id: str, namespace: str
    ) -> LogicalSourceResource:
        if not all((source_variant_id, artifact_id, namespace)):
            raise ValueError("Source resource identity is required")
        existing = self.sources.get(source_variant_id)
        if existing is not None:
            if (existing.artifact_id, existing.namespace) != (artifact_id, namespace):
                raise ValueError("Source Variant resource identity changed")
            return existing
        resource = LogicalSourceResource(source_variant_id, artifact_id, namespace)
        self.sources[source_variant_id] = resource
        return resource

    def register_replica(self, replica: V8ReplicaResource) -> None:
        source = self.sources.get(replica.source_variant_id)
        if source is None or source.artifact_id != replica.artifact_id:
            raise ValueError("Replica does not belong to a registered Source Artifact")
        if any(
            item.tier is replica.tier and item.lifecycle is not ReplicaLifecycle.DELETED
            for item in source.replicas.values()
        ):
            raise ValueError("only one live full-KV Replica is allowed per tier")
        if replica.is_backing and any(
            item.is_backing and item.lifecycle is not ReplicaLifecycle.DELETED
            for item in source.replicas.values()
        ):
            raise ValueError("only one live backing Replica is allowed")
        source.replicas[replica.replica_id] = replica

    @staticmethod
    def ttl_for(predicted_remaining_s: float, floor_s: float = 30.0) -> float:
        if predicted_remaining_s < 0:
            raise ValueError("predicted remaining time must be non-negative")
        return max(floor_s, 4.0 * predicted_remaining_s)

    def _new_lease(
        self,
        *,
        request_id: str,
        request_generation: int,
        segment_id: str,
        source: LogicalSourceResource,
        replica_id: Optional[str],
        purpose: LeasePurpose,
        predicted_remaining_s: float,
        now_s: float,
    ) -> LeaseRecord:
        ttl = self.ttl_for(predicted_remaining_s, self.ttl_floor_s)
        record = LeaseRecord(
            lease_id=str(uuid.uuid4()),
            owner_request_id=request_id,
            owner_request_generation=request_generation,
            segment_id=segment_id,
            source_variant_id=source.source_variant_id,
            artifact_id=source.artifact_id,
            replica_id=replica_id,
            purpose=purpose,
            created_at_s=now_s,
            expires_at_s=now_s + ttl,
            heartbeat_at_s=now_s,
        )
        self.leases[record.lease_id] = record
        return record

    def freeze_and_acquire_logical(
        self,
        *,
        request_id: str,
        request_generation: int,
        segment_id: str,
        source_variant_id: str,
        predicted_remaining_s: float,
        now_s: Optional[float] = None,
    ) -> LeaseRecord:
        source = self.sources.get(source_variant_id)
        if source is None or not source.active or source.healthy_backing_replica_count < 1:
            raise RuntimeError("Source freeze requires one healthy backing Replica")
        key = (request_id, request_generation, segment_id)
        frozen = self._frozen_by_segment.get(key)
        if frozen is not None and frozen != source_variant_id:
            raise RuntimeError("Source was already frozen to another Variant")
        timestamp = time.monotonic() if now_s is None else now_s
        record = self._new_lease(
            request_id=request_id,
            request_generation=request_generation,
            segment_id=segment_id,
            source=source,
            replica_id=None,
            purpose=LeasePurpose.LOGICAL_SOURCE,
            predicted_remaining_s=predicted_remaining_s,
            now_s=timestamp,
        )
        self._frozen_by_segment[key] = source_variant_id
        source.logical_lease_refcount += 1
        return record

    def compare_and_lease_batch(
        self,
        *,
        request_id: str,
        request_generation: int,
        requests: Sequence[ReplicaLeaseRequest],
        predicted_remaining_s: float,
        hbm_capacity_bytes: Optional[int] = None,
        now_s: Optional[float] = None,
    ) -> Tuple[LeaseRecord, ...]:
        if len({item.segment_id for item in requests}) != len(requests):
            raise ValueError("a physical batch may bind at most one Replica per Segment")
        resolved = []
        gpu_bytes = 0
        for request in requests:
            source = self.sources.get(request.source_variant_id)
            if source is None or source.artifact_id != request.artifact_id:
                raise RuntimeError("stale Source/Artifact binding")
            frozen = self._frozen_by_segment.get(
                (request_id, request_generation, request.segment_id)
            )
            if frozen != request.source_variant_id:
                raise RuntimeError("physical binding differs from the frozen Source")
            replica = source.replicas.get(request.replica_id)
            if replica is None or not replica.healthy:
                raise RuntimeError("physical Replica is unavailable")
            if replica.generation != request.replica_generation:
                raise RuntimeError("stale Replica generation")
            if replica.placement_epoch != request.placement_epoch:
                raise RuntimeError("stale Replica placement")
            if replica.tier is KVLocation.GPU:
                gpu_bytes += replica.size_bytes
            resolved.append((source, replica, request))
        if hbm_capacity_bytes is not None and gpu_bytes > hbm_capacity_bytes:
            raise MemoryError("atomic Replica batch exceeds HBM capacity")

        timestamp = time.monotonic() if now_s is None else now_s
        records = []
        for source, replica, request in resolved:
            record = self._new_lease(
                request_id=request_id,
                request_generation=request_generation,
                segment_id=request.segment_id,
                source=source,
                replica_id=replica.replica_id,
                purpose=request.purpose,
                predicted_remaining_s=predicted_remaining_s,
                now_s=timestamp,
            )
            replica.lease_refcount += 1
            if request.purpose in {LeasePurpose.COPY_SOURCE, LeasePurpose.COPY_TARGET}:
                replica.copy_in_flight += 1
            elif request.purpose is LeasePurpose.EXECUTION:
                replica.execution_in_flight += 1
            records.append(record)
        return tuple(records)

    def heartbeat(self, lease_id: str, *, now_s: Optional[float] = None) -> LeaseRecord:
        record = self.leases[lease_id]
        if record.state not in {LeaseLifecycle.ACTIVE, LeaseLifecycle.SUSPECT}:
            raise RuntimeError("only live leases may be renewed")
        timestamp = time.monotonic() if now_s is None else now_s
        ttl = record.expires_at_s - record.heartbeat_at_s
        record.heartbeat_at_s = timestamp
        record.expires_at_s = timestamp + max(ttl, self.ttl_floor_s)
        record.state = LeaseLifecycle.ACTIVE
        record.suspect_at_s = None
        return record

    def release(self, lease_id: str, *, reason: str = "request_terminal") -> None:
        record = self.leases[lease_id]
        if record.state is LeaseLifecycle.RELEASED:
            return
        source = self.sources[record.source_variant_id]
        if record.replica_id is None:
            source.logical_lease_refcount -= 1
            if source.logical_lease_refcount < 0:
                raise RuntimeError("logical lease refcount underflow")
        else:
            replica = source.replicas[record.replica_id]
            replica.lease_refcount -= 1
            if record.purpose in {LeasePurpose.COPY_SOURCE, LeasePurpose.COPY_TARGET}:
                replica.copy_in_flight -= 1
            elif record.purpose is LeasePurpose.EXECUTION:
                replica.execution_in_flight -= 1
            if min(replica.lease_refcount, replica.copy_in_flight, replica.execution_in_flight) < 0:
                raise RuntimeError("physical lease counter underflow")
        record.state = LeaseLifecycle.RELEASED
        record.release_reason = reason

    def recover_expired(
        self,
        *,
        live_owners: Iterable[Tuple[str, int]],
        now_s: Optional[float] = None,
    ) -> Tuple[str, ...]:
        timestamp = time.monotonic() if now_s is None else now_s
        live = set(live_owners)
        released = []
        for record in self.leases.values():
            if record.state is LeaseLifecycle.RELEASED or timestamp < record.expires_at_s:
                continue
            owner = (record.owner_request_id, record.owner_request_generation)
            if owner in live:
                self.heartbeat(record.lease_id, now_s=timestamp)
                continue
            if record.state is LeaseLifecycle.ACTIVE:
                record.state = LeaseLifecycle.SUSPECT
                record.suspect_at_s = timestamp
                continue
            if (
                record.state is LeaseLifecycle.SUSPECT
                and record.suspect_at_s is not None
                and timestamp >= record.suspect_at_s + self.orphan_grace_s
            ):
                record.state = LeaseLifecycle.ORPHANED
                self.release(record.lease_id, reason="orphan_recovery")
                released.append(record.lease_id)
        return tuple(released)

    def evict_replica(self, source_variant_id: str, replica_id: str) -> None:
        source = self.sources[source_variant_id]
        replica = source.replicas[replica_id]
        if replica.busy:
            raise RuntimeError("leased/copying/executing Replica cannot be evicted")
        if replica.is_backing and source.logical_lease_refcount > 0:
            raise RuntimeError("logical Source lease protects the last healthy backing")
        replica.lifecycle = ReplicaLifecycle.DELETED

    def purge_namespace(self, namespace: str) -> None:
        targets = [item for item in self.sources.values() if item.namespace == namespace]
        if any(
            item.logical_lease_refcount
            or any(replica.busy for replica in item.replicas.values())
            for item in targets
        ):
            raise RuntimeError("namespace purge must wait for active leases")
        for source in targets:
            source.active = False
            for replica in source.replicas.values():
                replica.lifecycle = ReplicaLifecycle.DELETED

    def same_source_replan(
        self,
        source_variant_id: str,
        *,
        excluded_replica_ids: Iterable[str] = (),
        attempt: int,
    ) -> Optional[V8ReplicaResource]:
        if attempt not in {1, 2}:
            raise RuntimeError("same-Source Replica replan is limited to two attempts")
        excluded = set(excluded_replica_ids)
        source = self.sources[source_variant_id]
        candidates = [
            item
            for item in source.replicas.values()
            if item.healthy and item.replica_id not in excluded
        ]
        priority = {KVLocation.GPU: 0, KVLocation.PINNED_CPU: 1, KVLocation.SSD: 2}
        return min(
            candidates,
            key=lambda item: (priority[item.tier], item.replica_id),
            default=None,
        )

