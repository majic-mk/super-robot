from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, TYPE_CHECKING

from .contracts import KVLocation

if TYPE_CHECKING:
    from .v8_leases import V8LeaseManager, V8ReplicaResource


@dataclass
class TieredBackingEntry:
    source_variant_id: str
    size_bytes: int
    tier: KVLocation
    last_access_order: int
    backing_replica_id: str = ""
    busy: bool = False

    def __post_init__(self) -> None:
        self.tier = KVLocation(self.tier)
        if not self.source_variant_id or self.size_bytes <= 0:
            raise ValueError("backing entry requires identity and positive bytes")
        if self.tier not in {KVLocation.PINNED_CPU, KVLocation.SSD}:
            raise ValueError("schema-v8 backing must reside in CPU or SSD")


@dataclass(frozen=True)
class TieredBackingAction:
    action: str
    source_variant_id: str
    source_tier: Optional[KVLocation]
    target_tier: Optional[KVLocation]
    bytes: int


@dataclass(frozen=True)
class BackingMigrationTicket:
    source_variant_id: str
    old_replica_id: str
    destination_replica_id: str
    old_tier: KVLocation
    destination_tier: KVLocation
    old_generation: int
    destination_generation: int


class Schema8TieredBackingManager:
    """One backing copy with CPU preference and one request-use LRU epoch.

    The manager produces deterministic placement actions.  A runtime must
    complete and verify the destination copy before applying a move action to
    its PhysicalReplica registry; GPU hot Replicas remain outside this backing
    policy.
    """

    def __init__(self, *, cpu_capacity_bytes: int, ssd_capacity_bytes: int) -> None:
        if min(cpu_capacity_bytes, ssd_capacity_bytes) < 0:
            raise ValueError("tier capacities must be non-negative")
        self.cpu_capacity_bytes = int(cpu_capacity_bytes)
        self.ssd_capacity_bytes = int(ssd_capacity_bytes)
        self._clock = 0
        self._entries: Dict[str, TieredBackingEntry] = {}
        self._actions: list[TieredBackingAction] = []

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _checkpoint(self) -> Tuple[int, Dict[str, TieredBackingEntry], int]:
        return (
            self._clock,
            {
                key: TieredBackingEntry(
                    row.source_variant_id,
                    row.size_bytes,
                    row.tier,
                    row.last_access_order,
                    row.backing_replica_id,
                    row.busy,
                )
                for key, row in self._entries.items()
            },
            len(self._actions),
        )

    def _rollback(
        self,
        checkpoint: Tuple[int, Dict[str, TieredBackingEntry], int],
    ) -> None:
        self._clock, self._entries, action_count = checkpoint
        del self._actions[action_count:]

    @property
    def actions(self) -> Tuple[TieredBackingAction, ...]:
        return tuple(self._actions)

    def entry(self, source_variant_id: str) -> TieredBackingEntry:
        return self._entries[source_variant_id]

    def tier_usage(self, tier: KVLocation) -> int:
        tier = KVLocation(tier)
        return sum(row.size_bytes for row in self._entries.values() if row.tier is tier)

    def snapshot(self) -> Mapping[str, object]:
        return {
            "cpu_used_bytes": self.tier_usage(KVLocation.PINNED_CPU),
            "ssd_used_bytes": self.tier_usage(KVLocation.SSD),
            "entries": {
                key: {
                    "tier": row.tier.value,
                    "size_bytes": row.size_bytes,
                    "last_access_order": row.last_access_order,
                    "backing_replica_id": row.backing_replica_id,
                    "busy": row.busy,
                }
                for key, row in sorted(self._entries.items())
            },
        }

    def set_busy(self, source_variant_id: str, busy: bool) -> None:
        self.entry(source_variant_id).busy = bool(busy)

    def register(
        self,
        source_variant_id: str,
        *,
        size_bytes: int,
        initially_used: bool = True,
        backing_replica_id: str = "",
    ) -> Tuple[TieredBackingAction, ...]:
        if source_variant_id in self._entries:
            raise ValueError("a Source Variant may have one backing entry")
        if size_bytes <= 0:
            raise ValueError("backing size must be positive")
        checkpoint = self._checkpoint()
        start = len(self._actions)
        try:
            target = (
                KVLocation.PINNED_CPU
                if initially_used and size_bytes <= self.cpu_capacity_bytes
                else KVLocation.SSD
            )
            self._make_room(
                target,
                size_bytes,
                protected_source_ids=(source_variant_id,),
            )
            entry = TieredBackingEntry(
                source_variant_id,
                int(size_bytes),
                target,
                self._tick(),
                str(backing_replica_id),
            )
            self._entries[source_variant_id] = entry
            self._actions.append(
                TieredBackingAction(
                    "backing_registered", source_variant_id, None, target, size_bytes
                )
            )
        except Exception:
            self._rollback(checkpoint)
            raise
        return tuple(self._actions[start:])

    def access(self, source_variant_id: str) -> Tuple[TieredBackingAction, ...]:
        entry = self.entry(source_variant_id)
        checkpoint = self._checkpoint()
        start = len(self._actions)
        try:
            entry.last_access_order = self._tick()
            if entry.tier is KVLocation.PINNED_CPU:
                self._actions.append(
                    TieredBackingAction(
                        "cpu_backing_hit", source_variant_id,
                        KVLocation.PINNED_CPU, KVLocation.PINNED_CPU, entry.size_bytes,
                    )
                )
                return tuple(self._actions[start:])
            if entry.size_bytes > self.cpu_capacity_bytes:
                self._actions.append(
                    TieredBackingAction(
                        "ssd_backing_too_large_for_cpu", source_variant_id,
                        KVLocation.SSD, KVLocation.SSD, entry.size_bytes,
                    )
                )
                return tuple(self._actions[start:])
            self._make_room(
                KVLocation.PINNED_CPU,
                entry.size_bytes,
                protected_source_ids=(source_variant_id,),
            )
            # This manager records the post-transition policy state.  The
            # runtime applies it only after a verified copy and atomic registry
            # swap; failure must leave its pre-transition registry unchanged.
            entry = self.entry(source_variant_id)
            entry.tier = KVLocation.PINNED_CPU
            entry.last_access_order = self._tick()
            self._actions.append(
                TieredBackingAction(
                    "promote_ssd_to_cpu", source_variant_id,
                    KVLocation.SSD, KVLocation.PINNED_CPU, entry.size_bytes,
                )
            )
        except Exception:
            self._rollback(checkpoint)
            raise
        return tuple(self._actions[start:])

    def _make_room(
        self,
        tier: KVLocation,
        requested_bytes: int,
        *,
        protected_source_ids: Tuple[str, ...],
    ) -> None:
        tier = KVLocation(tier)
        capacity = (
            self.cpu_capacity_bytes
            if tier is KVLocation.PINNED_CPU
            else self.ssd_capacity_bytes
        )
        if requested_bytes > capacity:
            raise MemoryError("backing object exceeds the target tier capacity")
        while self.tier_usage(tier) + requested_bytes > capacity:
            candidates = sorted(
                (
                    row
                    for row in self._entries.values()
                    if row.tier is tier
                    and not row.busy
                    and row.source_variant_id not in set(protected_source_ids)
                ),
                key=lambda row: (row.last_access_order, row.source_variant_id),
            )
            if not candidates:
                raise MemoryError("target tier is full and every LRU victim is busy")
            victim = candidates[0]
            if tier is KVLocation.PINNED_CPU:
                self._demote_cpu_victim(victim, protected_source_ids)
            else:
                self._evict_ssd_victim(victim)

    def _demote_cpu_victim(
        self,
        victim: TieredBackingEntry,
        additionally_protected_source_ids: Tuple[str, ...],
    ) -> None:
        if victim.size_bytes <= self.ssd_capacity_bytes:
            self._make_room(
                KVLocation.SSD,
                victim.size_bytes,
                protected_source_ids=(
                    victim.source_variant_id,
                    *additionally_protected_source_ids,
                ),
            )
            victim.tier = KVLocation.SSD
            self._actions.append(
                TieredBackingAction(
                    "demote_cpu_to_ssd", victim.source_variant_id,
                    KVLocation.PINNED_CPU, KVLocation.SSD, victim.size_bytes,
                )
            )
            return
        self._entries.pop(victim.source_variant_id)
        self._actions.append(
            TieredBackingAction(
                "evict_cpu_oversize_source", victim.source_variant_id,
                KVLocation.PINNED_CPU, None, victim.size_bytes,
            )
        )

    def _evict_ssd_victim(self, victim: TieredBackingEntry) -> None:
        self._entries.pop(victim.source_variant_id)
        self._actions.append(
            TieredBackingAction(
                "evict_ssd_lru_source", victim.source_variant_id,
                KVLocation.SSD, None, victim.size_bytes,
            )
        )


class Schema8TieredReplicaCoordinator:
    """Derive tier-LRU eviction protection from the authoritative lease graph.

    Callers must synchronize immediately before planning any pressure action.
    This removes the unsafe requirement for application code to remember a
    separate manual ``set_busy`` flag.
    """

    def __init__(
        self,
        *,
        backing_manager: Schema8TieredBackingManager,
        lease_manager: "V8LeaseManager",
    ) -> None:
        self.backing_manager = backing_manager
        self.lease_manager = lease_manager

    def synchronize_busy(self) -> Mapping[str, bool]:
        result: Dict[str, bool] = {}
        for source_id, entry in self.backing_manager._entries.items():
            source = self.lease_manager.sources.get(source_id)
            if source is None or not source.active:
                raise RuntimeError("tiered backing lacks an active lease resource")
            healthy = [
                replica
                for replica in source.replicas.values()
                if replica.healthy and replica.is_backing
            ]
            if len(healthy) != 1:
                raise RuntimeError("Source must have exactly one healthy backing Replica")
            replica = healthy[0]
            if entry.backing_replica_id and entry.backing_replica_id != replica.replica_id:
                raise RuntimeError("tiered backing points at another Replica")
            if replica.tier is not entry.tier:
                raise RuntimeError("tiered backing and Replica registry disagree")
            entry.backing_replica_id = replica.replica_id
            entry.busy = bool(source.logical_lease_refcount or replica.busy)
            result[source_id] = entry.busy
        return result

    def access(self, source_variant_id: str) -> Tuple[TieredBackingAction, ...]:
        self.synchronize_busy()
        return self.backing_manager.access(source_variant_id)

    def register(
        self,
        source_variant_id: str,
        *,
        size_bytes: int,
        initially_used: bool = True,
    ) -> Tuple[TieredBackingAction, ...]:
        source = self.lease_manager.sources.get(source_variant_id)
        if source is None:
            raise RuntimeError("backing registration requires a Source resource")
        healthy = [
            replica
            for replica in source.replicas.values()
            if replica.healthy and replica.is_backing
        ]
        if len(healthy) != 1:
            raise RuntimeError("backing registration requires exactly one healthy Replica")
        replica = healthy[0]
        actions = self.backing_manager.register(
            source_variant_id,
            size_bytes=size_bytes,
            initially_used=initially_used,
            backing_replica_id=replica.replica_id,
        )
        entry = self.backing_manager.entry(source_variant_id)
        if entry.tier is not replica.tier:
            # Placement actions must be executed and atomically reflected in
            # the Replica registry before this coordinator may expose them.
            self.backing_manager._entries.pop(source_variant_id, None)
            raise RuntimeError("policy placement differs from the registered backing Replica")
        self.synchronize_busy()
        return actions

    def begin_backing_migration(
        self, destination: "V8ReplicaResource"
    ) -> BackingMigrationTicket:
        """Create a non-authoritative destination while preserving old backing."""

        from .v8_leases import ReplicaLifecycle

        source = self.lease_manager.sources.get(destination.source_variant_id)
        if source is None or source.artifact_id != destination.artifact_id:
            raise RuntimeError("backing migration destination has wrong Source identity")
        old = [
            replica
            for replica in source.replicas.values()
            if replica.healthy and replica.is_backing
        ]
        if len(old) != 1:
            raise RuntimeError("backing migration requires one healthy old backing")
        old_replica = old[0]
        if old_replica.busy or source.logical_lease_refcount:
            raise RuntimeError("busy Source backing cannot migrate")
        if destination.tier is old_replica.tier:
            raise RuntimeError("exclusive backing migration must change tier")
        if destination.is_backing:
            raise ValueError("migration destination is non-authoritative until commit")
        if destination.lifecycle is not ReplicaLifecycle.ALLOCATING:
            raise ValueError("migration destination must start ALLOCATING")
        self.lease_manager.register_replica(destination)
        old_replica.copy_in_flight += 1
        destination.copy_in_flight += 1
        return BackingMigrationTicket(
            source.source_variant_id,
            old_replica.replica_id,
            destination.replica_id,
            old_replica.tier,
            destination.tier,
            old_replica.generation,
            destination.generation,
        )

    def finish_backing_migration(
        self,
        ticket: BackingMigrationTicket,
        *,
        copy_completed: bool,
        source_logical_digest: str,
        destination_logical_digest: str,
    ) -> None:
        """Atomically publish a verified destination or preserve the old backing."""

        from .v8_leases import ReplicaLifecycle

        source = self.lease_manager.sources[ticket.source_variant_id]
        old = source.replicas[ticket.old_replica_id]
        destination = source.replicas[ticket.destination_replica_id]
        if (
            old.generation != ticket.old_generation
            or destination.generation != ticket.destination_generation
            or old.tier is not ticket.old_tier
            or destination.tier is not ticket.destination_tier
        ):
            raise RuntimeError("stale backing migration ticket")
        valid = bool(
            copy_completed
            and source_logical_digest
            and source_logical_digest == destination_logical_digest
        )
        old.copy_in_flight -= 1
        destination.copy_in_flight -= 1
        if min(old.copy_in_flight, destination.copy_in_flight) < 0:
            raise RuntimeError("backing migration copy counter underflow")
        if not valid:
            destination.lifecycle = ReplicaLifecycle.DELETED
            # The original authoritative backing remains untouched.
            return
        if not old.is_backing or old.lifecycle is not ReplicaLifecycle.READY:
            raise RuntimeError("old backing changed before migration commit")
        # This block is the authoritative placement swap. Runtime integration
        # must execute it under the Replica registry lock.
        destination.lifecycle = ReplicaLifecycle.READY
        destination.is_backing = True
        old.is_backing = False
        old.lifecycle = ReplicaLifecycle.DELETED
        destination.placement_epoch = max(
            destination.placement_epoch, old.placement_epoch + 1
        )
        entry = self.backing_manager.entry(ticket.source_variant_id)
        entry.tier = destination.tier
        entry.backing_replica_id = destination.replica_id
        entry.last_access_order = self.backing_manager._tick()
        self.synchronize_busy()
