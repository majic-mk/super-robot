from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from .contracts import KVLocation


@dataclass
class TieredBackingEntry:
    source_variant_id: str
    size_bytes: int
    tier: KVLocation
    last_access_order: int
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


class Schema8TieredBackingManager:
    """One backing copy with CPU preference and independent per-tier LRU.

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
