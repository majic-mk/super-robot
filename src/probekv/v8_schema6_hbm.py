from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Tuple


GIB = 1024 ** 3


class HBMReservationKind(str, Enum):
    SELECTION_WORKSPACE = "selection_workspace"
    WINNER_PREFETCH = "winner_prefetch"
    COMMITTED_EXECUTION = "committed_execution"


@dataclass
class HBMReservation:
    reservation_id: str
    owner_request_id: str
    segment_id: str
    bytes: int
    kind: HBMReservationKind
    epoch: int
    released: bool = False


class UnifiedHBMReservationManager:
    """One allocator contract for comparison, prefetch and execution memory.

    ``allocator_free_reservable_bytes`` is already net of model/runtime memory
    and all active reservations managed here.  Callers must not subtract those
    reservations a second time.
    """

    def __init__(
        self,
        *,
        allocator_capacity_bytes: int,
        external_reserved_bytes: int = 0,
        safety_bytes: int = 4 * GIB,
    ) -> None:
        if min(allocator_capacity_bytes, external_reserved_bytes, safety_bytes) < 0:
            raise ValueError("HBM capacities must be non-negative")
        if external_reserved_bytes + safety_bytes > allocator_capacity_bytes:
            raise ValueError("HBM reservations exceed allocator capacity")
        self.allocator_capacity_bytes = int(allocator_capacity_bytes)
        self.external_reserved_bytes = int(external_reserved_bytes)
        self.safety_bytes = int(safety_bytes)
        self.epoch = 1
        self.reservations: Dict[str, HBMReservation] = {}

    @property
    def active_reserved_bytes(self) -> int:
        return sum(row.bytes for row in self.reservations.values() if not row.released)

    @property
    def allocator_free_reservable_bytes(self) -> int:
        return max(
            0,
            self.allocator_capacity_bytes
            - self.external_reserved_bytes
            - self.active_reserved_bytes,
        )

    @property
    def selector_lease_bytes(self) -> int:
        return max(0, self.allocator_free_reservable_bytes - self.safety_bytes)

    def reserve_batch(
        self,
        *,
        owner_request_id: str,
        rows: Iterable[Tuple[str, int, HBMReservationKind]],
    ) -> Tuple[HBMReservation, ...]:
        requests = tuple(rows)
        if not owner_request_id or not requests:
            raise ValueError("HBM reservation batch requires an owner and rows")
        if len({segment_id for segment_id, _, _ in requests}) != len(requests):
            raise ValueError("an HBM reservation batch may contain one row per Segment")
        if any(size <= 0 for _, size, _ in requests):
            raise ValueError("HBM reservation sizes must be positive")
        requested = sum(size for _, size, _ in requests)
        available = self.allocator_free_reservable_bytes - self.safety_bytes
        if requested > max(0, available):
            raise MemoryError("atomic HBM reservation batch exceeds safe headroom")
        self.epoch += 1
        created = []
        for segment_id, size, kind in requests:
            reservation = HBMReservation(
                str(uuid.uuid4()), owner_request_id, segment_id, int(size),
                HBMReservationKind(kind), self.epoch,
            )
            self.reservations[reservation.reservation_id] = reservation
            created.append(reservation)
        return tuple(created)

    def promote(
        self,
        reservation_id: str,
        *,
        expected: HBMReservationKind,
        target: HBMReservationKind,
    ) -> HBMReservation:
        reservation = self.reservations[reservation_id]
        if reservation.released or reservation.kind is not expected:
            raise RuntimeError("stale or incompatible HBM reservation promotion")
        if expected is not HBMReservationKind.WINNER_PREFETCH or target is not HBMReservationKind.COMMITTED_EXECUTION:
            raise RuntimeError("only winner-prefetch reservations may become execution reservations")
        self.epoch += 1
        reservation.kind = target
        reservation.epoch = self.epoch
        return reservation

    def release(self, reservation_id: str) -> None:
        reservation = self.reservations[reservation_id]
        if reservation.released:
            return
        reservation.released = True
        self.epoch += 1
