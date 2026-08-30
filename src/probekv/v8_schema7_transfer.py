from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Dict, Optional

from .v8_schema7_contracts import TransferPath


@dataclass(frozen=True)
class TransferCapabilities:
    gds_available: bool
    pinned_cpu_available: bool = True


@dataclass(frozen=True)
class TransferPlan:
    path: TransferPath
    source_tier: str
    requested_bytes: int
    requires_pinned_staging: bool
    gds_capability_verified: bool


class Schema7TransferPlanner:
    def choose(
        self,
        *,
        source_tier: str,
        requested_bytes: int,
        capabilities: TransferCapabilities,
        prefer_gds: bool = True,
    ) -> TransferPlan:
        if requested_bytes < 0:
            raise ValueError("transfer bytes must be non-negative")
        tier = source_tier.lower()
        if tier == "gpu":
            path = TransferPath.GPU_RESIDENT
        elif tier in {"cpu", "pinned_cpu"}:
            if not capabilities.pinned_cpu_available:
                raise RuntimeError("pinned CPU transfer path is unavailable")
            path = TransferPath.CPU_PINNED_TO_GPU
        elif tier == "ssd":
            path = (
                TransferPath.SSD_GDS_TO_GPU
                if prefer_gds and capabilities.gds_available
                else TransferPath.SSD_STAGED_TO_GPU
            )
        else:
            raise ValueError("unsupported Source Replica tier")
        return TransferPlan(
            path=path,
            source_tier=tier,
            requested_bytes=requested_bytes,
            requires_pinned_staging=path is TransferPath.SSD_STAGED_TO_GPU,
            gds_capability_verified=(
                path is TransferPath.SSD_GDS_TO_GPU and capabilities.gds_available
            ),
        )


@dataclass
class PinnedStagingLease:
    lease_id: str
    owner_request_id: str
    bytes_reserved: int
    double_buffer: bool
    released: bool = False


class PinnedStagingPool:
    """Global byte-accounted staging pool; tensor allocation stays runtime-owned."""

    def __init__(self, capacity_bytes: int = 2_147_483_648) -> None:
        if capacity_bytes <= 0:
            raise ValueError("pinned staging capacity must be positive")
        self.capacity_bytes = int(capacity_bytes)
        self._leases: Dict[str, PinnedStagingLease] = {}
        self._next_id = 1
        self._lock = RLock()

    @property
    def reserved_bytes(self) -> int:
        return sum(
            lease.bytes_reserved for lease in self._leases.values() if not lease.released
        )

    def acquire(
        self,
        *,
        owner_request_id: str,
        slot_bytes: int,
        double_buffer: bool = True,
    ) -> PinnedStagingLease:
        if not owner_request_id or slot_bytes <= 0:
            raise ValueError("invalid staging lease request")
        requested = int(slot_bytes) * (2 if double_buffer else 1)
        with self._lock:
            if self.reserved_bytes + requested > self.capacity_bytes:
                raise MemoryError("pinned staging pool is exhausted")
            lease = PinnedStagingLease(
                f"pinned-{self._next_id}", owner_request_id, requested, double_buffer
            )
            self._next_id += 1
            self._leases[lease.lease_id] = lease
            return lease

    def release(self, lease_id: str) -> None:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise KeyError("unknown pinned staging lease")
            if lease.released:
                raise RuntimeError("pinned staging lease was already released")
            lease.released = True

    def get(self, lease_id: str) -> Optional[PinnedStagingLease]:
        return self._leases.get(lease_id)
