from __future__ import annotations

from dataclasses import dataclass

from .v8_schema6_hbm import (
    HBMReservation,
    HBMReservationKind,
    UnifiedHBMReservationManager,
)


@dataclass(frozen=True)
class SelectionWorkspacePlan:
    compared_k: int
    microbatch_k: int
    one_shot: bool
    current_state_bytes: int
    per_source_state_bytes: int
    reserved_bytes: int
    reservation: HBMReservation


def acquire_elastic_selection_workspace(
    manager: UnifiedHBMReservationManager,
    *,
    owner_request_id: str,
    segment_id: str,
    compared_k: int,
    current_state_bytes: int,
    per_source_state_bytes: int,
) -> SelectionWorkspacePlan:
    if compared_k < 1 or min(current_state_bytes, per_source_state_bytes) <= 0:
        raise ValueError("selection workspace geometry is invalid")
    available = manager.selector_lease_bytes
    all_bytes = current_state_bytes + compared_k * per_source_state_bytes
    if all_bytes <= available:
        microbatch_k = compared_k
    else:
        microbatch_k = (available - current_state_bytes) // per_source_state_bytes
        if microbatch_k < 1:
            raise MemoryError("safe HBM headroom cannot compare even one Source")
        microbatch_k = min(compared_k, int(microbatch_k))
    reserved = current_state_bytes + microbatch_k * per_source_state_bytes
    reservation = manager.reserve_batch(
        owner_request_id=owner_request_id,
        rows=((segment_id, reserved, HBMReservationKind.SELECTION_WORKSPACE),),
    )[0]
    return SelectionWorkspacePlan(
        compared_k,
        microbatch_k,
        microbatch_k == compared_k,
        current_state_bytes,
        per_source_state_bytes,
        reserved,
        reservation,
    )
