from __future__ import annotations

from typing import Mapping, Sequence

from .v8_schema8_contracts import DenseSelectionBarrierDecision


def close_dense_selection_barrier(
    *,
    segment_ids: Sequence[str],
    resolved_completed_depth_by_segment: Mapping[str, int],
) -> DenseSelectionBarrierDecision:
    inventory = tuple(str(value) for value in segment_ids)
    if not inventory or len(set(inventory)) != len(inventory):
        raise ValueError("dense selection barrier requires a unique inventory")
    if set(resolved_completed_depth_by_segment) - set(inventory):
        raise ValueError("barrier resolution references an unknown Segment")
    unresolved = tuple(
        segment_id
        for segment_id in inventory
        if segment_id not in resolved_completed_depth_by_segment
    )
    # An unresolved d=1 Segment receives the single allowed d=2 rescue pass.
    completed = dict(resolved_completed_depth_by_segment)
    for segment_id in unresolved:
        completed[segment_id] = 2
    barrier_depth = max(completed.values())
    return DenseSelectionBarrierDecision(
        completed,
        barrier_depth,
        barrier_depth + 1,
        unresolved,
    )
