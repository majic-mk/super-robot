from __future__ import annotations

from typing import Mapping, Sequence

from .v8_schema8_contracts import BarrierResolution, DenseSelectionBarrierDecision


def close_dense_selection_barrier(
    *,
    segment_ids: Sequence[str],
    resolved_completed_depth_by_segment: Mapping[str, int],
    source_frozen_segment_ids: Sequence[str],
    abstained_segment_ids: Sequence[str],
) -> DenseSelectionBarrierDecision:
    inventory = tuple(str(value) for value in segment_ids)
    if not inventory or len(set(inventory)) != len(inventory):
        raise ValueError("dense selection barrier requires a unique inventory")
    if set(resolved_completed_depth_by_segment) != set(inventory):
        raise ValueError("barrier close requires a final depth for every Segment")
    frozen = set(source_frozen_segment_ids)
    abstained = set(abstained_segment_ids)
    if frozen & abstained or frozen | abstained != set(inventory):
        raise ValueError("barrier close requires one explicit terminal resolution")
    if set(resolved_completed_depth_by_segment) - set(inventory):
        raise ValueError("barrier resolution references an unknown Segment")
    completed = dict(resolved_completed_depth_by_segment)
    rescued = tuple(
        segment_id for segment_id in inventory if completed[segment_id] == 2
    )
    barrier_depth = max(completed.values())
    return DenseSelectionBarrierDecision(
        completed,
        {
            segment_id: (
                BarrierResolution.SOURCE_FROZEN
                if segment_id in frozen
                else BarrierResolution.ABSTAINED_DENSE
            )
            for segment_id in inventory
        },
        barrier_depth,
        barrier_depth + 1,
        rescued,
    )
