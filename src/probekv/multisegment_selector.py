from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence

from .candidate_budget import RequestComparisonAllocation
from .contracts import CandidateBounds, SelectionReason, SourceDecision
from .selector import DynamicProbeSelector
from .v6_contracts import (
    RequestSelectionPlan,
    SegmentSelectionDecision,
)


class MultiSegmentProbeSelector:
    """Run one conservative Source selector per segment.

    The request allocation is linear in the number of candidate summaries.
    Segments without comparison budget abstain rather than falling through to
    a metadata, latest, or default Source.
    """

    def __init__(self, selector: DynamicProbeSelector) -> None:
        self.selector = selector

    def select(
        self,
        request_id: str,
        allocation: RequestComparisonAllocation,
        bounds_by_segment_layer: Mapping[
            str, Mapping[int, Sequence[CandidateBounds]]
        ],
        on_source_locked: Optional[
            Callable[[str, SourceDecision], None]
        ] = None,
    ) -> RequestSelectionPlan:
        decisions: Dict[str, SegmentSelectionDecision] = {}
        known_segments = {audit.segment_id for audit in allocation.audits}
        unexpected = set(bounds_by_segment_layer) - known_segments
        if unexpected:
            raise ValueError(
                "bounds supplied for unallocated segments: %s"
                % ", ".join(sorted(unexpected))
            )
        audit_by_id = {audit.segment_id: audit for audit in allocation.audits}
        filtered_by_segment = {}
        pending = []
        saw_quality = {}
        for audit in allocation.audits:
            compared = set(audit.compared_source_ids)
            if not compared:
                source_decision = SourceDecision(
                    selected_source_id=None,
                    probe_layer=self.selector.policy.max_layer,
                    reuse_layer=None,
                    safe_repair_ratio_upper=None,
                    prefetch_m=0,
                    selection_reason=(
                        SelectionReason.NO_QUALITY_SAFE_SOURCE
                        if audit.eligible_k == 0
                        else SelectionReason.COMPARISON_BUDGET_EXHAUSTED
                    ),
                )
                decisions[audit.segment_id] = SegmentSelectionDecision(
                    audit.segment_id, source_decision, audit
                )
            else:
                filtered = {}
                for layer, bounds in bounds_by_segment_layer.get(
                    audit.segment_id, {}
                ).items():
                    filtered[layer] = tuple(
                        candidate
                        for candidate in bounds
                        if candidate.source_id in compared
                    )
                filtered_by_segment[audit.segment_id] = filtered
                pending.append(audit.segment_id)
                saw_quality[audit.segment_id] = False

        # Layer-major progress permits an already-locked segment to start its
        # winner-only prefetch while unresolved segments continue probing.
        for layer in self.selector.policy.checkpoints:
            for segment_id in tuple(pending):
                source_decision, covered = self.selector.evaluate_checkpoint(
                    filtered_by_segment[segment_id].get(layer, ()),
                    layer,
                    allocation.full_reference_ms,
                    saw_quality[segment_id],
                )
                saw_quality[segment_id] = covered
                if source_decision is None:
                    continue
                audit = audit_by_id[segment_id]
                decisions[segment_id] = SegmentSelectionDecision(
                    segment_id, source_decision, audit
                )
                pending.remove(segment_id)
                if (
                    on_source_locked is not None
                    and not source_decision.abstained
                ):
                    on_source_locked(segment_id, source_decision)

        for segment_id in pending:
            reason = (
                SelectionReason.MAX_PROBE_UNCERTAIN
                if saw_quality[segment_id]
                else SelectionReason.NO_QUALITY_SAFE_SOURCE
            )
            source_decision = SourceDecision(
                None,
                self.selector.policy.checkpoints[-1],
                None,
                None,
                0,
                reason,
            )
            audit = audit_by_id[segment_id]
            decisions[segment_id] = SegmentSelectionDecision(
                segment_id, source_decision, audit
            )
        return RequestSelectionPlan(
            request_id=request_id,
            segment_decisions=tuple(
                decisions[audit.segment_id] for audit in allocation.audits
            ),
            probe_ms=allocation.probe_ms,
            metadata_ms=allocation.metadata_ms,
            compare_ms=allocation.budget_used_ms,
            full_reference_ms=allocation.full_reference_ms,
        )
