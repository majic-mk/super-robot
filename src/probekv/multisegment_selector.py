from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Set, Union

from .candidate_budget import RequestComparisonAllocation
from .contracts import CandidateBounds, SelectionReason, SourceDecision
from .selector import DynamicProbeSelector
from .v6_contracts import (
    RequestSelectionPlan,
    SelectionExecutionPolicy,
    SegmentSelectionDecision,
)


class MultiSegmentProbeSelector:
    """Run one conservative Source selector per segment.

    The request allocation is linear in the number of candidate summaries.
    Segments without comparison budget abstain rather than falling through to
    a metadata, latest, or default Source.
    """

    def __init__(
        self,
        selector: DynamicProbeSelector,
        selection_execution_policy: SelectionExecutionPolicy = (
            SelectionExecutionPolicy.LEGACY_COMMON_AFTER_SELECTION
        ),
    ) -> None:
        self.selector = selector
        self.selection_execution_policy = selection_execution_policy

    def _eligible_floors(
        self,
        ordered_segment_ids: Sequence[str],
        decisions: Mapping[str, SegmentSelectionDecision],
    ) -> Dict[str, int]:
        if self.selection_execution_policy is (
            SelectionExecutionPolicy.LEGACY_COMMON_AFTER_SELECTION
        ):
            return {}
        result: Dict[str, int] = {}
        for index, segment_id in enumerate(ordered_segment_ids):
            decision = decisions.get(segment_id)
            if decision is None or decision.source_decision.abstained:
                continue
            if self.selection_execution_policy is (
                SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP
            ):
                result[segment_id] = decision.source_decision.probe_layer + 1
                continue
            downstream = ordered_segment_ids[index:]
            if any(item not in decisions for item in downstream):
                continue
            result[segment_id] = 1 + max(
                decisions[item].source_decision.probe_layer for item in downstream
            )
        return result

    def select(
        self,
        request_id: str,
        allocation: RequestComparisonAllocation,
        bounds_by_segment_layer: Union[
            Mapping[str, Mapping[int, Sequence[CandidateBounds]]],
            Callable[[str, int], Sequence[CandidateBounds]],
        ],
        on_source_locked: Optional[
            Callable[[str, SourceDecision], None]
        ] = None,
        on_reuse_eligible: Optional[Callable[[str, int], None]] = None,
        probe_state_origin: Optional[str] = None,
    ) -> RequestSelectionPlan:
        decisions: Dict[str, SegmentSelectionDecision] = {}
        known_segments = {audit.segment_id for audit in allocation.audits}
        dynamic_provider = callable(bounds_by_segment_layer)
        if (
            self.selection_execution_policy is (
                SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP
            )
            and not dynamic_provider
        ):
            raise ValueError(
                "immediate staggered selection requires checkpoint-dynamic bounds"
            )
        if not dynamic_provider:
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
        first_checkpoint = self.selector.policy.checkpoints[0]
        for audit in allocation.audits:
            compared = set(audit.compared_source_ids)
            if not compared:
                source_decision = SourceDecision(
                    selected_source_id=None,
                    probe_layer=first_checkpoint,
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
                if not dynamic_provider:
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

        # Layer-major progress permits winner-only prefetch at lock time.  A
        # separate policy-controlled eligibility event determines whether the
        # winner may affect subsequent probe states.
        ordered_segment_ids = tuple(audit.segment_id for audit in allocation.audits)
        emitted_reuse: Set[str] = set()
        for layer in self.selector.policy.checkpoints:
            for segment_id in tuple(pending):
                compared = set(
                    audit_by_id[segment_id].compared_source_ids
                )
                checkpoint_bounds = (
                    tuple(bounds_by_segment_layer(segment_id, layer))
                    if dynamic_provider
                    else filtered_by_segment[segment_id].get(layer, ())
                )
                checkpoint_bounds = tuple(
                    candidate
                    for candidate in checkpoint_bounds
                    if candidate.source_id in compared
                )
                source_decision, covered = self.selector.evaluate_checkpoint(
                    checkpoint_bounds,
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

            floors = self._eligible_floors(ordered_segment_ids, decisions)
            if on_reuse_eligible is not None:
                for segment_id in ordered_segment_ids:
                    if segment_id in floors and segment_id not in emitted_reuse:
                        on_reuse_eligible(segment_id, floors[segment_id])
                        emitted_reuse.add(segment_id)

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
        floors = self._eligible_floors(ordered_segment_ids, decisions)
        if on_reuse_eligible is not None:
            for segment_id in ordered_segment_ids:
                if segment_id in floors and segment_id not in emitted_reuse:
                    on_reuse_eligible(segment_id, floors[segment_id])
        if probe_state_origin is None:
            if self.selection_execution_policy is (
                SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP
            ):
                raise ValueError(
                    "immediate staggered selection requires an explicit "
                    "policy-conditioned probe-state origin"
                )
            probe_state_origin = "dense_clean"
        return RequestSelectionPlan(
            request_id=request_id,
            segment_decisions=tuple(
                decisions[audit.segment_id] for audit in allocation.audits
            ),
            probe_ms=allocation.probe_ms,
            metadata_ms=allocation.metadata_ms,
            compare_ms=allocation.budget_used_ms,
            full_reference_ms=allocation.full_reference_ms,
            selection_execution_policy=self.selection_execution_policy,
            earliest_reuse_layer_by_segment=floors,
            probe_state_origin=probe_state_origin,
        )
