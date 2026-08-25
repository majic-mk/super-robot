from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from .candidate_budget import RequestComparisonAllocation
from .contracts import (
    CandidateBounds,
    ReuseAdmissionState,
    SourceDecision,
    SourceSelectionState,
)
from .multisegment_selector import MultiSegmentProbeSelector
from .v6_contracts import (
    RequestExecutionMode,
    RequestExecutionPlan,
    RequestRefinedCost,
    RequestSchedulingFeedback,
    RequestSelectionPlan,
    SelectionExecutionPolicy,
    SegmentExecutionDecision,
    SegmentExecutionPath,
)


class MultiSegmentClosedLoopRuntime(Protocol):
    """Runtime boundary for the v6 multi-segment closed loop."""

    def load_and_schedule(
        self, selection: RequestSelectionPlan
    ) -> RequestSchedulingFeedback:
        ...

    def measure_refined_cost(
        self,
        selection: RequestSelectionPlan,
        scheduling: RequestSchedulingFeedback,
        boundary: int,
        active_segment_ids: Tuple[str, ...],
    ) -> RequestRefinedCost:
        ...

    def measure_staggered_refined_cost(
        self,
        selection: RequestSelectionPlan,
        scheduling: RequestSchedulingFeedback,
        boundary_by_segment: Mapping[str, int],
        active_segment_ids: Tuple[str, ...],
    ) -> RequestRefinedCost:
        """Profile one non-Cartesian per-segment boundary plan."""
        ...

    def execute_reuse(
        self,
        selection: RequestSelectionPlan,
        execution: RequestExecutionPlan,
    ) -> Any:
        ...

    def execute_dense(
        self,
        selection: RequestSelectionPlan,
        execution: RequestExecutionPlan,
    ) -> Any:
        ...


class StreamingMultiSegmentRuntime(MultiSegmentClosedLoopRuntime, Protocol):
    probe_state_origin: str

    def on_source_locked(
        self, segment_id: str, decision: SourceDecision
    ) -> None:
        """Start or register winner-only prefetch at the lock checkpoint."""
        ...

    def on_reuse_eligible(self, segment_id: str, earliest_layer: int) -> None:
        """Allow policy A/C to expose a locked winner to the execution lane."""
        ...


@dataclass(frozen=True)
class MultiSegmentClosedLoopResult:
    selection: RequestSelectionPlan
    scheduling: Optional[RequestSchedulingFeedback]
    refined_cost: Optional[RequestRefinedCost]
    execution: RequestExecutionPlan
    runtime_result: Any

    def to_audit_record(self) -> Dict[str, Any]:
        execution_by_id = {
            item.segment_id: item for item in self.execution.segment_decisions
        }
        selection_rows = []
        for item in self.selection.segment_decisions:
            execution = execution_by_id[item.segment_id]
            source = item.source_decision
            selection_rows.append(
                {
                    "segment_id": item.segment_id,
                    "stored_k": item.comparison.stored_k,
                    "eligible_k": item.comparison.eligible_k,
                    "compared_k": item.comparison.compared_k,
                    "compared_source_ids": item.comparison.compared_source_ids,
                    "dropped_source_ids": item.comparison.dropped_source_ids,
                    "comparison_budget_used_ms": (
                        item.comparison.budget_used_ms
                    ),
                    "selected_source_id": source.selected_source_id,
                    "probe_layer": source.probe_layer,
                    "earliest_reuse_layer": (
                        self.selection.earliest_reuse_layer_by_segment.get(
                            item.segment_id
                        )
                    ),
                    "safe_repair_ratio_upper": source.safe_repair_ratio_upper,
                    "predicted_cost_upper_ms": source.predicted_cost_upper_ms,
                    "selection_state": source.selection_state.value,
                    "selection_reason": source.selection_reason.value,
                    "admission_state": execution.admission_state.value,
                    "execution_path": execution.path.value,
                    "actual_reuse_boundary": execution.actual_reuse_boundary,
                    "rejection_reason": execution.rejection_reason,
                }
            )
        scheduling_rows = []
        if self.scheduling is not None:
            for item in self.scheduling.segments:
                scheduling_rows.append(
                    {
                        "segment_id": item.segment_id,
                        "selected_source_id": item.selected_source_id,
                        "source_load_start_ms": item.source_load_start_ms,
                        "source_load_finish_ms": item.source_load_finish_ms,
                        "source_ready_ms": item.source_ready_ms,
                        "source_ready": item.source_ready,
                        "first_ready_layer": item.first_ready_layer,
                        "ready_through_layer": item.ready_through_layer,
                        "layer_ready_ms": item.layer_ready_ms,
                        "transferred_bytes": item.transferred_bytes,
                        "wasted_bytes": item.wasted_bytes,
                    }
                )
        return {
            "request_id": self.selection.request_id,
            "protocol_version": 6,
            "probe_ms": self.selection.probe_ms,
            "metadata_ms": self.selection.metadata_ms,
            "compare_ms": self.selection.compare_ms,
            "detected_segment_count": len(self.selection.segment_decisions),
            "selected_segment_count": len(self.selection.selected),
            "compared_segment_count": sum(
                item.comparison.compared_k > 0
                for item in self.selection.segment_decisions
            ),
            "loaded_segment_count": (
                len(self.scheduling.segments)
                if self.scheduling is not None
                else 0
            ),
            "ready_source_count": (
                sum(item.source_ready for item in self.scheduling.segments)
                if self.scheduling is not None
                else 0
            ),
            "accepted_segment_count": sum(
                item.path is SegmentExecutionPath.REUSE
                for item in self.execution.segment_decisions
            ),
            "dense_segment_count": sum(
                item.path is SegmentExecutionPath.DENSE
                for item in self.execution.segment_decisions
            ),
            "rejected_segment_count": sum(
                item.admission_state is ReuseAdmissionState.REJECTED
                for item in self.execution.segment_decisions
            ),
            "execution_mode": self.execution.mode.value,
            "selection_execution_policy": (
                self.selection.selection_execution_policy.value
            ),
            "probe_state_origin": self.selection.probe_state_origin,
            "actual_reuse_boundary": self.execution.actual_reuse_boundary,
            "actual_reuse_boundary_by_segment": dict(
                self.execution.actual_reuse_boundary_by_segment
            ),
            "transferred_bytes": self.execution.transferred_bytes,
            "wasted_loaded_bytes": self.execution.wasted_loaded_bytes,
            "refined_reuse_total_ms": (
                self.refined_cost.reuse_total_ms
                if self.refined_cost is not None
                else None
            ),
            "full_reference_ms": self.selection.full_reference_ms,
            "joint_quality_covered": (
                self.refined_cost.joint_quality_covered
                if self.refined_cost is not None
                else None
            ),
            "interference_accounting_mode": (
                self.refined_cost.interference_accounting_mode.value
                if self.refined_cost is not None
                else None
            ),
            "scheduled_step_finish_ms": (
                self.scheduling.scheduled_step_finish_ms
                if self.scheduling is not None
                else None
            ),
            "a_resume_ms": (
                self.scheduling.a_resume_ms
                if self.scheduling is not None
                else None
            ),
            "post_ready_blocking_ms": (
                self.scheduling.post_ready_blocking_ms
                if self.scheduling is not None
                else None
            ),
            "load_interference_ms": (
                self.scheduling.load_interference_ms
                if self.scheduling is not None
                else None
            ),
            "useful_a_dense_ms": (
                self.scheduling.useful_a_dense_ms
                if self.scheduling is not None
                else None
            ),
            "useful_other_request_work_ms": (
                self.scheduling.useful_other_request_work_ms
                if self.scheduling is not None
                else None
            ),
            "candidate_boundaries": (
                self.scheduling.candidate_boundaries
                if self.scheduling is not None
                else ()
            ),
            "refined_cost_components": (
                {
                    "probe_ms": self.refined_cost.probe_ms,
                    "metadata_ms": self.refined_cost.metadata_ms,
                    "compare_ms": self.refined_cost.compare_ms,
                    "visible_load_ms": self.refined_cost.visible_load_ms,
                    "post_ready_blocking_ms": (
                        self.refined_cost.post_ready_blocking_ms
                    ),
                    "load_interference_ms": (
                        self.refined_cost.load_interference_ms
                    ),
                    "repair_selection_ms": (
                        self.refined_cost.repair_selection_ms
                    ),
                    "repair_ms": self.refined_cost.repair_ms,
                    "remaining_ms": self.refined_cost.remaining_ms,
                    "marginal_saved_ms": dict(
                        self.refined_cost.marginal_saved_ms
                    ),
                    "repair_ratio_upper_by_segment": dict(
                        self.refined_cost.repair_ratio_upper_by_segment
                    ),
                }
                if self.refined_cost is not None
                else None
            ),
            "union_repair_mask_digest": getattr(
                self.runtime_result, "union_mask_digest", None
            ),
            "segments": selection_rows,
            "scheduling": scheduling_rows,
        }


class MultiSegmentReuseController:
    """Selector -> multi-source scheduler -> refined cost -> final admission."""

    def __init__(self, gamma: float = 0.8) -> None:
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma

    @staticmethod
    def _selected_map(selection: RequestSelectionPlan) -> Dict[str, str]:
        return {
            item.segment_id: str(item.source_decision.selected_source_id)
            for item in selection.selected
        }

    @classmethod
    def _validate_scheduling(
        cls,
        selection: RequestSelectionPlan,
        scheduling: RequestSchedulingFeedback,
    ) -> None:
        if scheduling.request_id != selection.request_id:
            raise ValueError("scheduler returned another request")
        expected = cls._selected_map(selection)
        observed = {
            item.segment_id: item.selected_source_id
            for item in scheduling.segments
        }
        if observed != expected:
            raise ValueError("scheduler changed or omitted locked Sources")

    @classmethod
    def _validate_refined(
        cls,
        selection: RequestSelectionPlan,
        refined: RequestRefinedCost,
        boundary: int,
        active: Tuple[str, ...],
    ) -> None:
        if refined.request_id != selection.request_id:
            raise ValueError("refined cost belongs to another request")
        if refined.boundary != boundary:
            raise ValueError("refined cost changed the common boundary")
        if tuple(refined.active_segment_ids) != tuple(active):
            raise ValueError("refined cost changed the active segment set")
        locked = cls._selected_map(selection)
        expected_sources = {segment_id: locked[segment_id] for segment_id in active}
        if dict(refined.selected_source_ids) != expected_sources:
            raise ValueError("refined cost changed a locked Source")
        if (
            refined.cost_origin != "request_arrival"
            or refined.cost_endpoint != "first_token_ready"
        ):
            raise ValueError("refined cost uses a different accounting scope")
        if abs(refined.full_reference_ms - selection.full_reference_ms) > 1e-6:
            raise ValueError("selection and admission use different dense references")

    @staticmethod
    def _dense_decisions(
        selection: RequestSelectionPlan,
        accepted: Sequence[str] = (),
        refined_ratios: Optional[Mapping[str, float]] = None,
        boundary_by_segment: Optional[Mapping[str, int]] = None,
    ) -> Tuple[SegmentExecutionDecision, ...]:
        accepted_set = set(accepted)
        refined_ratios = dict(refined_ratios or {})
        boundary_by_segment = dict(boundary_by_segment or {})
        result = []
        for item in selection.segment_decisions:
            source = item.source_decision
            if item.segment_id in accepted_set:
                result.append(
                    SegmentExecutionDecision(
                        segment_id=item.segment_id,
                        path=SegmentExecutionPath.REUSE,
                        selected_source_id=source.selected_source_id,
                        selection_state=SourceSelectionState.SELECTED,
                        admission_state=ReuseAdmissionState.ACCEPTED,
                        repair_ratio_upper=refined_ratios[item.segment_id],
                        actual_reuse_boundary=boundary_by_segment[item.segment_id],
                    )
                )
            else:
                selected = source.selected_source_id is not None
                result.append(
                    SegmentExecutionDecision(
                        segment_id=item.segment_id,
                        path=SegmentExecutionPath.DENSE,
                        selected_source_id=source.selected_source_id,
                        selection_state=(
                            SourceSelectionState.SELECTED
                            if selected
                            else SourceSelectionState.NOT_SELECTED
                        ),
                        admission_state=(
                            ReuseAdmissionState.REJECTED
                            if selected
                            else ReuseAdmissionState.NOT_EVALUATED
                        ),
                        repair_ratio_upper=(
                            source.safe_repair_ratio_upper if selected else None
                        ),
                        rejection_reason=(
                            "final_request_cost_or_readiness"
                            if selected
                            else source.selection_reason.value
                        ),
                    )
                )
        return tuple(result)

    def _full_plan(
        self,
        selection: RequestSelectionPlan,
        scheduling: Optional[RequestSchedulingFeedback],
        refined: Optional[RequestRefinedCost],
    ) -> RequestExecutionPlan:
        transferred = sum(
            item.transferred_bytes for item in scheduling.segments
        ) if scheduling is not None else 0
        return RequestExecutionPlan(
            request_id=selection.request_id,
            mode=RequestExecutionMode.FULL_RECOMPUTE,
            segment_decisions=self._dense_decisions(selection),
            actual_reuse_boundary=None,
            refined_cost=refined,
            transferred_bytes=transferred,
            wasted_loaded_bytes=transferred,
            actual_reuse_boundary_by_segment={},
        )

    def execute(
        self,
        selection: RequestSelectionPlan,
        runtime: MultiSegmentClosedLoopRuntime,
    ) -> MultiSegmentClosedLoopResult:
        if not selection.selected:
            execution = self._full_plan(selection, None, None)
            runtime_result = runtime.execute_dense(selection, execution)
            return MultiSegmentClosedLoopResult(
                selection, None, None, execution, runtime_result
            )

        scheduling = runtime.load_and_schedule(selection)
        self._validate_scheduling(selection, scheduling)
        feedback_by_id = {item.segment_id: item for item in scheduling.segments}
        feasible = []
        last_refined = None
        for boundary in scheduling.candidate_boundaries:
            active = tuple(
                sorted(
                    segment_id
                    for segment_id, feedback in feedback_by_id.items()
                    if feedback.ready_at(boundary)
                )
            )
            while active:
                refined = runtime.measure_refined_cost(
                    selection, scheduling, boundary, active
                )
                self._validate_refined(selection, refined, boundary, active)
                last_refined = refined
                positive = tuple(
                    segment_id for segment_id in active
                    if refined.marginal_saved_ms[segment_id] > 0
                )
                if positive != active:
                    active = positive
                    continue
                if (
                    refined.joint_quality_covered
                    and refined.reuse_total_ms
                    <= self.gamma * refined.full_reference_ms
                ):
                    feasible.append(
                        (
                            refined.reuse_total_ms,
                            boundary,
                            active,
                            refined,
                        )
                    )
                break

        if not feasible:
            execution = self._full_plan(selection, scheduling, last_refined)
            runtime_result = runtime.execute_dense(selection, execution)
            return MultiSegmentClosedLoopResult(
                selection, scheduling, last_refined, execution, runtime_result
            )

        _, boundary, accepted, refined = min(
            feasible,
            key=lambda item: (item[0], item[1], item[2]),
        )
        decisions = self._dense_decisions(
            selection,
            accepted,
            refined.repair_ratio_upper_by_segment,
            {segment_id: boundary for segment_id in accepted},
        )
        if len(accepted) == len(decisions):
            mode = RequestExecutionMode.ALL_REUSE
        else:
            mode = RequestExecutionMode.PARTIAL_REUSE
        accepted_set = set(accepted)
        transferred = sum(item.transferred_bytes for item in scheduling.segments)
        wasted = sum(
            (
                item.wasted_bytes
                if item.segment_id in accepted_set
                else item.transferred_bytes
            )
            for item in scheduling.segments
        )
        execution = RequestExecutionPlan(
            request_id=selection.request_id,
            mode=mode,
            segment_decisions=decisions,
            actual_reuse_boundary=boundary,
            refined_cost=refined,
            transferred_bytes=transferred,
            wasted_loaded_bytes=wasted,
            actual_reuse_boundary_by_segment={
                segment_id: boundary for segment_id in accepted
            },
        )
        runtime_result = runtime.execute_reuse(selection, execution)
        return MultiSegmentClosedLoopResult(
            selection, scheduling, refined, execution, runtime_result
        )


class StaggeredMultiSegmentReuseController(MultiSegmentReuseController):
    """Refined admission for the A/C per-segment boundary policies.

    Each selected segment receives exactly one earliest feasible boundary.  The
    controller never enumerates a Cartesian product of boundary choices; a
    runtime profile may prune segments with non-positive marginal saving, then
    the same request-level quality and gamma gates used by the common policy
    decide final reuse.
    """

    @classmethod
    def _validate_staggered_refined(
        cls,
        selection: RequestSelectionPlan,
        refined: RequestRefinedCost,
        boundaries: Mapping[str, int],
        active: Tuple[str, ...],
    ) -> None:
        if refined.request_id != selection.request_id:
            raise ValueError("refined cost belongs to another request")
        if tuple(refined.active_segment_ids) != tuple(active):
            raise ValueError("refined cost changed the active segment set")
        expected_boundaries = {
            segment_id: int(boundaries[segment_id]) for segment_id in active
        }
        if dict(refined.boundary_by_segment) != expected_boundaries:
            raise ValueError("refined cost changed staggered reuse boundaries")
        locked = cls._selected_map(selection)
        if dict(refined.selected_source_ids) != {
            segment_id: locked[segment_id] for segment_id in active
        }:
            raise ValueError("refined cost changed a locked Source")
        if (
            refined.cost_origin != "request_arrival"
            or refined.cost_endpoint != "first_token_ready"
        ):
            raise ValueError("refined cost uses a different accounting scope")
        if abs(refined.full_reference_ms - selection.full_reference_ms) > 1e-6:
            raise ValueError("selection and admission use different dense references")

    def execute(
        self,
        selection: RequestSelectionPlan,
        runtime: MultiSegmentClosedLoopRuntime,
    ) -> MultiSegmentClosedLoopResult:
        if selection.selection_execution_policy not in {
            SelectionExecutionPolicy.CAUSAL_COMMIT_WAIT,
            SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP,
        }:
            raise ValueError("staggered controller requires policy A or C")
        if not selection.selected:
            execution = self._full_plan(selection, None, None)
            runtime_result = runtime.execute_dense(selection, execution)
            return MultiSegmentClosedLoopResult(
                selection, None, None, execution, runtime_result
            )

        scheduling = runtime.load_and_schedule(selection)
        self._validate_scheduling(selection, scheduling)
        floors = dict(selection.earliest_reuse_layer_by_segment)
        boundaries: Dict[str, int] = {}
        for feedback in scheduling.segments:
            if not feedback.source_ready:
                continue
            boundary = max(
                floors[feedback.segment_id], feedback.first_ready_layer
            )
            if boundary <= feedback.ready_through_layer:
                boundaries[feedback.segment_id] = boundary

        active = tuple(
            item.segment_id
            for item in selection.segment_decisions
            if item.segment_id in boundaries
        )
        last_refined = None
        while active:
            active_boundaries = {
                segment_id: boundaries[segment_id] for segment_id in active
            }
            refined = runtime.measure_staggered_refined_cost(
                selection, scheduling, active_boundaries, active
            )
            self._validate_staggered_refined(
                selection, refined, active_boundaries, active
            )
            last_refined = refined
            positive = tuple(
                segment_id
                for segment_id in active
                if refined.marginal_saved_ms[segment_id] > 0
            )
            if positive != active:
                active = positive
                continue
            break

        if (
            not active
            or last_refined is None
            or not last_refined.joint_quality_covered
            or last_refined.reuse_total_ms
            > self.gamma * last_refined.full_reference_ms
        ):
            execution = self._full_plan(selection, scheduling, last_refined)
            runtime_result = runtime.execute_dense(selection, execution)
            return MultiSegmentClosedLoopResult(
                selection, scheduling, last_refined, execution, runtime_result
            )

        accepted_boundaries = {
            segment_id: boundaries[segment_id] for segment_id in active
        }
        decisions = self._dense_decisions(
            selection,
            active,
            last_refined.repair_ratio_upper_by_segment,
            accepted_boundaries,
        )
        mode = (
            RequestExecutionMode.ALL_REUSE
            if len(active) == len(decisions)
            else RequestExecutionMode.PARTIAL_REUSE
        )
        accepted_set = set(active)
        transferred = sum(item.transferred_bytes for item in scheduling.segments)
        wasted = sum(
            (
                item.wasted_bytes
                if item.segment_id in accepted_set
                else item.transferred_bytes
            )
            for item in scheduling.segments
        )
        unique_boundaries = set(accepted_boundaries.values())
        common_boundary = (
            next(iter(unique_boundaries)) if len(unique_boundaries) == 1 else None
        )
        execution = RequestExecutionPlan(
            request_id=selection.request_id,
            mode=mode,
            segment_decisions=decisions,
            actual_reuse_boundary=common_boundary,
            refined_cost=last_refined,
            transferred_bytes=transferred,
            wasted_loaded_bytes=wasted,
            actual_reuse_boundary_by_segment=accepted_boundaries,
        )
        runtime_result = runtime.execute_reuse(selection, execution)
        return MultiSegmentClosedLoopResult(
            selection, scheduling, last_refined, execution, runtime_result
        )


class MultiSegmentOnlinePipeline:
    """Layer-major selection with immediate lock events and final closure."""

    def __init__(
        self,
        selector: MultiSegmentProbeSelector,
        controller: MultiSegmentReuseController,
    ) -> None:
        self.selector = selector
        self.controller = controller

    def execute(
        self,
        request_id: str,
        allocation: RequestComparisonAllocation,
        bounds_by_segment_layer: Union[
            Mapping[str, Mapping[int, Sequence[CandidateBounds]]],
            Callable[[str, int], Sequence[CandidateBounds]],
        ],
        runtime: StreamingMultiSegmentRuntime,
    ) -> MultiSegmentClosedLoopResult:
        policy = self.selector.selection_execution_policy
        reuse_callback = None
        probe_state_origin = "dense_clean"
        if policy is not SelectionExecutionPolicy.LEGACY_COMMON_AFTER_SELECTION:
            reuse_callback = getattr(runtime, "on_reuse_eligible", None)
            if reuse_callback is None:
                raise RuntimeError(
                    "A/C selection execution requires a reuse-eligibility runtime hook"
                )
            probe_state_origin = str(
                getattr(runtime, "probe_state_origin", "")
            )
        selection = self.selector.select(
            request_id,
            allocation,
            bounds_by_segment_layer,
            on_source_locked=runtime.on_source_locked,
            on_reuse_eligible=reuse_callback,
            probe_state_origin=probe_state_origin,
        )
        return self.controller.execute(selection, runtime)
