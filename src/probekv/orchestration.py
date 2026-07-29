from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Protocol

from .contracts import ExecutionDecision, SourceDecision
from .cost import (
    DynamicReusePlanner,
    LayerOption,
    ReusePlan,
    finalize_execution,
)


class ClosedLoopPolicy(str, Enum):
    """Execution protocols.

    ``LEGACY_PRE_SCHEDULE_ADMISSION`` is retained only to reproduce artifacts
    produced before protocol v4.  New experiments must use
    ``TWO_STAGE_REFINED_ADMISSION``.
    """

    LEGACY_PRE_SCHEDULE_ADMISSION = "legacy_pre_schedule_admission"
    TWO_STAGE_REFINED_ADMISSION = "two_stage_refined_admission"


@dataclass(frozen=True)
class SchedulingFeedback:
    selected_source_id: str
    evaluated_reuse_boundary: int
    source_ready: bool
    load_ms: float
    overlap_ms: float
    source_ready_ms: float
    scheduled_step_finish_ms: float
    a_resume_ms: float
    post_ready_blocking_ms: float
    load_interference_ms: float = 0.0
    useful_a_dense_ms: float = 0.0
    useful_other_request_work_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.selected_source_id:
            raise ValueError("scheduling feedback requires a selected source")
        if self.evaluated_reuse_boundary < 1:
            raise ValueError("reuse boundary must be 1-based")
        if min(
            self.load_ms,
            self.overlap_ms,
            self.source_ready_ms,
            self.scheduled_step_finish_ms,
            self.a_resume_ms,
            self.post_ready_blocking_ms,
            self.load_interference_ms,
            self.useful_a_dense_ms,
            self.useful_other_request_work_ms,
        ) < 0:
            raise ValueError("scheduler measurements must be non-negative")
        if self.a_resume_ms + 1e-12 < self.source_ready_ms:
            raise ValueError("A cannot resume before source-ready")
        if abs(
            self.post_ready_blocking_ms
            - (self.a_resume_ms - self.source_ready_ms)
        ) > 1e-6:
            raise ValueError(
                "post-ready blocking must equal A resume minus source-ready"
            )


@dataclass(frozen=True)
class RefinedCostMeasurement:
    selected_source_id: str
    evaluated_reuse_boundary: int
    repair_ratio_upper: float
    probe_ms: float
    compare_ms: float
    repair_ms: float
    full_ms: float

    def __post_init__(self) -> None:
        if not self.selected_source_id:
            raise ValueError("refined cost requires a selected source")
        if self.evaluated_reuse_boundary < 1:
            raise ValueError("reuse boundary must be 1-based")
        if not 0.0 <= self.repair_ratio_upper <= 1.0:
            raise ValueError("repair ratio must be in [0, 1]")
        if min(
            self.probe_ms,
            self.compare_ms,
            self.repair_ms,
            self.full_ms,
        ) < 0:
            raise ValueError("refined cost measurements must be non-negative")
        if self.full_ms <= 0:
            raise ValueError("full recomputation time must be positive")


class ClosedLoopRuntime(Protocol):
    """Runtime boundary used by the two-stage controller.

    Loading and scheduling happen before cost refinement.  Implementations
    must not execute selective reuse from ``load_and_schedule``.
    """

    def load_and_schedule(
        self, selection: SourceDecision
    ) -> SchedulingFeedback:
        ...

    def measure_refined_cost(
        self,
        selection: SourceDecision,
        scheduling: SchedulingFeedback,
    ) -> RefinedCostMeasurement:
        ...

    def execute_reuse(
        self,
        selection: SourceDecision,
        decision: ExecutionDecision,
    ) -> Any:
        ...

    def execute_full(
        self,
        selection: SourceDecision,
        decision: ExecutionDecision,
    ) -> Any:
        ...


@dataclass(frozen=True)
class ClosedLoopResult:
    selection: SourceDecision
    scheduling: Optional[SchedulingFeedback]
    refined_cost: Optional[RefinedCostMeasurement]
    execution: ExecutionDecision
    runtime_result: Any
    closure_policy: ClosedLoopPolicy

    def to_audit_record(self) -> dict:
        timing = self.execution.timing
        return {
            "closure_policy": self.closure_policy.value,
            "selection_state": self.execution.selection_state.value,
            "probe_admission_state": self.selection.admission_state.value,
            "admission_state": self.execution.admission_state.value,
            "selected_source_id": self.execution.selected_source_id,
            "selection_reason": self.execution.selection_reason.value,
            "reuse_accepted": self.execution.reuse_accepted,
            "execution_mode": self.execution.execution_mode.value,
            "rejection_reason": (
                self.execution.rejection_reason.value
                if self.execution.rejection_reason is not None
                else None
            ),
            "predicted_cost_upper_ms": (
                self.selection.predicted_cost_upper_ms
            ),
            "predicted_repair_ratio_upper": (
                self.selection.safe_repair_ratio_upper
            ),
            "evaluated_reuse_boundary": (
                self.scheduling.evaluated_reuse_boundary
                if self.scheduling is not None
                else None
            ),
            "actual_reuse_boundary": (
                self.execution.actual_reuse_boundary
            ),
            "source_ready": (
                self.scheduling.source_ready
                if self.scheduling is not None
                else None
            ),
            "source_ready_ms": (
                self.scheduling.source_ready_ms
                if self.scheduling is not None
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
            "load_ms": timing.load_ms if timing is not None else None,
            "overlap_ms": timing.overlap_ms if timing is not None else None,
            "visible_load_ms": (
                timing.visible_load_ms if timing is not None else None
            ),
            "load_interference_ms": (
                timing.load_interference_ms if timing is not None else None
            ),
            "repair_ms": timing.repair_ms if timing is not None else None,
            "full_ms": timing.full_ms if timing is not None else None,
            "refined_reuse_total_ms": (
                timing.reuse_total_ms if timing is not None else None
            ),
        }


class TwoStageReuseController:
    """Enforce selector -> scheduler -> refined cost -> final admission."""

    def __init__(
        self,
        gamma: float = 0.8,
        policy: ClosedLoopPolicy = (
            ClosedLoopPolicy.TWO_STAGE_REFINED_ADMISSION
        ),
    ) -> None:
        self.planner = DynamicReusePlanner(gamma)
        self.policy = policy

    @staticmethod
    def _validate_feedback(
        selection: SourceDecision,
        scheduling: SchedulingFeedback,
        refined: RefinedCostMeasurement,
    ) -> None:
        source_id = selection.selected_source_id
        if source_id is None:
            raise ValueError("abstention cannot produce runtime feedback")
        if scheduling.selected_source_id != source_id:
            raise ValueError("scheduler returned feedback for another source")
        if refined.selected_source_id != source_id:
            raise ValueError("cost model refined another source")
        if (
            refined.evaluated_reuse_boundary
            != scheduling.evaluated_reuse_boundary
        ):
            raise ValueError(
                "scheduler and cost planner disagree on reuse boundary"
            )

    @staticmethod
    def _refined_option(
        scheduling: SchedulingFeedback,
        refined: RefinedCostMeasurement,
    ) -> LayerOption:
        return LayerOption(
            layer=scheduling.evaluated_reuse_boundary,
            repair_ratio_upper=refined.repair_ratio_upper,
            probe_ms=refined.probe_ms,
            compare_ms=refined.compare_ms,
            # load_ms is elapsed copy time including measured interference.
            load_ms=scheduling.load_ms,
            overlap_ms=scheduling.overlap_ms,
            repair_ms=refined.repair_ms,
            full_ms=refined.full_ms,
            buffer_ready=scheduling.source_ready,
            post_ready_blocking_ms=scheduling.post_ready_blocking_ms,
            load_interference_ms=scheduling.load_interference_ms,
            source_ready_ms=scheduling.source_ready_ms,
            a_resume_ms=scheduling.a_resume_ms,
            scheduled_step_finish_ms=(
                scheduling.scheduled_step_finish_ms
            ),
        )

    def execute(
        self,
        selection: SourceDecision,
        runtime: ClosedLoopRuntime,
        legacy_plan: Optional[ReusePlan] = None,
    ) -> ClosedLoopResult:
        # This branch is intentionally before every runtime call: abstention
        # can never load, schedule, or fall through to latest/default Source.
        if selection.abstained:
            execution = finalize_execution(selection)
            result = runtime.execute_full(selection, execution)
            return ClosedLoopResult(
                selection,
                None,
                None,
                execution,
                result,
                self.policy,
            )

        if self.policy is ClosedLoopPolicy.LEGACY_PRE_SCHEDULE_ADMISSION:
            if legacy_plan is None:
                raise ValueError("legacy closure requires a pre-schedule plan")
            provisional = finalize_execution(selection, legacy_plan)
            if not provisional.reuse_accepted:
                result = runtime.execute_full(selection, provisional)
                return ClosedLoopResult(
                    selection,
                    None,
                    None,
                    provisional,
                    result,
                    self.policy,
                )
            scheduling = runtime.load_and_schedule(selection)
            # Deliberately reproduce the legacy behavior: scheduling feedback
            # is recorded but does not alter its already-final admission.
            result = runtime.execute_reuse(selection, provisional)
            return ClosedLoopResult(
                selection,
                scheduling,
                None,
                provisional,
                result,
                self.policy,
            )

        scheduling = runtime.load_and_schedule(selection)
        refined = runtime.measure_refined_cost(selection, scheduling)
        self._validate_feedback(selection, scheduling, refined)
        plan = self.planner.plan(
            [self._refined_option(scheduling, refined)]
        )
        execution = finalize_execution(selection, plan)
        if execution.reuse_accepted:
            result = runtime.execute_reuse(selection, execution)
        else:
            result = runtime.execute_full(selection, execution)
        return ClosedLoopResult(
            selection,
            scheduling,
            refined,
            execution,
            result,
            self.policy,
        )
