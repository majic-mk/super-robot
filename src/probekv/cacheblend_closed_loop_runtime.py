from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Tuple

from .contracts import (
    ExecutionDecision,
    ExecutionMode,
    HistoricalSource,
    SourceDecision,
)
from .orchestration import (
    ClosedLoopRuntime,
    RefinedCostMeasurement,
    SchedulingFeedback,
)


class CacheBlendRuntimeState(str, Enum):
    IDLE = "idle"
    SCHEDULED = "scheduled"
    PROFILED = "profiled"
    EXECUTED = "executed"


@dataclass(frozen=True)
class CacheBlendRuntimeCapabilities:
    """Capabilities that the pinned CacheBlend fork must expose.

    A case-level ``generate()`` wrapper does not satisfy this contract.  The
    runtime must be able to pause a prefill at a layer boundary, overlap an
    asynchronous Source load with useful work, and return the resulting
    boundary and event timings.
    """

    backend_name: str
    async_source_loading: bool
    layer_resumable_prefill: bool
    scheduler_feedback: bool
    boundary_conditioned_profiles: bool
    canonical_sources_read_only: bool
    cuda_event_timing: bool

    def require_closed_loop(self, require_cuda_events: bool = True) -> None:
        missing = []
        for name in (
            "async_source_loading",
            "layer_resumable_prefill",
            "scheduler_feedback",
            "boundary_conditioned_profiles",
            "canonical_sources_read_only",
        ):
            if not getattr(self, name):
                missing.append(name)
        if require_cuda_events and not self.cuda_event_timing:
            missing.append("cuda_event_timing")
        if missing:
            raise RuntimeError(
                "CacheBlend runtime is not closed-loop capable: %s"
                % ", ".join(missing)
            )
        if not self.backend_name:
            raise ValueError("CacheBlend backend name must be explicit")


@dataclass(frozen=True)
class SourceLoadTicket:
    selected_source_id: str
    source_load_start_ms: float
    requested_bytes: int
    opaque_handle: Any = None

    def __post_init__(self) -> None:
        if not self.selected_source_id:
            raise ValueError("load ticket requires a selected Source")
        if self.source_load_start_ms < 0 or self.requested_bytes < 0:
            raise ValueError("invalid Source load ticket")


@dataclass(frozen=True)
class CacheBlendScheduleObservation:
    selected_source_id: str
    evaluated_reuse_boundary: int
    source_ready: bool
    source_ready_ms: float
    scheduled_step_finish_ms: float
    a_resume_ms: float
    overlap_ms: float
    load_interference_ms: float
    useful_a_dense_ms: float
    useful_other_request_work_ms: float
    probe_ms: float
    compare_ms: float
    transferred_bytes: int

    def __post_init__(self) -> None:
        if not self.selected_source_id:
            raise ValueError("schedule observation requires a Source")
        if self.evaluated_reuse_boundary < 1:
            raise ValueError("runtime boundary must be 1-based")
        if min(
            self.source_ready_ms,
            self.scheduled_step_finish_ms,
            self.a_resume_ms,
            self.overlap_ms,
            self.load_interference_ms,
            self.useful_a_dense_ms,
            self.useful_other_request_work_ms,
            self.probe_ms,
            self.compare_ms,
        ) < 0:
            raise ValueError("runtime schedule timings must be non-negative")
        if self.a_resume_ms + 1e-12 < self.source_ready_ms:
            raise ValueError("A cannot resume before the Source-ready event")
        if self.transferred_bytes < 0:
            raise ValueError("transferred bytes must be non-negative")

    @property
    def post_ready_blocking_ms(self) -> float:
        return self.a_resume_ms - self.source_ready_ms


@dataclass(frozen=True)
class CacheBlendBoundaryProfile:
    selected_source_id: str
    evaluated_reuse_boundary: int
    repair_ratio_upper: float
    repair_selection_ms_upper: float
    repair_ms_upper: float
    remaining_layer_ms_upper: float
    full_total_ms: float
    profile_key: str
    cost_origin: str = "request_arrival"
    cost_endpoint: str = "first_token_ready"
    future_timing_source: str = "a800_boundary_profile_upper"

    def __post_init__(self) -> None:
        if not self.selected_source_id:
            raise ValueError("boundary profile requires a Source")
        if self.evaluated_reuse_boundary < 1:
            raise ValueError("profile boundary must be 1-based")
        if not 0.0 <= self.repair_ratio_upper <= 1.0:
            raise ValueError("repair ratio upper must be in [0, 1]")
        if min(
            self.repair_selection_ms_upper,
            self.repair_ms_upper,
            self.remaining_layer_ms_upper,
        ) < 0:
            raise ValueError("profile costs must be non-negative")
        if self.full_total_ms <= 0:
            raise ValueError("full total profile must be positive")
        if not (
            self.profile_key
            and self.cost_origin
            and self.cost_endpoint
            and self.future_timing_source
        ):
            raise ValueError("boundary profile provenance must be explicit")


@dataclass(frozen=True)
class CacheBlendExecutionObservation:
    execution_mode: ExecutionMode
    selected_source_id: Optional[str]
    actual_reuse_boundary: Optional[int]
    started_ms: float
    first_token_ready_ms: float
    total_host_ms: float
    total_gpu_ms: float
    output_token_ids: Tuple[int, ...] = ()
    output_hash: str = ""
    source_digest_before: str = ""
    source_digest_after: str = ""
    wasted_loaded_bytes: int = 0

    def __post_init__(self) -> None:
        if min(
            self.started_ms,
            self.first_token_ready_ms,
            self.total_host_ms,
            self.total_gpu_ms,
        ) < 0:
            raise ValueError("execution timings must be non-negative")
        if self.first_token_ready_ms + 1e-12 < self.started_ms:
            raise ValueError("first token cannot precede execution start")
        if self.wasted_loaded_bytes < 0:
            raise ValueError("wasted loaded bytes must be non-negative")
        if self.execution_mode is ExecutionMode.REUSE:
            if self.selected_source_id is None:
                raise ValueError("reuse execution requires a Source")
            if self.actual_reuse_boundary is None:
                raise ValueError("reuse execution requires an actual boundary")
        elif self.actual_reuse_boundary is not None:
            raise ValueError("full execution cannot report a reuse boundary")
        if (
            self.source_digest_before
            and self.source_digest_after
            and self.source_digest_before != self.source_digest_after
        ):
            raise RuntimeError("CacheBlend mutated the canonical Source")

    @property
    def realized_ttft_ms(self) -> float:
        return self.first_token_ready_ms

    def to_audit_record(self) -> Mapping[str, Any]:
        return {
            "runtime_execution_mode": self.execution_mode.value,
            "runtime_selected_source_id": self.selected_source_id,
            "runtime_actual_reuse_boundary": self.actual_reuse_boundary,
            "runtime_started_ms": self.started_ms,
            "runtime_first_token_ready_ms": self.first_token_ready_ms,
            "runtime_total_host_ms": self.total_host_ms,
            "runtime_total_gpu_ms": self.total_gpu_ms,
            "runtime_realized_ttft_ms": self.realized_ttft_ms,
            "runtime_output_hash": self.output_hash,
            "runtime_wasted_loaded_bytes": self.wasted_loaded_bytes,
        }


class CacheBlendOnlineEngine(Protocol):
    """Stack-specific hooks implemented in the pinned CacheBlend fork."""

    def capabilities(self) -> CacheBlendRuntimeCapabilities:
        ...

    def begin_source_load(
        self, source: HistoricalSource
    ) -> SourceLoadTicket:
        ...

    def schedule_waiting_window(
        self,
        selection: SourceDecision,
        ticket: SourceLoadTicket,
    ) -> CacheBlendScheduleObservation:
        ...

    def profile_boundary(
        self,
        source: HistoricalSource,
        boundary: int,
        repair_ratio_upper: float,
    ) -> CacheBlendBoundaryProfile:
        ...

    def execute_selective_reuse(
        self,
        source: HistoricalSource,
        boundary: int,
        repair_ratio_upper: float,
    ) -> CacheBlendExecutionObservation:
        ...

    def execute_full_recompute(
        self,
        retained_source_id: Optional[str],
    ) -> CacheBlendExecutionObservation:
        ...


class CacheBlendClosedLoopRuntime(ClosedLoopRuntime):
    """Validate and adapt real CacheBlend events to the v5 controller.

    The adapter is deliberately request-scoped.  It locks the Source selected
    during probe, prevents a second Source from being introduced during cost
    refinement, and makes final admission the only path to selective reuse.
    """

    def __init__(
        self,
        engine: CacheBlendOnlineEngine,
        sources: Mapping[str, HistoricalSource],
        total_layers: int,
        require_cuda_events: bool = True,
    ) -> None:
        if total_layers <= 0:
            raise ValueError("total_layers must be positive")
        self.engine = engine
        self.sources = dict(sources)
        if set(self.sources) != {
            source.source_id for source in self.sources.values()
        }:
            raise ValueError("Source mapping keys must match source_id")
        for source in self.sources.values():
            source.validate_canonical()
        self.total_layers = total_layers
        self.capability_record = engine.capabilities()
        self.capability_record.require_closed_loop(require_cuda_events)
        self.state = CacheBlendRuntimeState.IDLE
        self._selected_source_id: Optional[str] = None
        self._ticket: Optional[SourceLoadTicket] = None
        self._observation: Optional[CacheBlendScheduleObservation] = None
        self._profile: Optional[CacheBlendBoundaryProfile] = None

    def _source_for_selection(
        self, selection: SourceDecision
    ) -> HistoricalSource:
        source_id = selection.selected_source_id
        if source_id is None:
            raise ValueError("abstention cannot load a CacheBlend Source")
        if source_id not in self.sources:
            raise ValueError("selector chose an unknown CacheBlend Source")
        if (
            self._selected_source_id is not None
            and self._selected_source_id != source_id
        ):
            raise RuntimeError("selected CacheBlend Source is already locked")
        return self.sources[source_id]

    def load_and_schedule(
        self, selection: SourceDecision
    ) -> SchedulingFeedback:
        if self.state is not CacheBlendRuntimeState.IDLE:
            raise RuntimeError("Source loading may occur only once")
        source = self._source_for_selection(selection)
        ticket = self.engine.begin_source_load(source)
        if ticket.selected_source_id != source.source_id:
            raise RuntimeError("loader returned a ticket for another Source")
        observation = self.engine.schedule_waiting_window(selection, ticket)
        if observation.selected_source_id != source.source_id:
            raise RuntimeError("scheduler observed another Source")
        if observation.evaluated_reuse_boundary < selection.probe_layer:
            raise RuntimeError("reuse boundary cannot precede probe exit")
        if observation.evaluated_reuse_boundary > self.total_layers:
            raise RuntimeError("reuse boundary exceeds the model")
        if observation.source_ready_ms + 1e-12 < ticket.source_load_start_ms:
            raise RuntimeError("Source-ready precedes load start")
        if observation.transferred_bytes > ticket.requested_bytes:
            raise RuntimeError("loader transferred more bytes than requested")
        load_ms = (
            observation.source_ready_ms - ticket.source_load_start_ms
        )
        if observation.overlap_ms > load_ms + 1e-6:
            raise RuntimeError("load overlap cannot exceed elapsed load time")
        self._selected_source_id = source.source_id
        self._ticket = ticket
        self._observation = observation
        self.state = CacheBlendRuntimeState.SCHEDULED
        return SchedulingFeedback(
            selected_source_id=source.source_id,
            evaluated_reuse_boundary=(
                observation.evaluated_reuse_boundary
            ),
            source_ready=observation.source_ready,
            load_ms=load_ms,
            overlap_ms=observation.overlap_ms,
            source_ready_ms=observation.source_ready_ms,
            scheduled_step_finish_ms=(
                observation.scheduled_step_finish_ms
            ),
            a_resume_ms=observation.a_resume_ms,
            post_ready_blocking_ms=(
                observation.post_ready_blocking_ms
            ),
            load_interference_ms=observation.load_interference_ms,
            useful_a_dense_ms=observation.useful_a_dense_ms,
            useful_other_request_work_ms=(
                observation.useful_other_request_work_ms
            ),
            source_load_start_ms=ticket.source_load_start_ms,
            source_load_bytes=observation.transferred_bytes,
        )

    def measure_refined_cost(
        self,
        selection: SourceDecision,
        scheduling: SchedulingFeedback,
    ) -> RefinedCostMeasurement:
        if self.state is not CacheBlendRuntimeState.SCHEDULED:
            raise RuntimeError(
                "refined cost requires completed load/scheduling"
            )
        source = self._source_for_selection(selection)
        if scheduling.selected_source_id != source.source_id:
            raise RuntimeError("refined scheduling changed the Source")
        if self._observation is None:
            raise RuntimeError("missing CacheBlend schedule observation")
        if (
            scheduling.evaluated_reuse_boundary
            != self._observation.evaluated_reuse_boundary
        ):
            raise RuntimeError("scheduler boundary changed before profiling")
        profile = self.engine.profile_boundary(
            source,
            scheduling.evaluated_reuse_boundary,
            float(selection.safe_repair_ratio_upper),
        )
        if profile.selected_source_id != source.source_id:
            raise RuntimeError("boundary profiler changed the Source")
        if (
            profile.evaluated_reuse_boundary
            != scheduling.evaluated_reuse_boundary
        ):
            raise RuntimeError("boundary profiler changed the boundary")
        if (
            abs(
                profile.repair_ratio_upper
                - float(selection.safe_repair_ratio_upper)
            )
            > 1e-12
        ):
            raise RuntimeError("boundary profiler changed the safe ratio")
        self._profile = profile
        self.state = CacheBlendRuntimeState.PROFILED
        return RefinedCostMeasurement(
            selected_source_id=source.source_id,
            evaluated_reuse_boundary=profile.evaluated_reuse_boundary,
            repair_ratio_upper=profile.repair_ratio_upper,
            probe_ms=self._observation.probe_ms,
            compare_ms=self._observation.compare_ms,
            repair_ms=profile.repair_ms_upper,
            full_ms=profile.full_total_ms,
            repair_selection_ms=profile.repair_selection_ms_upper,
            remaining_layer_ms=profile.remaining_layer_ms_upper,
            cost_origin=profile.cost_origin,
            cost_endpoint=profile.cost_endpoint,
            past_timing_source="cacheblend_runtime_events",
            future_timing_source=profile.future_timing_source,
            profile_key=profile.profile_key,
        )

    def _validate_execution_source(
        self,
        selection: SourceDecision,
        decision: ExecutionDecision,
    ) -> Optional[HistoricalSource]:
        if decision.selected_source_id != selection.selected_source_id:
            raise RuntimeError("execution decision changed the Source")
        if selection.selected_source_id is None:
            return None
        return self._source_for_selection(selection)

    def execute_reuse(
        self,
        selection: SourceDecision,
        decision: ExecutionDecision,
    ) -> CacheBlendExecutionObservation:
        if self.state is not CacheBlendRuntimeState.PROFILED:
            raise RuntimeError("reuse requires a refined boundary profile")
        source = self._validate_execution_source(selection, decision)
        if source is None or not decision.reuse_accepted:
            raise RuntimeError("rejected or source-free request cannot reuse")
        if decision.actual_reuse_boundary is None:
            raise RuntimeError("accepted reuse requires an actual boundary")
        observation = self.engine.execute_selective_reuse(
            source,
            decision.actual_reuse_boundary,
            float(decision.safe_repair_ratio_upper),
        )
        if observation.execution_mode is not ExecutionMode.REUSE:
            raise RuntimeError("CacheBlend did not execute the admitted reuse")
        if observation.selected_source_id != source.source_id:
            raise RuntimeError("CacheBlend executed reuse with another Source")
        if (
            observation.actual_reuse_boundary
            != decision.actual_reuse_boundary
        ):
            raise RuntimeError("CacheBlend executed another reuse boundary")
        self.state = CacheBlendRuntimeState.EXECUTED
        return observation

    def execute_full(
        self,
        selection: SourceDecision,
        decision: ExecutionDecision,
    ) -> CacheBlendExecutionObservation:
        if self.state is CacheBlendRuntimeState.EXECUTED:
            raise RuntimeError("request has already executed")
        source = self._validate_execution_source(selection, decision)
        if decision.reuse_accepted:
            raise RuntimeError("accepted reuse cannot execute full")
        observation = self.engine.execute_full_recompute(
            source.source_id if source is not None else None
        )
        if observation.execution_mode is not ExecutionMode.FULL_RECOMPUTE:
            raise RuntimeError("CacheBlend did not execute full recomputation")
        if observation.actual_reuse_boundary is not None:
            raise RuntimeError("full fallback reported a reuse boundary")
        wasted = 0
        if self._observation is not None:
            wasted = self._observation.transferred_bytes
        if observation.wasted_loaded_bytes != wasted:
            raise RuntimeError(
                "full fallback did not account for transferred Source bytes"
            )
        self.state = CacheBlendRuntimeState.EXECUTED
        return observation
