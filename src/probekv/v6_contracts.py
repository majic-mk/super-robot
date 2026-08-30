from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

from .contracts import (
    HistoricalSource,
    InterferenceAccountingMode,
    ReuseAdmissionState,
    SourceDecision,
    SourceSelectionState,
)
from .model_signature import validate_v6_model_signature
from .manifest import token_content_hash


class RegionKind(str, Enum):
    PREFIX_EXACT = "prefix_exact"
    REUSE_CANDIDATE = "reuse_candidate"
    DENSE = "dense"
    MANDATORY_SUFFIX = "mandatory_suffix"


class RequestExecutionMode(str, Enum):
    ALL_REUSE = "all_reuse"
    PARTIAL_REUSE = "partial_reuse"
    FULL_RECOMPUTE = "full_recompute"


class SegmentExecutionPath(str, Enum):
    PREFIX_EXACT = "prefix_exact"
    REUSE = "reuse"
    DENSE = "dense"


class SelectionExecutionPolicy(str, Enum):
    """How locked reuse interacts with still-unresolved later segments.

    ``LEGACY_COMMON_AFTER_SELECTION`` preserves the original v6 protocol.
    The two experiment policies are deliberately limited to the user-approved
    A/C alternatives; a shadow-dense probe path is not part of the protocol.
    """

    LEGACY_COMMON_AFTER_SELECTION = "legacy_common_after_selection"
    CAUSAL_COMMIT_WAIT = "causal_commit_wait"
    IMMEDIATE_STAGGERED_CLOSED_LOOP = "immediate_staggered_closed_loop"
    DENSE_SELECTION_BARRIER = "dense_selection_barrier"


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    kind: RegionKind
    start: int
    end: int
    segment_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("region_id must be non-empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("region token span must be non-empty")
        if self.kind is RegionKind.REUSE_CANDIDATE:
            if not self.segment_id:
                raise ValueError("reuse candidate region requires segment_id")
        elif self.segment_id is not None:
            raise ValueError("only reuse candidate regions may bind segment_id")

    @property
    def token_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SegmentSpec:
    segment_id: str
    order: int
    token_start: int
    token_end: int
    content_hash: str
    token_ids: Tuple[int, ...]
    sources: Tuple[HistoricalSource, ...] = field(default_factory=tuple)

    def validate(self, model_signature: str, max_sources: int = 16) -> None:
        if not self.segment_id or not self.content_hash:
            raise ValueError("segment identifiers must be non-empty")
        if self.order < 0:
            raise ValueError("segment order must be non-negative")
        if self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("segment token span must be non-empty")
        if len(self.token_ids) != self.token_end - self.token_start:
            raise ValueError("segment token IDs do not match its token span")
        if self.content_hash != token_content_hash(self.token_ids):
            raise ValueError("segment content hash does not match token IDs")
        if len(self.sources) > max_sources:
            raise ValueError("segment source count exceeds v6 maximum")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must be unique within a segment")
        context_ids = [source.context_id for source in self.sources]
        if len(context_ids) != len(set(context_ids)):
            raise ValueError("historical contexts must be independent")
        for source in self.sources:
            source.validate_canonical()
            if source.model_signature != model_signature:
                raise ValueError("segment source belongs to another model")
            if source.content_hash != self.content_hash:
                raise ValueError("segment source has a different content hash")
            if source.token_count != len(self.token_ids):
                raise ValueError("segment source token count mismatch")

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    model_signature: str
    token_ids: Tuple[int, ...]
    regions: Tuple[RegionSpec, ...]
    segments: Tuple[SegmentSpec, ...]
    exact_prefix_tokens: int = 0
    mandatory_suffix_tokens: int = 0
    current_context_id: str = ""

    def validate(self, max_sources_per_segment: int = 16) -> None:
        if not self.request_id or not self.model_signature:
            raise ValueError("request and model identifiers are required")
        validate_v6_model_signature(self.model_signature)
        if not self.token_ids:
            raise ValueError("request token IDs must not be empty")
        if not 0 <= self.exact_prefix_tokens <= len(self.token_ids):
            raise ValueError("invalid exact prefix length")
        if not 0 <= self.mandatory_suffix_tokens <= len(self.token_ids):
            raise ValueError("invalid mandatory suffix length")
        if not self.regions:
            raise ValueError("request requires ordered regions")
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region IDs must be unique")
        cursor = 0
        for region in self.regions:
            if region.start != cursor:
                raise ValueError("regions must cover the request contiguously")
            cursor = region.end
        if cursor != len(self.token_ids):
            raise ValueError("regions must cover every request token")
        prefix_regions = [
            region for region in self.regions
            if region.kind is RegionKind.PREFIX_EXACT
        ]
        if self.exact_prefix_tokens:
            if len(prefix_regions) != 1:
                raise ValueError("exact prefix requires exactly one prefix region")
            if prefix_regions[0].start != 0 or (
                prefix_regions[0].end != self.exact_prefix_tokens
            ):
                raise ValueError("prefix region does not match exact prefix length")
        elif prefix_regions:
            raise ValueError("zero exact prefix cannot have a prefix region")
        suffix_regions = [
            region for region in self.regions
            if region.kind is RegionKind.MANDATORY_SUFFIX
        ]
        if self.mandatory_suffix_tokens:
            if len(suffix_regions) != 1:
                raise ValueError("mandatory suffix requires one suffix region")
            suffix = suffix_regions[0]
            if suffix.end != len(self.token_ids) or (
                suffix.token_count != self.mandatory_suffix_tokens
            ):
                raise ValueError("suffix region does not match suffix length")
        elif suffix_regions:
            raise ValueError("zero suffix cannot have a suffix region")
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("segment IDs must be unique")
        if [segment.order for segment in self.segments] != list(
            range(len(self.segments))
        ):
            raise ValueError("segments must use dense zero-based ordering")
        by_id = {segment.segment_id: segment for segment in self.segments}
        candidate_regions = [
            region for region in self.regions
            if region.kind is RegionKind.REUSE_CANDIDATE
        ]
        if {region.segment_id for region in candidate_regions} != set(by_id):
            raise ValueError("reuse regions and segment specifications disagree")
        for region in candidate_regions:
            segment = by_id[str(region.segment_id)]
            if (region.start, region.end) != (
                segment.token_start,
                segment.token_end,
            ):
                raise ValueError("segment and region spans disagree")
        for segment in self.segments:
            segment.validate(self.model_signature, max_sources_per_segment)
            current_context_id = self.current_context_id or self.request_id
            if any(
                source.context_id == current_context_id
                for source in segment.sources
            ):
                raise ValueError(
                    "current request context cannot be a historical Source"
                )


@dataclass(frozen=True)
class VariantComparisonAudit:
    segment_id: str
    stored_k: int
    eligible_k: int
    compared_source_ids: Tuple[str, ...]
    dropped_source_ids: Tuple[str, ...]
    budget_used_ms: float
    budget_limit_ms: float

    def __post_init__(self) -> None:
        if min(self.stored_k, self.eligible_k) < 0:
            raise ValueError("candidate counts must be non-negative")
        if self.eligible_k > self.stored_k:
            raise ValueError("eligible candidates cannot exceed stored candidates")
        if len(self.compared_source_ids) > self.eligible_k:
            raise ValueError("compared candidates exceed eligible candidates")
        if len(self.compared_source_ids) + len(self.dropped_source_ids) != (
            self.stored_k
        ):
            raise ValueError("comparison audit does not cover stored candidates")
        if set(self.compared_source_ids) & set(self.dropped_source_ids):
            raise ValueError("compared and dropped candidates must be disjoint")
        if min(self.budget_used_ms, self.budget_limit_ms) < 0:
            raise ValueError("comparison budgets must be non-negative")

    @property
    def compared_k(self) -> int:
        return len(self.compared_source_ids)


@dataclass(frozen=True)
class SegmentSelectionDecision:
    segment_id: str
    source_decision: SourceDecision
    comparison: VariantComparisonAudit

    def __post_init__(self) -> None:
        if self.segment_id != self.comparison.segment_id:
            raise ValueError("selection and comparison segment IDs disagree")
        selected = self.source_decision.selected_source_id
        if selected is not None and selected not in self.comparison.compared_source_ids:
            raise ValueError("selected Source was not current-state compared")

    @property
    def selection_state(self) -> SourceSelectionState:
        return self.source_decision.selection_state  # type: ignore[return-value]


@dataclass(frozen=True)
class RequestSelectionPlan:
    request_id: str
    segment_decisions: Tuple[SegmentSelectionDecision, ...]
    probe_ms: float
    metadata_ms: float
    compare_ms: float
    full_reference_ms: float
    selection_execution_policy: SelectionExecutionPolicy = (
        SelectionExecutionPolicy.LEGACY_COMMON_AFTER_SELECTION
    )
    earliest_reuse_layer_by_segment: Mapping[str, int] = field(
        default_factory=dict
    )
    probe_state_origin: str = "dense_clean"

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request selection requires request_id")
        ids = [decision.segment_id for decision in self.segment_decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("request selection contains duplicate segments")
        if min(
            self.probe_ms,
            self.metadata_ms,
            self.compare_ms,
            self.full_reference_ms,
        ) < 0:
            raise ValueError("selection timings must be non-negative")
        if self.full_reference_ms <= 0:
            raise ValueError("full reference time must be positive")
        selected_ids = {
            decision.segment_id
            for decision in self.segment_decisions
            if not decision.source_decision.abstained
        }
        floors = {
            str(segment_id): int(layer)
            for segment_id, layer in self.earliest_reuse_layer_by_segment.items()
        }
        if self.selection_execution_policy is (
            SelectionExecutionPolicy.LEGACY_COMMON_AFTER_SELECTION
        ):
            if floors:
                raise ValueError("legacy common selection cannot carry staggered floors")
        else:
            if set(floors) != selected_ids:
                raise ValueError(
                    "staggered selection requires one reuse floor per locked Source"
                )
            decision_by_id = {
                decision.segment_id: decision for decision in self.segment_decisions
            }
            for segment_id, layer in floors.items():
                if layer <= decision_by_id[segment_id].source_decision.probe_layer:
                    raise ValueError("reuse must start after the Source decision layer")
        if self.selection_execution_policy is (
            SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP
        ) and self.probe_state_origin != "policy_conditioned_closed_loop":
            raise ValueError(
                "immediate staggered selection requires policy-conditioned probe state"
            )
        if self.selection_execution_policy is (
            SelectionExecutionPolicy.CAUSAL_COMMIT_WAIT
        ) and self.probe_state_origin != "dense_clean":
            raise ValueError("causal commit selection requires clean dense probe state")
        object.__setattr__(self, "earliest_reuse_layer_by_segment", floors)

    @property
    def selected(self) -> Tuple[SegmentSelectionDecision, ...]:
        return tuple(
            decision for decision in self.segment_decisions
            if not decision.source_decision.abstained
        )


@dataclass(frozen=True)
class SegmentSchedulingFeedback:
    segment_id: str
    selected_source_id: str
    source_load_start_ms: float
    source_ready_ms: float
    first_ready_layer: int
    ready_through_layer: int
    transferred_bytes: int
    wasted_bytes: int = 0
    source_ready: bool = True
    source_load_finish_ms: Optional[float] = None
    layer_ready_ms: Tuple[Tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.segment_id or not self.selected_source_id:
            raise ValueError("segment scheduling identifiers are required")
        if min(self.source_load_start_ms, self.source_ready_ms) < 0:
            raise ValueError("source times must be non-negative")
        if self.source_ready_ms < self.source_load_start_ms:
            raise ValueError("source ready cannot precede load start")
        load_finish = (
            self.source_ready_ms
            if self.source_load_finish_ms is None
            else float(self.source_load_finish_ms)
        )
        object.__setattr__(self, "source_load_finish_ms", load_finish)
        if load_finish < self.source_load_start_ms:
            raise ValueError("source load finish cannot precede load start")
        if self.source_ready and load_finish < self.source_ready_ms:
            raise ValueError("source load finish cannot precede Source ready")
        if self.source_ready:
            if self.first_ready_layer < 1:
                raise ValueError("ready layers are 1-based")
            if self.ready_through_layer < self.first_ready_layer:
                raise ValueError("invalid contiguous ready layer range")
        elif self.first_ready_layer != 0 or self.ready_through_layer != 0:
            raise ValueError("unready Source must use zero ready-layer sentinels")
        if self.layer_ready_ms:
            layers = tuple(int(layer) for layer, _ in self.layer_ready_ms)
            times = tuple(float(ready_ms) for _, ready_ms in self.layer_ready_ms)
            if layers != tuple(sorted(set(layers))):
                raise ValueError("layer readiness must be sorted and unique")
            if any(layer < 1 for layer in layers):
                raise ValueError("layer readiness uses 1-based layers")
            if any(
                ready_ms < self.source_load_start_ms
                or ready_ms > load_finish + 1e-9
                for ready_ms in times
            ):
                raise ValueError("layer readiness falls outside the load interval")
            if times != tuple(sorted(times)):
                raise ValueError("layer readiness times must be monotonic")
            if self.source_ready and (
                layers[0] != self.first_ready_layer
                or layers[-1] != self.ready_through_layer
            ):
                raise ValueError("layer readiness does not match ready-layer range")
        elif self.source_ready:
            endpoints = ((self.first_ready_layer, self.source_ready_ms),)
            if self.ready_through_layer != self.first_ready_layer:
                endpoints += ((self.ready_through_layer, load_finish),)
            object.__setattr__(self, "layer_ready_ms", endpoints)
        if min(self.transferred_bytes, self.wasted_bytes) < 0:
            raise ValueError("byte counts must be non-negative")
        if self.wasted_bytes > self.transferred_bytes:
            raise ValueError("wasted bytes cannot exceed transferred bytes")

    def ready_at(self, layer: int) -> bool:
        return self.source_ready and (
            self.first_ready_layer <= layer <= self.ready_through_layer
        )


@dataclass(frozen=True)
class RequestSchedulingFeedback:
    request_id: str
    segments: Tuple[SegmentSchedulingFeedback, ...]
    scheduled_step_finish_ms: float
    a_resume_ms: float
    post_ready_blocking_ms: float
    load_interference_ms: float
    useful_a_dense_ms: float
    useful_other_request_work_ms: float
    candidate_boundaries: Tuple[int, ...]

    def __post_init__(self) -> None:
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("scheduler returned duplicate segment feedback")
        if min(
            self.scheduled_step_finish_ms,
            self.a_resume_ms,
            self.post_ready_blocking_ms,
            self.load_interference_ms,
            self.useful_a_dense_ms,
            self.useful_other_request_work_ms,
        ) < 0:
            raise ValueError("scheduler timings must be non-negative")
        if tuple(sorted(set(self.candidate_boundaries))) != self.candidate_boundaries:
            raise ValueError("candidate boundaries must be sorted and unique")
        if any(layer < 1 for layer in self.candidate_boundaries):
            raise ValueError("candidate boundaries are 1-based")


@dataclass(frozen=True)
class RequestRefinedCost:
    request_id: str
    boundary: int
    active_segment_ids: Tuple[str, ...]
    selected_source_ids: Mapping[str, str]
    marginal_saved_ms: Mapping[str, float]
    repair_ratio_upper_by_segment: Mapping[str, float]
    reuse_total_ms: float
    full_reference_ms: float
    probe_ms: float
    metadata_ms: float
    compare_ms: float
    visible_load_ms: float
    post_ready_blocking_ms: float
    load_interference_ms: float
    repair_selection_ms: float
    repair_ms: float
    remaining_ms: float
    boundary_by_segment: Mapping[str, int] = field(default_factory=dict)
    joint_quality_covered: bool = True
    cost_origin: str = "request_arrival"
    cost_endpoint: str = "first_token_ready"
    interference_accounting_mode: InterferenceAccountingMode = (
        InterferenceAccountingMode.EXPLICIT_PENALTY
    )

    def __post_init__(self) -> None:
        if self.boundary < 1:
            raise ValueError("refined boundary must be 1-based")
        if len(self.active_segment_ids) != len(set(self.active_segment_ids)):
            raise ValueError("active segment IDs must be unique")
        boundaries = {
            str(segment_id): int(layer)
            for segment_id, layer in self.boundary_by_segment.items()
        }
        if not boundaries:
            boundaries = {
                segment_id: self.boundary for segment_id in self.active_segment_ids
            }
        if set(boundaries) != set(self.active_segment_ids):
            raise ValueError("every active segment needs an actual reuse boundary")
        if any(layer < 1 for layer in boundaries.values()):
            raise ValueError("segment reuse boundaries are 1-based")
        if boundaries and self.boundary != min(boundaries.values()):
            raise ValueError(
                "request boundary must equal the earliest active segment boundary"
            )
        object.__setattr__(self, "boundary_by_segment", boundaries)
        if set(self.active_segment_ids) != set(self.selected_source_ids):
            raise ValueError("active segments and locked Sources disagree")
        if set(self.active_segment_ids) != set(self.marginal_saved_ms):
            raise ValueError("every active segment needs a marginal saving")
        if set(self.active_segment_ids) != set(
            self.repair_ratio_upper_by_segment
        ):
            raise ValueError("every active segment needs a refined safe ratio")
        if any(
            not 0 <= float(value) <= 1
            for value in self.repair_ratio_upper_by_segment.values()
        ):
            raise ValueError("refined repair ratios must be in [0, 1]")
        timings = (
            self.reuse_total_ms,
            self.full_reference_ms,
            self.probe_ms,
            self.metadata_ms,
            self.compare_ms,
            self.visible_load_ms,
            self.post_ready_blocking_ms,
            self.load_interference_ms,
            self.repair_selection_ms,
            self.repair_ms,
            self.remaining_ms,
        )
        if min(timings) < 0:
            raise ValueError("refined costs must be non-negative")
        if not all(math.isfinite(value) for value in timings):
            raise ValueError("refined costs must be finite")
        if not all(
            math.isfinite(float(value))
            for value in self.marginal_saved_ms.values()
        ):
            raise ValueError("marginal savings must be finite")
        if self.full_reference_ms <= 0:
            raise ValueError("full reference cost must be positive")
        if (
            self.interference_accounting_mode
            is not InterferenceAccountingMode.EXPLICIT_PENALTY
        ):
            raise ValueError(
                "v6 requires interference as one explicit non-load component"
            )
        component_total = (
            self.probe_ms
            + self.metadata_ms
            + self.compare_ms
            + self.visible_load_ms
            + self.post_ready_blocking_ms
            + self.load_interference_ms
            + self.repair_selection_ms
            + self.repair_ms
            + self.remaining_ms
        )
        if abs(component_total - self.reuse_total_ms) > 1e-6:
            raise ValueError("refined total does not equal its cost components")


@dataclass(frozen=True)
class SegmentExecutionDecision:
    segment_id: str
    path: SegmentExecutionPath
    selected_source_id: Optional[str]
    selection_state: SourceSelectionState
    admission_state: ReuseAdmissionState
    repair_ratio_upper: Optional[float] = None
    rejection_reason: Optional[str] = None
    actual_reuse_boundary: Optional[int] = None

    def __post_init__(self) -> None:
        if self.path is SegmentExecutionPath.REUSE:
            if self.selected_source_id is None:
                raise ValueError("reuse requires a selected Source")
            if self.selection_state is not SourceSelectionState.SELECTED:
                raise ValueError("reuse requires selected state")
            if self.admission_state is not ReuseAdmissionState.ACCEPTED:
                raise ValueError("reuse requires accepted admission")
            if self.actual_reuse_boundary is None:
                raise ValueError("reuse requires a per-segment actual boundary")
        elif self.admission_state is ReuseAdmissionState.ACCEPTED:
            raise ValueError("only reuse path may have accepted admission")
        elif self.actual_reuse_boundary is not None:
            raise ValueError("dense segment cannot have an actual reuse boundary")
        if self.selected_source_id is None and (
            self.selection_state is SourceSelectionState.SELECTED
        ):
            raise ValueError("selected state requires Source ID")


@dataclass(frozen=True)
class RequestExecutionPlan:
    request_id: str
    mode: RequestExecutionMode
    segment_decisions: Tuple[SegmentExecutionDecision, ...]
    actual_reuse_boundary: Optional[int]
    refined_cost: Optional[RequestRefinedCost]
    transferred_bytes: int
    wasted_loaded_bytes: int
    actual_reuse_boundary_by_segment: Mapping[str, int] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if min(self.transferred_bytes, self.wasted_loaded_bytes) < 0:
            raise ValueError("execution byte counts must be non-negative")
        if self.wasted_loaded_bytes > self.transferred_bytes:
            raise ValueError("wasted bytes cannot exceed transferred bytes")
        reused = [
            decision for decision in self.segment_decisions
            if decision.path is SegmentExecutionPath.REUSE
        ]
        boundaries = {
            str(segment_id): int(layer)
            for segment_id, layer in self.actual_reuse_boundary_by_segment.items()
        }
        if reused and not boundaries:
            boundaries = {
                decision.segment_id: int(decision.actual_reuse_boundary)
                for decision in reused
            }
        reused_ids = {decision.segment_id for decision in reused}
        if set(boundaries) != reused_ids:
            raise ValueError("execution requires one boundary per reused segment")
        if any(layer < 1 for layer in boundaries.values()):
            raise ValueError("execution boundaries are 1-based")
        if reused:
            unique = set(boundaries.values())
            if self.actual_reuse_boundary is not None and (
                len(unique) != 1
                or self.actual_reuse_boundary != next(iter(unique))
            ):
                raise ValueError("common boundary disagrees with segment boundaries")
        elif self.actual_reuse_boundary is not None:
            raise ValueError("dense execution cannot have reuse boundary")
        object.__setattr__(self, "actual_reuse_boundary_by_segment", boundaries)
        if self.mode is RequestExecutionMode.FULL_RECOMPUTE and reused:
            raise ValueError("full recomputation cannot contain reused segments")
        if self.mode is RequestExecutionMode.ALL_REUSE and (
            len(reused) != len(self.segment_decisions)
        ):
            raise ValueError("all_reuse requires every segment to reuse")
        if self.mode is RequestExecutionMode.PARTIAL_REUSE and (
            not reused or len(reused) == len(self.segment_decisions)
        ):
            raise ValueError("partial_reuse requires mixed segment paths")
