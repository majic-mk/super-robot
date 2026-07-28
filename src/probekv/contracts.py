from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple


class KVLocation(str, Enum):
    GPU = "gpu"
    PINNED_CPU = "pinned_cpu"
    SSD = "ssd"


class SourceOrigin(str, Enum):
    FULL_PREFILL = "full_prefill"
    SELECTIVE_REPAIR = "selective_repair"


class DecisionReason(str, Enum):
    CONFIDENT = "confident"
    MAX_PROBE_UNCERTAIN = "max_probe_uncertain"
    QUALITY_UNCOVERED = "quality_uncovered"
    ECONOMIC_REJECT = "economic_reject"
    NO_FEASIBLE_LAYER = "no_feasible_layer"


class SelectionReason(str, Enum):
    EARLY_CONFIDENT = "early_confident"
    FINAL_ECONOMIC_MIN_COST = "final_economic_min_cost"
    FINAL_MAX_REUSE_LOWER_BOUND = "final_max_reuse_lower_bound"
    MAX_PROBE_UNCERTAIN = "max_probe_uncertain"
    NO_QUALITY_SAFE_SOURCE = "no_quality_safe_source"
    NO_ECONOMIC_SOURCE = "no_economic_source"


class RejectionReason(str, Enum):
    QUALITY_GATE_FAILED = "quality_gate_failed"
    PREDICTED_TIME_GATE_FAILED = "predicted_time_gate_failed"
    FINAL_TIME_GATE_FAILED = "final_time_gate_failed"
    SELECTION_UNCERTAIN = "selection_uncertain"
    NO_FEASIBLE_REUSE_LAYER = "no_feasible_reuse_layer"


class ExecutionMode(str, Enum):
    REUSE = "reuse"
    FULL_RECOMPUTE = "full_recompute"


@dataclass(frozen=True)
class HistoricalSource:
    source_id: str
    content_hash: str
    context_id: str
    model_signature: str
    token_count: int
    exact: bool
    origin: SourceOrigin
    kv_location: KVLocation
    kv_handles: Tuple[str, ...] = field(default_factory=tuple)
    probe_summary: Mapping[int, Tuple[float, ...]] = field(default_factory=dict)

    def validate_canonical(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if not self.exact:
            raise ValueError("canonical HistoricalSource must be exact")
        if self.origin is not SourceOrigin.FULL_PREFILL:
            raise ValueError(
                "selective-repair KV is approximate and must never be promoted"
            )


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    current_prompt: str
    content_hash: str
    model_signature: str
    sources: Tuple[HistoricalSource, ...]
    split: str
    regime: str
    segment_length: int

    def validate(self, max_sources: int = 8) -> None:
        if self.split not in {"train", "calibration", "test", "pilot"}:
            raise ValueError("unsupported split: %s" % self.split)
        if not 1 <= len(self.sources) <= max_sources:
            raise ValueError("source count must be in [1, %d]" % max_sources)
        if self.segment_length <= 0:
            raise ValueError("segment_length must be positive")
        ids = [source.source_id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source_id values must be unique within a case")
        for source in self.sources:
            source.validate_canonical()
            if source.content_hash != self.content_hash:
                raise ValueError("all sources must represent the exact same segment")
            if source.model_signature != self.model_signature:
                raise ValueError("source and current model signatures must match")


@dataclass(frozen=True)
class ProbeObservation:
    case_id: str
    source_id: str
    layer: int
    k_drift: float
    v_drift: float
    hidden_drift: float
    query_score: float
    prefix_overlap: float
    order_score: float
    comparison_latency_ms: float

    def feature_map(self) -> Dict[str, float]:
        return {
            "k_drift": self.k_drift,
            "v_drift": self.v_drift,
            "hidden_drift": self.hidden_drift,
            "query_score": self.query_score,
            "prefix_overlap": self.prefix_overlap,
            "order_score": self.order_score,
        }


@dataclass(frozen=True)
class SafeBudgetLabel:
    case_id: str
    source_id: str
    reuse_layer: int
    safe_repair_ratio: float
    quality_score: float
    token_f1: float
    repair_latency_ms: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.safe_repair_ratio <= 1.0:
            raise ValueError("safe_repair_ratio must be in [0, 1]")


@dataclass(frozen=True)
class CandidateBounds:
    source_id: str
    repair_ratio_upper: float
    cost_lower_ms: float
    cost_upper_ms: float
    quality_covered: bool = True
    cost_scope: str = "predicted_total_reuse"

    def __post_init__(self) -> None:
        if self.cost_lower_ms > self.cost_upper_ms:
            raise ValueError("cost lower bound exceeds upper bound")
        if not 0.0 <= self.repair_ratio_upper <= 1.0:
            raise ValueError("repair ratio upper bound must be in [0, 1]")
        if not self.cost_scope:
            raise ValueError("candidate cost scope must be explicit")


@dataclass(frozen=True)
class SourceDecision:
    selected_source_id: Optional[str]
    probe_layer: int
    reuse_layer: Optional[int]
    safe_repair_ratio_upper: Optional[float]
    prefetch_m: int
    selection_reason: SelectionReason
    predicted_cost_upper_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.probe_layer < 1:
            raise ValueError("probe_layer must be 1-based")
        if self.selected_source_id is None:
            if self.safe_repair_ratio_upper is not None:
                raise ValueError("abstention cannot retain a repair ratio")
            if self.prefetch_m != 0:
                raise ValueError("abstention cannot prefetch a winner")
            if self.predicted_cost_upper_ms is not None:
                raise ValueError("abstention cannot retain a winner cost")
        elif self.safe_repair_ratio_upper is None:
            raise ValueError("selected source requires a safe repair upper bound")

    @property
    def abstained(self) -> bool:
        return self.selected_source_id is None

    @property
    def reason(self) -> DecisionReason:
        """Legacy reason used by pre-v3 artifacts and callers."""
        if self.selection_reason in {
            SelectionReason.EARLY_CONFIDENT,
            SelectionReason.FINAL_ECONOMIC_MIN_COST,
            SelectionReason.FINAL_MAX_REUSE_LOWER_BOUND,
        }:
            return DecisionReason.CONFIDENT
        if self.selection_reason is SelectionReason.MAX_PROBE_UNCERTAIN:
            return DecisionReason.MAX_PROBE_UNCERTAIN
        if self.selection_reason is SelectionReason.NO_QUALITY_SAFE_SOURCE:
            return DecisionReason.QUALITY_UNCOVERED
        return DecisionReason.ECONOMIC_REJECT


@dataclass(frozen=True)
class TimingBreakdown:
    probe_ms: float
    compare_ms: float
    load_ms: float
    visible_load_ms: float
    repair_ms: float
    full_ms: float
    post_ready_blocking_ms: float = 0.0
    load_interference_ms: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.probe_ms,
            self.compare_ms,
            self.load_ms,
            self.visible_load_ms,
            self.repair_ms,
            self.full_ms,
            self.post_ready_blocking_ms,
            self.load_interference_ms,
        ) < 0:
            raise ValueError("timing values must be non-negative")

    @property
    def reuse_total_ms(self) -> float:
        return (
            self.probe_ms
            + self.compare_ms
            + self.visible_load_ms
            + self.post_ready_blocking_ms
            + self.repair_ms
        )


@dataclass(frozen=True)
class ExecutionDecision:
    selected_source_id: Optional[str]
    selection_reason: SelectionReason
    reuse_accepted: bool
    rejection_reason: Optional[RejectionReason]
    execution_mode: ExecutionMode
    probe_layer: int
    reuse_layer: Optional[int]
    safe_repair_ratio_upper: Optional[float]
    timing: Optional[TimingBreakdown]

    def __post_init__(self) -> None:
        if self.selected_source_id is None and self.reuse_accepted:
            raise ValueError("reuse cannot be accepted without a selected source")
        if self.reuse_accepted and self.execution_mode is not ExecutionMode.REUSE:
            raise ValueError("accepted reuse requires REUSE execution mode")
        if self.execution_mode is ExecutionMode.FULL_RECOMPUTE and self.reuse_accepted:
            raise ValueError("full recompute cannot accept reuse")
        if self.execution_mode is ExecutionMode.REUSE and not self.reuse_accepted:
            raise ValueError("REUSE execution mode requires accepted reuse")
        if self.reuse_accepted and self.rejection_reason is not None:
            raise ValueError("accepted reuse cannot have a rejection reason")
        if not self.reuse_accepted and self.rejection_reason is None:
            raise ValueError("rejected reuse requires a rejection reason")
        if self.reuse_accepted and (
            self.reuse_layer is None or self.timing is None
        ):
            raise ValueError("accepted reuse requires a layer and timing")

    @property
    def abstained(self) -> bool:
        return self.selected_source_id is None
