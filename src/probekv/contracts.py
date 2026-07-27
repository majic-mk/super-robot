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

    def __post_init__(self) -> None:
        if self.cost_lower_ms > self.cost_upper_ms:
            raise ValueError("cost lower bound exceeds upper bound")
        if not 0.0 <= self.repair_ratio_upper <= 1.0:
            raise ValueError("repair ratio upper bound must be in [0, 1]")


@dataclass(frozen=True)
class SourceDecision:
    selected_source_id: Optional[str]
    probe_layer: int
    reuse_layer: Optional[int]
    safe_repair_ratio_upper: Optional[float]
    prefetch_m: int
    reason: DecisionReason

    @property
    def abstained(self) -> bool:
        return self.selected_source_id is None


@dataclass(frozen=True)
class TimingBreakdown:
    probe_ms: float
    compare_ms: float
    load_ms: float
    visible_load_ms: float
    repair_ms: float
    full_ms: float

    @property
    def reuse_total_ms(self) -> float:
        return (
            self.probe_ms
            + self.compare_ms
            + self.visible_load_ms
            + self.repair_ms
        )
