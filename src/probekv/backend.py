from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Tuple

from .contracts import HistoricalSource, KVLocation
from .repair_semantics import repaired_segment_token_count


@dataclass(frozen=True)
class RepairResult:
    quality_score: float
    token_f1: float
    latency_ms: float
    repaired_ratio: float
    requested_ratio: float = 0.0
    eligible_segment_tokens: int = 0
    selected_segment_tokens: int = 0
    effective_ratio: float = 0.0
    mandatory_suffix_tokens: int = 0
    reuse_start_layer: int = 1
    repair_gpu_ms: float = 0.0
    repair_host_ms: float = 0.0
    source_digest_before: str = ""
    source_digest_after: str = ""
    output_token_ids: Tuple[int, ...] = ()
    output_hash: str = ""
    output_text: str = ""


class RepairBackend(ABC):
    """Interface implemented by CacheBlend and any later repair backend."""

    @abstractmethod
    def prepare_source(self, source: HistoricalSource, target: KVLocation) -> float:
        """Return measured source preparation latency in milliseconds."""

    @abstractmethod
    def repair(
        self, source: HistoricalSource, start_layer: int, ratio: float
    ) -> RepairResult:
        """Repair an existing canonical source without mutating it."""

    @abstractmethod
    def full_remaining(self, token_count: int, start_layer: int) -> float:
        """Return full recompute latency from a layer in milliseconds."""


class DeterministicSimulationBackend(RepairBackend):
    """Local-only backend for logic tests, never for paper performance claims."""

    paper_evidence = False

    def __init__(
        self,
        total_layers: int = 32,
        layer_ms_per_1k_tokens: float = 0.45,
        load_gb_per_second: float = 12.0,
        safe_ratio_by_source: Mapping[str, float] = None,
    ) -> None:
        self.total_layers = total_layers
        self.layer_ms_per_1k_tokens = layer_ms_per_1k_tokens
        self.load_gb_per_second = load_gb_per_second
        self.safe_ratio_by_source = dict(safe_ratio_by_source or {})

    def prepare_source(self, source: HistoricalSource, target: KVLocation) -> float:
        if target is KVLocation.GPU and source.kv_location is not KVLocation.GPU:
            estimated_bytes = source.token_count * self.total_layers * 4096 * 2 * 2
            return (estimated_bytes / 1_000_000_000.0) / self.load_gb_per_second * 1000.0
        return 0.0

    def repair(
        self, source: HistoricalSource, start_layer: int, ratio: float
    ) -> RepairResult:
        if not 1 <= start_layer <= self.total_layers:
            raise ValueError("invalid start_layer")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be in [0, 1]")
        remaining = self.total_layers - start_layer + 1
        latency = (
            remaining
            * self.layer_ms_per_1k_tokens
            * (source.token_count / 1024.0)
            * ratio
        )
        threshold = self.safe_ratio_by_source.get(source.source_id, 0.2)
        deficit = max(0.0, threshold - ratio)
        token_f1 = max(0.0, 1.0 - deficit * 2.0)
        quality = max(0.0, 1.0 - deficit)
        selected = repaired_segment_token_count(source.token_count, ratio)
        effective = selected / float(source.token_count)
        return RepairResult(
            quality,
            token_f1,
            latency,
            effective,
            requested_ratio=ratio,
            eligible_segment_tokens=source.token_count,
            selected_segment_tokens=selected,
            effective_ratio=effective,
            mandatory_suffix_tokens=0,
            reuse_start_layer=start_layer,
            repair_gpu_ms=latency,
            repair_host_ms=latency,
        )

    def full_remaining(self, token_count: int, start_layer: int) -> float:
        if not 1 <= start_layer <= self.total_layers:
            raise ValueError("invalid start_layer")
        return (
            (self.total_layers - start_layer + 1)
            * self.layer_ms_per_1k_tokens
            * (token_count / 1024.0)
        )
