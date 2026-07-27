from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from .contracts import HistoricalSource, KVLocation


@dataclass(frozen=True)
class RepairResult:
    quality_score: float
    token_f1: float
    latency_ms: float
    repaired_ratio: float


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
        if not 0 <= start_layer < self.total_layers:
            raise ValueError("invalid start_layer")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be in [0, 1]")
        remaining = self.total_layers - start_layer
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
        return RepairResult(quality, token_f1, latency, ratio)

    def full_remaining(self, token_count: int, start_layer: int) -> float:
        if not 0 <= start_layer < self.total_layers:
            raise ValueError("invalid start_layer")
        return (
            (self.total_layers - start_layer)
            * self.layer_ms_per_1k_tokens
            * (token_count / 1024.0)
        )
