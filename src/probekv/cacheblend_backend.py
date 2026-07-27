from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .backend import RepairBackend, RepairResult
from .contracts import HistoricalSource, KVLocation


@dataclass(frozen=True)
class RuntimeRepairMeasurement:
    quality_score: float
    token_f1: float
    latency_ms: float
    source_digest_before: str
    source_digest_after: str


class CacheBlendRuntime(Protocol):
    """Narrow shim implemented inside the pinned CacheBlend fork.

    Keeping this protocol free of vLLM tensor types lets all orchestration and
    invariants be tested locally. Only the shim implementation is stack-specific.
    """

    def stage_canonical_source(
        self, source: HistoricalSource, target: KVLocation
    ) -> float:
        ...

    def selective_repair(
        self, source: HistoricalSource, start_layer: int, ratio: float
    ) -> RuntimeRepairMeasurement:
        ...

    def dense_remaining_ms(self, token_count: int, start_layer: int) -> float:
        ...

    def provenance(self) -> Mapping[str, Any]:
        ...


class CacheBlendBackend(RepairBackend):
    """Validated adapter around a pinned CacheBlend runtime shim."""

    def __init__(self, runtime: CacheBlendRuntime, total_layers: int) -> None:
        if total_layers <= 0:
            raise ValueError("total_layers must be positive")
        self.runtime = runtime
        self.total_layers = total_layers

    def prepare_source(self, source: HistoricalSource, target: KVLocation) -> float:
        source.validate_canonical()
        latency = float(self.runtime.stage_canonical_source(source, target))
        if latency < 0:
            raise ValueError("source preparation latency must be non-negative")
        return latency

    def repair(
        self, source: HistoricalSource, start_layer: int, ratio: float
    ) -> RepairResult:
        source.validate_canonical()
        if not 0 <= start_layer < self.total_layers:
            raise ValueError("invalid start_layer")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be in [0, 1]")
        measurement = self.runtime.selective_repair(source, start_layer, ratio)
        if measurement.source_digest_before != measurement.source_digest_after:
            raise RuntimeError("CacheBlend runtime mutated a canonical source")
        if measurement.latency_ms < 0:
            raise ValueError("repair latency must be non-negative")
        if not 0.0 <= measurement.quality_score <= 1.0:
            raise ValueError("quality score must be in [0, 1]")
        if not 0.0 <= measurement.token_f1 <= 1.0:
            raise ValueError("token F1 must be in [0, 1]")
        return RepairResult(
            measurement.quality_score,
            measurement.token_f1,
            measurement.latency_ms,
            ratio,
        )

    def full_remaining(self, token_count: int, start_layer: int) -> float:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if not 0 <= start_layer < self.total_layers:
            raise ValueError("invalid start_layer")
        latency = float(self.runtime.dense_remaining_ms(token_count, start_layer))
        if latency < 0:
            raise ValueError("dense latency must be non-negative")
        return latency

    def provenance(self) -> Mapping[str, Any]:
        record = dict(self.runtime.provenance())
        required = {"cacheblend_commit", "vllm", "torch", "cuda"}
        missing = sorted(required - set(record))
        if missing:
            raise ValueError("missing CacheBlend provenance: %s" % ", ".join(missing))
        return record
