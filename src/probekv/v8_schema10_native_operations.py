"""Staged operation dispatcher used by the real CUDA premeasurement session.

It deliberately does not manufacture an operation. Every callback owns the
native context and must return a real first-token endpoint for request-cost
operations; absent support is an error, not a zero-cost cell.
"""
from dataclasses import dataclass

from .v8_schema10_cost_collection import CudaCostCollector
from .v8_schema10_execution import digest_json


@dataclass(frozen=True)
class OperationSpec:
    category: str
    query: dict
    joint: bool = False
    warmup: int = 2
    repeats: int = 5


class NativeOperationDispatcher:
    def __init__(self, *, adapter, collector: CudaCostCollector):
        if not getattr(collector, "provenance", None):
            raise ValueError("native dispatcher requires a real CUDA collector")
        self.adapter, self.collector = adapter, collector

    def run(self, spec: OperationSpec, *, reset, operation):
        if not isinstance(spec, OperationSpec) or not callable(reset) or not callable(operation):
            raise TypeError("operation requires a preregistered spec, reset and native callback")
        if spec.category in {"dense_reference", "source_future", "joint_future"} and not spec.joint:
            # Source-future is also measured to the first token even when its
            # callback does not use the joint estimator.
            pass
        return self.collector.measure(category=spec.category, query=spec.query,
            operation=operation, reset=reset, warmup=spec.warmup, repeats=spec.repeats,
            joint=spec.joint)

    @staticmethod
    def expected_cells(specs):
        cells = [(s.category, digest_json(s.query)) for s in specs]
        if len(cells) != len(set(cells)):
            raise ValueError("duplicate preregistered native operation cell")
        return set(cells)
