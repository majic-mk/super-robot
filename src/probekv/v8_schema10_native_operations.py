"""Staged operation dispatcher used by the real CUDA premeasurement session.

It deliberately does not manufacture an operation. Every callback owns the
native context and must return a real first-token endpoint for request-cost
operations; absent support is an error, not a zero-cost cell.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Mapping

from .v8_schema10_cost_collection import CudaCostCollector
from .v8_schema10_execution import digest_json


@dataclass(frozen=True)
class OperationSpec:
    category: str
    query: dict
    joint: bool = False
    warmup: int = 2
    repeats: int = 5


@dataclass(frozen=True)
class RegisteredOperation:
    """A concrete, preregistered native operation.

    The callbacks are deliberately part of the execution object rather than
    being inferred from a cost row.  This prevents a measurement table from
    becoming an executable claim by itself.
    """
    spec: OperationSpec
    reset: Callable
    operation: Callable


@dataclass
class NativeMeasurementSession:
    """Run an immutable operation list and fail closed on missing cells."""
    dispatcher: "NativeOperationDispatcher"
    operations: List[RegisteredOperation] = field(default_factory=list)

    def register(self, operation: RegisteredOperation):
        if not isinstance(operation, RegisteredOperation):
            raise TypeError("native measurement session requires RegisteredOperation")
        key = (operation.spec.category, digest_json(operation.spec.query))
        if any((item.spec.category, digest_json(item.spec.query)) == key for item in self.operations):
            raise ValueError("duplicate registered native operation cell")
        self.operations.append(operation)

    def run(self):
        specs = [item.spec for item in self.operations]
        expected = self.dispatcher.expected_cells(specs)
        rows = []
        seen = set()
        for item in self.operations:
            key = (item.spec.category, digest_json(item.spec.query))
            if key in seen:
                raise ValueError("duplicate registered native operation cell")
            row = self.dispatcher.run(item.spec, reset=item.reset, operation=item.operation)
            if not isinstance(row, Mapping):
                raise ValueError("native operation did not return a measurement row")
            rows.append(dict(row))
            seen.add(key)
        if seen != expected:
            raise RuntimeError("native measurement session did not cover its immutable operation plan")
        return rows


class NativeOperationDispatcher:
    def __init__(self, *, adapter, collector: CudaCostCollector):
        if not getattr(collector, "provenance", None):
            raise ValueError("native dispatcher requires a real CUDA collector")
        self.adapter, self.collector = adapter, collector

    def run(self, spec: OperationSpec, *, reset, operation):
        if not isinstance(spec, OperationSpec) or not callable(reset) or not callable(operation):
            raise TypeError("operation requires a preregistered spec, reset and native callback")
        # ``joint`` is a cost-table attribution flag.  The collector still
        # requires a first-token endpoint for these categories even when an
        # individual primitive is measured outside the joint planner.
        return self.collector.measure(category=spec.category, query=spec.query,
            operation=operation, reset=reset, warmup=spec.warmup, repeats=spec.repeats,
            joint=spec.joint)

    @staticmethod
    def expected_cells(specs):
        cells = [(s.category, digest_json(s.query)) for s in specs]
        if len(cells) != len(set(cells)):
            raise ValueError("duplicate preregistered native operation cell")
        return set(cells)
