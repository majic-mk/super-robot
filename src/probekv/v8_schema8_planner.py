from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .v8_schema7_planner import FinalCommitPlanner


@dataclass(frozen=True)
class Gate1LayerCost:
    layer_1based: int
    repair_ratio: float
    predicted_load_upper_ms: float
    predicted_repair_upper_ms: float
    predicted_nonoverlap_upper_ms: float

    def __post_init__(self) -> None:
        if self.layer_1based < 1 or not 0 <= self.repair_ratio <= 0.15:
            raise ValueError("invalid schema-v8 Gate1 layer plan")
        if min(
            self.predicted_load_upper_ms,
            self.predicted_repair_upper_ms,
            self.predicted_nonoverlap_upper_ms,
        ) < 0:
            raise ValueError("Gate1 layer costs must be non-negative")

    @property
    def critical_path_upper_ms(self) -> float:
        return (
            max(self.predicted_load_upper_ms, self.predicted_repair_upper_ms)
            + self.predicted_nonoverlap_upper_ms
        )


@dataclass(frozen=True)
class Gate1LocalPlan:
    source_variant_id: str
    selection_completed_depth: int
    repair_check_completed_depth: int
    first_selective_reuse_layer: int
    dense_repair_check_upper_ms: float
    support_build_upper_ms: float
    layer_costs: Tuple[Gate1LayerCost, ...]
    dense_remaining_same_origin_ms: float
    gate1_gamma: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_variant_id or self.selection_completed_depth < 1:
            raise ValueError("Gate1 plan requires a selected Source and legal depth")
        if self.repair_check_completed_depth < self.selection_completed_depth:
            raise ValueError("Gate1 repair check cannot precede Source selection")
        if self.first_selective_reuse_layer != self.repair_check_completed_depth + 1:
            raise ValueError("Gate1 reuse must follow the dense repair-check layer")
        if min(
            self.dense_repair_check_upper_ms,
            self.support_build_upper_ms,
        ) < 0 or self.dense_remaining_same_origin_ms <= 0:
            raise ValueError("Gate1 costs use one positive same-origin horizon")
        if self.gate1_gamma != 1.0:
            raise ValueError("schema-v8 Gate1 gamma is frozen at 1.0")
        layers = tuple(row.layer_1based for row in self.layer_costs)
        if layers and (
            layers != tuple(sorted(set(layers)))
            or layers[0] != self.first_selective_reuse_layer
        ):
            raise ValueError("Gate1 layer costs must start at the first reuse layer")

    @property
    def predicted_reuse_future_upper_ms(self) -> float:
        return (
            self.dense_repair_check_upper_ms
            + self.support_build_upper_ms
            + sum(row.critical_path_upper_ms for row in self.layer_costs)
        )

    @property
    def passed(self) -> bool:
        return (
            self.predicted_reuse_future_upper_ms
            <= self.gate1_gamma * self.dense_remaining_same_origin_ms + 1e-12
        )


def gate1_candidate_costs(plans: Sequence[Gate1LocalPlan]) -> Tuple[float, ...]:
    if not plans:
        raise ValueError("Gate1 requires at least one Source-local plan")
    return tuple(plan.predicted_reuse_future_upper_ms for plan in plans)


__all__ = [
    "FinalCommitPlanner",
    "Gate1LayerCost",
    "Gate1LocalPlan",
    "gate1_candidate_costs",
]
