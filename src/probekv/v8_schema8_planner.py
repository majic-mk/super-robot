from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .v8_schema7_planner import FinalCommitPlanner


@dataclass(frozen=True)
class Gate1MarginalLowerBound:
    """Unavoidable Source-local costs under optimistic overlap.

    Gate1 is deliberately not a miniature request planner. In particular it
    must not assign the same base Transformer work, union repair kernel, or
    aggregate copy-stream critical path to every Segment independently.
    Those shared costs belong to request-level FinalCommitAdmission.
    """

    support_build_marginal_lower_ms: float
    visible_load_marginal_lower_ms: float
    repair_marginal_lower_ms: float

    def __post_init__(self) -> None:
        if min(
            self.support_build_marginal_lower_ms,
            self.visible_load_marginal_lower_ms,
            self.repair_marginal_lower_ms,
        ) < 0:
            raise ValueError("Gate1 marginal lower-bound costs must be non-negative")

    @property
    def total_ms(self) -> float:
        return (
            self.support_build_marginal_lower_ms
            + self.visible_load_marginal_lower_ms
            + self.repair_marginal_lower_ms
        )


@dataclass(frozen=True)
class Gate1LocalPlan:
    """Optimistic Segment-local feasibility screen.

    ``dense_repair_check_sunk_ms`` is retained for audit and later request-level
    accounting, but is common/sunk at this decision point and therefore is not
    charged again inside the marginal Gate1 comparison.
    """

    source_variant_id: str
    selection_completed_depth: int
    repair_check_completed_depth: int
    first_selective_reuse_layer: int
    dense_repair_check_sunk_ms: float
    marginal_lower_bound: Gate1MarginalLowerBound
    dense_marginal_same_origin_ms: float
    gate1_gamma: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_variant_id or self.selection_completed_depth < 1:
            raise ValueError("Gate1 plan requires a selected Source and legal depth")
        if self.repair_check_completed_depth < self.selection_completed_depth:
            raise ValueError("Gate1 repair check cannot precede Source selection")
        if self.first_selective_reuse_layer != self.repair_check_completed_depth + 1:
            raise ValueError("Gate1 reuse must follow the dense repair-check layer")
        if self.dense_repair_check_sunk_ms < 0:
            raise ValueError("Gate1 sunk repair-check cost must be non-negative")
        if self.dense_marginal_same_origin_ms <= 0:
            raise ValueError("Gate1 requires positive same-origin dense marginal cost")
        if self.gate1_gamma != 1.0:
            raise ValueError("schema-v8 Gate1 gamma is frozen at 1.0")

    @property
    def predicted_reuse_marginal_lower_ms(self) -> float:
        return self.marginal_lower_bound.total_ms

    @property
    def predicted_reuse_future_upper_ms(self) -> float:
        raise AttributeError(
            "schema-v8 Gate1 no longer predicts a per-Segment future upper bound; "
            "use predicted_reuse_marginal_lower_ms"
        )

    @property
    def passed(self) -> bool:
        return (
            self.predicted_reuse_marginal_lower_ms
            <= self.gate1_gamma * self.dense_marginal_same_origin_ms + 1e-12
        )


def gate1_candidate_costs(plans: Sequence[Gate1LocalPlan]) -> Tuple[float, ...]:
    if not plans:
        raise ValueError("Gate1 requires at least one Source-local plan")
    return tuple(plan.predicted_reuse_marginal_lower_ms for plan in plans)


__all__ = [
    "FinalCommitPlanner",
    "Gate1MarginalLowerBound",
    "Gate1LocalPlan",
    "gate1_candidate_costs",
]
