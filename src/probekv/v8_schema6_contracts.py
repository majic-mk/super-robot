from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Tuple


V8_SCHEMA6_VERSION = 6
V8_SCHEMA6_PROTOCOL_VERSION = 8


class SelectionAxisState(str, Enum):
    PROBING = "probing"
    DECISION_READY = "decision_ready"
    SOURCE_FROZEN = "source_frozen"
    ABSTAINED = "abstained"


class Gate2AxisState(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    DEFERRED = "deferred"
    PROVISIONAL_REUSE = "provisional_reuse"
    PREDICTED_DENSE = "predicted_dense"


class PreparationAxisState(str, Enum):
    NONE = "none"
    PREFETCHING = "prefetching"
    READY = "ready"


class CommitAxisState(str, Enum):
    UNCOMMITTED = "uncommitted"
    REUSE_COMMIT = "reuse_commit"
    REFINED_DENSE = "refined_dense"


@dataclass(frozen=True)
class PlannerSnapshot:
    request_generation: int
    segment_inventory_generation: int
    scheduler_snapshot_id: str
    hbm_reservation_epoch: int
    runtime_cost_profile_sha: str

    def __post_init__(self) -> None:
        if min(
            self.request_generation,
            self.segment_inventory_generation,
            self.hbm_reservation_epoch,
        ) < 1:
            raise ValueError("Planner generations must be positive")
        if not self.scheduler_snapshot_id or not self.runtime_cost_profile_sha:
            raise ValueError("Planner snapshot provenance is incomplete")

    def assert_current(self, current: "PlannerSnapshot") -> None:
        if self != current:
            raise RuntimeError("stale Planner snapshot cannot be applied")


@dataclass
class Schema6SegmentRuntimeState:
    segment_id: str
    order: int
    selection_state: SelectionAxisState = SelectionAxisState.PROBING
    gate2_state: Gate2AxisState = Gate2AxisState.NOT_EVALUATED
    preparation_state: PreparationAxisState = PreparationAxisState.NONE
    commit_state: CommitAxisState = CommitAxisState.UNCOMMITTED
    source_variant_id: Optional[str] = None
    decision_completed_depth: Optional[int] = None
    actual_reuse_boundary: Optional[int] = None
    logical_lease_id: Optional[str] = None
    physical_lease_id: Optional[str] = None
    hbm_reservation_id: Optional[str] = None
    speculative_realized_overrun_ms: float = 0.0
    speculative_prefetch_disabled: bool = False
    reason: str = ""

    def validate(self) -> None:
        if not self.segment_id or self.order < 0:
            raise ValueError("invalid Segment runtime identity")
        if self.speculative_realized_overrun_ms < 0:
            raise ValueError("speculative overrun must be non-negative")
        if self.selection_state is SelectionAxisState.SOURCE_FROZEN:
            if not self.source_variant_id or not self.logical_lease_id:
                raise RuntimeError("frozen Source requires identity and logical lease")
        if self.selection_state is SelectionAxisState.ABSTAINED:
            if self.source_variant_id is not None:
                raise RuntimeError("abstained Segment cannot expose a Source")
            if self.gate2_state is not Gate2AxisState.NOT_EVALUATED:
                raise RuntimeError("abstained Segment cannot enter Gate 2")
        if self.preparation_state is not PreparationAxisState.NONE:
            if self.selection_state is not SelectionAxisState.SOURCE_FROZEN:
                raise RuntimeError("physical preparation requires a frozen Source")
            if not self.physical_lease_id or not self.hbm_reservation_id:
                raise RuntimeError("physical preparation requires lease and HBM reservation")
        if self.commit_state is CommitAxisState.REUSE_COMMIT:
            if self.gate2_state is not Gate2AxisState.PROVISIONAL_REUSE:
                raise RuntimeError("reuse commit requires Gate 2 admission")
            if self.preparation_state is not PreparationAxisState.READY:
                raise RuntimeError("reuse commit requires a ready winner")
        if self.gate2_state is Gate2AxisState.PREDICTED_DENSE:
            if self.commit_state is CommitAxisState.REUSE_COMMIT:
                raise RuntimeError("predicted dense cannot be promoted to reuse")


@dataclass(frozen=True)
class Gate3SubsetDecision:
    accepted_ready_segment_ids: Tuple[str, ...]
    rejected_ready_segment_ids: Tuple[str, ...]
    untouched_segment_ids: Tuple[str, ...]
    request_total_ms: float
    dense_reference_total_ms: float
    planner_snapshot: PlannerSnapshot
    reason_by_segment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        accepted = set(self.accepted_ready_segment_ids)
        rejected = set(self.rejected_ready_segment_ids)
        untouched = set(self.untouched_segment_ids)
        if accepted & rejected or accepted & untouched or rejected & untouched:
            raise ValueError("Gate 3 decision sets must be disjoint")
        if min(self.request_total_ms, self.dense_reference_total_ms) < 0:
            raise ValueError("Gate 3 costs must be non-negative")
        if set(self.reason_by_segment) - (accepted | rejected | untouched):
            raise ValueError("Gate 3 reason references an unknown Segment")


@dataclass(frozen=True)
class SpeculativeWasteAdmission:
    admitted: bool
    deferred_base_ms: float
    dense_reference_total_ms: float
    waste_safety_budget_ms: float
    predicted_visible_copy_ms: float
    predicted_copy_interference_ms: float
    reason: str


def evaluate_speculative_waste_admission(
    *,
    actual_sunk_ms: float,
    dense_fallback_joint_future_ms: float,
    dense_reference_total_ms: float,
    predicted_visible_copy_ms: float,
    predicted_copy_interference_ms: float,
    hbm_available: bool,
    preserves_existing_reservations: bool,
) -> SpeculativeWasteAdmission:
    values = (
        actual_sunk_ms,
        dense_fallback_joint_future_ms,
        dense_reference_total_ms,
        predicted_visible_copy_ms,
        predicted_copy_interference_ms,
    )
    if min(values) < 0 or dense_reference_total_ms <= 0:
        raise ValueError("speculative admission costs must be valid")
    deferred_base = actual_sunk_ms + dense_fallback_joint_future_ms
    budget = max(0.0, dense_reference_total_ms - deferred_base)
    preparation = predicted_visible_copy_ms + predicted_copy_interference_ms
    admitted = (
        preparation <= budget + 1e-12
        and hbm_available
        and preserves_existing_reservations
    )
    if not hbm_available:
        reason = "insufficient_hbm"
    elif not preserves_existing_reservations:
        reason = "would_violate_existing_reservation"
    elif preparation > budget + 1e-12:
        reason = "speculative_waste_budget_exceeded"
    else:
        reason = "speculative_waste_safe"
    return SpeculativeWasteAdmission(
        admitted,
        deferred_base,
        dense_reference_total_ms,
        budget,
        predicted_visible_copy_ms,
        predicted_copy_interference_ms,
        reason,
    )
