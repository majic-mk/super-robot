from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

from .v7_contracts import PredictedAccessPlan


@dataclass(frozen=True)
class SharedSunkCost:
    shared_probe_ms: float
    metadata_ms: float
    summary_ms: float
    compare_batch_ms: float
    speculative_visible_ms: float = 0.0

    def __post_init__(self) -> None:
        if min(self.__dict__.values()) < 0:
            raise ValueError("shared costs must be non-negative")

    @property
    def total_ms(self) -> float:
        return sum(self.__dict__.values())

    def within_probe_budget(self, dense_reference_ms: float, fraction: float = 0.05) -> bool:
        if dense_reference_ms <= 0 or not 0 < fraction <= 1:
            raise ValueError("invalid dense reference or probe budget")
        # Speculative visible work is execution overlap, not selector overhead.
        selector_overhead = (
            self.shared_probe_ms
            + self.metadata_ms
            + self.summary_ms
            + self.compare_batch_ms
        )
        return selector_overhead <= fraction * dense_reference_ms + 1e-9


@dataclass(frozen=True)
class SourceCostCandidate:
    source_variant_id: str
    access_plan: PredictedAccessPlan
    safe: bool
    current_state_cost_upper_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.access_plan.source_variant_id != self.source_variant_id:
            raise ValueError("Source cost and access plan disagree")
        if self.current_state_cost_upper_ms < 0:
            raise ValueError("current-state cost must be non-negative")

    @property
    def future_upper_ms(self) -> float:
        return self.current_state_cost_upper_ms + self.access_plan.future_cost_upper_ms


def choose_source_variant(
    candidates: Sequence[SourceCostCandidate],
    *,
    sunk: SharedSunkCost,
    dense_reference_ms: float,
    gamma: float,
) -> Optional[SourceCostCandidate]:
    """Select once; final runtime may accept/reject but never substitute."""
    if dense_reference_ms <= 0 or not 0 < gamma <= 1:
        raise ValueError("invalid economic boundary")
    safe = [candidate for candidate in candidates if candidate.safe]
    if not safe:
        return None
    best = min(safe, key=lambda item: (item.future_upper_ms, item.source_variant_id))
    if sunk.total_ms + best.future_upper_ms > gamma * dense_reference_ms + 1e-9:
        return None
    return best


def repair_token_count(token_count: int, ratio: float) -> int:
    if token_count < 0 or not 0 <= ratio <= 1:
        raise ValueError("invalid repair token count or ratio")
    if ratio == 0 or token_count == 0:
        return 0
    if ratio == 1:
        return token_count
    return min(token_count, int(math.ceil(ratio * token_count)))


class JointSegmentPath(str, Enum):
    REUSE = "reuse"
    DENSE = "dense"


@dataclass(frozen=True)
class LockedSegmentOption:
    segment_id: str
    source_variant_id: str
    replica_id: str
    actual_reuse_boundary: int
    repair_ratio_upper: float
    segment_tokens: int
    resident_hbm_bytes: int
    load_ms: float
    post_ready_blocking_ms: float
    interference_ms: float
    repair_selection_ms: float
    repair_ms: float
    remaining_ms: float
    dense_remaining_ms: float
    source_ready: bool = True

    def __post_init__(self) -> None:
        if not all((self.segment_id, self.source_variant_id, self.replica_id)):
            raise ValueError("locked Segment identity is required")
        if self.actual_reuse_boundary < 1:
            raise ValueError("reuse boundary is 1-based")
        if not 0 <= self.repair_ratio_upper <= 1:
            raise ValueError("repair ratio must be in [0, 1]")
        if self.segment_tokens < 1 or self.resident_hbm_bytes < 0:
            raise ValueError("invalid Segment size")
        if min(
            self.load_ms,
            self.post_ready_blocking_ms,
            self.interference_ms,
            self.repair_selection_ms,
            self.repair_ms,
            self.remaining_ms,
            self.dense_remaining_ms,
        ) < 0:
            raise ValueError("joint plan costs must be non-negative")

    @property
    def reuse_future_ms(self) -> float:
        return (
            self.load_ms
            + self.post_ready_blocking_ms
            + self.interference_ms
            + self.repair_selection_ms
            + self.repair_ms
            + self.remaining_ms
        )

    @property
    def marginal_saved_ms(self) -> float:
        return self.dense_remaining_ms - self.reuse_future_ms


@dataclass(frozen=True)
class JointSegmentDecision:
    segment_id: str
    source_variant_id: str
    path: JointSegmentPath
    replica_id: Optional[str]
    actual_reuse_boundary: Optional[int]
    repair_ratio_upper: Optional[float]
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class JointRequestPlan:
    request_plan_id: str
    decisions: Tuple[JointSegmentDecision, ...]
    shared_sunk_ms: float
    dense_reference_ms: float
    joint_future_ms: float
    joint_total_ms: float
    hbm_bytes: int
    transferred_bytes: int
    wasted_bytes: int

    @property
    def reuse_accepted(self) -> bool:
        return any(item.path is JointSegmentPath.REUSE for item in self.decisions)


class JointRequestPlanner:
    """Greedy request-level admission after per-Segment Source freeze.

    It never enumerates Source combinations. Options already contain one locked
    Source per Segment; resource conflicts are resolved by marginal saving.
    """

    def __init__(self, *, gamma: float = 0.8, hbm_capacity_bytes: int) -> None:
        if not 0 < gamma <= 1 or hbm_capacity_bytes < 0:
            raise ValueError("invalid joint-planner constraints")
        self.gamma = gamma
        self.hbm_capacity_bytes = hbm_capacity_bytes

    def plan(
        self,
        request_plan_id: str,
        options: Sequence[LockedSegmentOption],
        *,
        shared_sunk_ms: float,
        dense_reference_ms: float,
        joint_shared_interference_ms: float = 0.0,
    ) -> JointRequestPlan:
        if not request_plan_id or shared_sunk_ms < 0 or dense_reference_ms <= 0:
            raise ValueError("invalid request-level planning input")
        if joint_shared_interference_ms < 0:
            raise ValueError("joint interference must be non-negative")
        if len({item.segment_id for item in options}) != len(options):
            raise ValueError("one locked Source option is allowed per Segment")
        accepted = []
        used_hbm = 0
        for option in sorted(
            options,
            key=lambda item: (-item.marginal_saved_ms, item.segment_id),
        ):
            if not option.source_ready or option.marginal_saved_ms <= 0:
                continue
            if used_hbm + option.resident_hbm_bytes > self.hbm_capacity_bytes:
                continue
            accepted.append(option)
            used_hbm += option.resident_hbm_bytes

        def total_for(rows: Sequence[LockedSegmentOption]) -> float:
            accepted_ids = {row.segment_id for row in rows}
            future = joint_shared_interference_ms
            for option in options:
                future += (
                    option.reuse_future_ms
                    if option.segment_id in accepted_ids
                    else option.dense_remaining_ms
                )
            return shared_sunk_ms + future

        while accepted and total_for(accepted) > self.gamma * dense_reference_ms + 1e-9:
            victim = min(accepted, key=lambda item: (item.marginal_saved_ms, item.segment_id))
            accepted.remove(victim)
            used_hbm -= victim.resident_hbm_bytes

        accepted_ids = {item.segment_id for item in accepted}
        decisions = []
        transferred = sum(item.resident_hbm_bytes for item in options if item.source_ready)
        used_transferred = sum(item.resident_hbm_bytes for item in accepted)
        for option in sorted(options, key=lambda item: item.segment_id):
            if option.segment_id in accepted_ids:
                decisions.append(
                    JointSegmentDecision(
                        segment_id=option.segment_id,
                        source_variant_id=option.source_variant_id,
                        path=JointSegmentPath.REUSE,
                        replica_id=option.replica_id,
                        actual_reuse_boundary=option.actual_reuse_boundary,
                        repair_ratio_upper=option.repair_ratio_upper,
                    )
                )
            else:
                reason = (
                    "source_not_ready"
                    if not option.source_ready
                    else "nonpositive_marginal_or_joint_admission"
                )
                decisions.append(
                    JointSegmentDecision(
                        segment_id=option.segment_id,
                        source_variant_id=option.source_variant_id,
                        path=JointSegmentPath.DENSE,
                        replica_id=None,
                        actual_reuse_boundary=None,
                        repair_ratio_upper=None,
                        rejection_reason=reason,
                    )
                )
        joint_total = total_for(accepted)
        return JointRequestPlan(
            request_plan_id=request_plan_id,
            decisions=tuple(decisions),
            shared_sunk_ms=shared_sunk_ms,
            dense_reference_ms=dense_reference_ms,
            joint_future_ms=joint_total - shared_sunk_ms,
            joint_total_ms=joint_total,
            hbm_bytes=used_hbm,
            transferred_bytes=transferred,
            wasted_bytes=max(0, transferred - used_transferred),
        )
