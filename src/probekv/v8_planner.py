from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple


class SegmentPlanState(str, Enum):
    UNDECIDED = "undecided"
    PROVISIONAL_REUSE = "provisional_reuse"
    PREDICTED_DENSE = "predicted_dense"
    FINAL_REUSE = "final_reuse"
    REFINED_DENSE = "refined_dense"


@dataclass(frozen=True)
class UnifiedCostComponents:
    probe_ms: float = 0.0
    metadata_ms: float = 0.0
    selection_ms: float = 0.0
    visible_load_ms: float = 0.0
    post_ready_blocking_ms: float = 0.0
    interference_ms: float = 0.0
    repair_selection_ms: float = 0.0
    repair_ms: float = 0.0
    remaining_ms: float = 0.0
    origin: str = "request_arrival"
    endpoint: str = "first_token_ready"

    def __post_init__(self) -> None:
        if min(self.numeric_components) < 0:
            raise ValueError("unified cost components must be non-negative")
        if self.origin != "request_arrival" or self.endpoint != "first_token_ready":
            raise ValueError("v8 costs must use one frozen origin and endpoint")

    @property
    def numeric_components(self) -> Tuple[float, ...]:
        return (
            self.probe_ms,
            self.metadata_ms,
            self.selection_ms,
            self.visible_load_ms,
            self.post_ready_blocking_ms,
            self.interference_ms,
            self.repair_selection_ms,
            self.repair_ms,
            self.remaining_ms,
        )

    @property
    def total_ms(self) -> float:
        return sum(self.numeric_components)


@dataclass(frozen=True)
class PredictedSegmentOption:
    segment_id: str
    source_variant_id: str
    artifact_id: str
    replica_id: str
    replica_generation: int
    placement_epoch: int
    predicted_boundary: int
    resident_hbm_bytes: int
    future_cost_upper: UnifiedCostComponents
    dense_remaining_ms: float
    already_prefetched: bool = False

    def __post_init__(self) -> None:
        if not all(
            (self.segment_id, self.source_variant_id, self.artifact_id, self.replica_id)
        ):
            raise ValueError("complete predicted Segment option is required")
        if min(
            self.replica_generation,
            self.placement_epoch,
            self.predicted_boundary,
        ) < 1:
            raise ValueError("invalid predicted Replica or boundary generation")
        if self.resident_hbm_bytes < 0 or self.dense_remaining_ms < 0:
            raise ValueError("invalid predicted Segment resource cost")
        if any(
            (
                self.future_cost_upper.probe_ms,
                self.future_cost_upper.metadata_ms,
                self.future_cost_upper.selection_ms,
            )
        ):
            raise ValueError("shared selection sunk cost cannot be repeated per Segment")

    @property
    def predicted_reuse_future_ms(self) -> float:
        return self.future_cost_upper.total_ms

    @property
    def marginal_saved_ms(self) -> float:
        return self.dense_remaining_ms - self.predicted_reuse_future_ms


@dataclass(frozen=True)
class PredictedSegmentDecision:
    segment_id: str
    source_variant_id: str
    state: SegmentPlanState
    replica_id: Optional[str]
    predicted_boundary: Optional[int]
    reason: str


@dataclass(frozen=True)
class PredictedRequestPlan:
    request_plan_id: str
    decisions: Tuple[PredictedSegmentDecision, ...]
    shared_sunk_ms: float
    predicted_total_upper_ms: float
    dense_reference_ms: float
    hbm_bytes: int
    gamma_bound_met: bool
    dense_remaining_ms_by_segment: Mapping[str, float]

    @property
    def provisional_segment_ids(self) -> Tuple[str, ...]:
        return tuple(
            item.segment_id
            for item in self.decisions
            if item.state is SegmentPlanState.PROVISIONAL_REUSE
        )


class PredictedJointPlanner:
    """Request-level resource planning before physical Replica prefetch."""

    def __init__(self, *, gamma: float = 0.8, hbm_capacity_bytes: int) -> None:
        if not 0 < gamma <= 1 or hbm_capacity_bytes < 0:
            raise ValueError("invalid predicted planner constraints")
        self.gamma = gamma
        self.hbm_capacity_bytes = hbm_capacity_bytes

    def plan(
        self,
        request_plan_id: str,
        options: Sequence[PredictedSegmentOption],
        *,
        shared_sunk: UnifiedCostComponents,
        dense_reference_ms: float,
        joint_interference_upper_ms: float = 0.0,
    ) -> PredictedRequestPlan:
        if not request_plan_id or dense_reference_ms <= 0 or joint_interference_upper_ms < 0:
            raise ValueError("invalid predicted request input")
        if any(
            (
                shared_sunk.visible_load_ms,
                shared_sunk.post_ready_blocking_ms,
                shared_sunk.interference_ms,
                shared_sunk.repair_selection_ms,
                shared_sunk.repair_ms,
                shared_sunk.remaining_ms,
            )
        ):
            raise ValueError("shared sunk cost may contain only probe/metadata/selection")
        if len({item.segment_id for item in options}) != len(options):
            raise ValueError("one frozen Source option is allowed per Segment")

        forced = [item for item in options if item.already_prefetched]
        accepted = list(forced)
        used_hbm = sum(item.resident_hbm_bytes for item in forced)
        if used_hbm > self.hbm_capacity_bytes:
            raise MemoryError("already-prefetched Segments exceed the HBM contract")
        for option in sorted(
            (item for item in options if not item.already_prefetched),
            key=lambda item: (-item.marginal_saved_ms, item.segment_id),
        ):
            if option.marginal_saved_ms <= 0:
                continue
            if used_hbm + option.resident_hbm_bytes > self.hbm_capacity_bytes:
                continue
            accepted.append(option)
            used_hbm += option.resident_hbm_bytes

        forced_ids = {item.segment_id for item in forced}

        def request_total(rows: Sequence[PredictedSegmentOption]) -> float:
            accepted_ids = {item.segment_id for item in rows}
            future = joint_interference_upper_ms
            for option in options:
                future += (
                    option.predicted_reuse_future_ms
                    if option.segment_id in accepted_ids
                    else option.dense_remaining_ms
                )
            return shared_sunk.total_ms + future

        while (
            any(item.segment_id not in forced_ids for item in accepted)
            and request_total(accepted) > self.gamma * dense_reference_ms + 1e-12
        ):
            victim = min(
                (item for item in accepted if item.segment_id not in forced_ids),
                key=lambda item: (item.marginal_saved_ms, item.segment_id),
            )
            accepted.remove(victim)
            used_hbm -= victim.resident_hbm_bytes

        accepted_ids = {item.segment_id for item in accepted}
        total = request_total(accepted)
        decisions = []
        for option in options:
            if option.segment_id in accepted_ids:
                decisions.append(
                    PredictedSegmentDecision(
                        option.segment_id,
                        option.source_variant_id,
                        SegmentPlanState.PROVISIONAL_REUSE,
                        option.replica_id,
                        option.predicted_boundary,
                        "already_prefetched_monotonic"
                        if option.already_prefetched
                        else "predicted_joint_economic",
                    )
                )
            else:
                decisions.append(
                    PredictedSegmentDecision(
                        option.segment_id,
                        option.source_variant_id,
                        SegmentPlanState.PREDICTED_DENSE,
                        None,
                        None,
                        "hbm_marginal_or_gamma",
                    )
                )
        return PredictedRequestPlan(
            request_plan_id,
            tuple(decisions),
            shared_sunk.total_ms,
            total,
            dense_reference_ms,
            used_hbm,
            total <= self.gamma * dense_reference_ms + 1e-12,
            {item.segment_id: item.dense_remaining_ms for item in options},
        )


@dataclass(frozen=True)
class RefinedSegmentMeasurement:
    segment_id: str
    source_variant_id: str
    actual_boundary: int
    actual_and_profiled_future: UnifiedCostComponents
    dense_remaining_ms: float
    source_ready: bool
    transferred_bytes: int = 0
    wasted_bytes_if_dense: int = 0

    @property
    def refined_reuse_future_ms(self) -> float:
        return self.actual_and_profiled_future.total_ms

    @property
    def marginal_saved_ms(self) -> float:
        return self.dense_remaining_ms - self.refined_reuse_future_ms


@dataclass(frozen=True)
class RefinedRequestPlan:
    request_plan_id: str
    decisions: Tuple[PredictedSegmentDecision, ...]
    refined_total_ms: float
    dense_reference_ms: float
    transferred_bytes: int
    wasted_bytes: int
    gamma_bound_met: bool

    @property
    def final_reuse_segment_ids(self) -> Tuple[str, ...]:
        return tuple(
            item.segment_id
            for item in self.decisions
            if item.state is SegmentPlanState.FINAL_REUSE
        )


class RefinedJointPlanner:
    """Final event-driven admission; it may only downgrade provisional reuse."""

    def __init__(self, *, gamma: float = 0.8) -> None:
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma

    def plan(
        self,
        predicted: PredictedRequestPlan,
        measurements: Mapping[str, RefinedSegmentMeasurement],
        *,
        actual_shared_sunk_ms: float,
        joint_actual_interference_ms: float = 0.0,
    ) -> RefinedRequestPlan:
        if min(actual_shared_sunk_ms, joint_actual_interference_ms) < 0:
            raise ValueError("refined actual costs must be non-negative")
        provisional = set(predicted.provisional_segment_ids)
        if set(measurements) != provisional:
            raise ValueError("refined measurements must exactly cover provisional reuse")
        for decision in predicted.decisions:
            if decision.segment_id in measurements:
                measurement = measurements[decision.segment_id]
                if measurement.source_variant_id != decision.source_variant_id:
                    raise ValueError("refined planner changed a frozen Source")

        active = [
            item
            for item in measurements.values()
            if item.source_ready and item.marginal_saved_ms > 0
        ]

        predicted_dense = {
            item.segment_id
            for item in predicted.decisions
            if item.state is SegmentPlanState.PREDICTED_DENSE
        }

        def total(rows: Sequence[RefinedSegmentMeasurement]) -> float:
            accepted = {item.segment_id for item in rows}
            value = actual_shared_sunk_ms + joint_actual_interference_ms
            for segment_id in provisional:
                measurement = measurements[segment_id]
                value += (
                    measurement.refined_reuse_future_ms
                    if segment_id in accepted
                    else measurement.dense_remaining_ms
                )
            for segment_id in predicted_dense:
                value += predicted.dense_remaining_ms_by_segment[segment_id]
            return value

        while active and total(active) > self.gamma * predicted.dense_reference_ms + 1e-12:
            victim = min(active, key=lambda item: (item.marginal_saved_ms, item.segment_id))
            active.remove(victim)
        accepted = {item.segment_id for item in active}
        decisions = []
        for decision in predicted.decisions:
            if decision.segment_id in predicted_dense:
                decisions.append(decision)
            elif decision.segment_id in accepted:
                measurement = measurements[decision.segment_id]
                decisions.append(
                    PredictedSegmentDecision(
                        decision.segment_id,
                        decision.source_variant_id,
                        SegmentPlanState.FINAL_REUSE,
                        decision.replica_id,
                        measurement.actual_boundary,
                        "refined_gamma_accepted",
                    )
                )
            else:
                decisions.append(
                    PredictedSegmentDecision(
                        decision.segment_id,
                        decision.source_variant_id,
                        SegmentPlanState.REFINED_DENSE,
                        None,
                        None,
                        "not_ready_marginal_or_refined_gamma",
                    )
                )
        transferred = sum(item.transferred_bytes for item in measurements.values())
        wasted = sum(
            item.wasted_bytes_if_dense
            if item.segment_id in accepted
            else item.transferred_bytes
            for item in measurements.values()
        )
        refined_total = total(active)
        return RefinedRequestPlan(
            predicted.request_plan_id,
            tuple(decisions),
            refined_total,
            predicted.dense_reference_ms,
            transferred,
            wasted,
            refined_total <= self.gamma * predicted.dense_reference_ms + 1e-12,
        )
