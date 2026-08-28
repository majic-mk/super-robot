from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, Sequence, Tuple

from .v8_schema6_contracts import Gate3SubsetDecision, PlannerSnapshot


@dataclass(frozen=True)
class JointTimelineContext:
    inventory_segment_ids: Tuple[str, ...]
    reuse_segment_ids: Tuple[str, ...]
    dense_fallback_segment_ids: Tuple[str, ...]
    committed_segment_ids: Tuple[str, ...]
    boundary_by_segment: Mapping[str, int]
    union_mask_digest: str
    scheduler_state_id: str

    def __post_init__(self) -> None:
        inventory = set(self.inventory_segment_ids)
        reuse = set(self.reuse_segment_ids)
        dense = set(self.dense_fallback_segment_ids)
        committed = set(self.committed_segment_ids)
        if len(inventory) != len(self.inventory_segment_ids):
            raise ValueError("joint inventory Segment IDs must be unique")
        if reuse & dense or reuse & committed or dense & committed:
            raise ValueError("joint execution sets must be disjoint")
        if reuse | dense | committed != inventory:
            raise ValueError("joint timeline must assign every Segment exactly once")
        if set(self.boundary_by_segment) != reuse:
            raise ValueError("reuse boundaries must exactly cover the reuse set")
        if any(value < 1 for value in self.boundary_by_segment.values()):
            raise ValueError("reuse boundaries are 1-based")
        if not self.union_mask_digest or not self.scheduler_state_id:
            raise ValueError("joint execution context provenance is incomplete")


@dataclass(frozen=True)
class JointTimelineEstimate:
    joint_future_ms: float
    critical_path_components_ms: Mapping[str, float]
    per_segment_attribution_ms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.joint_future_ms < 0:
            raise ValueError("joint future must be non-negative")
        if min(self.critical_path_components_ms.values(), default=0.0) < 0:
            raise ValueError("critical-path components must be non-negative")
        if min(self.per_segment_attribution_ms.values(), default=0.0) < 0:
            raise ValueError("Segment attribution must be non-negative")
        component_total = sum(self.critical_path_components_ms.values())
        if abs(component_total - self.joint_future_ms) > 1e-6:
            raise ValueError("critical-path components must account for joint future once")


class JointTimelineEstimator(Protocol):
    def estimate(self, context: JointTimelineContext) -> JointTimelineEstimate:
        ...


class Gate2Disposition(str, Enum):
    PROVISIONAL_REUSE = "provisional_reuse"
    DEFERRED = "deferred"
    PREDICTED_DENSE = "predicted_dense"


@dataclass(frozen=True)
class FrozenSegmentCandidate:
    segment_id: str
    source_variant_id: str
    predicted_boundary: int
    resident_hbm_bytes: int

    def __post_init__(self) -> None:
        if not self.segment_id or not self.source_variant_id:
            raise ValueError("frozen Segment candidate identity is required")
        if self.predicted_boundary < 1 or self.resident_hbm_bytes < 0:
            raise ValueError("invalid frozen Segment boundary or HBM size")


@dataclass(frozen=True)
class Gate2JointDecision:
    disposition_by_segment: Mapping[str, Gate2Disposition]
    predicted_request_total_ms: float
    dense_reference_total_ms: float
    joint_future_ms: float
    snapshot: PlannerSnapshot
    inventory_segment_ids: Tuple[str, ...]
    source_variant_by_segment: Mapping[str, str]


class PredictedJointPlannerV6:
    """Incremental Gate 2 using one request critical-path estimate.

    The estimator receives the complete inventory.  Unresolved/deferred rows are
    represented as dense fallback, never omitted and never added as independent
    per-Segment TTFTs.
    """

    def __init__(self, estimator: JointTimelineEstimator, *, gamma: float = 0.8) -> None:
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.estimator = estimator
        self.gamma = gamma

    def plan_incremental(
        self,
        *,
        inventory_segment_ids: Sequence[str],
        frozen_candidates: Sequence[FrozenSegmentCandidate],
        existing_provisional_segment_ids: Sequence[str],
        existing_deferred_segment_ids: Sequence[str],
        predicted_dense_segment_ids: Sequence[str],
        committed_segment_ids: Sequence[str],
        actual_sunk_ms: float,
        dense_reference_total_ms: float,
        selection_closed: bool,
        snapshot: PlannerSnapshot,
        current_snapshot: PlannerSnapshot,
        union_mask_digest: str,
    ) -> Gate2JointDecision:
        snapshot.assert_current(current_snapshot)
        inventory = tuple(inventory_segment_ids)
        if not inventory or len(set(inventory)) != len(inventory):
            raise ValueError("Gate 2 requires one complete unique Segment inventory")
        if actual_sunk_ms < 0 or dense_reference_total_ms <= 0:
            raise ValueError("Gate 2 costs are invalid")
        candidates = {row.segment_id: row for row in frozen_candidates}
        if len(candidates) != len(frozen_candidates) or set(candidates) - set(inventory):
            raise ValueError("Gate 2 frozen candidates are invalid")
        provisional = set(existing_provisional_segment_ids)
        deferred = set(existing_deferred_segment_ids)
        predicted_dense = set(predicted_dense_segment_ids)
        committed = set(committed_segment_ids)
        fixed_sets = (provisional, deferred, predicted_dense, committed)
        if any(left & right for index, left in enumerate(fixed_sets) for right in fixed_sets[index + 1:]):
            raise ValueError("Gate 2 state sets must be disjoint")
        if set().union(*fixed_sets, set(candidates)) - set(inventory):
            raise ValueError("Gate 2 state references an unknown Segment")
        if provisional - set(candidates):
            raise ValueError("existing provisional Segments require frozen candidate boundaries")
        disposition = {
            **{segment_id: Gate2Disposition.PROVISIONAL_REUSE for segment_id in provisional},
            **{segment_id: Gate2Disposition.DEFERRED for segment_id in deferred},
            **{segment_id: Gate2Disposition.PREDICTED_DENSE for segment_id in predicted_dense},
        }
        boundary_by_segment = {
            segment_id: candidates[segment_id].predicted_boundary
            for segment_id in provisional
            if segment_id in candidates
        }
        source_by_segment = {
            row.segment_id: row.source_variant_id for row in frozen_candidates
        }
        latest = None
        new_ids = [
            segment_id for segment_id in inventory
            if segment_id in candidates
            and segment_id not in provisional
            and segment_id not in predicted_dense
            and segment_id not in committed
        ]
        for segment_id in new_ids:
            candidate = candidates[segment_id]
            trial_reuse = provisional | {segment_id}
            trial_boundaries = dict(boundary_by_segment)
            trial_boundaries[segment_id] = candidate.predicted_boundary
            dense = set(inventory) - trial_reuse - committed
            context = JointTimelineContext(
                inventory,
                tuple(sorted(trial_reuse)),
                tuple(sorted(dense)),
                tuple(sorted(committed)),
                trial_boundaries,
                union_mask_digest,
                snapshot.scheduler_snapshot_id,
            )
            latest = self.estimator.estimate(context)
            total = actual_sunk_ms + latest.joint_future_ms
            if total <= self.gamma * dense_reference_total_ms + 1e-12:
                provisional.add(segment_id)
                deferred.discard(segment_id)
                boundary_by_segment = trial_boundaries
                disposition[segment_id] = Gate2Disposition.PROVISIONAL_REUSE
            elif selection_closed:
                predicted_dense.add(segment_id)
                deferred.discard(segment_id)
                disposition[segment_id] = Gate2Disposition.PREDICTED_DENSE
            else:
                deferred.add(segment_id)
                disposition[segment_id] = Gate2Disposition.DEFERRED
        final_dense = set(inventory) - provisional - committed
        final_context = JointTimelineContext(
            inventory,
            tuple(sorted(provisional)),
            tuple(sorted(final_dense)),
            tuple(sorted(committed)),
            boundary_by_segment,
            union_mask_digest,
            snapshot.scheduler_snapshot_id,
        )
        latest = self.estimator.estimate(final_context)
        return Gate2JointDecision(
            disposition,
            actual_sunk_ms + latest.joint_future_ms,
            dense_reference_total_ms,
            latest.joint_future_ms,
            snapshot,
            inventory,
            source_by_segment,
        )


class RefinedJointPlannerV6:
    """Gate 3 subset admission with deterministic marginal pruning."""

    def __init__(self, estimator: JointTimelineEstimator, *, gamma: float = 0.8) -> None:
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.estimator = estimator
        self.gamma = gamma

    def plan_subset(
        self,
        *,
        inventory_segment_ids: Sequence[str],
        eligible_ready_segment_ids: Sequence[str],
        committed_segment_ids: Sequence[str],
        actual_boundary_by_segment: Mapping[str, int],
        actual_sunk_ms: float,
        dense_reference_total_ms: float,
        snapshot: PlannerSnapshot,
        current_snapshot: PlannerSnapshot,
        union_mask_digest: str,
    ) -> Gate3SubsetDecision:
        snapshot.assert_current(current_snapshot)
        inventory = tuple(inventory_segment_ids)
        eligible = set(eligible_ready_segment_ids)
        committed = set(committed_segment_ids)
        if len(set(inventory)) != len(inventory) or not inventory:
            raise ValueError("Gate 3 requires a complete unique inventory")
        if eligible & committed or (eligible | committed) - set(inventory):
            raise ValueError("Gate 3 eligible/committed sets are invalid")
        if set(actual_boundary_by_segment) != eligible:
            raise ValueError("actual boundaries must exactly cover ready eligible Segments")
        if min(actual_sunk_ms, dense_reference_total_ms) < 0 or dense_reference_total_ms == 0:
            raise ValueError("Gate 3 costs are invalid")

        order = {segment_id: index for index, segment_id in enumerate(inventory)}

        def evaluate(active: set[str]) -> JointTimelineEstimate:
            dense = set(inventory) - active - committed
            return self.estimator.estimate(
                JointTimelineContext(
                    inventory,
                    tuple(sorted(active)),
                    tuple(sorted(dense)),
                    tuple(sorted(committed)),
                    {segment_id: actual_boundary_by_segment[segment_id] for segment_id in active},
                    union_mask_digest,
                    snapshot.scheduler_snapshot_id,
                )
            )

        active = set(eligible)
        estimate = evaluate(active)
        while active and actual_sunk_ms + estimate.joint_future_ms > self.gamma * dense_reference_total_ms + 1e-12:
            marginal = []
            for segment_id in active:
                without = evaluate(active - {segment_id})
                saving = without.joint_future_ms - estimate.joint_future_ms
                marginal.append((saving, order[segment_id], segment_id))
            _, _, victim = min(marginal)
            active.remove(victim)
            estimate = evaluate(active)
        accepted = tuple(segment_id for segment_id in inventory if segment_id in active)
        rejected = tuple(segment_id for segment_id in inventory if segment_id in eligible - active)
        untouched = tuple(segment_id for segment_id in inventory if segment_id not in eligible)
        reasons = {
            **{segment_id: "refined_joint_gamma_accepted" for segment_id in accepted},
            **{segment_id: "refined_marginal_pruned" for segment_id in rejected},
            **{segment_id: "not_gate3_eligible" for segment_id in untouched},
        }
        return Gate3SubsetDecision(
            accepted,
            rejected,
            untouched,
            actual_sunk_ms + estimate.joint_future_ms,
            dense_reference_total_ms,
            snapshot,
            reasons,
        )


class DeterministicJointTimelineEstimator:
    """No-GPU joint estimator used by tests/simulation, never paper evidence.

    A single request base path is charged once.  Reuse changes only active-row
    work and adds request-level load/block/interference maxima, so Segment TTFTs
    are never summed as independent requests.
    """

    def __init__(
        self,
        *,
        base_future_ms: float,
        dense_cost_ms_by_segment: Mapping[str, float],
        reuse_cost_ms_by_segment: Mapping[str, float],
        committed_cost_ms_by_segment: Mapping[str, float] | None = None,
        joint_overhead_ms: float = 0.0,
    ) -> None:
        if min(
            base_future_ms,
            joint_overhead_ms,
            *dense_cost_ms_by_segment.values(),
            *reuse_cost_ms_by_segment.values(),
            *(committed_cost_ms_by_segment or {}).values(),
        ) < 0:
            raise ValueError("deterministic joint costs must be non-negative")
        self.base = float(base_future_ms)
        self.dense = dict(dense_cost_ms_by_segment)
        self.reuse = dict(reuse_cost_ms_by_segment)
        self.committed = dict(committed_cost_ms_by_segment or {})
        self.overhead = float(joint_overhead_ms)

    def estimate(self, context: JointTimelineContext) -> JointTimelineEstimate:
        missing = set(context.inventory_segment_ids) - set(self.dense)
        if missing or set(context.reuse_segment_ids) - set(self.reuse):
            raise ValueError("joint estimator lacks a Segment profile")
        row_work = sum(
            self.reuse[segment_id]
            if segment_id in set(context.reuse_segment_ids)
            else self.dense[segment_id]
            for segment_id in context.inventory_segment_ids
            if segment_id not in set(context.committed_segment_ids)
        )
        committed_work = sum(
            self.committed.get(segment_id, 0.0)
            for segment_id in context.committed_segment_ids
        )
        total = self.base + row_work + committed_work + self.overhead
        return JointTimelineEstimate(
            total,
            {
                "base_remaining": self.base,
                "active_row_work": row_work,
                "committed_fixed_path": committed_work,
                "joint_overhead": self.overhead,
            },
            {
                segment_id: (
                    self.reuse[segment_id]
                    if segment_id in set(context.reuse_segment_ids)
                    else self.dense[segment_id]
                )
                for segment_id in context.inventory_segment_ids
                if segment_id not in set(context.committed_segment_ids)
            },
        )
