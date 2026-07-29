from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import (
    CostValueKind,
    DecisionReason,
    ExecutionDecision,
    ExecutionMode,
    InterferenceAccountingMode,
    RejectionReason,
    SelectionReason,
    SourceDecision,
    TimingBreakdown,
)


@dataclass(frozen=True)
class LayerOption:
    layer: int
    repair_ratio_upper: float
    probe_ms: float
    compare_ms: float
    load_ms: float
    overlap_ms: float
    repair_ms: float
    full_ms: float
    buffer_ready: bool = True
    post_ready_blocking_ms: float = 0.0
    load_interference_ms: float = 0.0
    source_ready_ms: Optional[float] = None
    a_resume_ms: Optional[float] = None
    scheduled_step_finish_ms: Optional[float] = None
    repair_selection_ms: float = 0.0
    remaining_layer_ms: float = 0.0
    cost_origin: str = "request_arrival"
    cost_endpoint: str = "first_token_ready"
    cost_value_kind: CostValueKind = CostValueKind.REFINED_ACTUAL
    interference_accounting_mode: InterferenceAccountingMode = (
        InterferenceAccountingMode.INCLUDED_IN_LOAD
    )

    def timing(self) -> TimingBreakdown:
        return TimingBreakdown(
            probe_ms=self.probe_ms,
            compare_ms=self.compare_ms,
            load_ms=self.load_ms,
            visible_load_ms=max(0.0, self.load_ms - self.overlap_ms),
            repair_ms=self.repair_ms,
            full_ms=self.full_ms,
            post_ready_blocking_ms=self.post_ready_blocking_ms,
            load_interference_ms=self.load_interference_ms,
            source_ready_ms=self.source_ready_ms,
            a_resume_ms=self.a_resume_ms,
            scheduled_step_finish_ms=self.scheduled_step_finish_ms,
            overlap_ms=self.overlap_ms,
            evaluated_reuse_boundary=self.layer,
            repair_selection_ms=self.repair_selection_ms,
            remaining_layer_ms=self.remaining_layer_ms,
            cost_origin=self.cost_origin,
            cost_endpoint=self.cost_endpoint,
            cost_value_kind=self.cost_value_kind,
            interference_accounting_mode=(
                self.interference_accounting_mode
            ),
        )


@dataclass(frozen=True)
class ReusePlan:
    accepted: bool
    layer: Optional[int]
    repair_ratio_upper: Optional[float]
    timing: Optional[TimingBreakdown]
    reason: DecisionReason
    evaluated_layer: Optional[int] = None


class DynamicReusePlanner:
    def __init__(self, gamma: float = 0.8) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma

    def plan(self, options: Sequence[LayerOption]) -> ReusePlan:
        feasible = []
        evaluated = []
        for option in options:
            timing = option.timing()
            evaluated.append(
                (timing.reuse_total_ms, option.layer, option, timing)
            )
            if not option.buffer_ready:
                continue
            if timing.reuse_total_ms <= self.gamma * timing.full_ms:
                feasible.append((timing.reuse_total_ms, option.layer, option, timing))
        if not feasible:
            reason = (
                DecisionReason.NO_FEASIBLE_LAYER
                if not any(option.buffer_ready for option in options)
                else DecisionReason.ECONOMIC_REJECT
            )
            if not evaluated:
                return ReusePlan(False, None, None, None, reason)
            _, evaluated_layer, _, observed_timing = min(
                evaluated, key=lambda item: (item[0], item[1])
            )
            return ReusePlan(
                False,
                None,
                None,
                observed_timing,
                reason,
                evaluated_layer=evaluated_layer,
            )
        _, _, selected, timing = min(feasible, key=lambda item: (item[0], item[1]))
        return ReusePlan(
            True,
            selected.layer,
            selected.repair_ratio_upper,
            timing,
            DecisionReason.CONFIDENT,
            evaluated_layer=selected.layer,
        )


def cost_breakdown_from_total(
    total_ms: float,
    full_total_ms: float,
    boundary: int,
    value_kind: CostValueKind,
    *,
    probe_ms: float = 0.0,
    compare_ms: float = 0.0,
    visible_load_ms: float = 0.0,
    post_ready_blocking_ms: float = 0.0,
    repair_selection_ms: float = 0.0,
    load_interference_ms: float = 0.0,
    interference_accounting_mode: InterferenceAccountingMode = (
        InterferenceAccountingMode.INCLUDED_IN_LOAD
    ),
    cost_origin: str = "request_arrival",
    cost_endpoint: str = "first_token_ready",
) -> TimingBreakdown:
    """Build one canonical accounting identity from an aggregate estimate.

    The unexplained remainder is deliberately assigned to
    ``remaining_layer_ms``.  This helper is intended for legacy predictors
    during migration; new hardware predictors should populate every component
    directly.
    """

    known = (
        probe_ms
        + compare_ms
        + visible_load_ms
        + post_ready_blocking_ms
        + repair_selection_ms
        + (
            load_interference_ms
            if interference_accounting_mode
            is InterferenceAccountingMode.EXPLICIT_PENALTY
            else 0.0
        )
    )
    if total_ms + 1e-12 < known:
        raise ValueError("aggregate total is smaller than known components")
    return TimingBreakdown(
        probe_ms=probe_ms,
        compare_ms=compare_ms,
        load_ms=visible_load_ms,
        visible_load_ms=visible_load_ms,
        repair_ms=0.0,
        full_ms=full_total_ms,
        post_ready_blocking_ms=post_ready_blocking_ms,
        load_interference_ms=load_interference_ms,
        overlap_ms=0.0,
        evaluated_reuse_boundary=boundary,
        repair_selection_ms=repair_selection_ms,
        remaining_layer_ms=max(0.0, total_ms - known),
        cost_origin=cost_origin,
        cost_endpoint=cost_endpoint,
        cost_value_kind=value_kind,
        interference_accounting_mode=interference_accounting_mode,
    )


def finalize_execution(
    selection: SourceDecision,
    reuse_plan: Optional[ReusePlan] = None,
) -> ExecutionDecision:
    """Combine source selection with the later, refined time admission.

    Selection and admission are deliberately separate: a selected source is
    retained when refined timing rejects reuse, while abstention can never
    fall through to an arbitrary historical source.
    """

    if selection.abstained:
        rejection_by_reason = {
            SelectionReason.NO_QUALITY_SAFE_SOURCE:
                RejectionReason.QUALITY_GATE_FAILED,
            SelectionReason.NO_ECONOMIC_SOURCE:
                RejectionReason.PREDICTED_TIME_GATE_FAILED,
            SelectionReason.MAX_PROBE_UNCERTAIN:
                RejectionReason.SELECTION_UNCERTAIN,
        }
        rejection = rejection_by_reason.get(
            selection.selection_reason,
            RejectionReason.SELECTION_UNCERTAIN,
        )
        return ExecutionDecision(
            selected_source_id=None,
            selection_reason=selection.selection_reason,
            reuse_accepted=False,
            rejection_reason=rejection,
            execution_mode=ExecutionMode.FULL_RECOMPUTE,
            probe_layer=selection.probe_layer,
            reuse_layer=None,
            safe_repair_ratio_upper=None,
            timing=None,
            selection_state=selection.selection_state,
        )
    if reuse_plan is None:
        raise ValueError("selected source requires a refined reuse plan")
    if reuse_plan.accepted:
        return ExecutionDecision(
            selected_source_id=selection.selected_source_id,
            selection_reason=selection.selection_reason,
            reuse_accepted=True,
            rejection_reason=None,
            execution_mode=ExecutionMode.REUSE,
            probe_layer=selection.probe_layer,
            reuse_layer=reuse_plan.layer,
            safe_repair_ratio_upper=reuse_plan.repair_ratio_upper,
            timing=reuse_plan.timing,
            selection_state=selection.selection_state,
            actual_reuse_boundary=reuse_plan.layer,
        )
    rejection = (
        RejectionReason.NO_FEASIBLE_REUSE_LAYER
        if reuse_plan.reason is DecisionReason.NO_FEASIBLE_LAYER
        else RejectionReason.FINAL_TIME_GATE_FAILED
    )
    return ExecutionDecision(
        selected_source_id=selection.selected_source_id,
        selection_reason=selection.selection_reason,
        reuse_accepted=False,
        rejection_reason=rejection,
        execution_mode=ExecutionMode.FULL_RECOMPUTE,
        probe_layer=selection.probe_layer,
        reuse_layer=None,
        safe_repair_ratio_upper=selection.safe_repair_ratio_upper,
        timing=reuse_plan.timing,
        selection_state=selection.selection_state,
    )


def conservative_ratio_for_layer(
    anchor_ratios: Mapping[int, float], target_layer: int
) -> float:
    """Use the larger value from adjacent calibration anchors."""
    if not anchor_ratios:
        raise ValueError("at least one anchor is required")
    if target_layer in anchor_ratios:
        return anchor_ratios[target_layer]
    layers = sorted(anchor_ratios)
    lower = [layer for layer in layers if layer < target_layer]
    upper = [layer for layer in layers if layer > target_layer]
    neighbours = []
    if lower:
        neighbours.append(anchor_ratios[lower[-1]])
    if upper:
        neighbours.append(anchor_ratios[upper[0]])
    if not neighbours:
        raise ValueError("unable to locate a neighbouring anchor")
    return max(neighbours)


def bandwidth_sufficient(kv_bytes_per_layer: float, compute_ms_per_layer: float, available_bytes_per_ms: float) -> bool:
    """Whether transfer can keep pace with per-layer consumption."""
    if kv_bytes_per_layer < 0 or compute_ms_per_layer <= 0 or available_bytes_per_ms < 0:
        raise ValueError("invalid bandwidth inputs")
    required = kv_bytes_per_layer / compute_ms_per_layer
    return available_bytes_per_ms >= required
