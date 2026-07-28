from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import (
    DecisionReason,
    ExecutionDecision,
    ExecutionMode,
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
        )


@dataclass(frozen=True)
class ReusePlan:
    accepted: bool
    layer: Optional[int]
    repair_ratio_upper: Optional[float]
    timing: Optional[TimingBreakdown]
    reason: DecisionReason


class DynamicReusePlanner:
    def __init__(self, gamma: float = 0.8) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma

    def plan(self, options: Sequence[LayerOption]) -> ReusePlan:
        feasible = []
        for option in options:
            if not option.buffer_ready:
                continue
            timing = option.timing()
            if timing.reuse_total_ms <= self.gamma * timing.full_ms:
                feasible.append((timing.reuse_total_ms, option.layer, option, timing))
        if not feasible:
            reason = (
                DecisionReason.NO_FEASIBLE_LAYER
                if not any(option.buffer_ready for option in options)
                else DecisionReason.ECONOMIC_REJECT
            )
            return ReusePlan(False, None, None, None, reason)
        _, _, selected, timing = min(feasible, key=lambda item: (item[0], item[1]))
        return ReusePlan(
            True,
            selected.layer,
            selected.repair_ratio_upper,
            timing,
            DecisionReason.CONFIDENT,
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
