from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import DecisionReason, TimingBreakdown


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

    def timing(self) -> TimingBreakdown:
        return TimingBreakdown(
            probe_ms=self.probe_ms,
            compare_ms=self.compare_ms,
            load_ms=self.load_ms,
            visible_load_ms=max(0.0, self.load_ms - self.overlap_ms),
            repair_ms=self.repair_ms,
            full_ms=self.full_ms,
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
