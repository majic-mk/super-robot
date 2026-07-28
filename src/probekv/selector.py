from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import CandidateBounds, SelectionReason, SourceDecision


class SelectorPolicy(str, Enum):
    STRICT_INTERVAL = "strict_interval"
    FINAL_ECONOMIC_MIN_COST = "final_economic_min_cost"
    FINAL_ECONOMIC_MAX_REUSE = "final_economic_max_reuse"


@dataclass(frozen=True)
class ProbePolicy:
    checkpoints: Tuple[int, ...]
    max_layer: int
    selector_policy: SelectorPolicy = SelectorPolicy.STRICT_INTERVAL
    gamma: float = 0.8
    reuse_ratio_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if not self.checkpoints:
            raise ValueError("at least one checkpoint is required")
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be sorted and unique")
        if self.checkpoints[-1] > self.max_layer:
            raise ValueError("checkpoint exceeds max_layer")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.reuse_ratio_tolerance <= 1.0:
            raise ValueError("reuse_ratio_tolerance must be in [0, 1]")


class DynamicProbeSelector:
    """Conservative early-exit selector using calibrated cost intervals."""

    def __init__(self, policy: ProbePolicy) -> None:
        self.policy = policy

    @staticmethod
    def _confident_winner(
        candidates: Sequence[CandidateBounds],
    ) -> Optional[CandidateBounds]:
        if not candidates:
            return None
        quality_candidates = [candidate for candidate in candidates if candidate.quality_covered]
        if not quality_candidates:
            return None
        best = min(quality_candidates, key=lambda item: item.cost_upper_ms)
        competitors = [
            candidate
            for candidate in quality_candidates
            if candidate.source_id != best.source_id
        ]
        if not competitors:
            return best
        second_lower = min(candidate.cost_lower_ms for candidate in competitors)
        if best.cost_upper_ms < second_lower:
            return best
        return None

    @staticmethod
    def _selected(
        candidate: CandidateBounds,
        layer: int,
        reason: SelectionReason,
    ) -> SourceDecision:
        return SourceDecision(
            selected_source_id=candidate.source_id,
            probe_layer=layer,
            reuse_layer=None,
            safe_repair_ratio_upper=candidate.repair_ratio_upper,
            prefetch_m=1,
            selection_reason=reason,
            predicted_cost_upper_ms=candidate.cost_upper_ms,
        )

    @staticmethod
    def _abstained(layer: int, reason: SelectionReason) -> SourceDecision:
        return SourceDecision(
            selected_source_id=None,
            probe_layer=layer,
            reuse_layer=None,
            safe_repair_ratio_upper=None,
            prefetch_m=0,
            selection_reason=reason,
        )

    def _final_winner(
        self,
        candidates: Sequence[CandidateBounds],
        full_recompute_ms: Optional[float],
    ) -> Tuple[Optional[CandidateBounds], SelectionReason]:
        quality_candidates = [
            candidate for candidate in candidates if candidate.quality_covered
        ]
        if not quality_candidates:
            return None, SelectionReason.NO_QUALITY_SAFE_SOURCE
        if full_recompute_ms is None or full_recompute_ms <= 0:
            raise ValueError(
                "final economic selection requires positive full_recompute_ms"
            )
        economic = [
            candidate
            for candidate in quality_candidates
            if candidate.cost_upper_ms <= self.policy.gamma * full_recompute_ms
        ]
        if not economic:
            return None, SelectionReason.NO_ECONOMIC_SOURCE
        if self.policy.selector_policy is SelectorPolicy.FINAL_ECONOMIC_MIN_COST:
            return (
                min(
                    economic,
                    key=lambda item: (
                        item.cost_upper_ms,
                        item.repair_ratio_upper,
                        item.source_id,
                    ),
                ),
                SelectionReason.FINAL_ECONOMIC_MIN_COST,
            )
        minimum_repair = min(
            candidate.repair_ratio_upper for candidate in economic
        )
        near_max_reuse = [
            candidate
            for candidate in economic
            if candidate.repair_ratio_upper
            <= minimum_repair + self.policy.reuse_ratio_tolerance
        ]
        return (
            min(
                near_max_reuse,
                key=lambda item: (item.cost_upper_ms, item.source_id),
            ),
            SelectionReason.FINAL_MAX_REUSE_LOWER_BOUND,
        )

    def select(
        self,
        bounds_by_layer: Mapping[int, Sequence[CandidateBounds]],
        full_recompute_ms: Optional[float] = None,
    ) -> SourceDecision:
        last_layer = self.policy.checkpoints[-1]
        saw_quality_coverage = False
        for layer in self.policy.checkpoints:
            candidates = tuple(bounds_by_layer.get(layer, ()))
            saw_quality_coverage = saw_quality_coverage or any(
                candidate.quality_covered for candidate in candidates
            )
            at_final_layer = layer == self.policy.max_layer
            if (
                at_final_layer
                and self.policy.selector_policy
                is not SelectorPolicy.STRICT_INTERVAL
            ):
                winner, reason = self._final_winner(
                    candidates, full_recompute_ms
                )
                if winner is None:
                    return self._abstained(layer, reason)
                return self._selected(winner, layer, reason)
            winner = self._confident_winner(candidates)
            if winner is not None:
                return self._selected(
                    winner, layer, SelectionReason.EARLY_CONFIDENT
                )
            last_layer = layer
        reason = (
            SelectionReason.MAX_PROBE_UNCERTAIN
            if saw_quality_coverage
            else SelectionReason.NO_QUALITY_SAFE_SOURCE
        )
        return self._abstained(last_layer, reason)


def normalized_oracle_regret(
    selected_cost: float, oracle_cost: float, worst_cost: float
) -> float:
    denominator = worst_cost - oracle_cost
    if denominator <= 0:
        return 0.0
    return max(0.0, (selected_cost - oracle_cost) / denominator)


def default_probe_checkpoints(total_layers: int) -> Tuple[int, ...]:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if total_layers == 32:
        return (1, 2, 4, 6, 8)
    fractions = (0.03, 0.06, 0.125, 0.1875, 0.25)
    maximum = max(1, int(total_layers * 0.25))
    values = {
        max(1, min(maximum, int(round(total_layers * fraction))))
        for fraction in fractions
    }
    return tuple(sorted(values))


def dense_probe_checkpoints(total_layers: int) -> Tuple[int, ...]:
    """Check every layer up to the inclusive 25% probe ceiling."""
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    maximum = max(1, int(total_layers * 0.25))
    return tuple(range(1, maximum + 1))
