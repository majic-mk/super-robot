from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import CandidateBounds, DecisionReason, SourceDecision


@dataclass(frozen=True)
class ProbePolicy:
    checkpoints: Tuple[int, ...]
    max_layer: int

    def __post_init__(self) -> None:
        if not self.checkpoints:
            raise ValueError("at least one checkpoint is required")
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be sorted and unique")
        if self.checkpoints[-1] > self.max_layer:
            raise ValueError("checkpoint exceeds max_layer")


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

    def select(
        self, bounds_by_layer: Mapping[int, Sequence[CandidateBounds]]
    ) -> SourceDecision:
        last_layer = self.policy.checkpoints[-1]
        saw_quality_coverage = False
        for layer in self.policy.checkpoints:
            candidates = tuple(bounds_by_layer.get(layer, ()))
            saw_quality_coverage = saw_quality_coverage or any(
                candidate.quality_covered for candidate in candidates
            )
            winner = self._confident_winner(candidates)
            if winner is not None:
                return SourceDecision(
                    selected_source_id=winner.source_id,
                    probe_layer=layer,
                    reuse_layer=None,
                    safe_repair_ratio_upper=winner.repair_ratio_upper,
                    prefetch_m=1,
                    reason=DecisionReason.CONFIDENT,
                )
            last_layer = layer
        reason = (
            DecisionReason.MAX_PROBE_UNCERTAIN
            if saw_quality_coverage
            else DecisionReason.QUALITY_UNCOVERED
        )
        return SourceDecision(
            selected_source_id=None,
            probe_layer=last_layer,
            reuse_layer=None,
            safe_repair_ratio_upper=None,
            prefetch_m=0,
            reason=reason,
        )


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
