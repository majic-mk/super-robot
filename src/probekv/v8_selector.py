from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .v8_contracts import (
    AbstainReason,
    CandidateCounts,
    InsufficientRankingPolicy,
    ResidualCandidate,
    ResidualLockReason,
    ResidualSelectionDecision,
    ResidualSelectionState,
    SelectionScope,
    SelectorPolicyProfile,
)


def score_repair_token_count(token_count: int, ratio: float = 0.15) -> int:
    """Count ignored only for the residual Source score; keeps N-m positive."""
    if token_count < 2 or not 0 <= ratio <= 1:
        raise ValueError("residual scoring requires N>=2 and ratio in [0, 1]")
    return min(token_count - 1, int(math.ceil(ratio * token_count)))


def cacheblend_repair_token_count(token_count: int, ratio: float = 0.15) -> int:
    """Conservative CacheBlend repair count, including the r=1 endpoint."""
    if token_count <= 0 or not 0 <= ratio <= 1:
        raise ValueError("repair requires positive N and ratio in [0, 1]")
    if ratio == 0:
        return 0
    if ratio == 1:
        return token_count
    return min(token_count, int(math.ceil(ratio * token_count)))


def normalized_token_k_drifts(
    current_k: Sequence[Sequence[float]],
    source_k: Sequence[Sequence[float]],
    *,
    epsilon: float = 1e-12,
) -> Tuple[float, ...]:
    if epsilon <= 0 or len(current_k) != len(source_k) or not current_k:
        raise ValueError("aligned non-empty K rows and positive epsilon are required")
    result = []
    width = None
    for current, source in zip(current_k, source_k):
        if len(current) != len(source) or not current:
            raise ValueError("current and Source K geometry differs")
        if width is None:
            width = len(current)
        elif len(current) != width:
            raise ValueError("K rows must have one fixed geometry")
        numerator = math.sqrt(
            sum((float(left) - float(right)) ** 2 for left, right in zip(current, source))
        )
        denominator = max(
            math.sqrt(sum(float(value) ** 2 for value in current)), epsilon
        )
        result.append(numerator / denominator)
    return tuple(result)


def residual_k_score(
    token_drifts: Sequence[float],
    *,
    ratio: float = 0.15,
    absolute_positions: Optional[Sequence[int]] = None,
) -> Tuple[float, Tuple[int, ...]]:
    """Return residual mean and deterministic repair indices.

    Equal drifts are ordered by absolute token position, then local row index.
    """
    if len(token_drifts) < 2 or any(float(value) < 0 for value in token_drifts):
        raise ValueError("residual score requires at least two non-negative drifts")
    positions = tuple(
        range(len(token_drifts)) if absolute_positions is None else absolute_positions
    )
    if len(positions) != len(token_drifts) or len(set(positions)) != len(positions):
        raise ValueError("absolute positions must uniquely cover all token rows")
    count = score_repair_token_count(len(token_drifts), ratio)
    ranked = sorted(
        range(len(token_drifts)),
        key=lambda index: (-float(token_drifts[index]), positions[index], index),
    )
    repair = tuple(sorted(ranked[:count], key=lambda index: positions[index]))
    repaired = set(repair)
    residual = [
        float(value) for index, value in enumerate(token_drifts) if index not in repaired
    ]
    return sum(residual) / len(residual), repair


@dataclass(frozen=True)
class ComparisonBudget:
    dense_reference_ms: float
    shared_probe_ms: float
    metadata_ms: float
    other_selection_sunk_ms: float
    budget_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.dense_reference_ms <= 0 or not 0 < self.budget_fraction <= 1:
            raise ValueError("invalid comparison budget")
        if min(
            self.shared_probe_ms,
            self.metadata_ms,
            self.other_selection_sunk_ms,
        ) < 0:
            raise ValueError("selection sunk costs must be non-negative")

    @property
    def available_ms(self) -> float:
        return max(
            0.0,
            self.budget_fraction * self.dense_reference_ms
            - self.shared_probe_ms
            - self.metadata_ms
            - self.other_selection_sunk_ms,
        )

    def largest_batch(
        self,
        predicted_batch_ms: Mapping[int, float],
        eligible_k: int,
        max_compared_k: int = 16,
    ) -> int:
        if eligible_k < 0 or not 1 <= max_compared_k <= 16:
            raise ValueError("invalid comparison candidate count")
        feasible = [
            count
            for count, cost in predicted_batch_ms.items()
            if 0 <= count <= min(eligible_k, max_compared_k)
            and cost >= 0
            and cost <= self.available_ms + 1e-12
        ]
        return max(feasible, default=0)

    def new_ledger(self) -> "RequestSelectionBudgetLedger":
        return RequestSelectionBudgetLedger(self.available_ms)


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    predicted_upper_ms: float


@dataclass(frozen=True)
class BudgetSettlement:
    reservation_id: str
    predicted_upper_ms: float
    actual_critical_path_ms: float
    realized_overrun: bool
    total_budget_exceeded: bool


class RequestSelectionBudgetLedger:
    """One request-level reserve/settle ledger shared by every Segment.

    Admission uses predicted upper bounds.  A legal batch may still exceed its
    reservation at runtime; that is an observed prediction overrun, not an
    admission invariant violation.  Once the realized total exceeds the fixed
    budget, no further comparison batch may start.
    """

    def __init__(self, total_budget_ms: float) -> None:
        if total_budget_ms < 0:
            raise ValueError("selection budget must be non-negative")
        self.total_budget_ms = float(total_budget_ms)
        self.settled_actual_ms = 0.0
        self._reservations: Dict[str, float] = {}
        self.budget_admission_violation_count = 0
        self.selection_budget_realized_overrun_count = 0
        self.comparison_closed = total_budget_ms == 0

    @property
    def outstanding_reserved_ms(self) -> float:
        return sum(self._reservations.values())

    @property
    def available_ms(self) -> float:
        return max(
            0.0,
            self.total_budget_ms
            - self.settled_actual_ms
            - self.outstanding_reserved_ms,
        )

    def reserve(self, reservation_id: str, predicted_upper_ms: float) -> BudgetReservation:
        if not reservation_id or predicted_upper_ms < 0:
            raise ValueError("invalid comparison reservation")
        if reservation_id in self._reservations:
            raise ValueError("comparison reservation IDs must be unique")
        if self.comparison_closed or predicted_upper_ms > self.available_ms + 1e-12:
            self.budget_admission_violation_count += 1
            raise RuntimeError("comparison batch exceeds the request-level budget")
        self._reservations[reservation_id] = float(predicted_upper_ms)
        return BudgetReservation(reservation_id, float(predicted_upper_ms))

    def cancel(self, reservation_id: str) -> None:
        if reservation_id not in self._reservations:
            raise KeyError("unknown comparison reservation")
        del self._reservations[reservation_id]

    def settle(
        self,
        reservation_id: str,
        actual_critical_path_ms: float,
    ) -> BudgetSettlement:
        if actual_critical_path_ms < 0:
            raise ValueError("actual comparison time must be non-negative")
        if reservation_id not in self._reservations:
            raise KeyError("unknown comparison reservation")
        predicted = self._reservations.pop(reservation_id)
        previous = self.settled_actual_ms
        self.settled_actual_ms += float(actual_critical_path_ms)
        if self.settled_actual_ms + 1e-12 < previous:
            raise RuntimeError("settled selection cost must be monotonic")
        realized = actual_critical_path_ms > predicted + 1e-12
        exceeded = self.settled_actual_ms > self.total_budget_ms + 1e-12
        if realized:
            self.selection_budget_realized_overrun_count += 1
        if exceeded:
            self.comparison_closed = True
        return BudgetSettlement(
            reservation_id,
            predicted,
            float(actual_critical_path_ms),
            realized,
            exceeded,
        )

    def audit(self) -> Mapping[str, float | int | bool]:
        return {
            "total_budget_ms": self.total_budget_ms,
            "settled_actual_ms": self.settled_actual_ms,
            "outstanding_reserved_ms": self.outstanding_reserved_ms,
            "available_ms": self.available_ms,
            "budget_admission_violation_count": self.budget_admission_violation_count,
            "selection_budget_realized_overrun_count": (
                self.selection_budget_realized_overrun_count
            ),
            "comparison_closed": self.comparison_closed,
        }


@dataclass(frozen=True)
class SelectionScratchPlan:
    compared_k: int
    microbatch_k: int
    batches: Tuple[Tuple[int, ...], ...]
    peak_bytes: int
    capacity_bytes: int


def plan_selection_scratch(
    *,
    compared_k: int,
    source_state_bytes: int,
    current_state_bytes: int,
    capacity_bytes: int,
) -> SelectionScratchPlan:
    """Bound vectorized K-only comparison without loading any full-KV Artifact."""
    if compared_k < 1 or min(source_state_bytes, current_state_bytes) < 0:
        raise ValueError("invalid SelectionState comparison geometry")
    if source_state_bytes == 0 or capacity_bytes <= current_state_bytes:
        raise MemoryError("SelectionState scratch cannot hold one Source")
    microbatch_k = min(
        compared_k,
        (capacity_bytes - current_state_bytes) // source_state_bytes,
    )
    if microbatch_k < 1:
        raise MemoryError("SelectionState scratch cannot hold one Source")
    batches = tuple(
        tuple(range(start, min(compared_k, start + microbatch_k)))
        for start in range(0, compared_k, microbatch_k)
    )
    return SelectionScratchPlan(
        compared_k=compared_k,
        microbatch_k=microbatch_k,
        batches=batches,
        peak_bytes=current_state_bytes + microbatch_k * source_state_bytes,
        capacity_bytes=capacity_bytes,
    )


class TrainingFreeResidualKSelector:
    def __init__(
        self,
        profile: SelectorPolicyProfile,
        *,
        gamma: float = 0.8,
        insufficient_ranking_policy: InsufficientRankingPolicy = (
            InsufficientRankingPolicy.ABSTAIN_DENSE
        ),
    ) -> None:
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.profile = profile
        self.gamma = gamma
        self.insufficient_ranking_policy = InsufficientRankingPolicy(
            insufficient_ranking_policy
        )

    @staticmethod
    def _scope(counts: CandidateCounts) -> SelectionScope:
        if counts.correctness_eligible_k == 1:
            return SelectionScope.SINGLE_CORRECTNESS_ELIGIBLE
        if counts.compared_k < counts.correctness_eligible_k:
            return SelectionScope.CFO_BUDGET_TRUNCATED
        return SelectionScope.FULL_CORRECTNESS_SET

    def _economic(
        self,
        candidate: ResidualCandidate,
        dense_remaining_ms: float,
    ) -> bool:
        return (
            candidate.predicted_future_upper_ms
            <= self.gamma * dense_remaining_ms + 1e-12
        )

    def evaluate_checkpoint(
        self,
        *,
        completed_depth: int,
        counts: CandidateCounts,
        candidates: Sequence[ResidualCandidate],
        shared_sunk_ms: float,
        dense_reference_ms: float,
        previous_winner_source_id: Optional[str] = None,
        gate1_dense_remaining_ms: Optional[float] = None,
    ) -> ResidualSelectionDecision:
        if completed_depth < 0 or shared_sunk_ms < 0 or dense_reference_ms <= 0:
            raise ValueError("invalid checkpoint accounting")
        if len(candidates) != counts.compared_k:
            raise ValueError("compared_k must equal the number of scored candidates")
        if len({item.source_variant_id for item in candidates}) != len(candidates):
            raise ValueError("a Source Variant may be compared only once per checkpoint")
        if completed_depth == 0:
            return ResidualSelectionDecision(
                ResidualSelectionState.PENDING,
                completed_depth,
                counts,
                selection_scope=self._scope(counts),
            )
        if completed_depth not in self.profile.checkpoint_depths:
            raise ValueError("completed depth is not a frozen online checkpoint")
        at_max = completed_depth == self.profile.max_completed_depth
        scope = self._scope(counts)
        economic_dense_remaining_ms = (
            dense_reference_ms
            if gate1_dense_remaining_ms is None
            else float(gate1_dense_remaining_ms)
        )
        if economic_dense_remaining_ms <= 0:
            raise ValueError("Gate 1 dense remaining time must be positive")

        if counts.correctness_eligible_k == 0:
            return ResidualSelectionDecision(
                ResidualSelectionState.ABSTAINED,
                completed_depth,
                counts,
                abstain_reason=AbstainReason.NO_CORRECTNESS_ELIGIBLE_SOURCE,
                selection_scope=scope,
            )
        if counts.compared_k == 0:
            reason = (
                AbstainReason.SELECTION_STATE_UNAVAILABLE
                if counts.selection_state_available_k == 0
                else AbstainReason.COMPARISON_BUDGET_EXHAUSTED
            )
            return ResidualSelectionDecision(
                ResidualSelectionState.ABSTAINED,
                completed_depth,
                counts,
                abstain_reason=reason,
                selection_scope=scope,
            )

        ranked = sorted(
            candidates,
            key=lambda item: (
                item.residual_score,
                item.predicted_future_upper_ms,
                item.source_variant_id,
            ),
        )
        best = ranked[0]

        if counts.correctness_eligible_k == 1:
            if self._economic(best, economic_dense_remaining_ms):
                return ResidualSelectionDecision(
                    ResidualSelectionState.LOCKED,
                    completed_depth,
                    counts,
                    selected_source_variant_id=best.source_variant_id,
                    lock_reason=ResidualLockReason.SINGLE_CORRECTNESS_ELIGIBLE_SOURCE,
                    selection_scope=SelectionScope.SINGLE_CORRECTNESS_ELIGIBLE,
                    best_source_variant_id=best.source_variant_id,
                    predicted_total_upper_ms=(
                        shared_sunk_ms + best.predicted_future_upper_ms
                    ),
                )
            return ResidualSelectionDecision(
                ResidualSelectionState.ABSTAINED if at_max else ResidualSelectionState.PENDING,
                completed_depth,
                counts,
                abstain_reason=(
                    AbstainReason.PRELIMINARY_ECONOMIC_REJECTION
                    if at_max else AbstainReason.NONE
                ),
                selection_scope=SelectionScope.SINGLE_CORRECTNESS_ELIGIBLE,
                best_source_variant_id=best.source_variant_id,
            )

        if counts.compared_k < 2:
            if (
                at_max
                and self.insufficient_ranking_policy
                is InsufficientRankingPolicy.CFO_TOP1_FALLBACK
                and self._economic(best, economic_dense_remaining_ms)
            ):
                return ResidualSelectionDecision(
                    ResidualSelectionState.LOCKED,
                    completed_depth,
                    counts,
                    selected_source_variant_id=best.source_variant_id,
                    lock_reason=ResidualLockReason.CFO_TOP1_FALLBACK,
                    selection_scope=SelectionScope.CFO_BUDGET_TRUNCATED,
                    best_source_variant_id=best.source_variant_id,
                    predicted_total_upper_ms=(
                        shared_sunk_ms + best.predicted_future_upper_ms
                    ),
                )
            return ResidualSelectionDecision(
                ResidualSelectionState.ABSTAINED,
                completed_depth,
                counts,
                abstain_reason=AbstainReason.INSUFFICIENT_RANKING_COVERAGE,
                selection_scope=SelectionScope.INSUFFICIENT_RANKING_COVERAGE,
                best_source_variant_id=best.source_variant_id,
            )

        second = ranked[1]
        margin = (second.residual_score - best.residual_score) / max(
            second.residual_score, 1e-12
        )
        economic = self._economic(best, economic_dense_remaining_ms)
        common = dict(
            completed_depth=completed_depth,
            counts=counts,
            selection_scope=scope,
            best_source_variant_id=best.source_variant_id,
            runner_up_source_variant_id=second.source_variant_id,
            margin_defined=True,
            margin_value=margin,
            current_state_ranking_performed=True,
        )
        if economic and margin >= self.profile.eta_strong:
            return ResidualSelectionDecision(
                ResidualSelectionState.LOCKED,
                selected_source_variant_id=best.source_variant_id,
                lock_reason=ResidualLockReason.STRONG_MARGIN_EARLY_EXIT,
                predicted_total_upper_ms=shared_sunk_ms + best.predicted_future_upper_ms,
                **common,
            )
        if (
            economic
            and previous_winner_source_id == best.source_variant_id
            and margin >= self.profile.eta
        ):
            return ResidualSelectionDecision(
                ResidualSelectionState.LOCKED,
                selected_source_variant_id=best.source_variant_id,
                lock_reason=ResidualLockReason.STABLE_MARGIN_EARLY_EXIT,
                predicted_total_upper_ms=shared_sunk_ms + best.predicted_future_upper_ms,
                **common,
            )
        if not at_max:
            return ResidualSelectionDecision(ResidualSelectionState.PENDING, **common)

        limit = (
            (1 + self.profile.residual_band_relative_tolerance)
            * best.residual_score
            + self.profile.residual_band_numeric_slack
        )
        final = [
            item
            for item in ranked
            if item.residual_score <= limit
            and self._economic(item, economic_dense_remaining_ms)
        ]
        if not final:
            return ResidualSelectionDecision(
                ResidualSelectionState.ABSTAINED,
                abstain_reason=AbstainReason.MAX_DEPTH_NO_ECONOMIC_CANDIDATE,
                **common,
            )
        selected = min(
            final,
            key=lambda item: (
                item.predicted_future_upper_ms,
                item.residual_score,
                item.source_variant_id,
            ),
        )
        return ResidualSelectionDecision(
            ResidualSelectionState.LOCKED,
            selected_source_variant_id=selected.source_variant_id,
            lock_reason=ResidualLockReason.MAX_DEPTH_RESIDUAL_COST,
            predicted_total_upper_ms=shared_sunk_ms + selected.predicted_future_upper_ms,
            **common,
        )

    def evaluate_checkpoint_trace(self, **kwargs):
        """Return the explicit selector-decision/Gate-1 transition.

        ``evaluate_checkpoint`` remains the schema-v4-compatible convenience
        result.  Schema-v5 runtimes consume this trace so a residual winner is
        never confused with a formally frozen Source.
        """
        final = self.evaluate_checkpoint(**kwargs)
        depth = int(kwargs["completed_depth"])
        counts = kwargs["counts"]
        candidates = tuple(kwargs["candidates"])
        if depth == 0 or not candidates or counts.correctness_eligible_k == 0:
            return (final,)
        ranked = sorted(
            candidates,
            key=lambda item: (
                item.residual_score,
                item.predicted_future_upper_ms,
                item.source_variant_id,
            ),
        )
        chosen = None
        reason = ResidualLockReason.NONE
        if counts.correctness_eligible_k == 1:
            chosen = ranked[0]
            reason = ResidualLockReason.SINGLE_CORRECTNESS_ELIGIBLE_SOURCE
        elif counts.compared_k >= 2:
            best, second = ranked[:2]
            margin = (second.residual_score - best.residual_score) / max(
                second.residual_score, 1e-12
            )
            previous = kwargs.get("previous_winner_source_id")
            if depth == self.profile.max_completed_depth:
                limit = (
                    (1 + self.profile.residual_band_relative_tolerance)
                    * best.residual_score
                    + self.profile.residual_band_numeric_slack
                )
                chosen = min(
                    (item for item in ranked if item.residual_score <= limit),
                    key=lambda item: (
                        item.predicted_future_upper_ms,
                        item.residual_score,
                        item.source_variant_id,
                    ),
                )
                reason = ResidualLockReason.MAX_DEPTH_RESIDUAL_COST
            elif margin >= self.profile.eta_strong:
                chosen, reason = best, ResidualLockReason.STRONG_MARGIN_EARLY_EXIT
            elif previous == best.source_variant_id and margin >= self.profile.eta:
                chosen, reason = best, ResidualLockReason.STABLE_MARGIN_EARLY_EXIT
        elif (
            depth == self.profile.max_completed_depth
            and self.insufficient_ranking_policy is InsufficientRankingPolicy.CFO_TOP1_FALLBACK
        ):
            chosen, reason = ranked[0], ResidualLockReason.CFO_TOP1_FALLBACK
        if chosen is None:
            return (final,)
        proposal = ResidualSelectionDecision(
            state=ResidualSelectionState.SELECTOR_DECISION_READY,
            completed_depth=depth,
            counts=counts,
            selected_source_variant_id=chosen.source_variant_id,
            lock_reason=reason,
            selection_scope=self._scope(counts),
            best_source_variant_id=ranked[0].source_variant_id,
            runner_up_source_variant_id=(
                ranked[1].source_variant_id if len(ranked) >= 2 else None
            ),
            margin_defined=len(ranked) >= 2,
            margin_value=(
                (ranked[1].residual_score - ranked[0].residual_score)
                / max(ranked[1].residual_score, 1e-12)
                if len(ranked) >= 2 else None
            ),
            current_state_ranking_performed=(counts.compared_k >= 2),
            predicted_total_upper_ms=(
                float(kwargs["shared_sunk_ms"]) + chosen.predicted_future_upper_ms
            ),
        )
        return proposal, final
