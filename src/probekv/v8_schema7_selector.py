from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .v8_contracts import (
    AbstainReason,
    CandidateCounts,
    ResidualCandidate,
    ResidualLockReason,
    ResidualSelectionDecision,
    ResidualSelectionState,
    SelectionScope,
    SelectorPolicyProfile,
)
from .v8_schema7_contracts import SourceSelectionDepthPolicy
from .v8_selector import TrainingFreeResidualKSelector


MISTRAL_LEGACY_DEPTHS = (1, 2, 4, 5, 8)
QWEN_LEGACY_DEPTHS = (1, 2, 4, 5, 7)


def schema7_checkpoint_depths(
    *, policy: SourceSelectionDepthPolicy, model_family: str
) -> Tuple[int, ...]:
    family = model_family.lower()
    legacy = MISTRAL_LEGACY_DEPTHS if "mistral" in family else QWEN_LEGACY_DEPTHS
    if policy is SourceSelectionDepthPolicy.D1_ONLY:
        return (1,)
    if policy is SourceSelectionDepthPolicy.D1_D2_RESCUE:
        return (1, 2)
    return legacy


@dataclass(frozen=True)
class DepthPolicyTrace:
    policy: SourceSelectionDepthPolicy
    decisions: Tuple[ResidualSelectionDecision, ...]

    @property
    def final_decision(self) -> ResidualSelectionDecision:
        if not self.decisions:
            raise RuntimeError("Source-selection trace is empty")
        return self.decisions[-1]

    @property
    def locked_completed_depth(self) -> Optional[int]:
        decision = self.final_decision
        return decision.completed_depth if decision.source_frozen else None


@dataclass(frozen=True)
class WrongEarlyLockShadow:
    chosen_source_variant_id: str
    oracle_source_variant_id: str
    chosen_score: float
    oracle_score: float
    absolute_regret: float
    normalized_regret: float
    wrong_early_lock: bool


def evaluate_wrong_early_lock_shadow(
    *,
    chosen_source_variant_id: str,
    deep_candidates: Sequence[ResidualCandidate],
    epsilon: float = 1e-12,
) -> WrongEarlyLockShadow:
    """Offline-only shadow evaluation; it never changes an online frozen Source."""
    if not chosen_source_variant_id or not deep_candidates or epsilon <= 0:
        raise ValueError("shadow evaluation requires a chosen Source and candidates")
    by_id = {row.source_variant_id: row for row in deep_candidates}
    if len(by_id) != len(deep_candidates) or chosen_source_variant_id not in by_id:
        raise ValueError("deep shadow candidate set is incomplete")
    oracle = min(
        deep_candidates,
        key=lambda row: (row.residual_score, row.source_variant_id),
    )
    chosen = by_id[chosen_source_variant_id]
    absolute = max(0.0, chosen.residual_score - oracle.residual_score)
    return WrongEarlyLockShadow(
        chosen_source_variant_id,
        oracle.source_variant_id,
        chosen.residual_score,
        oracle.residual_score,
        absolute,
        absolute / max(oracle.residual_score, epsilon),
        chosen.source_variant_id != oracle.source_variant_id,
    )


class Schema7DepthSelector:
    """Apply one frozen depth policy without coupling selection to repair masks."""

    def __init__(
        self,
        *,
        policy: SourceSelectionDepthPolicy,
        profile: SelectorPolicyProfile,
        gamma: float = 0.8,
    ) -> None:
        if tuple(profile.checkpoint_depths) != tuple(
            sorted(set(profile.checkpoint_depths))
        ):
            raise ValueError("depth Profile checkpoints are invalid")
        self.policy = SourceSelectionDepthPolicy(policy)
        self.selector = TrainingFreeResidualKSelector(profile, gamma=gamma)

    def evaluate_trace(
        self,
        *,
        counts_by_depth: Mapping[int, CandidateCounts],
        candidates_by_depth: Mapping[int, Sequence[ResidualCandidate]],
        shared_sunk_ms_by_depth: Mapping[int, float],
        dense_reference_ms: float,
        gate1_dense_remaining_ms_by_depth: Mapping[int, float],
    ) -> DepthPolicyTrace:
        checkpoints = self.selector.profile.checkpoint_depths
        if set(checkpoints) - set(candidates_by_depth):
            raise ValueError("candidate trace misses a frozen checkpoint")
        if self.policy is SourceSelectionDepthPolicy.DEEP_FULL_CANDIDATE_ORACLE:
            depth = checkpoints[-1]
            counts = counts_by_depth[depth]
            candidates = tuple(candidates_by_depth[depth])
            if counts.compared_k != counts.correctness_eligible_k:
                raise ValueError("deep full-candidate oracle requires full comparison coverage")
            scope = (
                SelectionScope.SINGLE_CORRECTNESS_ELIGIBLE
                if counts.correctness_eligible_k == 1
                else (
                    SelectionScope.CFO_BUDGET_TRUNCATED
                    if counts.compared_k < counts.correctness_eligible_k
                    else SelectionScope.FULL_CORRECTNESS_SET
                )
            )
            if not candidates:
                decision = ResidualSelectionDecision(
                    ResidualSelectionState.ABSTAINED,
                    depth,
                    counts,
                    abstain_reason=AbstainReason.SELECTION_STATE_UNAVAILABLE,
                    selection_scope=scope,
                )
            else:
                best = min(
                    candidates,
                    key=lambda row: (row.residual_score, row.source_variant_id),
                )
                economic = best.predicted_future_upper_ms <= (
                    self.selector.gamma * gate1_dense_remaining_ms_by_depth[depth]
                    + 1e-12
                )
                decision = ResidualSelectionDecision(
                    (
                        ResidualSelectionState.SOURCE_FROZEN
                        if economic else ResidualSelectionState.ABSTAINED
                    ),
                    depth,
                    counts,
                    selected_source_variant_id=(
                        best.source_variant_id if economic else None
                    ),
                    lock_reason=(
                        ResidualLockReason.MAX_DEPTH_RESIDUAL_COST
                        if economic else ResidualLockReason.NONE
                    ),
                    abstain_reason=(
                        AbstainReason.NONE
                        if economic else AbstainReason.MAX_DEPTH_NO_ECONOMIC_CANDIDATE
                    ),
                    selection_scope=scope,
                    best_source_variant_id=best.source_variant_id,
                    current_state_ranking_performed=True,
                    predicted_total_upper_ms=(
                        shared_sunk_ms_by_depth[depth]
                        + best.predicted_future_upper_ms
                        if economic else None
                    ),
                )
            return DepthPolicyTrace(self.policy, (decision,))
        decisions = []
        previous_winner = None
        for depth in checkpoints:
            decision = self.selector.evaluate_checkpoint(
                completed_depth=depth,
                counts=counts_by_depth[depth],
                candidates=candidates_by_depth[depth],
                shared_sunk_ms=shared_sunk_ms_by_depth[depth],
                dense_reference_ms=dense_reference_ms,
                previous_winner_source_id=previous_winner,
                gate1_dense_remaining_ms=gate1_dense_remaining_ms_by_depth[depth],
            )
            decisions.append(decision)
            if decision.best_source_variant_id:
                previous_winner = decision.best_source_variant_id
            if decision.state in {
                ResidualSelectionState.SOURCE_FROZEN,
                ResidualSelectionState.ABSTAINED,
            }:
                break
        return DepthPolicyTrace(self.policy, tuple(decisions))
