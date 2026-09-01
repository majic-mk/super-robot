from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .v8_contracts import CandidateCounts, ResidualCandidate
from .v8_schema8_planner import Gate1LocalPlan
from .v8_schema9_profile import VariantAdmissionProfile


@dataclass(frozen=True)
class Schema9SourceDecision:
    state: str
    completed_depth: int
    selected_source_variant_id: Optional[str]
    gate1_plan: Optional[Gate1LocalPlan]
    best_residual_source_variant_id: Optional[str]
    best_residual: Optional[float]
    absolute_threshold: float
    absolute_compatible_source_variant_ids: Tuple[str, ...]
    margin: Optional[float]
    reason: str
    considered_source_variant_ids: Tuple[str, ...]
    selection_scope_complete: bool
    materialization_candidate: bool

    def __post_init__(self) -> None:
        if self.state not in {"continue_probe", "decision_ready", "abstained"}:
            raise ValueError("unknown schema9 Source decision state")
        if self.completed_depth not in {1, 2}:
            raise ValueError("schema9 decision depth must be d1/d2")
        if self.state == "decision_ready":
            if not self.selected_source_variant_id or self.gate1_plan is None:
                raise ValueError("decision-ready Source requires Gate1 evidence")
            if self.selected_source_variant_id not in (
                self.absolute_compatible_source_variant_ids
            ):
                raise ValueError("selected Source failed absolute compatibility")
            if self.gate1_plan.source_variant_id != self.selected_source_variant_id:
                raise ValueError("selected Source and Gate1 plan disagree")
        elif self.selected_source_variant_id is not None or self.gate1_plan is not None:
            raise ValueError("only decision-ready state may expose a Source")
        if self.materialization_candidate and not (
            self.state == "abstained"
            and self.completed_depth == 2
            and self.selection_scope_complete
            and not self.absolute_compatible_source_variant_ids
        ):
            raise ValueError("only complete-scope d2 mismatch may materialize")


class Schema9D1D2Selector:
    """Schema8 ranking plus an independently frozen absolute compatibility Gate."""

    def __init__(
        self,
        *,
        profile: VariantAdmissionProfile,
        strong_margin: float,
        stable_margin: float,
        residual_band_relative_tolerance: float,
        residual_band_numeric_slack: float = 1e-6,
    ) -> None:
        if not 0 <= stable_margin <= strong_margin <= 1:
            raise ValueError("invalid schema9 early-exit margins")
        if residual_band_relative_tolerance < 0 or residual_band_numeric_slack < 0:
            raise ValueError("invalid schema9 residual band")
        self.profile = profile
        self.strong_margin = strong_margin
        self.stable_margin = stable_margin
        self.residual_band_relative_tolerance = residual_band_relative_tolerance
        self.residual_band_numeric_slack = residual_band_numeric_slack

    @staticmethod
    def _ordered(candidates: Sequence[ResidualCandidate]) -> Tuple[ResidualCandidate, ...]:
        return tuple(
            sorted(candidates, key=lambda row: (row.residual_score, row.source_variant_id))
        )

    @staticmethod
    def _scope_complete(counts: CandidateCounts) -> bool:
        return (
            counts.correctness_eligible_k
            == counts.selection_state_available_k
            == counts.metadata_ranked_k
            == counts.compared_k
        )

    def decide(
        self,
        *,
        completed_depth: int,
        counts: CandidateCounts,
        candidates: Sequence[ResidualCandidate],
        gate1_plan_by_source: Mapping[str, Gate1LocalPlan],
        previous_best_source_variant_id: Optional[str] = None,
    ) -> Schema9SourceDecision:
        if completed_depth not in {1, 2}:
            raise ValueError("schema9 online selector only evaluates d1/d2")
        ordered = self._ordered(candidates)
        if len(ordered) != counts.compared_k:
            raise ValueError("compared candidates differ from compared_k")
        threshold = self.profile.threshold_for_depth(completed_depth)
        complete = self._scope_complete(counts)
        best = ordered[0] if ordered else None
        best_id = best.source_variant_id if best else None
        best_score = best.residual_score if best else None
        margin = None
        if len(ordered) >= 2:
            second = ordered[1]
            margin = (second.residual_score - ordered[0].residual_score) / max(
                second.residual_score, 1e-12
            )
        compatible = tuple(row for row in ordered if row.residual_score <= threshold)
        compatible_ids = tuple(row.source_variant_id for row in compatible)
        considered_ids = tuple(row.source_variant_id for row in ordered)

        def result(
            state: str,
            reason: str,
            *,
            chosen: Optional[ResidualCandidate] = None,
            plan: Optional[Gate1LocalPlan] = None,
            materialize: bool = False,
            considered: Tuple[str, ...] = considered_ids,
        ) -> Schema9SourceDecision:
            return Schema9SourceDecision(
                state,
                completed_depth,
                chosen.source_variant_id if chosen else None,
                plan,
                best_id,
                best_score,
                threshold,
                compatible_ids,
                margin,
                reason,
                considered,
                complete,
                materialize,
            )

        if counts.correctness_eligible_k > 1 and counts.compared_k < 2:
            return result(
                "continue_probe" if completed_depth == 1 else "abstained",
                "insufficient_ranking_coverage",
            )
        if not ordered:
            return result(
                "continue_probe" if completed_depth == 1 else "abstained",
                "content_or_selection_state_miss",
            )
        if not compatible:
            if completed_depth == 1:
                return result("continue_probe", "d1_absolute_residual_failed_rescue")
            return result(
                "abstained",
                "d2_no_absolute_compatible_source",
                materialize=complete,
            )

        if completed_depth == 1:
            single = counts.correctness_eligible_k == 1
            strong = margin is not None and margin >= self.strong_margin
            stable = (
                margin is not None
                and margin >= self.stable_margin
                and previous_best_source_variant_id == best_id
            )
            if not (single or strong or stable):
                return result("continue_probe", "d1_not_decisive")
            assert best is not None
            plan = gate1_plan_by_source.get(best.source_variant_id)
            if plan is None or not plan.passed:
                return result("continue_probe", "d1_gate1_failed_continue")
            return result(
                "decision_ready",
                "single_source" if single else (
                    "strong_margin" if strong else "stable_margin"
                ),
                chosen=best,
                plan=plan,
            )

        compatible_best = compatible[0]
        limit = (
            (1 + self.residual_band_relative_tolerance)
            * compatible_best.residual_score
            + self.residual_band_numeric_slack
        )
        band = tuple(row for row in compatible if row.residual_score <= limit)
        economic = tuple(
            row
            for row in band
            if row.source_variant_id in gate1_plan_by_source
            and gate1_plan_by_source[row.source_variant_id].passed
        )
        if not economic:
            return result(
                "abstained",
                "d2_no_economic_source_in_compatible_band",
                considered=tuple(row.source_variant_id for row in band),
            )
        chosen = min(
            economic,
            key=lambda row: (
                gate1_plan_by_source[
                    row.source_variant_id
                ].predicted_reuse_marginal_lower_ms,
                row.residual_score,
                row.source_variant_id,
            ),
        )
        return result(
            "decision_ready",
            "d2_absolute_compatible_band_min_cost",
            chosen=chosen,
            plan=gate1_plan_by_source[chosen.source_variant_id],
            considered=tuple(row.source_variant_id for row in band),
        )
