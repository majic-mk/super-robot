from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .v8_contracts import CandidateCounts, ResidualCandidate
from .v8_schema8_planner import Gate1LocalPlan


@dataclass(frozen=True)
class Schema8SourceDecision:
    state: str
    completed_depth: int
    selected_source_variant_id: Optional[str]
    gate1_plan: Optional[Gate1LocalPlan]
    best_residual_source_variant_id: Optional[str]
    margin: Optional[float]
    reason: str
    considered_source_variant_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in {"continue_probe", "decision_ready", "abstained"}:
            raise ValueError("unknown schema-v8 Source decision state")
        if self.completed_depth not in {1, 2}:
            raise ValueError("schema-v8 decision depth must be d1/d2")
        if self.state == "decision_ready":
            if not self.selected_source_variant_id or self.gate1_plan is None:
                raise ValueError("decision-ready Source requires a Gate1 plan")
            if self.selected_source_variant_id != self.gate1_plan.source_variant_id:
                raise ValueError("selected Source and Gate1 plan disagree")
        elif self.selected_source_variant_id is not None or self.gate1_plan is not None:
            raise ValueError("only decision-ready state may expose a Source")


class Schema8D1D2Selector:
    """Quality-first d1/d2 selection with a separate Gate1 economy check."""

    def __init__(
        self,
        *,
        strong_margin: float,
        stable_margin: float,
        residual_band_relative_tolerance: float,
        residual_band_numeric_slack: float = 1e-6,
    ) -> None:
        if not 0 <= stable_margin <= strong_margin <= 1:
            raise ValueError("invalid schema-v8 early-exit margins")
        if residual_band_relative_tolerance < 0 or residual_band_numeric_slack < 0:
            raise ValueError("invalid schema-v8 residual band")
        self.strong_margin = strong_margin
        self.stable_margin = stable_margin
        self.residual_band_relative_tolerance = residual_band_relative_tolerance
        self.residual_band_numeric_slack = residual_band_numeric_slack

    @staticmethod
    def _ordered(candidates: Sequence[ResidualCandidate]) -> Tuple[ResidualCandidate, ...]:
        return tuple(
            sorted(candidates, key=lambda row: (row.residual_score, row.source_variant_id))
        )

    def decide(
        self,
        *,
        completed_depth: int,
        counts: CandidateCounts,
        candidates: Sequence[ResidualCandidate],
        gate1_plan_by_source: Mapping[str, Gate1LocalPlan],
        previous_best_source_variant_id: Optional[str] = None,
    ) -> Schema8SourceDecision:
        if completed_depth not in {1, 2}:
            raise ValueError("schema-v8 online selector only evaluates d1/d2")
        ordered = self._ordered(candidates)
        if len(ordered) != counts.compared_k:
            raise ValueError("compared candidates differ from compared_k")
        if counts.correctness_eligible_k > 1 and counts.compared_k < 2:
            return Schema8SourceDecision(
                "continue_probe" if completed_depth == 1 else "abstained",
                completed_depth, None, None,
                ordered[0].source_variant_id if ordered else None,
                None, "insufficient_ranking_coverage",
                tuple(row.source_variant_id for row in ordered),
            )
        if not ordered:
            return Schema8SourceDecision(
                "continue_probe" if completed_depth == 1 else "abstained",
                completed_depth, None, None, None, None,
                "selection_state_unavailable", (),
            )
        best = ordered[0]
        margin = None
        if len(ordered) >= 2:
            second = ordered[1]
            margin = (second.residual_score - best.residual_score) / max(
                second.residual_score, 1e-12
            )

        if completed_depth == 1:
            single = counts.correctness_eligible_k == 1
            strong = margin is not None and margin >= self.strong_margin
            stable = (
                margin is not None
                and margin >= self.stable_margin
                and previous_best_source_variant_id == best.source_variant_id
            )
            if not (single or strong or stable):
                return Schema8SourceDecision(
                    "continue_probe", 1, None, None, best.source_variant_id,
                    margin, "d1_not_decisive",
                    tuple(row.source_variant_id for row in ordered),
                )
            plan = gate1_plan_by_source.get(best.source_variant_id)
            if plan is None or not plan.passed:
                return Schema8SourceDecision(
                    "continue_probe", 1, None, None, best.source_variant_id,
                    margin, "d1_gate1_failed_continue",
                    tuple(row.source_variant_id for row in ordered),
                )
            return Schema8SourceDecision(
                "decision_ready", 1, best.source_variant_id, plan,
                best.source_variant_id, margin,
                "single_source" if single else ("strong_margin" if strong else "stable_margin"),
                tuple(row.source_variant_id for row in ordered),
            )

        limit = (
            (1 + self.residual_band_relative_tolerance) * best.residual_score
            + self.residual_band_numeric_slack
        )
        band = tuple(row for row in ordered if row.residual_score <= limit)
        economic = tuple(
            row for row in band
            if row.source_variant_id in gate1_plan_by_source
            and gate1_plan_by_source[row.source_variant_id].passed
        )
        if not economic:
            return Schema8SourceDecision(
                "abstained", 2, None, None, best.source_variant_id, margin,
                "d2_no_economic_source_in_residual_band",
                tuple(row.source_variant_id for row in band),
            )
        chosen = min(
            economic,
            key=lambda row: (
                gate1_plan_by_source[row.source_variant_id].predicted_reuse_future_upper_ms,
                row.residual_score,
                row.source_variant_id,
            ),
        )
        return Schema8SourceDecision(
            "decision_ready", 2, chosen.source_variant_id,
            gate1_plan_by_source[chosen.source_variant_id],
            best.source_variant_id, margin, "d2_residual_band_min_cost",
            tuple(row.source_variant_id for row in band),
        )
