from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence, Tuple

from .v8_contracts import CandidateCounts, ResidualCandidate
from .v8_schema8_planner import Gate1LocalPlan
from .v8_schema10_contracts import Gate1Mode, VariantMaterializationReasonV10
from .v8_schema10_profile import PreparationPolicyProfile, VariantAdmissionProfileV10


@dataclass(frozen=True)
class Schema10SourceDecision:
    state: str
    completed_depth: int
    selected_source_variant_id: Optional[str]
    gate1_plan: Optional[Gate1LocalPlan]
    gate1_was_advisory_failure: bool
    best_residual: Optional[float]
    absolute_threshold: float
    margin: Optional[float]
    reason: str
    considered_source_variant_ids: Tuple[str, ...]
    selection_scope_complete: bool
    materialization_reason: Optional[VariantMaterializationReasonV10]

    def __post_init__(self) -> None:
        if self.state not in {"continue_probe", "decision_ready", "abstained"}:
            raise ValueError("unknown schema10 Source decision state")
        if self.completed_depth < 1:
            raise ValueError("schema10 decision depth must be positive")
        if self.state == "decision_ready":
            if not self.selected_source_variant_id or self.gate1_plan is None:
                raise ValueError("decision-ready Source requires Gate1 evidence")
        elif self.selected_source_variant_id is not None or self.gate1_plan is not None:
            raise ValueError("only decision-ready state may expose a Source")
        if self.materialization_reason is not None and self.state != "abstained":
            raise ValueError("only dense fallback may propose materialization")


class Schema10CheckpointSelector:
    def __init__(
        self,
        *,
        variant_profile: VariantAdmissionProfileV10,
        preparation_profile: PreparationPolicyProfile,
        strong_margin: float,
        stable_margin: float,
        residual_band_relative_tolerance: float,
        residual_band_numeric_slack: float = 1e-6,
        checkpoint_depths: Tuple[int, ...] = (1, 2),
        source_cost_selection_policy: str = "legacy_residual_band",
        depth2_keep_fraction: Optional[float] = None,
    ) -> None:
        if not 0 <= stable_margin <= strong_margin <= 1:
            raise ValueError("invalid schema10 early-exit margins")
        if not math.isfinite(residual_band_relative_tolerance) or not 0 <= residual_band_relative_tolerance <= 1:
            raise ValueError("invalid residual band tolerance")
        if residual_band_numeric_slack != 1e-6:
            raise ValueError("schema10 numeric slack must be 1e-6")
        self.variant_profile = variant_profile
        self.preparation_profile = preparation_profile
        self.strong_margin = strong_margin
        self.stable_margin = stable_margin
        self.residual_band_relative_tolerance = residual_band_relative_tolerance
        self.residual_band_numeric_slack = residual_band_numeric_slack
        if source_cost_selection_policy not in {"legacy_residual_band", "absolute_qualified_future_cost"}:
            raise ValueError("unknown Source cost selection policy")
        self.source_cost_selection_policy = source_cost_selection_policy
        if not checkpoint_depths or tuple(sorted(set(checkpoint_depths))) != checkpoint_depths:
            raise ValueError("selector checkpoints must be ordered and unique")
        if checkpoint_depths[0] < 1:
            raise ValueError("d0 is a negative control, not an online checkpoint")
        for depth in checkpoint_depths:
            variant_profile.threshold_for_depth(depth)
        self.checkpoint_depths = checkpoint_depths
        if depth2_keep_fraction is not None and (
                depth2_keep_fraction not in (.5, 1.) or isinstance(depth2_keep_fraction, bool)
                or checkpoint_depths != (1, 2)
                or variant_profile.source_residual_trim_ratio not in (.05, .15)):
            raise ValueError("cascade candidate requires d1/d2 and fixed 5/15 reference")
        self.depth2_keep_fraction = depth2_keep_fraction

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
    ) -> Schema10SourceDecision:
        if completed_depth not in self.checkpoint_depths:
            raise ValueError("depth is outside the frozen selector checkpoints")
        at_max = completed_depth == self.checkpoint_depths[-1]
        ordered = tuple(
            sorted(candidates, key=lambda row: (row.residual_score, row.source_variant_id))
        )
        if len(ordered) != counts.compared_k:
            raise ValueError("compared candidates differ from compared_k")
        if len({row.source_variant_id for row in ordered}) != len(ordered):
            raise ValueError("duplicate Source in current-state comparison")
        if any(not math.isfinite(row.residual_score) for row in ordered):
            raise ValueError("non-finite residual cannot select a Source")
        for row in ordered:
            if row.predicted_future_upper_ms is not None and (
                    isinstance(row.predicted_future_upper_ms, bool)
                    or not math.isfinite(row.predicted_future_upper_ms) or row.predicted_future_upper_ms < 0):
                raise ValueError("invalid Source future cost")
        for source_id, plan in gate1_plan_by_source.items():
            if plan.source_variant_id != source_id or plan.selection_completed_depth != completed_depth:
                raise ValueError("Gate1 plan Source/depth does not match this decision")
        complete = self._scope_complete(counts)
        threshold = self.variant_profile.threshold_for_depth(completed_depth)
        best = ordered[0] if ordered else None
        margin = None
        if len(ordered) >= 2:
            margin = (ordered[1].residual_score - ordered[0].residual_score) / max(
                ordered[1].residual_score, 1e-12
            )

        def result(
            state: str,
            reason: str,
            *,
            chosen: Optional[ResidualCandidate] = None,
            plan: Optional[Gate1LocalPlan] = None,
            materialization_reason: Optional[VariantMaterializationReasonV10] = None,
            considered: Optional[Sequence[ResidualCandidate]] = None,
        ) -> Schema10SourceDecision:
            return Schema10SourceDecision(
                state=state,
                completed_depth=completed_depth,
                selected_source_variant_id=(chosen.source_variant_id if chosen else None),
                gate1_plan=plan,
                gate1_was_advisory_failure=bool(
                    plan is not None
                    and not plan.passed
                    and self.preparation_profile.gate1_mode is Gate1Mode.FUSED_ADVISORY
                ),
                best_residual=(best.residual_score if best else None),
                absolute_threshold=threshold,
                margin=margin,
                reason=reason,
                considered_source_variant_ids=tuple(
                    row.source_variant_id for row in (considered or ordered)
                ),
                selection_scope_complete=complete,
                materialization_reason=materialization_reason,
            )

        if not ordered:
            return result(
                "abstained" if at_max else "continue_probe",
                "content_or_selection_state_miss",
                materialization_reason=(
                    VariantMaterializationReasonV10.CONTENT_MISS
                    if at_max and counts.correctness_eligible_k == 0
                    else None
                ),
            )
        if counts.correctness_eligible_k > 1 and counts.compared_k < 2:
            return result(
                "abstained" if at_max else "continue_probe",
                "insufficient_ranking_coverage",
                materialization_reason=(
                    VariantMaterializationReasonV10.BUDGET_TRUNCATED_EXPLORATION
                    if at_max
                    else None
                ),
            )
        compatible = tuple(row for row in ordered if row.residual_score <= threshold)
        if not compatible:
            if not at_max:
                return result("continue_probe", "d1_absolute_residual_failed_rescue")
            return result(
                "abstained",
                "d2_no_absolute_compatible_source",
                materialization_reason=(
                    VariantMaterializationReasonV10.COMPLETE_SCOPE_ABSOLUTE_MISMATCH
                    if complete
                    else VariantMaterializationReasonV10.BUDGET_TRUNCATED_EXPLORATION
                ),
            )

        if not at_max:
            single = counts.correctness_eligible_k == 1
            strong = margin is not None and margin >= self.strong_margin
            stable = (
                margin is not None
                and margin >= self.stable_margin
                and previous_best_source_variant_id == best.source_variant_id
            )
            if not (single or strong or stable):
                return result("continue_probe", "d1_not_decisive")
            plan = gate1_plan_by_source.get(best.source_variant_id)
            if plan is None or best.predicted_future_upper_ms is None:
                return result("continue_probe", "d1_gate1_evidence_missing")
            if self.source_cost_selection_policy == "absolute_qualified_future_cost":
                priced = [row for row in compatible if row.predicted_future_upper_ms is not None
                          and row.source_variant_id in gate1_plan_by_source
                          and (gate1_plan_by_source[row.source_variant_id].passed
                               or self.preparation_profile.gate1_mode is Gate1Mode.FUSED_ADVISORY)]
                winner = min(priced, key=lambda row: (row.predicted_future_upper_ms, row.residual_score,
                                                     row.source_variant_id)) if priced else None
                if winner is None or winner.source_variant_id != best.source_variant_id:
                    return result("continue_probe", "early_residual_and_cost_winners_disagree")
            if (
                not plan.passed
                and self.preparation_profile.gate1_mode is Gate1Mode.EXPLICIT_BARRIER
            ):
                return result("continue_probe", "d1_gate1_failed_continue")
            return result(
                "decision_ready",
                "d1_source_selected",
                chosen=best,
                plan=plan,
            )

        compatible_best = compatible[0]
        limit = (
            (1 + self.residual_band_relative_tolerance)
            * compatible_best.residual_score
            + self.residual_band_numeric_slack
        )
        band = (compatible if self.source_cost_selection_policy == "absolute_qualified_future_cost"
                else tuple(row for row in compatible if row.residual_score <= limit))
        eligible = []
        for row in band:
            plan = gate1_plan_by_source.get(row.source_variant_id)
            if plan is None or row.predicted_future_upper_ms is None:
                continue
            if plan.passed or self.preparation_profile.gate1_mode is Gate1Mode.FUSED_ADVISORY:
                eligible.append(row)
        if not eligible:
            return result("abstained", "d2_no_preparation_candidate", considered=band)
        chosen = min(
            eligible,
            key=lambda row: (
                (row.predicted_future_upper_ms if self.source_cost_selection_policy == "absolute_qualified_future_cost"
                 else gate1_plan_by_source[row.source_variant_id].predicted_reuse_marginal_lower_ms),
                row.residual_score,
                row.source_variant_id,
            ),
        )
        return result(
            "decision_ready",
            ("absolute_qualified_min_future_cost" if self.source_cost_selection_policy == "absolute_qualified_future_cost"
             else "d2_absolute_compatible_band_min_cost"),
            chosen=chosen,
            plan=gate1_plan_by_source[chosen.source_variant_id],
            considered=band,
        )


class Schema10D1D2Selector(Schema10CheckpointSelector):
    """Historical d1/d2 API; its checkpoint contract remains unchanged."""

    def __init__(self, **kwargs: object) -> None:
        if "checkpoint_depths" in kwargs:
            raise ValueError("use Schema10CheckpointSelector for another dispatch")
        super().__init__(checkpoint_depths=(1, 2), **kwargs)
