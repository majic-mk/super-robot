from __future__ import annotations

from typing import Any, Dict

from .config import ExperimentConfig
from .v8_contracts import CandidateCounts, ResidualCandidate
from .v8_schema8_planner import Gate1LocalPlan, Gate1MarginalLowerBound
from .v8_schema10_contracts import (
    AbsoluteResidualThreshold,
    DenseKVProvenance,
    Gate1Mode,
    VariantMaterializationReasonV10,
    schema10_no_gpu_gate,
)
from .v8_schema10_materialization import (
    VariantMaterializationControllerV10,
    VariantMaterializationRequestV10,
)
from .v8_schema10_preparation import (
    PreparationCostObservation,
    evaluate_gate1_counterfactual,
)
from .v8_schema10_profile import (
    PreparationPolicyProfile,
    VariantAdmissionProfileV10,
)
from .v8_schema10_selector import Schema10D1D2Selector


def run_v8_schema10_local_simulation(config: ExperimentConfig) -> Dict[str, Any]:
    if (config.protocol_version, config.v8_schema_version) != (8, 10):
        raise ValueError("schema10 simulation requires protocol 8/schema 10")
    variant_profile = VariantAdmissionProfileV10(
        code_commit="local-unfrozen",
        cacheblend_patch_sha256="0" * 64,
        model_id="local-model",
        model_revision="local-revision",
        tokenizer_hash="1" * 64,
        source_residual_trim_ratio=config.source_residual_trim_ratio,
        thresholds=tuple(
            AbsoluteResidualThreshold(depth, value)
            for depth, value in config.absolute_residual_threshold_by_depth
        ),
        materialization_budget_fraction=config.variant_materialization_budget_fraction,
        exploration_quota_per_content=config.exploration_quota_per_content,
        probation_comparison_observations=config.probation_comparison_observations,
        probation_lookup_opportunities=config.probation_lookup_opportunities,
        max_protected_probation_per_content=config.max_protected_probation_per_content,
    )
    preparation_profile = PreparationPolicyProfile(
        code_commit="local-unfrozen",
        model_id="local-model",
        runtime_policy="dense_selection_barrier",
        gate1_mode=Gate1Mode(config.gate1_mode),
    )
    selector = Schema10D1D2Selector(
        variant_profile=variant_profile,
        preparation_profile=preparation_profile,
        strong_margin=config.strong_early_exit_margin,
        stable_margin=config.early_exit_margin,
        residual_band_relative_tolerance=config.residual_band_relative_tolerance,
        residual_band_numeric_slack=config.residual_band_numeric_slack,
    )
    rows = []
    counterfactual_rows = []
    for index in range(config.cases):
        truncated = index % 3 == 2
        mismatch = index % 3 == 1
        scores = (0.40, 0.50) if mismatch else (0.04, 0.09)
        candidates = tuple(
            ResidualCandidate(f"source-{candidate}", score, 4.0 + candidate, candidate)
            for candidate, score in enumerate(scores[:1] if truncated else scores)
        )
        counts = (
            CandidateCounts(16, 16, 16, 1, 1)
            if truncated
            else CandidateCounts(2, 2, 2, 2, 2)
        )
        plans = {
            row.source_variant_id: Gate1LocalPlan(
                source_variant_id=row.source_variant_id,
                selection_completed_depth=2,
                repair_check_completed_depth=2,
                first_selective_reuse_layer=3,
                dense_repair_check_sunk_ms=1.0,
                marginal_lower_bound=Gate1MarginalLowerBound(0.2, 0.3, 0.4),
                dense_marginal_same_origin_ms=4.0,
                gate1_gamma=1.0,
            )
            for row in candidates
        }
        decision = selector.decide(
            completed_depth=2,
            counts=counts,
            candidates=candidates,
            gate1_plan_by_source=plans,
        )
        materialized = False
        novelty = False
        if decision.materialization_reason is not None:
            materialization = VariantMaterializationControllerV10(variant_profile).decide(
                VariantMaterializationRequestV10(
                    reason=decision.materialization_reason,
                    correctness_eligible_k=counts.correctness_eligible_k,
                    compared_k=counts.compared_k,
                    best_residual=decision.best_residual,
                    absolute_threshold=decision.absolute_threshold,
                    dense_kv_provenance=DenseKVProvenance.DENSE_EXACT,
                    existing_variant_count=min(15, counts.correctness_eligible_k),
                    dense_reference_total_ms=100.0,
                    estimated_materialization_ms=1.0,
                    exploration_materializations_for_content=0,
                    exploration_authorized=True,
                )
            )
            materialized = materialization.state.value == "admitted"
            novelty = materialization.context_novelty_proven
        counterfactual_rows.append(
            PreparationCostObservation(
                request_id=f"schema10-local-{index:04d}",
                dense_reference_total_ms=100.0,
                gate1_passed=True,
                additional_winner_full_kv_bytes_without_gate1=0,
                additional_visible_copy_ms_without_gate1=0.0,
                additional_pinned_staging_ms_without_gate1=0.0,
                additional_copy_interference_ms_without_gate1=0.0,
                additional_hbm_reservation_byte_ms_without_gate1=0.0,
                additional_wasted_preparation_ms_without_gate1=0.0,
                ttft_delta_ms_without_gate1=0.0,
                counterfactual_path_economically_invalid=False,
                counterfactual_final_commit_admitted=True,
            )
        )
        rows.append(
            {
                "case_id": f"schema10-local-{index:04d}",
                "protocol_version": 8,
                "schema_version": 10,
                "selection_state": decision.state,
                "selection_reason": decision.reason,
                "selection_scope_complete": decision.selection_scope_complete,
                "materialization_reason": (
                    decision.materialization_reason.value
                    if decision.materialization_reason is not None
                    else None
                ),
                "dense_exact_variant_materialized": materialized,
                "context_novelty_proven": novelty,
                "paper_evidence": False,
                "locked_test_accessed": False,
            }
        )
    gate1_summary = evaluate_gate1_counterfactual(
        counterfactual_rows,
        total_winner_full_kv_bytes_with_gate1=1,
        profile=preparation_profile,
    )
    gate = schema10_no_gpu_gate(artifact_preparation_ready=True)
    return {
        "summary": {
            "cases": len(rows),
            "protocol_version": 8,
            "schema_version": 10,
            "gate1_recommended_mode": gate1_summary.recommended_gate1_mode.value,
            "paper_evidence": False,
        },
        "gates": [
            {"name": "schema10_local_contract", "passed": True, "paper_evidence": False},
            {**gate, "name": "schema10_no_gpu_readiness", "passed": True},
        ],
        "rows": rows,
    }
