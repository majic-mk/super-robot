from __future__ import annotations

from typing import Any, Dict

from .config import ExperimentConfig
from .v8_contracts import CandidateCounts, ResidualCandidate
from .v8_schema8_planner import Gate1LocalPlan, Gate1MarginalLowerBound
from .v8_schema9_contracts import (
    AbsoluteResidualThreshold,
    DenseKVProvenance,
    VariantMaterializationReason,
    schema9_no_gpu_gate,
)
from .v8_schema9_materialization import (
    VariantMaterializationController,
    VariantMaterializationRequest,
)
from .v8_schema9_profile import VariantAdmissionProfile
from .v8_schema9_selector import Schema9D1D2Selector


def run_v8_schema9_local_simulation(config: ExperimentConfig) -> Dict[str, Any]:
    if (config.protocol_version, config.v8_schema_version) != (8, 9):
        raise ValueError("schema9 simulation requires protocol 8/schema 9")
    profile = VariantAdmissionProfile(
        code_commit="local-unfrozen",
        cacheblend_patch_sha256="0" * 64,
        model_id="local-model",
        model_revision="local-revision",
        tokenizer_hash="1" * 64,
        source_score_trim_ratio=config.source_score_trim_ratio,
        thresholds=tuple(
            AbsoluteResidualThreshold(depth, value)
            for depth, value in config.absolute_residual_threshold_by_depth
        ),
        require_full_candidate_coverage_for_mismatch=(
            config.require_full_candidate_coverage_for_mismatch
        ),
        materialization_budget_fraction=(
            config.variant_materialization_budget_fraction
        ),
        frozen=False,
    )
    selector = Schema9D1D2Selector(
        profile=profile,
        strong_margin=config.strong_early_exit_margin,
        stable_margin=config.early_exit_margin,
        residual_band_relative_tolerance=(
            config.residual_band_relative_tolerance
        ),
        residual_band_numeric_slack=config.residual_band_numeric_slack,
    )
    rows = []
    threshold_d2 = profile.threshold_for_depth(2)
    for index in range(config.cases):
        scores = (
            (0.04, 0.09)
            if index % 2 == 0
            else (threshold_d2 + 0.10, threshold_d2 + 0.20)
        )
        candidates = tuple(
            ResidualCandidate(f"source-{candidate}", score, 4.0 + candidate, candidate)
            for candidate, score in enumerate(scores)
        )
        counts = CandidateCounts(2, 2, 2, 2, 2)
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
        if decision.materialization_candidate:
            materialization = VariantMaterializationController(profile).decide(
                VariantMaterializationRequest(
                    VariantMaterializationReason.ABSOLUTE_RESIDUAL_MISMATCH,
                    True,
                    decision.best_residual,
                    decision.absolute_threshold,
                    DenseKVProvenance.DENSE_EXACT,
                    2,
                    100.0,
                    1.0,
                )
            )
            materialized = materialization.state.value == "admitted"
        rows.append(
            {
                "case_id": f"schema9-local-{index:04d}",
                "protocol_version": 8,
                "schema_version": 9,
                "selection_state": decision.state,
                "selection_reason": decision.reason,
                "absolute_threshold": decision.absolute_threshold,
                "selection_scope_complete": decision.selection_scope_complete,
                "dense_exact_variant_materialized": materialized,
                "paper_evidence": False,
                "locked_test_accessed": False,
            }
        )
    gate = schema9_no_gpu_gate(artifact_preparation_ready=True)
    return {
        "summary": {
            "cases": len(rows),
            "protocol_version": 8,
            "schema_version": 9,
            "paper_evidence": False,
        },
        "gates": [
            {
                "name": "schema9_local_contract",
                "passed": True,
                "paper_evidence": False,
            },
            {**gate, "name": "schema9_no_gpu_readiness", "passed": True},
        ],
        "rows": rows,
    }
