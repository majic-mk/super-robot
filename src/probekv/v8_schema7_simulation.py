from __future__ import annotations

import statistics
from typing import Any, Dict

from .config import ExperimentConfig
from .v8_contracts import CandidateCounts, ResidualCandidate, SelectorPolicyProfile
from .v8_schema7_contracts import (
    RepairMetric,
    RepairPolicy,
    SourceSelectionDepthPolicy,
    schema7_no_gpu_gate,
)
from .v8_schema7_repair import (
    LoadRecomputeAwareRepairController,
    build_initial_repair_support,
    shrink_repair_support,
)
from .v8_schema7_selector import Schema7DepthSelector


def run_v8_schema7_local_simulation(config: ExperimentConfig) -> Dict[str, Any]:
    """Exercise schema-v7 contracts without CUDA timing or paper evidence."""
    if (config.protocol_version, config.v8_schema_version) != (8, 7):
        raise ValueError("schema-v7 simulation requires protocol 8/schema 7")
    profile = SelectorPolicyProfile(
        profile_id="schema7-local-unfrozen",
        model_math_signature="deterministic-no-gpu",
        selection_execution_policy=config.selection_execution_policy.value,
        checkpoint_depths=config.probe_checkpoints,
        max_completed_depth=config.max_selection_layer,
        eta=config.early_exit_margin,
        eta_strong=config.strong_early_exit_margin,
        residual_band_relative_tolerance=config.residual_band_relative_tolerance,
    )
    selector = Schema7DepthSelector(
        policy=SourceSelectionDepthPolicy(config.source_selection_depth_policy),
        profile=profile,
        gamma=config.gamma,
    )
    rows = []
    for case_index in range(config.cases):
        counts_by_depth = {}
        candidates_by_depth = {}
        for depth in config.probe_checkpoints:
            candidate_count = 4
            counts_by_depth[depth] = CandidateCounts(4, 4, 4, 4, 4)
            close_at_d1 = case_index % 2 == 1 and depth == 1
            scores = (0.040, 0.045, 0.050, 0.055) if close_at_d1 else (
                0.020,
                0.060,
                0.080,
                0.100,
            )
            candidates_by_depth[depth] = tuple(
                ResidualCandidate(f"source-{index}", score, 4.0 + index, index)
                for index, score in enumerate(scores[:candidate_count])
            )
        trace = selector.evaluate_trace(
            counts_by_depth=counts_by_depth,
            candidates_by_depth=candidates_by_depth,
            shared_sunk_ms_by_depth={depth: float(depth) for depth in config.probe_checkpoints},
            dense_reference_ms=100.0,
            gate1_dense_remaining_ms_by_depth={depth: 40.0 for depth in config.probe_checkpoints},
        )
        decision = trace.final_decision
        token_count = 128
        positions = tuple(range(64, 64 + token_count))
        repair_drifts = tuple(float(index) / token_count for index in range(token_count))
        support = build_initial_repair_support(
            segment_id=f"segment-{case_index}",
            source_variant_id=decision.selected_source_variant_id or "dense",
            metric=RepairMetric(config.repair_metric),
            repair_check_completed_depth=max(1, decision.completed_depth),
            segment_absolute_positions=positions,
            drift_scores=repair_drifts,
            initial_cap=config.initial_repair_cap,
            repair_floor=config.repair_floor,
        )
        initial_count = support.candidate_count
        if config.repair_policy != RepairPolicy.FIXED_15.value:
            support = shrink_repair_support(
                support,
                next_ratio=config.repair_floor,
                drift_score_by_absolute_position={
                    position: float(position)
                    for position in support.candidate_absolute_positions
                },
            )
        controller = LoadRecomputeAwareRepairController()
        selected_plan = controller.choose(
            parent_ratio=0.15,
            certified_floor=config.repair_floor,
            repair_ms_by_ratio={0.15: 2.0, config.repair_floor: 1.0},
            load_ms_by_path={"cpu_pinned_to_gpu": 1.0},
            nonoverlap_ms=0.5,
        )
        rows.append(
            {
                "case_id": f"schema7-sim-{case_index:04d}",
                "protocol_version": 8,
                "schema_version": 7,
                "depth_policy": config.source_selection_depth_policy,
                "selection_completed_depth": decision.completed_depth,
                "selected_source_variant_id": decision.selected_source_variant_id,
                "repair_metric": config.repair_metric,
                "repair_policy": config.repair_policy,
                "initial_support_count": initial_count,
                "final_support_count": support.candidate_count,
                "load_recompute_candidate_ratio": selected_plan.ratio,
                "per_request_full_digest_verified": False,
                "paper_evidence": False,
                "locked_test_accessed": False,
            }
        )
    gate = schema7_no_gpu_gate()
    return {
        "summary": {
            "cases": len(rows),
            "protocol_version": 8,
            "schema_version": 7,
            "mean_selection_completed_depth": statistics.mean(
                row["selection_completed_depth"] for row in rows
            ),
            "paper_evidence": False,
        },
        "gates": [
            {
                "name": "schema7_local_contract",
                "passed": all(
                    row["initial_support_count"] >= row["final_support_count"]
                    and not row["per_request_full_digest_verified"]
                    for row in rows
                ),
                "paper_evidence": False,
            },
            dict(gate, name="schema7_no_gpu_readiness", passed=True),
        ],
        "rows": rows,
    }
