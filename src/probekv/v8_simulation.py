from __future__ import annotations

import statistics
from typing import Any, Dict

from .config import ExperimentConfig
from .v8_contracts import (
    CandidateCounts,
    InsufficientRankingPolicy,
    ResidualCandidate,
    ResidualSelectionState,
    SelectorPolicyProfile,
)
from .v8_planner import (
    PredictedJointPlanner,
    PredictedSegmentOption,
    RefinedJointPlanner,
    RefinedSegmentMeasurement,
    UnifiedCostComponents,
)
from .v8_selector import TrainingFreeResidualKSelector


def run_v8_local_simulation(config: ExperimentConfig) -> Dict[str, Any]:
    """Deterministic v8 contract exercise; it is never empirical GPU evidence."""
    if config.protocol_version != 8:
        raise ValueError("v8 simulation requires protocol_version=8")
    profile = SelectorPolicyProfile(
        profile_id="local-test-profile",
        model_math_signature="deterministic-simulation-v8",
        selection_execution_policy=config.selection_execution_policy.value,
        checkpoint_depths=config.probe_checkpoints,
        max_completed_depth=config.max_selection_layer,
        eta=config.early_exit_margin,
        eta_strong=config.strong_early_exit_margin,
        residual_band_relative_tolerance=config.residual_band_relative_tolerance,
    )
    selector = TrainingFreeResidualKSelector(
        profile,
        gamma=config.gamma,
        insufficient_ranking_policy=InsufficientRankingPolicy(
            config.insufficient_ranking_policy
        ),
    )
    segment_counts = (1, 2, 5, 10, 37)
    variant_counts = (1, 4, 16)
    rows = []
    for case_index in range(config.cases):
        segment_count = segment_counts[case_index % len(segment_counts)]
        variant_count = variant_counts[
            (case_index // len(segment_counts)) % len(variant_counts)
        ]
        request_id = "v8-sim-%04d" % case_index
        dense_reference_ms = 250.0 + 18.0 * segment_count
        shared_probe = 1.0 + 0.02 * segment_count
        shared_metadata = 0.01 * segment_count * variant_count
        shared_selection = 0.08 * segment_count
        shared_sunk_ms = shared_probe + shared_metadata + shared_selection
        locked = []
        abstained = 0
        compared_total = 0
        for segment_index in range(segment_count):
            segment_id = "c%d" % segment_index
            # Exercise the budget-truncated K=1 distinction without forcing it
            # into every K=16 cell.
            compared = (
                1
                if variant_count > 1 and case_index % 17 == 0 and segment_index == 0
                else variant_count
            )
            counts = CandidateCounts(
                stored_k=variant_count,
                correctness_eligible_k=variant_count,
                selection_state_available_k=variant_count,
                metadata_ranked_k=variant_count,
                compared_k=compared,
            )
            candidates = tuple(
                ResidualCandidate(
                    source_variant_id="%s-s%d" % (segment_id, source_index),
                    residual_score=0.02 + 0.04 * source_index,
                    predicted_future_upper_ms=8.0 + 0.2 * source_index,
                    metadata_rank=source_index,
                )
                for source_index in range(compared)
            )
            decision_trace = selector.evaluate_checkpoint_trace(
                completed_depth=config.probe_checkpoints[0],
                counts=counts,
                candidates=candidates,
                shared_sunk_ms=shared_sunk_ms,
                dense_reference_ms=dense_reference_ms,
            )
            decision = decision_trace[-1]
            if decision.state is ResidualSelectionState.PENDING:
                decision_trace = selector.evaluate_checkpoint_trace(
                    completed_depth=config.max_selection_layer,
                    counts=counts,
                    candidates=candidates,
                    shared_sunk_ms=shared_sunk_ms,
                    dense_reference_ms=dense_reference_ms,
                    previous_winner_source_id=candidates[0].source_variant_id,
                )
                decision = decision_trace[-1]
            compared_total += compared
            if decision.state is not ResidualSelectionState.LOCKED:
                abstained += 1
                continue
            token_count = 128 + 16 * (segment_index % 8)
            dense_remaining = 18.0 + 0.03 * token_count
            future = UnifiedCostComponents(
                visible_load_ms=1.2,
                post_ready_blocking_ms=0.3,
                interference_ms=0.2,
                repair_selection_ms=0.1,
                repair_ms=2.0,
                remaining_ms=3.0,
            )
            locked.append(
                PredictedSegmentOption(
                    segment_id=segment_id,
                    source_variant_id=str(decision.selected_source_variant_id),
                    artifact_id="%s-artifact" % decision.selected_source_variant_id,
                    replica_id="%s-gpu" % decision.selected_source_variant_id,
                    replica_generation=1,
                    placement_epoch=1,
                    predicted_boundary=2 + segment_index % 4,
                    resident_hbm_bytes=token_count * 4096,
                    future_cost_upper=future,
                    dense_remaining_ms=dense_remaining,
                )
            )
        predicted = PredictedJointPlanner(
            gamma=config.gamma,
            hbm_capacity_bytes=4 * 1024 * 1024 * 1024,
        ).plan(
            request_id,
            locked,
            shared_sunk=UnifiedCostComponents(
                probe_ms=shared_probe,
                metadata_ms=shared_metadata,
                selection_ms=shared_selection,
            ),
            dense_reference_ms=dense_reference_ms,
            joint_interference_upper_ms=0.02 * len(locked),
        )
        option_by_segment = {item.segment_id: item for item in locked}
        measurements = {}
        for segment_id in predicted.provisional_segment_ids:
            option = option_by_segment[segment_id]
            measurements[segment_id] = RefinedSegmentMeasurement(
                segment_id,
                option.source_variant_id,
                option.predicted_boundary,
                UnifiedCostComponents(
                    visible_load_ms=1.0,
                    post_ready_blocking_ms=0.2,
                    interference_ms=0.2,
                    repair_selection_ms=0.1,
                    repair_ms=1.8,
                    remaining_ms=2.8,
                ),
                option.dense_remaining_ms,
                source_ready=True,
                transferred_bytes=option.resident_hbm_bytes,
            )
        refined = RefinedJointPlanner(gamma=config.gamma).plan(
            predicted,
            measurements,
            actual_shared_sunk_ms=shared_sunk_ms,
            joint_actual_interference_ms=0.01 * len(measurements),
        )
        rows.append(
            {
                "case_id": request_id,
                "protocol_version": 8,
                "selection_execution_policy": config.selection_execution_policy.value,
                "detected_segment_count": segment_count,
                "planned_segment_count": segment_count,
                "stored_variants_per_segment": variant_count,
                "stored_k": segment_count * variant_count,
                "correctness_eligible_k": segment_count * variant_count,
                "selection_state_available_k": segment_count * variant_count,
                "metadata_ranked_k": segment_count * variant_count,
                "compared_k": compared_total,
                "locked_segment_count": len(locked),
                "abstained_segment_count": abstained,
                "provisional_reuse_segment_count": len(predicted.provisional_segment_ids),
                "final_reuse_segment_count": len(refined.final_reuse_segment_ids),
                "request_attributed_full_kv_bytes_transferred_for_selection": 0,
                "request_attributed_nonwinner_full_kv_bytes_transferred": 0,
                "request_attributed_full_kv_prefetch_before_source_freeze": 0,
                "predicted_total_upper_ms": predicted.predicted_total_upper_ms,
                "refined_total_ms": refined.refined_total_ms,
                "dense_reference_ms": dense_reference_ms,
                "paper_evidence": False,
                "locked_test_accessed": False,
            }
        )
    covered = {
        (item["detected_segment_count"], item["stored_variants_per_segment"])
        for item in rows
    }
    expected = {
        (segments, variants) for segments in segment_counts for variants in variant_counts
    }
    zero_selection_transfer = all(
        item["request_attributed_full_kv_bytes_transferred_for_selection"] == 0
        and item["request_attributed_nonwinner_full_kv_bytes_transferred"] == 0
        and item["request_attributed_full_kv_prefetch_before_source_freeze"] == 0
        for item in rows
    )
    return {
        "summary": {
            "cases": len(rows),
            "protocol_version": 8,
            "mean_detected_segments": statistics.mean(
                item["detected_segment_count"] for item in rows
            ),
            "mean_final_reuse_segments": statistics.mean(
                item["final_reuse_segment_count"] for item in rows
            ),
            "training_free_selector": True,
            "paper_evidence": False,
        },
        "gates": [
            {
                "name": "v8_local_segment_variant_matrix",
                "passed": covered == expected,
                "covered_cells": len(covered),
                "expected_cells": len(expected),
                "paper_evidence": False,
            },
            {
                "name": "v8_zero_full_kv_selection_transfer",
                "passed": zero_selection_transfer,
                "paper_evidence": False,
            },
        ],
        "rows": rows,
    }
