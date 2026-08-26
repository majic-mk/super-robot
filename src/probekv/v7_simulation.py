from __future__ import annotations

import random
import statistics
from typing import Any, Dict

from .config import ExperimentConfig
from .candidate_budget import VariantComparisonCandidate, allocate_variant_comparisons
from .v7_planner import (
    JointRequestPlanner,
    LockedSegmentOption,
    SharedSunkCost,
    repair_token_count,
)


def run_v7_local_simulation(config: ExperimentConfig) -> Dict[str, Any]:
    """Deterministic contract exercise; never empirical model evidence."""
    if config.protocol_version != 7:
        raise ValueError("v7 simulation requires protocol_version=7")
    randomizer = random.Random(config.seed)
    rows = []
    segment_count_samples = (1, 2, 5, 10, 37)
    variant_counts = (1, 4, 16)
    for case_index in range(config.cases):
        segment_count = segment_count_samples[case_index % len(segment_count_samples)]
        variant_count = variant_counts[(case_index // len(segment_count_samples)) % len(variant_counts)]
        request_id = "v7-sim-%04d" % case_index
        dense_reference_ms = 100.0 + 15.0 * segment_count
        shared_probe_ms = 0.8 + 0.025 * segment_count
        metadata_ms = 0.03 * segment_count * variant_count
        summary_ms = 0.15 + 0.01 * segment_count
        candidates = []
        for segment_index in range(segment_count):
            for source_index in range(variant_count):
                candidates.append(
                    VariantComparisonCandidate(
                        segment_id="c%d" % segment_index,
                        source_id="c%d-s%d" % (segment_index, source_index),
                        metadata_score=float(source_index),
                        predicted_saved_ms=20.0 + source_index,
                        comparison_upper_ms=0.003,
                    )
                )
        allocation = allocate_variant_comparisons(
            candidates,
            full_reference_ms=dense_reference_ms,
            probe_ms=shared_probe_ms + summary_ms,
            metadata_ms=metadata_ms,
            budget_fraction=config.probe_compare_budget_fraction,
            max_per_segment=config.max_compared_variants_per_segment,
        )
        sunk = SharedSunkCost(
            shared_probe_ms=shared_probe_ms,
            metadata_ms=metadata_ms,
            summary_ms=summary_ms,
            compare_batch_ms=allocation.budget_used_ms,
        )
        options = []
        compared_by_segment = allocation.compared_by_segment()
        for segment_index in range(segment_count):
            segment_id = "c%d" % segment_index
            compared = compared_by_segment.get(segment_id, ())
            if not compared:
                continue
            winner = compared[-1] if case_index % 5 == 0 else compared[0]
            ratio = (0.05, 0.10, 0.16, 0.20)[segment_index % 4]
            token_count = 128 + 16 * (segment_index % 8)
            repair_tokens = repair_token_count(token_count, ratio)
            dense_remaining_ms = 12.0 + 0.15 * token_count
            reuse_future_ms = dense_remaining_ms * (0.42 + 0.02 * (segment_index % 3))
            # Keep all cost components explicit and sum to the synthetic future.
            load = reuse_future_ms * 0.18
            blocking = reuse_future_ms * 0.08
            interference = reuse_future_ms * 0.04
            selection = reuse_future_ms * 0.02
            repair = reuse_future_ms * 0.38
            remaining = reuse_future_ms - load - blocking - interference - selection - repair
            ready = not (case_index % 11 == 0 and segment_index == segment_count - 1)
            options.append(
                LockedSegmentOption(
                    segment_id=segment_id,
                    source_variant_id=winner,
                    replica_id="%s-bf16-gpu" % winner,
                    actual_reuse_boundary=2 + segment_index % max(1, config.max_selection_layer - 1),
                    repair_ratio_upper=ratio,
                    segment_tokens=token_count,
                    resident_hbm_bytes=token_count * 4096,
                    load_ms=load,
                    post_ready_blocking_ms=blocking,
                    interference_ms=interference,
                    repair_selection_ms=selection,
                    repair_ms=repair,
                    remaining_ms=remaining,
                    dense_remaining_ms=dense_remaining_ms,
                    source_ready=ready,
                )
            )
            if repair_tokens < ratio * token_count - 1e-9:
                raise RuntimeError("v7 repair count is not conservative")
        planner = JointRequestPlanner(
            gamma=config.gamma,
            hbm_capacity_bytes=2 * 1024 * 1024 * 1024,
        )
        plan = planner.plan(
            request_id,
            options,
            shared_sunk_ms=sunk.total_ms,
            dense_reference_ms=dense_reference_ms,
            joint_shared_interference_ms=0.05 * len(options),
        )
        accepted = sum(decision.path.value == "reuse" for decision in plan.decisions)
        row = {
            "case_id": request_id,
            "protocol_version": 7,
            "selection_execution_policy": config.selection_execution_policy.value,
            "detected_segment_count": segment_count,
            "planned_segment_count": segment_count,
            "stored_k": segment_count * variant_count,
            "eligible_k": segment_count * variant_count,
            "compared_k": sum(len(value) for value in compared_by_segment.values()),
            "stored_variants_per_segment": variant_count,
            "canonical_artifacts_per_source": 1,
            "accepted_segment_count": accepted,
            "dense_segment_count": segment_count - accepted,
            "actual_reuse_boundary_by_segment": {
                decision.segment_id: decision.actual_reuse_boundary
                for decision in plan.decisions
                if decision.actual_reuse_boundary is not None
            },
            "shared_probe_ms": shared_probe_ms,
            "shared_metadata_ms": metadata_ms,
            "shared_summary_ms": summary_ms,
            "shared_compare_batch_ms": allocation.budget_used_ms,
            "probe_budget_passed": sunk.within_probe_budget(
                dense_reference_ms, config.probe_compare_budget_fraction
            ),
            "joint_total_ms": plan.joint_total_ms,
            "dense_reference_ms": dense_reference_ms,
            "paper_evidence": False,
            "locked_test_accessed": False,
        }
        rows.append(row)
    covered = {
        (row["detected_segment_count"], row["stored_variants_per_segment"])
        for row in rows
    }
    expected = {
        (segments, variants)
        for segments in segment_count_samples
        for variants in variant_counts
    }
    return {
        "summary": {
            "cases": len(rows),
            "protocol_version": 7,
            "mean_detected_segments": statistics.mean(
                row["detected_segment_count"] for row in rows
            ),
            "mean_accepted_segments": statistics.mean(
                row["accepted_segment_count"] for row in rows
            ),
            "single_artifact_policy": True,
            "paper_evidence": False,
        },
        "gates": [
            {
                "name": "v7_local_segment_variant_matrix",
                "passed": covered == expected,
                "covered_cells": len(covered),
                "expected_cells": len(expected),
                "paper_evidence": False,
            },
            {
                "name": "v7_single_artifact_and_probe_budget",
                "passed": all(
                    row["canonical_artifacts_per_source"] == 1
                    and row["probe_budget_passed"]
                    for row in rows
                ),
                "paper_evidence": False,
            },
        ],
        "rows": rows,
    }
