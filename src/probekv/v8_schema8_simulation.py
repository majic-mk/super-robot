from __future__ import annotations

import statistics
from typing import Any, Dict, Tuple

from .config import ExperimentConfig
from .v8_schema8_barrier import close_dense_selection_barrier
from .v8_schema8_contracts import RepairRatioScope, schema8_no_gpu_gate
from .v8_schema8_planner import Gate1LayerCost, Gate1LocalPlan
from .v8_schema8_repair import (
    SegmentLayerRepairRatio,
    validate_union_repair_ratio_plan,
)
from .v8_schema8_storage import Schema8TieredBackingManager


def _ratio_rows(
    config: ExperimentConfig,
    *,
    segment_ids: Tuple[str, ...],
    first_reuse_layer: int,
) -> Tuple[SegmentLayerRepairRatio, ...]:
    rows = []
    for segment_index, segment_id in enumerate(segment_ids):
        for age in range(2):
            if config.repair_ratio_scope == RepairRatioScope.UNIFORM_FIXED.value:
                ratio = 0.15
            elif config.repair_ratio_scope == RepairRatioScope.SHARED_RELATIVE_SCHEDULE.value:
                ratio = 0.15 if age == 0 else max(config.repair_floor, 0.12)
            else:
                # A frozen Profile is required before this branch can execute.
                ratio = (0.15, 0.12, 0.10)[min(segment_index, 2)]
            rows.append(
                SegmentLayerRepairRatio(
                    segment_id,
                    first_reuse_layer + age,
                    first_reuse_layer,
                    ratio,
                )
            )
    return tuple(rows)


def run_v8_schema8_local_simulation(config: ExperimentConfig) -> Dict[str, Any]:
    """Exercise schema-v8 contracts without CUDA timing or paper evidence."""
    if (config.protocol_version, config.v8_schema_version) != (8, 8):
        raise ValueError("schema-v8 simulation requires protocol 8/schema 8")

    rows = []
    manager = Schema8TieredBackingManager(
        cpu_capacity_bytes=3 * 1024,
        ssd_capacity_bytes=6 * 1024,
    )
    for case_index in range(config.cases):
        segment_ids = tuple(
            f"case-{case_index:04d}-segment-{index}" for index in range(3)
        )
        # Odd cases resolve every Segment at d=1.  Even cases exercise the
        # single d=2 rescue barrier without introducing an A/C policy.
        resolved = (
            {segment_id: 1 for segment_id in segment_ids}
            if case_index % 2
            else {segment_ids[0]: 1}
        )
        barrier = close_dense_selection_barrier(
            segment_ids=segment_ids,
            resolved_completed_depth_by_segment=resolved,
        )

        gate1_plans = []
        for segment_id in segment_ids:
            first_layer = barrier.first_selective_reuse_layer
            gate1_plans.append(
                Gate1LocalPlan(
                    source_variant_id=f"{segment_id}-winner",
                    selection_completed_depth=(
                        barrier.resolved_completed_depth_by_segment[segment_id]
                    ),
                    first_selective_reuse_layer=first_layer,
                    dense_repair_check_upper_ms=1.0,
                    support_build_upper_ms=0.5,
                    layer_costs=(
                        Gate1LayerCost(first_layer, 0.15, 1.0, 2.0, 0.5),
                        Gate1LayerCost(first_layer + 1, 0.12, 1.0, 1.5, 0.5),
                    ),
                    dense_remaining_same_origin_ms=8.0,
                    gate1_gamma=config.gate1_gamma,
                )
            )

        ratio_plan = validate_union_repair_ratio_plan(
            scope=RepairRatioScope(config.repair_ratio_scope),
            rows=_ratio_rows(
                config,
                segment_ids=segment_ids,
                first_reuse_layer=barrier.first_selective_reuse_layer,
            ),
            certified_floor=config.repair_floor,
            profile_frozen=config.selector_profile_status == "frozen",
        )

        for segment_id in segment_ids:
            if segment_id not in manager.snapshot()["entries"]:
                manager.register(segment_id, size_bytes=1024)
            manager.access(segment_id)

        rows.append(
            {
                "case_id": f"schema8-sim-{case_index:04d}",
                "protocol_version": 8,
                "schema_version": 8,
                "selection_execution_policy": config.selection_execution_policy.value,
                "barrier_completed_depth": barrier.barrier_completed_depth,
                "first_selective_reuse_layer": barrier.first_selective_reuse_layer,
                "d2_rescue_segment_count": len(barrier.d2_rescue_segment_ids),
                "gate1_gamma": config.gate1_gamma,
                "gate1_all_passed": all(plan.passed for plan in gate1_plans),
                "final_commit_gamma": config.gamma,
                "repair_ratio_scope": ratio_plan.scope.value,
                "cpu_backing_used_bytes": manager.tier_usage("pinned_cpu"),
                "ssd_backing_used_bytes": manager.tier_usage("ssd"),
                "paper_evidence": False,
                "locked_test_accessed": False,
            }
        )

    gate = schema8_no_gpu_gate()
    return {
        "summary": {
            "cases": len(rows),
            "protocol_version": 8,
            "schema_version": 8,
            "mean_barrier_completed_depth": statistics.mean(
                row["barrier_completed_depth"] for row in rows
            ),
            "paper_evidence": False,
        },
        "gates": [
            {
                "name": "schema8_local_contract",
                "passed": all(
                    row["gate1_all_passed"]
                    and row["gate1_gamma"] == 1.0
                    and row["final_commit_gamma"] == 0.8
                    for row in rows
                ),
                "paper_evidence": False,
            },
            dict(gate, name="schema8_no_gpu_readiness", passed=True),
        ],
        "rows": rows,
    }
