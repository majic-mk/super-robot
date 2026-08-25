from __future__ import annotations

import random
import statistics
from dataclasses import replace
from typing import Any, Dict, Tuple

from .candidate_budget import (
    VariantComparisonCandidate,
    allocate_variant_comparisons,
)
from .config import ExperimentConfig
from .contracts import (
    CandidateBounds,
    CostValueKind,
    InterferenceAccountingMode,
)
from .cost import cost_breakdown_from_total
from .multisegment_orchestration import (
    MultiSegmentOnlinePipeline,
    MultiSegmentReuseController,
    StaggeredMultiSegmentReuseController,
)
from .multisegment_selector import MultiSegmentProbeSelector
from .selector import DynamicProbeSelector, ProbePolicy
from .v6_contracts import (
    RequestRefinedCost,
    RequestSchedulingFeedback,
    SelectionExecutionPolicy,
    SegmentSchedulingFeedback,
)


class _PreparedV6Runtime:
    def __init__(
        self,
        total_layers: int,
        selection_execution_policy: SelectionExecutionPolicy,
    ) -> None:
        self.total_layers = total_layers
        self.selection_execution_policy = selection_execution_policy
        self.probe_state_origin = (
            "policy_conditioned_closed_loop"
            if selection_execution_policy is (
                SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP
            )
            else "dense_clean"
        )
        self.action = None
        self.lock_events = []
        self.reuse_eligible_events = []

    def on_source_locked(self, segment_id, decision):
        self.lock_events.append(
            (decision.probe_layer, segment_id, decision.selected_source_id)
        )

    def on_reuse_eligible(self, segment_id, earliest_layer):
        self.reuse_eligible_events.append((earliest_layer, segment_id))

    def load_and_schedule(self, selection):
        feedback = []
        for index, decision in enumerate(selection.selected):
            source = decision.source_decision
            first_ready = min(
                self.total_layers,
                source.probe_layer + 1 + index % 3,
            )
            load_start = source.probe_layer * 0.25
            ready_ms = load_start + 1.0 + index * 0.1
            feedback.append(
                SegmentSchedulingFeedback(
                    segment_id=decision.segment_id,
                    selected_source_id=str(source.selected_source_id),
                    source_load_start_ms=load_start,
                    source_ready_ms=ready_ms,
                    first_ready_layer=first_ready,
                    ready_through_layer=self.total_layers,
                    transferred_bytes=32_000_000,
                )
            )
        first_boundary = min(item.first_ready_layer for item in feedback)
        last_boundary = min(self.total_layers, max(first_boundary, 16))
        return RequestSchedulingFeedback(
            request_id=selection.request_id,
            segments=tuple(feedback),
            scheduled_step_finish_ms=max(item.source_ready_ms for item in feedback),
            a_resume_ms=max(item.source_ready_ms for item in feedback),
            post_ready_blocking_ms=0.0,
            load_interference_ms=0.0,
            useful_a_dense_ms=1.0,
            useful_other_request_work_ms=2.0,
            candidate_boundaries=tuple(range(first_boundary, last_boundary + 1)),
        )

    def measure_refined_cost(
        self, selection, scheduling, boundary, active_segment_ids
    ):
        sources = {
            item.segment_id: str(item.source_decision.selected_source_id)
            for item in selection.selected
            if item.segment_id in active_segment_ids
        }
        marginal = {
            segment_id: 4.0 + (index % 3)
            for index, segment_id in enumerate(active_segment_ids)
        }
        repair_selection = 0.05 * len(active_segment_ids)
        repair = 1.5 * len(active_segment_ids)
        visible_load = max(0.0, 1.5 - 0.1 * boundary)
        remaining = max(
            0.0,
            0.55 * selection.full_reference_ms
            - sum(marginal.values()),
        )
        total = (
            selection.probe_ms
            + selection.metadata_ms
            + selection.compare_ms
            + visible_load
            + repair_selection
            + repair
            + remaining
        )
        return RequestRefinedCost(
            request_id=selection.request_id,
            boundary=boundary,
            active_segment_ids=tuple(active_segment_ids),
            selected_source_ids=sources,
            marginal_saved_ms=marginal,
            repair_ratio_upper_by_segment={
                segment_id: 0.20 for segment_id in active_segment_ids
            },
            reuse_total_ms=total,
            full_reference_ms=selection.full_reference_ms,
            probe_ms=selection.probe_ms,
            metadata_ms=selection.metadata_ms,
            compare_ms=selection.compare_ms,
            visible_load_ms=visible_load,
            post_ready_blocking_ms=0.0,
            load_interference_ms=0.0,
            repair_selection_ms=repair_selection,
            repair_ms=repair,
            remaining_ms=remaining,
            joint_quality_covered=True,
        )

    def measure_staggered_refined_cost(
        self, selection, scheduling, boundary_by_segment, active_segment_ids
    ):
        boundary = min(boundary_by_segment.values())
        common_profile = self.measure_refined_cost(
            selection, scheduling, boundary, active_segment_ids
        )
        return replace(
            common_profile,
            boundary_by_segment=dict(boundary_by_segment),
        )

    def execute_reuse(self, selection, execution):
        self.action = "multisegment_prefill"
        return {"action": self.action}

    def execute_dense(self, selection, execution):
        self.action = "dense"
        return {"action": self.action}


def run_v6_local_simulation(config: ExperimentConfig) -> Dict[str, Any]:
    """Exercise v6 contracts with deterministic, non-paper synthetic data."""

    if config.protocol_version != 6:
        raise ValueError("v6 simulation requires protocol_version=6")
    randomizer = random.Random(config.seed)
    selector = MultiSegmentProbeSelector(
        DynamicProbeSelector(
            ProbePolicy(
                checkpoints=config.probe_checkpoints,
                max_layer=config.max_selection_layer,
                selector_policy=config.selector_policy,
                gamma=config.gamma,
                reuse_ratio_tolerance=config.reuse_ratio_tolerance,
                preliminary_economic_filter=True,
                require_component_cost_bounds=True,
            )
        ),
        config.selection_execution_policy,
    )
    controller = (
        MultiSegmentReuseController(config.gamma)
        if config.selection_execution_policy is (
            SelectionExecutionPolicy.LEGACY_COMMON_AFTER_SELECTION
        )
        else StaggeredMultiSegmentReuseController(config.gamma)
    )
    rows = []
    # These are deterministic coverage samples, never a runtime ceiling.  The
    # request contracts and planner accept every detected segment that fits the
    # model context window and available resources.
    segment_count_samples = (1, 2, 5, 10, 17)
    variant_counts = (1, 4, 16)
    for case_index in range(config.cases):
        request_id = "v6-sim-%04d" % case_index
        segment_count = segment_count_samples[
            case_index % len(segment_count_samples)
        ]
        variant_count = variant_counts[case_index % len(variant_counts)]
        full_ms = 180.0 + segment_count * 8.0
        probe_ms = 1.0 + 0.05 * segment_count
        metadata_ms = 0.1 + 0.01 * segment_count * variant_count
        candidates = []
        latent_costs: Dict[str, Dict[str, float]] = {}
        for segment_index in range(segment_count):
            segment_id = "c%d" % segment_index
            latent_costs[segment_id] = {}
            for source_index in range(variant_count):
                source_id = "%s-s%d" % (segment_id, source_index)
                cost = (
                    0.48 * full_ms
                    + source_index * 0.25
                    + randomizer.uniform(0.0, 2.0)
                )
                if source_index == variant_count - 1 and case_index % 5 == 0:
                    cost -= 6.0
                latent_costs[segment_id][source_id] = cost
                candidates.append(
                    VariantComparisonCandidate(
                        segment_id=segment_id,
                        source_id=source_id,
                        metadata_score=source_index / float(max(1, variant_count)),
                        predicted_saved_ms=max(0.0, full_ms - cost),
                        comparison_upper_ms=0.01,
                    )
                )
        allocation = allocate_variant_comparisons(
            candidates,
            full_reference_ms=full_ms,
            probe_ms=probe_ms,
            metadata_ms=metadata_ms,
            budget_fraction=config.probe_compare_budget_fraction,
            max_per_segment=config.max_compared_variants_per_segment,
        )
        compared = allocation.compared_by_segment()
        bounds = {}
        for segment_id in latent_costs:
            bounds[segment_id] = {}
            for layer in config.probe_checkpoints:
                width = 7.0 / max(1.0, layer)
                layer_bounds = []
                for source_id in compared.get(segment_id, ()):
                    cost = latent_costs[segment_id][source_id]
                    lower = max(0.0, cost - width)
                    upper = cost + width
                    layer_bounds.append(
                        CandidateBounds(
                            source_id=source_id,
                            repair_ratio_upper=min(1.0, 0.1 + cost / full_ms * 0.2),
                            cost_lower_ms=lower,
                            cost_upper_ms=upper,
                            quality_covered=True,
                            cost_lower_breakdown=cost_breakdown_from_total(
                                lower,
                                full_ms,
                                layer,
                                CostValueKind.PREDICTED_LOWER,
                                probe_ms=probe_ms,
                                metadata_ms=metadata_ms,
                                compare_ms=allocation.budget_used_ms,
                                interference_accounting_mode=(
                                    InterferenceAccountingMode.EXPLICIT_PENALTY
                                ),
                            ),
                            cost_upper_breakdown=cost_breakdown_from_total(
                                upper,
                                full_ms,
                                layer,
                                CostValueKind.PREDICTED_UPPER,
                                probe_ms=probe_ms,
                                metadata_ms=metadata_ms,
                                compare_ms=allocation.budget_used_ms,
                                interference_accounting_mode=(
                                    InterferenceAccountingMode.EXPLICIT_PENALTY
                                ),
                            ),
                        )
                    )
                bounds[segment_id][layer] = tuple(layer_bounds)
        runtime = _PreparedV6Runtime(
            config.total_layers, config.selection_execution_policy
        )
        bounds_input = (
            (lambda segment_id, layer: bounds.get(segment_id, {}).get(layer, ()))
            if config.selection_execution_policy is (
                SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP
            )
            else bounds
        )
        result = MultiSegmentOnlinePipeline(selector, controller).execute(
            request_id, allocation, bounds_input, runtime
        )
        row = result.to_audit_record()
        row.update(
            {
                "case_id": request_id,
                "stored_variants_per_segment": variant_count,
                "paper_evidence": False,
                "evidence_class": "local_simulation",
                "lock_events": runtime.lock_events,
                "reuse_eligible_events": runtime.reuse_eligible_events,
            }
        )
        rows.append(row)
    accepted = [row["accepted_segment_count"] for row in rows]
    covered_cells = sorted(
        {
            (
                row["detected_segment_count"],
                row["stored_variants_per_segment"],
            )
            for row in rows
        }
    )
    expected_cells = {
        (segments, variants)
        for segments in segment_count_samples
        for variants in variant_counts
    }
    return {
        "summary": {
            "cases": len(rows),
            "protocol_version": 6,
            "mean_detected_segments": statistics.mean(
                row["detected_segment_count"] for row in rows
            ),
            "mean_accepted_segments": statistics.mean(accepted),
            "covered_segment_variant_cells": [
                {"segments": segments, "variants": variants}
                for segments, variants in covered_cells
            ],
            "paper_evidence": False,
        },
        "gates": [
            {
                "name": "v6_local_segment_variant_matrix",
                "passed": set(covered_cells) == expected_cells,
                "covered_cells": len(covered_cells),
                "expected_cells": len(expected_cells),
                "paper_evidence": False,
            }
        ],
        "rows": rows,
    }
