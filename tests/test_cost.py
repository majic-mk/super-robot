import unittest

from probekv.contracts import (
    DecisionReason,
    ExecutionDecision,
    ExecutionMode,
    RejectionReason,
    SelectionReason,
    SourceDecision,
)
from probekv.cost import (
    DynamicReusePlanner,
    LayerOption,
    bandwidth_sufficient,
    conservative_ratio_for_layer,
    finalize_execution,
)


def option(layer, load, overlap, repair, full=100, ready=True):
    return LayerOption(layer, 0.2, 4, 1, load, overlap, repair, full, ready)


class CostTests(unittest.TestCase):
    def test_total_reuse_cost_includes_all_components(self):
        plan = DynamicReusePlanner(0.8).plan([option(4, 20, 5, 55)])
        self.assertTrue(plan.accepted)
        self.assertEqual(plan.timing.visible_load_ms, 15)
        self.assertEqual(plan.timing.reuse_total_ms, 75)

    def test_post_ready_blocking_is_charged_to_reuse_total(self):
        blocked = LayerOption(
            layer=4,
            repair_ratio_upper=0.2,
            probe_ms=4,
            compare_ms=1,
            load_ms=20,
            overlap_ms=5,
            repair_ms=50,
            full_ms=100,
            post_ready_blocking_ms=3,
        )
        plan = DynamicReusePlanner(0.8).plan([blocked])
        self.assertTrue(plan.accepted)
        self.assertEqual(plan.timing.reuse_total_ms, 73)
        rejected = DynamicReusePlanner(0.7).plan([blocked])
        self.assertFalse(rejected.accepted)

    def test_90_percent_reuse_is_rejected_at_gamma_08(self):
        plan = DynamicReusePlanner(0.8).plan([option(4, 20, 0, 65)])
        self.assertFalse(plan.accepted)
        self.assertEqual(plan.reason, DecisionReason.ECONOMIC_REJECT)

    def test_minimum_total_cost_layer_is_selected(self):
        plan = DynamicReusePlanner(0.8).plan(
            [option(4, 20, 0, 40), option(6, 20, 15, 38)]
        )
        self.assertEqual(plan.layer, 6)

    def test_not_ready_buffers_do_not_enter_plan(self):
        plan = DynamicReusePlanner().plan([option(4, 10, 10, 20, ready=False)])
        self.assertEqual(plan.reason, DecisionReason.NO_FEASIBLE_LAYER)

    def test_adjacent_anchor_uses_larger_ratio(self):
        self.assertEqual(conservative_ratio_for_layer({3: 0.2, 7: 0.3}, 5), 0.3)

    def test_bandwidth_condition(self):
        # 1 MB required every 2 ms => 0.5 MB/ms.
        self.assertTrue(bandwidth_sufficient(1_000_000, 2, 500_000))
        self.assertFalse(bandwidth_sufficient(1_000_000, 2, 499_999))

    def test_selected_source_is_retained_when_final_time_gate_fails(self):
        selection = SourceDecision(
            selected_source_id="s2",
            probe_layer=4,
            reuse_layer=None,
            safe_repair_ratio_upper=0.2,
            prefetch_m=1,
            selection_reason=SelectionReason.EARLY_CONFIDENT,
            predicted_cost_upper_ms=40,
        )
        plan = DynamicReusePlanner(0.8).plan(
            [option(4, 20, 0, 65)]
        )
        execution = finalize_execution(selection, plan)
        self.assertEqual(execution.selected_source_id, "s2")
        self.assertFalse(execution.reuse_accepted)
        self.assertEqual(
            execution.rejection_reason,
            RejectionReason.FINAL_TIME_GATE_FAILED,
        )
        self.assertEqual(
            execution.execution_mode, ExecutionMode.FULL_RECOMPUTE
        )

    def test_abstention_cannot_fall_through_to_reuse(self):
        selection = SourceDecision(
            selected_source_id=None,
            probe_layer=8,
            reuse_layer=None,
            safe_repair_ratio_upper=None,
            prefetch_m=0,
            selection_reason=SelectionReason.MAX_PROBE_UNCERTAIN,
        )
        execution = finalize_execution(selection)
        self.assertTrue(execution.abstained)
        self.assertFalse(execution.reuse_accepted)
        self.assertEqual(
            execution.rejection_reason,
            RejectionReason.SELECTION_UNCERTAIN,
        )
        with self.assertRaises(ValueError):
            ExecutionDecision(
                selected_source_id=None,
                selection_reason=SelectionReason.MAX_PROBE_UNCERTAIN,
                reuse_accepted=True,
                rejection_reason=None,
                execution_mode=ExecutionMode.REUSE,
                probe_layer=8,
                reuse_layer=8,
                safe_repair_ratio_upper=0.2,
                timing=option(8, 10, 5, 5).timing(),
            )


if __name__ == "__main__":
    unittest.main()
