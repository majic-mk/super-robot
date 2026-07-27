import unittest

from probekv.contracts import DecisionReason
from probekv.cost import (
    DynamicReusePlanner,
    LayerOption,
    bandwidth_sufficient,
    conservative_ratio_for_layer,
)


def option(layer, load, overlap, repair, full=100, ready=True):
    return LayerOption(layer, 0.2, 4, 1, load, overlap, repair, full, ready)


class CostTests(unittest.TestCase):
    def test_total_reuse_cost_includes_all_components(self):
        plan = DynamicReusePlanner(0.8).plan([option(4, 20, 5, 55)])
        self.assertTrue(plan.accepted)
        self.assertEqual(plan.timing.visible_load_ms, 15)
        self.assertEqual(plan.timing.reuse_total_ms, 75)

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


if __name__ == "__main__":
    unittest.main()
