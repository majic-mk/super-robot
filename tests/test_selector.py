import unittest

from probekv.contracts import (
    CandidateBounds,
    CostValueKind,
    DecisionReason,
    SelectionReason,
)
from probekv.cost import cost_breakdown_from_total
from probekv.selector import (
    DynamicProbeSelector,
    ProbePolicy,
    SelectorPolicy,
    dense_probe_checkpoints,
    default_probe_checkpoints,
    normalized_oracle_regret,
)


class SelectorTests(unittest.TestCase):
    @staticmethod
    def _component_candidate(source_id, ratio, lower, upper, layer):
        return CandidateBounds(
            source_id,
            ratio,
            lower,
            upper,
            cost_lower_breakdown=cost_breakdown_from_total(
                lower,
                100,
                layer,
                CostValueKind.PREDICTED_LOWER,
            ),
            cost_upper_breakdown=cost_breakdown_from_total(
                upper,
                100,
                layer,
                CostValueKind.PREDICTED_UPPER,
            ),
        )

    def test_unified_cost_keeps_early_source_selection_before_probe_max(self):
        selector = DynamicProbeSelector(
            ProbePolicy(
                (1, 2, 4),
                4,
                SelectorPolicy.FINAL_ECONOMIC_MIN_COST,
                gamma=0.8,
                preliminary_economic_filter=True,
                require_component_cost_bounds=True,
            )
        )
        decision = selector.select(
            {
                1: (
                    self._component_candidate("s1", 0.2, 10, 30, 1),
                    self._component_candidate("s2", 0.3, 20, 40, 1),
                ),
                2: (
                    self._component_candidate("s1", 0.2, 10, 18, 2),
                    self._component_candidate("s2", 0.3, 22, 35, 2),
                ),
                4: (
                    self._component_candidate("s1", 0.2, 10, 15, 4),
                    self._component_candidate("s2", 0.3, 25, 30, 4),
                ),
            },
            full_recompute_ms=100,
        )
        self.assertEqual(decision.selected_source_id, "s1")
        self.assertEqual(decision.probe_layer, 2)
        self.assertEqual(
            decision.selection_reason, SelectionReason.EARLY_CONFIDENT
        )
        self.assertIsNotNone(decision.predicted_cost_breakdown)

    def test_unified_cost_rejects_aggregate_only_candidate_bounds(self):
        selector = DynamicProbeSelector(
            ProbePolicy(
                (1,),
                1,
                SelectorPolicy.FINAL_ECONOMIC_MIN_COST,
                gamma=0.8,
                preliminary_economic_filter=True,
                require_component_cost_bounds=True,
            )
        )
        with self.assertRaisesRegex(ValueError, "component bounds"):
            selector.select(
                {1: (CandidateBounds("s1", 0.2, 10, 20),)},
                full_recompute_ms=100,
            )

    def test_early_exit_requires_separated_intervals(self):
        selector = DynamicProbeSelector(ProbePolicy((1, 2, 4), 4))
        decision = selector.select(
            {
                1: (
                    CandidateBounds("s1", 0.2, 10, 20),
                    CandidateBounds("s2", 0.3, 18, 25),
                ),
                2: (
                    CandidateBounds("s1", 0.2, 10, 14),
                    CandidateBounds("s2", 0.3, 16, 22),
                ),
            }
        )
        self.assertEqual(decision.selected_source_id, "s1")
        self.assertEqual(decision.probe_layer, 2)

    def test_max_probe_uncertain_abstains(self):
        selector = DynamicProbeSelector(ProbePolicy((1, 2), 2))
        overlapping = (
            CandidateBounds("s1", 0.2, 10, 20),
            CandidateBounds("s2", 0.3, 15, 25),
        )
        decision = selector.select({1: overlapping, 2: overlapping})
        self.assertTrue(decision.abstained)
        self.assertEqual(decision.reason, DecisionReason.MAX_PROBE_UNCERTAIN)

    def test_final_min_cost_selects_safe_economic_source_without_separation(self):
        selector = DynamicProbeSelector(
            ProbePolicy(
                (1, 2),
                2,
                SelectorPolicy.FINAL_ECONOMIC_MIN_COST,
                gamma=0.8,
            )
        )
        overlapping = (
            CandidateBounds("s1", 0.20, 10, 35),
            CandidateBounds("s2", 0.25, 12, 30),
        )
        decision = selector.select(
            {1: overlapping, 2: overlapping},
            full_recompute_ms=100,
        )
        self.assertEqual(decision.selected_source_id, "s2")
        self.assertEqual(
            decision.selection_reason,
            SelectionReason.FINAL_ECONOMIC_MIN_COST,
        )
        self.assertFalse(decision.abstained)

    def test_final_max_reuse_uses_tolerance_then_cost(self):
        selector = DynamicProbeSelector(
            ProbePolicy(
                (1,),
                1,
                SelectorPolicy.FINAL_ECONOMIC_MAX_REUSE,
                gamma=0.8,
                reuse_ratio_tolerance=0.02,
            )
        )
        decision = selector.select(
            {
                1: (
                    CandidateBounds("s1", 0.10, 20, 70),
                    CandidateBounds("s2", 0.115, 10, 20),
                    CandidateBounds("s3", 0.14, 5, 10),
                )
            },
            full_recompute_ms=100,
        )
        self.assertEqual(decision.selected_source_id, "s2")
        self.assertEqual(
            decision.selection_reason,
            SelectionReason.FINAL_MAX_REUSE_LOWER_BOUND,
        )

    def test_final_filter_skips_uneconomic_max_reuse_source(self):
        selector = DynamicProbeSelector(
            ProbePolicy(
                (1,),
                1,
                SelectorPolicy.FINAL_ECONOMIC_MAX_REUSE,
                gamma=0.8,
            )
        )
        decision = selector.select(
            {
                1: (
                    CandidateBounds("s1", 0.10, 80, 90),
                    CandidateBounds("s2", 0.20, 20, 40),
                )
            },
            full_recompute_ms=100,
        )
        self.assertEqual(decision.selected_source_id, "s2")

    def test_v4_preliminary_filter_never_selects_uneconomic_early_winner(self):
        selector = DynamicProbeSelector(
            ProbePolicy(
                (1, 2),
                2,
                SelectorPolicy.FINAL_ECONOMIC_MIN_COST,
                gamma=0.8,
                preliminary_economic_filter=True,
            )
        )
        decision = selector.select(
            {
                1: (
                    CandidateBounds("s1", 0.1, 10, 90),
                    CandidateBounds("s2", 0.2, 95, 100),
                ),
                2: (
                    CandidateBounds("s1", 0.1, 10, 90),
                    CandidateBounds("s2", 0.2, 20, 40),
                ),
            },
            full_recompute_ms=100,
        )
        self.assertEqual(decision.selected_source_id, "s2")
        self.assertEqual(decision.probe_layer, 2)

    def test_final_abstains_when_quality_or_economic_set_is_empty(self):
        policy = ProbePolicy(
            (1,),
            1,
            SelectorPolicy.FINAL_ECONOMIC_MIN_COST,
            gamma=0.8,
        )
        selector = DynamicProbeSelector(policy)
        no_quality = selector.select(
            {1: (CandidateBounds("s1", 0.1, 10, 20, False),)},
            full_recompute_ms=100,
        )
        self.assertTrue(no_quality.abstained)
        self.assertEqual(
            no_quality.selection_reason,
            SelectionReason.NO_QUALITY_SAFE_SOURCE,
        )
        no_economic = selector.select(
            {1: (CandidateBounds("s1", 0.1, 80, 90),)},
            full_recompute_ms=100,
        )
        self.assertTrue(no_economic.abstained)
        self.assertEqual(
            no_economic.selection_reason,
            SelectionReason.NO_ECONOMIC_SOURCE,
        )

    def test_32_layer_contract(self):
        self.assertEqual(default_probe_checkpoints(32), (1, 2, 4, 6, 8))
        self.assertEqual(dense_probe_checkpoints(32), tuple(range(1, 9)))
        self.assertLessEqual(default_probe_checkpoints(22)[-1], 5)

    def test_normalized_regret(self):
        self.assertAlmostEqual(normalized_oracle_regret(15, 10, 20), 0.5)


if __name__ == "__main__":
    unittest.main()
