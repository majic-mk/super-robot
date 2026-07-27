import unittest

from probekv.contracts import CandidateBounds, DecisionReason
from probekv.selector import (
    DynamicProbeSelector,
    ProbePolicy,
    dense_probe_checkpoints,
    default_probe_checkpoints,
    normalized_oracle_regret,
)


class SelectorTests(unittest.TestCase):
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

    def test_32_layer_contract(self):
        self.assertEqual(default_probe_checkpoints(32), (1, 2, 4, 6, 8))
        self.assertEqual(dense_probe_checkpoints(32), tuple(range(1, 9)))
        self.assertLessEqual(default_probe_checkpoints(22)[-1], 5)

    def test_normalized_regret(self):
        self.assertAlmostEqual(normalized_oracle_regret(15, 10, 20), 0.5)


if __name__ == "__main__":
    unittest.main()
