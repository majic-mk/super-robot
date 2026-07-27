import unittest

from probekv.backend import DeterministicSimulationBackend
from probekv.contracts import KVLocation
from probekv.gates import (
    gate_h1,
    gate_h2,
    gate_h3,
    gate_h4,
    publication_band,
)
from probekv.statistics import (
    ConfidenceInterval,
    clopper_pearson_upper_bound,
    grouped_paired_bootstrap,
    holm_bonferroni,
    minimum_zero_violation_trials,
    paired_hodges_lehmann,
    kendall_tau,
    spearman_correlation,
)

from tests.helpers import canonical_source


class BackendTests(unittest.TestCase):
    def test_simulation_backend_is_explicitly_not_paper_evidence(self):
        backend = DeterministicSimulationBackend(
            safe_ratio_by_source={"s1": 0.2}
        )
        self.assertFalse(backend.paper_evidence)
        source = canonical_source()
        low = backend.repair(source, 4, 0.1)
        high = backend.repair(source, 4, 0.2)
        self.assertLess(low.token_f1, high.token_f1)
        self.assertGreater(
            backend.prepare_source(source, KVLocation.GPU), 0
        )


class StatisticsTests(unittest.TestCase):
    def test_grouped_bootstrap_is_deterministic(self):
        values = {"a": [1, 2], "b": [3, 4]}
        first = grouped_paired_bootstrap(values, iterations=100, seed=1)
        second = grouped_paired_bootstrap(values, iterations=100, seed=1)
        self.assertEqual(first, second)

    def test_hodges_lehmann(self):
        self.assertEqual(paired_hodges_lehmann([1, 2, 3]), 2)

    def test_holm_stops_after_first_non_rejection(self):
        decisions = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.2})
        self.assertTrue(decisions["a"])
        self.assertFalse(decisions["b"])
        self.assertFalse(decisions["c"])

    def test_exact_tail_sample_requirement(self):
        self.assertGreater(clopper_pearson_upper_bound(0, 200), 0.01)
        self.assertEqual(minimum_zero_violation_trials(), 299)

    def test_rank_correlations(self):
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [10, 20, 30]), 1)
        self.assertAlmostEqual(kendall_tau([1, 2, 3], [30, 20, 10]), -1)


class GateTests(unittest.TestCase):
    def test_h1_contract(self):
        result = gate_h1(
            [0.2, 0.15, 0.01, 0.02],
            [0.2, 0.15, 0.1, 0.1],
            ConfidenceInterval(0.13, 0.05, 0.2),
        )
        self.assertTrue(result.passed)

    def test_h2_contract(self):
        result = gate_h2(
            0.2,
            0.1,
            ConfidenceInterval(0.1, 0.03, 0.15),
            0.85,
            0.04,
        )
        self.assertTrue(result.passed)

    def test_h3_tail_rule_is_strict(self):
        # With only 200 cases even 0 observed violations cannot prove <=1%.
        result = gate_h3(ConfidenceInterval(0, -0.005, 0.005), 0, 200)
        self.assertFalse(result.passed)

    def test_h4_and_publication_band(self):
        self.assertTrue(gate_h4([70, 79], [100, 100], 0.8).passed)
        self.assertEqual(publication_band(0.12, 0.11, 0.06), "q1_candidate")


if __name__ == "__main__":
    unittest.main()
