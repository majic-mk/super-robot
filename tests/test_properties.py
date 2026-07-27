import random
import unittest

from probekv.contracts import CandidateBounds
from probekv.cost import DynamicReusePlanner, LayerOption
from probekv.prefetch import PrefetchCandidate, PrefetchPolicy, choose_prefetch
from probekv.selector import DynamicProbeSelector, ProbePolicy


class RandomizedInvariantTests(unittest.TestCase):
    def test_admission_never_accepts_over_gamma(self):
        randomizer = random.Random(20260726)
        planner = DynamicReusePlanner(0.8)
        for _ in range(500):
            full = randomizer.uniform(10, 200)
            option = LayerOption(
                layer=randomizer.randint(1, 16),
                repair_ratio_upper=randomizer.random(),
                probe_ms=randomizer.uniform(0, 10),
                compare_ms=randomizer.uniform(0, 5),
                load_ms=randomizer.uniform(0, 50),
                overlap_ms=randomizer.uniform(0, 50),
                repair_ms=randomizer.uniform(0, 200),
                full_ms=full,
                buffer_ready=True,
            )
            result = planner.plan([option])
            if result.accepted:
                self.assertLessEqual(result.timing.reuse_total_ms, 0.8 * full)

    def test_dynamic_prefetch_never_exceeds_hbm(self):
        randomizer = random.Random(20260727)
        for _ in range(200):
            candidates = [
                PrefetchCandidate(
                    "s%d" % index,
                    probability,
                    randomizer.randint(1, 8) * 1000,
                    randomizer.uniform(0, 20),
                )
                for index, probability in enumerate([0.5, 0.3, 0.15, 0.05])
            ]
            budget = randomizer.randint(1, 20) * 1000
            decision = choose_prefetch(
                PrefetchPolicy.DYNAMIC, candidates, budget, randomizer.uniform(0, 10)
            )
            self.assertLessEqual(decision.transferred_bytes, budget)

    def test_selector_never_claims_confidence_when_intervals_overlap(self):
        selector = DynamicProbeSelector(ProbePolicy((1, 2, 4), 4))
        overlap = (
            CandidateBounds("s1", 0.2, 10, 20),
            CandidateBounds("s2", 0.3, 19, 30),
            CandidateBounds("s3", 0.4, 15, 25),
        )
        decision = selector.select({1: overlap, 2: overlap, 4: overlap})
        self.assertTrue(decision.abstained)


if __name__ == "__main__":
    unittest.main()
