import unittest

from probekv.prefetch import (
    PrefetchCandidate,
    PrefetchPolicy,
    choose_prefetch,
)
from probekv.scheduler import (
    ReadyRequest,
    SchedulerPolicy,
    SchedulerScenario,
    simulate_schedule,
    simulate_waiting_queue,
)


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            PrefetchCandidate("s1", 0.65, 1_000_000_000, 20),
            PrefetchCandidate("s2", 0.20, 1_000_000_000, 20),
            PrefetchCandidate("s3", 0.10, 1_000_000_000, 20),
            PrefetchCandidate("s4", 0.05, 1_000_000_000, 20),
        ]

    def test_p4_uploads_all_only_when_hbm_allows(self):
        accepted = choose_prefetch(
            PrefetchPolicy.P4, self.candidates, 5_000_000_000, 0
        )
        self.assertEqual(len(accepted.source_ids), 4)
        rejected = choose_prefetch(
            PrefetchPolicy.P4, self.candidates, 2_000_000_000, 0
        )
        self.assertEqual(rejected.source_ids, ())

    def test_dynamic_accounts_for_hbm_and_wasted_bytes(self):
        decision = choose_prefetch(
            PrefetchPolicy.DYNAMIC,
            self.candidates,
            2_000_000_000,
            overlap_ms=5,
            byte_penalty_ms_per_gb=2,
        )
        self.assertLessEqual(decision.transferred_bytes, 2_000_000_000)
        self.assertLessEqual(len(decision.source_ids), 2)

    def test_p1_loads_only_final_winner(self):
        decision = choose_prefetch(
            PrefetchPolicy.P1,
            self.candidates,
            2_000_000_000,
            overlap_ms=5,
            winner_source_id="s2",
        )
        self.assertEqual(decision.source_ids, ("s2",))
        self.assertEqual(decision.expected_visible_load_ms, 15)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.scenario = SchedulerScenario(
            load_ms=8,
            dense_layer_ms=2,
            max_extra_dense_layers=4,
            repair_ms=20,
            decode_start_ms=2,
            other_ready_work_ms=20,
            microbatch_ms=1,
        )

    def test_a_only_hides_load_and_reduces_a_ttft(self):
        baseline = simulate_schedule(SchedulerPolicy.NO_OVERLAP, self.scenario)
        a_only = simulate_schedule(SchedulerPolicy.A_ONLY, self.scenario)
        self.assertLess(a_only.a_ttft_ms, baseline.a_ttft_ms)
        self.assertEqual(a_only.other_work_completed_ms, 0)

    def test_b_only_uses_wait_without_delaying_a(self):
        baseline = simulate_schedule(SchedulerPolicy.NO_OVERLAP, self.scenario)
        b_only = simulate_schedule(SchedulerPolicy.B_ONLY, self.scenario)
        self.assertEqual(b_only.a_ttft_ms, baseline.a_ttft_ms)
        self.assertEqual(b_only.other_work_completed_ms, 8)

    def test_hybrid_combines_a_and_b(self):
        hybrid = simulate_schedule(
            SchedulerPolicy.HYBRID, self.scenario, hybrid_dense_budget=2
        )
        self.assertGreater(hybrid.useful_a_dense_ms, 0)
        self.assertGreater(hybrid.other_work_completed_ms, 0)

    def test_many_request_queue_stops_at_source_ready_event(self):
        requests = [
            ReadyRequest("b", 0, 4, 20),
            ReadyRequest("c", 0, 4, 20),
            ReadyRequest("d", 2, 7, 20),
        ]
        result = simulate_waiting_queue(
            SchedulerPolicy.HYBRID,
            self.scenario,
            requests,
            a_layer=4,
            hybrid_dense_budget=1,
        )
        self.assertEqual(result.elapsed_wait_window_ms, self.scenario.load_ms)
        self.assertGreater(result.service_ms_by_request["b"], 0)
        self.assertGreater(result.service_ms_by_request["c"], 0)
        self.assertLessEqual(result.a_ttft_ms, 30)
        self.assertGreater(result.jain_fairness, 0)


if __name__ == "__main__":
    unittest.main()
