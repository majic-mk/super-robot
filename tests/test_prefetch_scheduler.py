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

    def test_bounded_overrun_accounts_for_atomic_step_and_a_delay(self):
        scenario = SchedulerScenario(
            load_ms=5,
            dense_layer_ms=1,
            max_extra_dense_layers=0,
            repair_ms=10,
            decode_start_ms=1,
            other_ready_work_ms=2,
            microbatch_ms=2,
            max_post_ready_overrun_ms=1,
        )
        result = simulate_waiting_queue(
            SchedulerPolicy.HYBRID_BOUNDED_OVERRUN,
            scenario,
            [ReadyRequest("b", 4, 3, 2)],
            a_layer=4,
            hybrid_dense_budget=0,
        )
        self.assertEqual(result.source_ready_ms, 5)
        self.assertEqual(result.scheduled_step_finish_ms, 6)
        self.assertEqual(result.a_resume_ms, 6)
        self.assertEqual(result.post_ready_blocking_ms, 1)
        self.assertEqual(result.useful_other_request_work_ms, 2)
        self.assertEqual(result.hidden_work_ms, 1)

    def test_bounded_overrun_rejects_step_above_budget(self):
        scenario = SchedulerScenario(
            load_ms=5,
            dense_layer_ms=1,
            max_extra_dense_layers=0,
            repair_ms=10,
            decode_start_ms=1,
            other_ready_work_ms=2,
            microbatch_ms=2,
            max_post_ready_overrun_ms=0.5,
        )
        result = simulate_waiting_queue(
            SchedulerPolicy.HYBRID_BOUNDED_OVERRUN,
            scenario,
            [ReadyRequest("b", 4, 3, 2)],
            a_layer=4,
            hybrid_dense_budget=0,
        )
        self.assertEqual(result.a_resume_ms, 5)
        self.assertEqual(result.post_ready_blocking_ms, 0)
        self.assertEqual(result.useful_other_request_work_ms, 0)

    def test_strict_policy_never_splits_or_overruns_atomic_step(self):
        scenario = SchedulerScenario(
            load_ms=5,
            dense_layer_ms=1,
            max_extra_dense_layers=0,
            repair_ms=10,
            decode_start_ms=1,
            other_ready_work_ms=2,
            microbatch_ms=2,
            max_post_ready_overrun_ms=10,
        )
        result = simulate_waiting_queue(
            SchedulerPolicy.HYBRID_STRICT,
            scenario,
            [ReadyRequest("b", 4, 3, 2)],
            a_layer=4,
            hybrid_dense_budget=0,
        )
        self.assertEqual(result.a_resume_ms, result.source_ready_ms)
        self.assertEqual(result.service_ms_by_request["b"], 0)
        self.assertLessEqual(
            result.hidden_work_ms, result.source_ready_ms
        )

    def test_load_interference_moves_source_ready_explicitly(self):
        scenario = SchedulerScenario(
            load_ms=5,
            dense_layer_ms=1,
            max_extra_dense_layers=0,
            repair_ms=1,
            decode_start_ms=1,
            other_ready_work_ms=0,
            load_interference_ms=2,
        )
        result = simulate_schedule(SchedulerPolicy.NO_OVERLAP, scenario)
        self.assertEqual(result.source_ready_ms, 7)
        self.assertEqual(result.load_interference_ms, 2)


if __name__ == "__main__":
    unittest.main()
