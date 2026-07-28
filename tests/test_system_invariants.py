import unittest

from probekv.contracts import (
    CandidateBounds,
    ExecutionMode,
    RejectionReason,
)
from probekv.cost import (
    DynamicReusePlanner,
    LayerOption,
    finalize_execution,
)
from probekv.scheduler import (
    ReadyRequest,
    SchedulerPolicy,
    SchedulerScenario,
    simulate_waiting_queue,
)
from probekv.selector import (
    DynamicProbeSelector,
    ProbePolicy,
    SelectorPolicy,
)


class CrossModuleInvariantTests(unittest.TestCase):
    def test_selected_source_survives_scheduler_cost_rejection(self):
        selector = DynamicProbeSelector(
            ProbePolicy(
                (1,),
                1,
                SelectorPolicy.FINAL_ECONOMIC_MIN_COST,
                gamma=0.8,
            )
        )
        selection = selector.select(
            {
                1: (
                    CandidateBounds("s1", 0.2, 40, 60),
                    CandidateBounds("s2", 0.3, 45, 70),
                )
            },
            full_recompute_ms=100,
        )
        schedule = simulate_waiting_queue(
            SchedulerPolicy.HYBRID_BOUNDED_OVERRUN,
            SchedulerScenario(
                load_ms=5,
                dense_layer_ms=1,
                max_extra_dense_layers=0,
                repair_ms=60,
                decode_start_ms=1,
                other_ready_work_ms=2,
                microbatch_ms=2,
                max_post_ready_overrun_ms=1,
            ),
            [ReadyRequest("b", 4, 3, 2)],
            a_layer=1,
            hybrid_dense_budget=0,
        )
        plan = DynamicReusePlanner(0.8).plan(
            [
                LayerOption(
                    layer=1,
                    repair_ratio_upper=0.2,
                    probe_ms=4,
                    compare_ms=1,
                    load_ms=20,
                    overlap_ms=5,
                    repair_ms=60,
                    full_ms=100,
                    post_ready_blocking_ms=(
                        schedule.post_ready_blocking_ms
                    ),
                )
            ]
        )
        execution = finalize_execution(selection, plan)
        self.assertEqual(execution.selected_source_id, "s1")
        self.assertFalse(execution.reuse_accepted)
        self.assertEqual(
            execution.rejection_reason,
            RejectionReason.FINAL_TIME_GATE_FAILED,
        )
        self.assertEqual(
            execution.execution_mode, ExecutionMode.FULL_RECOMPUTE
        )
        self.assertEqual(schedule.useful_other_request_work_ms, 2)
        self.assertEqual(schedule.post_ready_blocking_ms, 1)

    def test_no_selected_source_never_reaches_reuse(self):
        selector = DynamicProbeSelector(
            ProbePolicy((1,), 1, SelectorPolicy.STRICT_INTERVAL)
        )
        selection = selector.select(
            {
                1: (
                    CandidateBounds("s1", 0.2, 10, 30),
                    CandidateBounds("s2", 0.3, 20, 40),
                )
            }
        )
        execution = finalize_execution(selection)
        self.assertIsNone(execution.selected_source_id)
        self.assertFalse(execution.reuse_accepted)
        self.assertEqual(
            execution.execution_mode, ExecutionMode.FULL_RECOMPUTE
        )


if __name__ == "__main__":
    unittest.main()
