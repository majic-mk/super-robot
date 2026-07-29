import unittest

from probekv.contracts import (
    CostAccountingPolicy,
    CostValueKind,
    ExecutionMode,
    RejectionReason,
    SelectionReason,
    SourceDecision,
)
from probekv.cost import cost_breakdown_from_total
from probekv.orchestration import (
    ClosedLoopPolicy,
    RefinedCostMeasurement,
    SchedulingFeedback,
    TwoStageReuseController,
)


def selected():
    return SourceDecision(
        selected_source_id="s1",
        probe_layer=4,
        reuse_layer=None,
        safe_repair_ratio_upper=0.2,
        prefetch_m=1,
        selection_reason=SelectionReason.FINAL_ECONOMIC_MIN_COST,
        predicted_cost_upper_ms=55,
    )


def selected_with_unified_cost():
    return SourceDecision(
        selected_source_id="s1",
        probe_layer=4,
        reuse_layer=None,
        safe_repair_ratio_upper=0.2,
        prefetch_m=1,
        selection_reason=SelectionReason.FINAL_ECONOMIC_MIN_COST,
        predicted_cost_upper_ms=55,
        predicted_cost_breakdown=cost_breakdown_from_total(
            55,
            100,
            4,
            CostValueKind.PREDICTED_UPPER,
        ),
    )


def feedback():
    return SchedulingFeedback(
        selected_source_id="s1",
        evaluated_reuse_boundary=6,
        source_ready=True,
        load_ms=20,
        overlap_ms=15,
        source_ready_ms=20,
        scheduled_step_finish_ms=21,
        a_resume_ms=21,
        post_ready_blocking_ms=1,
        load_interference_ms=2,
        useful_a_dense_ms=4,
        useful_other_request_work_ms=10,
    )


class FakeRuntime:
    def __init__(self, repair_ms=30):
        self.calls = []
        self.repair_ms = repair_ms

    def load_and_schedule(self, selection):
        self.calls.append("load_and_schedule:%s" % selection.selected_source_id)
        return feedback()

    def measure_refined_cost(self, selection, scheduling):
        self.calls.append("measure_refined_cost")
        return RefinedCostMeasurement(
            selected_source_id=selection.selected_source_id,
            evaluated_reuse_boundary=scheduling.evaluated_reuse_boundary,
            repair_ratio_upper=0.2,
            probe_ms=4,
            compare_ms=1,
            repair_ms=self.repair_ms,
            full_ms=100,
        )

    def execute_reuse(self, selection, decision):
        self.calls.append("execute_reuse")
        return "reuse-output"

    def execute_full(self, selection, decision):
        self.calls.append("execute_full")
        return "full-output"


class ClosedLoopTests(unittest.TestCase):
    def test_unified_protocol_checks_preliminary_breakdown_before_load(self):
        runtime = FakeRuntime(repair_ms=30)
        controller = TwoStageReuseController(
            0.8,
            ClosedLoopPolicy.TWO_STAGE_REFINED_ADMISSION,
            CostAccountingPolicy.UNIFIED_COMPONENTS_V1,
        )
        with self.assertRaisesRegex(
            ValueError, "predicted cost breakdown before loading"
        ):
            controller.execute(selected(), runtime)
        self.assertEqual(runtime.calls, [])

    def test_unified_protocol_selects_once_then_only_admits_or_rejects(self):
        controller = TwoStageReuseController(
            0.8,
            ClosedLoopPolicy.TWO_STAGE_REFINED_ADMISSION,
            CostAccountingPolicy.UNIFIED_COMPONENTS_V1,
        )
        accepted = controller.execute(
            selected_with_unified_cost(), FakeRuntime(repair_ms=30)
        )
        self.assertEqual(accepted.execution.selected_source_id, "s1")
        self.assertTrue(accepted.execution.reuse_accepted)
        rejected = controller.execute(
            selected_with_unified_cost(), FakeRuntime(repair_ms=75)
        )
        self.assertEqual(rejected.execution.selected_source_id, "s1")
        self.assertFalse(rejected.execution.reuse_accepted)
        self.assertEqual(rejected.runtime_result, "full-output")
        audit = rejected.to_audit_record()
        self.assertEqual(audit["source_locked_at_probe"], "s1")
        self.assertEqual(audit["refined_source_id"], "s1")
        self.assertFalse(audit["refined_source_changed"])

    def test_scheduler_feedback_precedes_final_admission_and_reuse(self):
        runtime = FakeRuntime(repair_ms=30)
        result = TwoStageReuseController(0.8).execute(
            selected(), runtime
        )
        self.assertEqual(
            runtime.calls,
            [
                "load_and_schedule:s1",
                "measure_refined_cost",
                "execute_reuse",
            ],
        )
        self.assertTrue(result.execution.reuse_accepted)
        self.assertEqual(result.execution.actual_reuse_boundary, 6)
        audit = result.to_audit_record()
        self.assertEqual(audit["source_ready_ms"], 20)
        self.assertEqual(audit["a_resume_ms"], 21)
        self.assertEqual(audit["post_ready_blocking_ms"], 1)
        self.assertEqual(audit["actual_reuse_boundary"], 6)
        self.assertEqual(
            audit["probe_admission_state"], "not_evaluated"
        )
        self.assertEqual(audit["admission_state"], "accepted")

    def test_refined_time_rejects_but_retains_selected_source(self):
        runtime = FakeRuntime(repair_ms=75)
        result = TwoStageReuseController(0.8).execute(
            selected(), runtime
        )
        self.assertEqual(runtime.calls[-1], "execute_full")
        self.assertEqual(result.execution.selected_source_id, "s1")
        self.assertFalse(result.execution.reuse_accepted)
        self.assertEqual(
            result.execution.rejection_reason,
            RejectionReason.FINAL_TIME_GATE_FAILED,
        )
        self.assertEqual(
            result.execution.execution_mode,
            ExecutionMode.FULL_RECOMPUTE,
        )
        self.assertIsNone(result.execution.actual_reuse_boundary)
        self.assertEqual(
            result.to_audit_record()["evaluated_reuse_boundary"], 6
        )

    def test_abstention_executes_full_without_loading_or_default_source(self):
        runtime = FakeRuntime()
        abstention = SourceDecision(
            selected_source_id=None,
            probe_layer=8,
            reuse_layer=None,
            safe_repair_ratio_upper=None,
            prefetch_m=0,
            selection_reason=SelectionReason.MAX_PROBE_UNCERTAIN,
        )
        result = TwoStageReuseController().execute(abstention, runtime)
        self.assertEqual(runtime.calls, ["execute_full"])
        self.assertIsNone(result.scheduling)
        self.assertIsNone(result.execution.selected_source_id)

    def test_scheduler_and_refined_cost_must_bind_same_boundary(self):
        class BadRuntime(FakeRuntime):
            def measure_refined_cost(self, selection, scheduling):
                value = super().measure_refined_cost(selection, scheduling)
                return RefinedCostMeasurement(
                    value.selected_source_id,
                    7,
                    value.repair_ratio_upper,
                    value.probe_ms,
                    value.compare_ms,
                    value.repair_ms,
                    value.full_ms,
                )

        with self.assertRaisesRegex(ValueError, "reuse boundary"):
            TwoStageReuseController().execute(selected(), BadRuntime())

    def test_refined_stage_cannot_switch_to_a_cheaper_source(self):
        class SourceSwitchingRuntime(FakeRuntime):
            def measure_refined_cost(self, selection, scheduling):
                value = super().measure_refined_cost(selection, scheduling)
                return RefinedCostMeasurement(
                    "s2",
                    value.evaluated_reuse_boundary,
                    value.repair_ratio_upper,
                    value.probe_ms,
                    value.compare_ms,
                    1,
                    value.full_ms,
                )

        with self.assertRaisesRegex(ValueError, "another source"):
            TwoStageReuseController().execute(
                selected(), SourceSwitchingRuntime()
            )

    def test_unified_protocol_rejects_mismatched_cost_scope(self):
        class MismatchedScopeRuntime(FakeRuntime):
            def measure_refined_cost(self, selection, scheduling):
                value = super().measure_refined_cost(selection, scheduling)
                return RefinedCostMeasurement(
                    selected_source_id=value.selected_source_id,
                    evaluated_reuse_boundary=(
                        value.evaluated_reuse_boundary
                    ),
                    repair_ratio_upper=value.repair_ratio_upper,
                    probe_ms=value.probe_ms,
                    compare_ms=value.compare_ms,
                    repair_ms=value.repair_ms,
                    full_ms=value.full_ms,
                    cost_origin="selection_complete",
                    cost_endpoint="first_token_ready",
                )

        controller = TwoStageReuseController(
            0.8,
            ClosedLoopPolicy.TWO_STAGE_REFINED_ADMISSION,
            CostAccountingPolicy.UNIFIED_COMPONENTS_V1,
        )
        with self.assertRaisesRegex(ValueError, "different accounting"):
            controller.execute(
                selected_with_unified_cost(), MismatchedScopeRuntime()
            )


if __name__ == "__main__":
    unittest.main()
