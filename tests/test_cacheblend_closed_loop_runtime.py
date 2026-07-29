import unittest

from probekv.cacheblend_closed_loop_runtime import (
    CacheBlendBoundaryProfile,
    CacheBlendClosedLoopRuntime,
    CacheBlendExecutionObservation,
    CacheBlendRuntimeCapabilities,
    CacheBlendRuntimeState,
    CacheBlendScheduleObservation,
    SourceLoadTicket,
)
from probekv.contracts import (
    CostAccountingPolicy,
    CostValueKind,
    ExecutionMode,
    SelectionReason,
    SourceDecision,
)
from probekv.cost import cost_breakdown_from_total
from probekv.orchestration import TwoStageReuseController

from tests.helpers import canonical_source


def selected(source_id="s1"):
    predicted = cost_breakdown_from_total(
        60,
        100,
        2,
        CostValueKind.PREDICTED_UPPER,
        probe_ms=4,
        compare_ms=1,
        visible_load_ms=5,
    )
    return SourceDecision(
        selected_source_id=source_id,
        probe_layer=2,
        reuse_layer=None,
        safe_repair_ratio_upper=0.2,
        prefetch_m=1,
        selection_reason=SelectionReason.EARLY_CONFIDENT,
        predicted_cost_upper_ms=predicted.reuse_total_ms,
        predicted_cost_breakdown=predicted,
    )


class FakeClosedLoopEngine:
    def __init__(
        self,
        repair_ms=20,
        boundary=6,
        scheduled_source_id="s1",
        capabilities=None,
    ):
        self.repair_ms = repair_ms
        self.boundary = boundary
        self.scheduled_source_id = scheduled_source_id
        self.calls = []
        self._capabilities = capabilities or CacheBlendRuntimeCapabilities(
            backend_name="fake-cacheblend-a800",
            async_source_loading=True,
            layer_resumable_prefill=True,
            scheduler_feedback=True,
            boundary_conditioned_profiles=True,
            canonical_sources_read_only=True,
            cuda_event_timing=True,
        )

    def capabilities(self):
        return self._capabilities

    def begin_source_load(self, source):
        self.calls.append("begin_load:%s" % source.source_id)
        return SourceLoadTicket(source.source_id, 5.0, 512)

    def schedule_waiting_window(self, selection, ticket):
        self.calls.append("schedule:%s" % ticket.selected_source_id)
        return CacheBlendScheduleObservation(
            selected_source_id=self.scheduled_source_id,
            evaluated_reuse_boundary=self.boundary,
            source_ready=True,
            source_ready_ms=20.0,
            scheduled_step_finish_ms=21.0,
            a_resume_ms=21.0,
            overlap_ms=10.0,
            load_interference_ms=2.0,
            useful_a_dense_ms=4.0,
            useful_other_request_work_ms=8.0,
            probe_ms=4.0,
            compare_ms=1.0,
            transferred_bytes=512,
        )

    def profile_boundary(self, source, boundary, repair_ratio_upper):
        self.calls.append("profile:%s:%d" % (source.source_id, boundary))
        return CacheBlendBoundaryProfile(
            selected_source_id=source.source_id,
            evaluated_reuse_boundary=boundary,
            repair_ratio_upper=repair_ratio_upper,
            repair_selection_ms_upper=2.0,
            repair_ms_upper=self.repair_ms,
            remaining_layer_ms_upper=20.0,
            full_total_ms=100.0,
            profile_key="a800:l%d:r%.2f" % (
                boundary,
                repair_ratio_upper,
            ),
        )

    def execute_selective_reuse(
        self, source, boundary, repair_ratio_upper
    ):
        self.calls.append("reuse:%s:%d" % (source.source_id, boundary))
        return CacheBlendExecutionObservation(
            execution_mode=ExecutionMode.REUSE,
            selected_source_id=source.source_id,
            actual_reuse_boundary=boundary,
            started_ms=21.0,
            first_token_ready_ms=55.0,
            total_host_ms=34.0,
            total_gpu_ms=30.0,
            output_token_ids=(1, 2),
            output_hash="reuse",
            source_digest_before="same",
            source_digest_after="same",
        )

    def execute_full_recompute(self, retained_source_id):
        self.calls.append("full:%s" % retained_source_id)
        wasted = 512 if retained_source_id is not None else 0
        return CacheBlendExecutionObservation(
            execution_mode=ExecutionMode.FULL_RECOMPUTE,
            selected_source_id=retained_source_id,
            actual_reuse_boundary=None,
            started_ms=21.0 if retained_source_id is not None else 0.0,
            first_token_ready_ms=100.0,
            total_host_ms=100.0,
            total_gpu_ms=95.0,
            output_token_ids=(1, 2),
            output_hash="full",
            wasted_loaded_bytes=wasted,
        )


class CacheBlendClosedLoopRuntimeTests(unittest.TestCase):
    def runtime(self, engine):
        source = canonical_source()
        return CacheBlendClosedLoopRuntime(
            engine,
            {source.source_id: source},
            total_layers=32,
        )

    def test_real_call_order_closes_selector_to_cacheblend_reuse(self):
        engine = FakeClosedLoopEngine()
        runtime = self.runtime(engine)
        result = TwoStageReuseController(
            cost_accounting_policy=CostAccountingPolicy.UNIFIED_COMPONENTS_V1
        ).execute(selected(), runtime)
        self.assertTrue(result.execution.reuse_accepted)
        self.assertEqual(result.execution.actual_reuse_boundary, 6)
        self.assertEqual(
            engine.calls,
            [
                "begin_load:s1",
                "schedule:s1",
                "profile:s1:6",
                "reuse:s1:6",
            ],
        )
        self.assertEqual(runtime.state, CacheBlendRuntimeState.EXECUTED)
        audit = result.to_audit_record()
        self.assertEqual(audit["source_load_start_ms"], 5.0)
        self.assertEqual(audit["source_ready_ms"], 20.0)
        self.assertEqual(audit["a_resume_ms"], 21.0)
        self.assertEqual(audit["post_ready_blocking_ms"], 1.0)
        self.assertEqual(audit["source_load_bytes"], 512)
        self.assertEqual(
            audit["refined_future_timing_source"],
            "a800_boundary_profile_upper",
        )
        self.assertEqual(audit["runtime_actual_reuse_boundary"], 6)
        self.assertEqual(audit["runtime_realized_ttft_ms"], 55.0)

    def test_refined_rejection_keeps_source_and_accounts_wasted_load(self):
        engine = FakeClosedLoopEngine(repair_ms=80)
        result = TwoStageReuseController(
            cost_accounting_policy=CostAccountingPolicy.UNIFIED_COMPONENTS_V1
        ).execute(selected(), self.runtime(engine))
        self.assertFalse(result.execution.reuse_accepted)
        self.assertEqual(result.execution.selected_source_id, "s1")
        self.assertIsNone(result.execution.actual_reuse_boundary)
        self.assertEqual(engine.calls[-1], "full:s1")
        audit = result.to_audit_record()
        self.assertEqual(audit["evaluated_reuse_boundary"], 6)
        self.assertEqual(audit["wasted_loaded_bytes"], 512)
        self.assertEqual(audit["runtime_wasted_loaded_bytes"], 512)

    def test_abstention_does_not_start_source_load(self):
        engine = FakeClosedLoopEngine()
        runtime = self.runtime(engine)
        abstention = SourceDecision(
            selected_source_id=None,
            probe_layer=8,
            reuse_layer=None,
            safe_repair_ratio_upper=None,
            prefetch_m=0,
            selection_reason=SelectionReason.MAX_PROBE_UNCERTAIN,
        )
        result = TwoStageReuseController().execute(abstention, runtime)
        self.assertFalse(result.execution.reuse_accepted)
        self.assertEqual(engine.calls, ["full:None"])

    def test_scheduler_cannot_switch_selected_source(self):
        engine = FakeClosedLoopEngine(scheduled_source_id="s2")
        with self.assertRaisesRegex(RuntimeError, "another Source"):
            TwoStageReuseController(
                cost_accounting_policy=(
                    CostAccountingPolicy.UNIFIED_COMPONENTS_V1
                )
            ).execute(selected(), self.runtime(engine))
        self.assertNotIn("profile:s2:6", engine.calls)

    def test_actual_boundary_may_move_after_early_probe(self):
        engine = FakeClosedLoopEngine(boundary=9)
        result = TwoStageReuseController(
            cost_accounting_policy=CostAccountingPolicy.UNIFIED_COMPONENTS_V1
        ).execute(selected(), self.runtime(engine))
        self.assertEqual(result.selection.probe_layer, 2)
        self.assertEqual(result.execution.actual_reuse_boundary, 9)
        self.assertIn("profile:s1:9", engine.calls)

    def test_non_resumable_case_runner_is_rejected_before_loading(self):
        capabilities = CacheBlendRuntimeCapabilities(
            backend_name="case-level-generate-wrapper",
            async_source_loading=False,
            layer_resumable_prefill=False,
            scheduler_feedback=False,
            boundary_conditioned_profiles=False,
            canonical_sources_read_only=True,
            cuda_event_timing=True,
        )
        engine = FakeClosedLoopEngine(capabilities=capabilities)
        with self.assertRaisesRegex(
            RuntimeError, "layer_resumable_prefill"
        ):
            self.runtime(engine)
        self.assertEqual(engine.calls, [])


if __name__ == "__main__":
    unittest.main()
