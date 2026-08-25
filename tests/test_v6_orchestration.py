import unittest
from dataclasses import replace

from probekv.candidate_budget import (
    VariantComparisonCandidate,
    allocate_variant_comparisons,
)
from probekv.contracts import (
    CandidateBounds,
    ReuseAdmissionState,
    SelectionReason,
    SourceDecision,
    SourceSelectionState,
)
from probekv.multisegment_orchestration import (
    MultiSegmentOnlinePipeline,
    MultiSegmentReuseController,
    StaggeredMultiSegmentReuseController,
)
from probekv.multisegment_selector import MultiSegmentProbeSelector
from probekv.selector import DynamicProbeSelector, ProbePolicy, SelectorPolicy
from probekv.v6_contracts import (
    RequestExecutionMode,
    RequestRefinedCost,
    RequestSchedulingFeedback,
    RequestSelectionPlan,
    SelectionExecutionPolicy,
    SegmentExecutionPath,
    SegmentSchedulingFeedback,
    SegmentSelectionDecision,
    VariantComparisonAudit,
)


def selected_segment(segment_id, source_id, probe_layer):
    decision = SourceDecision(
        source_id,
        probe_layer,
        None,
        0.20,
        1,
        SelectionReason.EARLY_CONFIDENT,
        predicted_cost_upper_ms=60.0,
    )
    return SegmentSelectionDecision(
        segment_id,
        decision,
        VariantComparisonAudit(
            segment_id, 1, 1, (source_id,), (), 0.1, 5.0
        ),
    )


def abstained_segment(segment_id):
    return SegmentSelectionDecision(
        segment_id,
        SourceDecision(
            None,
            8,
            None,
            None,
            0,
            SelectionReason.MAX_PROBE_UNCERTAIN,
        ),
        VariantComparisonAudit(segment_id, 1, 1, (segment_id + "-s",), (), 0.1, 5.0),
    )


def selection_plan(selected=3, total=5):
    rows = [
        selected_segment("c%d" % index, "c%d-s" % index, index + 1)
        for index in range(selected)
    ]
    rows.extend(abstained_segment("c%d" % index) for index in range(selected, total))
    return RequestSelectionPlan("request", tuple(rows), 1.0, 0.5, 0.5, 100.0)


def refined(active, boundary, total, *, quality=True, marginal=None):
    active = tuple(active)
    components = (1.0, 0.5, 0.5, 5.0, 2.0, 1.0, 1.0, 10.0)
    remaining = total - sum(components)
    return RequestRefinedCost(
        request_id="request",
        boundary=boundary,
        active_segment_ids=active,
        selected_source_ids={segment_id: segment_id + "-s" for segment_id in active},
        marginal_saved_ms=(
            dict(marginal)
            if marginal is not None
            else {segment_id: 5.0 for segment_id in active}
        ),
        repair_ratio_upper_by_segment={
            segment_id: 0.15 for segment_id in active
        },
        reuse_total_ms=total,
        full_reference_ms=100.0,
        probe_ms=components[0],
        metadata_ms=components[1],
        compare_ms=components[2],
        visible_load_ms=components[3],
        post_ready_blocking_ms=components[4],
        load_interference_ms=components[5],
        repair_selection_ms=components[6],
        repair_ms=components[7],
        remaining_ms=remaining,
        joint_quality_covered=quality,
    )


class PartialRuntime:
    def __init__(self):
        self.loaded = False
        self.refined_calls = []
        self.executed = None

    def load_and_schedule(self, selection):
        self.loaded = True
        return RequestSchedulingFeedback(
            "request",
            (
                SegmentSchedulingFeedback("c0", "c0-s", 0.0, 4.0, 5, 32, 100),
                SegmentSchedulingFeedback("c1", "c1-s", 0.0, 4.0, 5, 32, 200),
                SegmentSchedulingFeedback("c2", "c2-s", 0.0, 9.0, 10, 32, 300),
            ),
            10.0,
            5.0,
            1.0,
            0.5,
            4.0,
            3.0,
            (5, 10),
        )

    def measure_refined_cost(self, selection, scheduling, boundary, active):
        self.refined_calls.append((boundary, active))
        if boundary == 5 and active == ("c0", "c1"):
            return refined(active, boundary, 65.0, marginal={"c0": 8.0, "c1": -1.0})
        if boundary == 5 and active == ("c0",):
            return refined(active, boundary, 60.0)
        return refined(active, boundary, 72.0)

    def execute_reuse(self, selection, execution):
        self.executed = "reuse"
        return "reuse-output"

    def execute_dense(self, selection, execution):
        self.executed = "dense"
        return "dense-output"


class RejectRuntime(PartialRuntime):
    def load_and_schedule(self, selection):
        self.loaded = True
        item = selection.selected[0]
        source_id = item.source_decision.selected_source_id
        return RequestSchedulingFeedback(
            "request",
            (SegmentSchedulingFeedback(item.segment_id, source_id, 0, 2, 3, 32, 123),),
            3,
            2,
            0,
            0,
            1,
            0,
            (3,),
        )

    def measure_refined_cost(self, selection, scheduling, boundary, active):
        return refined(active, boundary, 90.0)


class V6ClosedLoopTests(unittest.TestCase):
    def test_a_and_c_close_selector_scheduler_refinement_with_staggered_boundaries(self):
        candidates = (
            VariantComparisonCandidate("c0", "c0-s", 0.0, 50.0, 0.01),
            VariantComparisonCandidate("c1", "c1-a", 0.0, 50.0, 0.01),
            VariantComparisonCandidate("c1", "c1-b", 0.1, 49.0, 0.01),
        )
        allocation = allocate_variant_comparisons(
            candidates,
            full_reference_ms=100.0,
            probe_ms=0.1,
            metadata_ms=0.1,
            segment_ids=("c0", "c1"),
        )
        bounds = {
            "c0": {1: (CandidateBounds("c0-s", 0.1, 10.0, 20.0),)},
            "c1": {
                layer: (
                    CandidateBounds("c1-a", 0.1, 10.0, 30.0),
                    CandidateBounds("c1-b", 0.1, 11.0, 31.0),
                )
                for layer in (1, 2, 3, 4)
            },
        }

        class StaggeredRuntime:
            def __init__(self, policy):
                self.probe_state_origin = (
                    "policy_conditioned_closed_loop"
                    if policy is SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP
                    else "dense_clean"
                )
                self.eligible = []
                self.action = None

            def on_source_locked(self, segment_id, decision):
                pass

            def on_reuse_eligible(self, segment_id, earliest_layer):
                self.eligible.append((segment_id, earliest_layer))

            def load_and_schedule(self, selection):
                return RequestSchedulingFeedback(
                    "request",
                    (
                        SegmentSchedulingFeedback("c0", "c0-s", 0, 1, 2, 32, 10),
                        SegmentSchedulingFeedback("c1", "c1-a", 0, 1, 2, 32, 10),
                    ),
                    5,
                    5,
                    0,
                    0,
                    4,
                    0,
                    tuple(range(2, 33)),
                )

            def measure_staggered_refined_cost(
                self, selection, scheduling, boundary_by_segment, active
            ):
                value = refined(active, min(boundary_by_segment.values()), 60.0)
                return replace(
                    value,
                    boundary_by_segment=dict(boundary_by_segment),
                    selected_source_ids={
                        item.segment_id: str(
                            item.source_decision.selected_source_id
                        )
                        for item in selection.selected
                        if item.segment_id in active
                    },
                )

            def execute_reuse(self, selection, execution):
                self.action = "reuse"
                return "reuse"

            def execute_dense(self, selection, execution):
                self.action = "dense"
                return "dense"

        observed = {}
        for policy in (
            SelectionExecutionPolicy.CAUSAL_COMMIT_WAIT,
            SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP,
        ):
            selector = MultiSegmentProbeSelector(
                DynamicProbeSelector(
                    ProbePolicy(
                        (1, 2, 3, 4),
                        4,
                        selector_policy=SelectorPolicy.FINAL_ECONOMIC_MIN_COST,
                        preliminary_economic_filter=True,
                    )
                ),
                policy,
            )
            runtime = StaggeredRuntime(policy)
            result = MultiSegmentOnlinePipeline(
                selector, StaggeredMultiSegmentReuseController()
            ).execute(
                "request",
                allocation,
                (
                    (lambda segment_id, layer: bounds.get(segment_id, {}).get(layer, ()))
                    if policy is SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP
                    else bounds
                ),
                runtime,
            )
            observed[policy] = dict(
                result.execution.actual_reuse_boundary_by_segment
            )
            self.assertEqual(runtime.action, "reuse")
            self.assertEqual(result.execution.mode, RequestExecutionMode.ALL_REUSE)

        self.assertEqual(
            observed[SelectionExecutionPolicy.CAUSAL_COMMIT_WAIT],
            {"c0": 5, "c1": 5},
        )
        self.assertEqual(
            observed[
                SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP
            ],
            {"c0": 2, "c1": 5},
        )

    def test_selector_scheduler_refined_cost_and_admission_form_one_chain(self):
        candidates = (
            VariantComparisonCandidate("c0", "c0-s", 0.1, 40.0, 0.01),
            VariantComparisonCandidate("c1", "c1-s", 0.1, 40.0, 0.01),
        )
        allocation = allocate_variant_comparisons(
            candidates,
            full_reference_ms=100.0,
            probe_ms=1.0,
            metadata_ms=0.5,
            segment_ids=("c0", "c1"),
        )
        bounds = {
            "c0": {
                1: (CandidateBounds("c0-s", 0.2, 40.0, 50.0),),
                2: (CandidateBounds("c0-s", 0.2, 40.0, 50.0),),
            },
            "c1": {
                1: (),
                2: (CandidateBounds("c1-s", 0.2, 45.0, 55.0),),
            },
        }
        selector = MultiSegmentProbeSelector(
            DynamicProbeSelector(
                ProbePolicy(
                    (1, 2),
                    2,
                    preliminary_economic_filter=True,
                )
            )
        )
        class EndToEndRuntime:
            def __init__(self):
                self.events = []

            def on_source_locked(self, segment_id, decision):
                self.events.append(
                    ("lock", decision.probe_layer, segment_id, decision.selected_source_id)
                )

            def load_and_schedule(self, plan):
                self.events.append(("schedule",))
                return RequestSchedulingFeedback(
                    "request",
                    (
                        SegmentSchedulingFeedback("c0", "c0-s", 0, 2, 3, 32, 10),
                        SegmentSchedulingFeedback("c1", "c1-s", 0, 3, 4, 32, 10),
                    ),
                    4,
                    3,
                    0,
                    0,
                    2,
                    0,
                    (3, 4),
                )

            def measure_refined_cost(self, plan, scheduling, boundary, active):
                self.events.append(("refined",))
                return refined(active, boundary, 70.0 if boundary == 3 else 60.0)

            def execute_reuse(self, plan, execution):
                self.events.append(("reuse",))
                return "reuse"

            def execute_dense(self, plan, execution):
                self.events.append(("dense",))
                return "dense"

        runtime = EndToEndRuntime()
        result = MultiSegmentOnlinePipeline(
            selector, MultiSegmentReuseController()
        ).execute("request", allocation, bounds, runtime)
        self.assertEqual(
            [
                item.source_decision.probe_layer
                for item in result.selection.segment_decisions
            ],
            [1, 2],
        )
        self.assertEqual(
            runtime.events[:2],
            [("lock", 1, "c0", "c0-s"), ("lock", 2, "c1", "c1-s")],
        )
        self.assertEqual(result.execution.mode, RequestExecutionMode.ALL_REUSE)
        self.assertEqual(result.execution.actual_reuse_boundary, 4)
        self.assertEqual(runtime.events[-1], ("reuse",))
        self.assertLess(runtime.events.index(("schedule",)), runtime.events.index(("refined",)))

    def test_partial_reuse_uses_common_boundary_and_marginal_pruning(self):
        runtime = PartialRuntime()
        result = MultiSegmentReuseController(0.8).execute(
            selection_plan(), runtime
        )
        self.assertEqual(result.execution.mode, RequestExecutionMode.PARTIAL_REUSE)
        self.assertEqual(result.execution.actual_reuse_boundary, 5)
        by_id = {item.segment_id: item for item in result.execution.segment_decisions}
        self.assertEqual(by_id["c0"].path, SegmentExecutionPath.REUSE)
        self.assertEqual(by_id["c0"].repair_ratio_upper, 0.15)
        self.assertEqual(by_id["c1"].path, SegmentExecutionPath.DENSE)
        self.assertEqual(by_id["c1"].selected_source_id, "c1-s")
        self.assertEqual(by_id["c1"].admission_state, ReuseAdmissionState.REJECTED)
        self.assertEqual(by_id["c3"].admission_state, ReuseAdmissionState.NOT_EVALUATED)
        self.assertEqual(result.execution.transferred_bytes, 600)
        self.assertEqual(result.execution.wasted_loaded_bytes, 500)
        self.assertEqual(runtime.executed, "reuse")
        self.assertIn((5, ("c0",)), runtime.refined_calls)

    def test_selector_abstention_never_loads_a_default_source(self):
        runtime = PartialRuntime()
        result = MultiSegmentReuseController().execute(
            selection_plan(selected=0, total=2), runtime
        )
        self.assertFalse(runtime.loaded)
        self.assertEqual(result.execution.mode, RequestExecutionMode.FULL_RECOMPUTE)
        self.assertEqual(runtime.executed, "dense")
        for item in result.execution.segment_decisions:
            self.assertIsNone(item.selected_source_id)
            self.assertEqual(item.selection_state, SourceSelectionState.NOT_SELECTED)
            self.assertEqual(item.admission_state, ReuseAdmissionState.NOT_EVALUATED)

    def test_refined_rejection_retains_locked_source_and_charges_waste(self):
        runtime = RejectRuntime()
        result = MultiSegmentReuseController(0.8).execute(
            selection_plan(selected=1, total=1), runtime
        )
        decision = result.execution.segment_decisions[0]
        self.assertEqual(result.execution.mode, RequestExecutionMode.FULL_RECOMPUTE)
        self.assertEqual(decision.selected_source_id, "c0-s")
        self.assertEqual(decision.selection_state, SourceSelectionState.SELECTED)
        self.assertEqual(decision.admission_state, ReuseAdmissionState.REJECTED)
        self.assertEqual(result.execution.wasted_loaded_bytes, 123)
        self.assertEqual(runtime.executed, "dense")

    def test_unready_locked_source_only_forces_its_segment_dense(self):
        class OneUnreadyRuntime(PartialRuntime):
            def load_and_schedule(self, selection):
                self.loaded = True
                return RequestSchedulingFeedback(
                    "request",
                    (
                        SegmentSchedulingFeedback("c0", "c0-s", 0, 3, 5, 32, 100),
                        SegmentSchedulingFeedback(
                            "c1", "c1-s", 0, 3, 0, 0, 50, 50, False
                        ),
                    ),
                    5,
                    3,
                    0,
                    0,
                    2,
                    0,
                    (5,),
                )

            def measure_refined_cost(self, selection, scheduling, boundary, active):
                self.refined_calls.append((boundary, active))
                return refined(active, boundary, 60.0)

        runtime = OneUnreadyRuntime()
        result = MultiSegmentReuseController().execute(
            selection_plan(selected=2, total=2), runtime
        )
        by_id = {item.segment_id: item for item in result.execution.segment_decisions}
        self.assertEqual(result.execution.mode, RequestExecutionMode.PARTIAL_REUSE)
        self.assertEqual(by_id["c0"].path, SegmentExecutionPath.REUSE)
        self.assertEqual(by_id["c1"].path, SegmentExecutionPath.DENSE)
        self.assertEqual(result.execution.wasted_loaded_bytes, 50)

    def test_scheduler_cannot_replace_locked_source(self):
        class WrongRuntime(RejectRuntime):
            def load_and_schedule(self, selection):
                feedback = super().load_and_schedule(selection)
                wrong = SegmentSchedulingFeedback("c0", "latest", 0, 2, 3, 32, 1)
                return RequestSchedulingFeedback(
                    feedback.request_id,
                    (wrong,),
                    feedback.scheduled_step_finish_ms,
                    feedback.a_resume_ms,
                    feedback.post_ready_blocking_ms,
                    feedback.load_interference_ms,
                    feedback.useful_a_dense_ms,
                    feedback.useful_other_request_work_ms,
                    feedback.candidate_boundaries,
                )

        with self.assertRaisesRegex(ValueError, "locked Sources"):
            MultiSegmentReuseController().execute(
                selection_plan(selected=1, total=1), WrongRuntime()
            )


if __name__ == "__main__":
    unittest.main()
