import unittest

from probekv.v8_planner import (
    PredictedJointPlanner,
    PredictedSegmentOption,
    RefinedJointPlanner,
    RefinedSegmentMeasurement,
    SegmentPlanState,
    UnifiedCostComponents,
)


def future(total):
    return UnifiedCostComponents(repair_ms=total)


def option(segment, reuse, dense, *, hbm=100, prefetched=False):
    return PredictedSegmentOption(
        segment, "source-" + segment, "artifact-" + segment,
        "gpu-" + segment, 1, 1, 2, hbm, future(reuse), dense,
        already_prefetched=prefetched,
    )


class V8PlannerTests(unittest.TestCase):
    def test_predicted_planner_supports_partial_reuse(self):
        plan = PredictedJointPlanner(gamma=0.8, hbm_capacity_bytes=150).plan(
            "request", [option("a", 10, 50), option("b", 20, 30)],
            shared_sunk=UnifiedCostComponents(probe_ms=1),
            dense_reference_ms=100,
        )
        states = {item.segment_id: item.state for item in plan.decisions}
        self.assertEqual(states["a"], SegmentPlanState.PROVISIONAL_REUSE)
        self.assertEqual(states["b"], SegmentPlanState.PREDICTED_DENSE)

    def test_prefetched_segment_is_monotonic_in_predicted_planner(self):
        plan = PredictedJointPlanner(gamma=0.8, hbm_capacity_bytes=100).plan(
            "request", [option("a", 90, 10, prefetched=True)],
            shared_sunk=UnifiedCostComponents(), dense_reference_ms=100,
        )
        self.assertEqual(plan.decisions[0].state, SegmentPlanState.PROVISIONAL_REUSE)
        self.assertFalse(plan.gamma_bound_met)

    def test_refined_planner_may_downgrade_but_never_promote(self):
        predicted = PredictedJointPlanner(gamma=0.8, hbm_capacity_bytes=100).plan(
            "request", [option("a", 10, 50), option("b", 60, 50)],
            shared_sunk=UnifiedCostComponents(), dense_reference_ms=100,
        )
        self.assertEqual(
            {item.segment_id: item.state for item in predicted.decisions}["b"],
            SegmentPlanState.PREDICTED_DENSE,
        )
        refined = RefinedJointPlanner(gamma=0.8).plan(
            predicted,
            {
                "a": RefinedSegmentMeasurement(
                    "a", "source-a", 3, future(70), 50, True,
                    transferred_bytes=100,
                )
            },
            actual_shared_sunk_ms=1,
        )
        states = {item.segment_id: item.state for item in refined.decisions}
        self.assertEqual(states["a"], SegmentPlanState.REFINED_DENSE)
        self.assertEqual(states["b"], SegmentPlanState.PREDICTED_DENSE)
        self.assertEqual(refined.final_reuse_segment_ids, ())
        self.assertEqual(refined.wasted_bytes, 100)

    def test_refined_source_cannot_change(self):
        predicted = PredictedJointPlanner(gamma=0.8, hbm_capacity_bytes=100).plan(
            "request", [option("a", 10, 50)],
            shared_sunk=UnifiedCostComponents(), dense_reference_ms=100,
        )
        with self.assertRaises(ValueError):
            RefinedJointPlanner().plan(
                predicted,
                {"a": RefinedSegmentMeasurement("a", "source-other", 2, future(5), 50, True)},
                actual_shared_sunk_ms=1,
            )


if __name__ == "__main__":
    unittest.main()
