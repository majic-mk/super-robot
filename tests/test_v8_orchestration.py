import unittest

from probekv.contracts import KVLocation
from probekv.v8_contracts import (
    CandidateCounts,
    ResidualCandidate,
    SelectorPolicyProfile,
)
from probekv.v8_leases import V8LeaseManager, V8ReplicaResource
from probekv.v8_orchestration import V8RequestOrchestrator
from probekv.v8_planner import (
    PredictedJointPlanner,
    PredictedSegmentOption,
    RefinedJointPlanner,
    RefinedSegmentMeasurement,
    SegmentPlanState,
    UnifiedCostComponents,
)
from probekv.v8_selector import TrainingFreeResidualKSelector


def setup_source(manager, suffix):
    source = "source-" + suffix
    artifact = "artifact-" + suffix
    replica = "gpu-" + suffix
    manager.register_source(source, artifact, "model")
    manager.register_replica(V8ReplicaResource(
        "cpu-" + suffix, source, artifact, KVLocation.PINNED_CPU,
        1, 1, 100, True,
    ))
    manager.register_replica(V8ReplicaResource(
        replica, source, artifact, KVLocation.GPU,
        1, 1, 100, False,
    ))
    return source, artifact, replica


class V8ClosedLoopTests(unittest.TestCase):
    def setUp(self):
        profile = SelectorPolicyProfile(
            "p", "m", "causal_commit_wait", (1, 2), 2,
            0.3, 0.6, 0.05,
        )
        self.selector = TrainingFreeResidualKSelector(profile)
        self.manager = V8LeaseManager()
        self.source, self.artifact, self.replica = setup_source(self.manager, "a")
        self.orchestrator = V8RequestOrchestrator(
            self.manager,
            PredictedJointPlanner(gamma=0.8, hbm_capacity_bytes=1000),
            RefinedJointPlanner(gamma=0.8),
        )

    def decision(self):
        return self.selector.evaluate_checkpoint(
            completed_depth=1,
            counts=CandidateCounts(1, 1, 1, 1, 1),
            candidates=[ResidualCandidate(self.source, 0.1, 10, 0)],
            shared_sunk_ms=1,
            dense_reference_ms=100,
        )

    def test_selector_to_leases_predicted_and_refined_rejection_is_one_chain(self):
        option = PredictedSegmentOption(
            "c", self.source, self.artifact, self.replica, 1, 1, 2, 100,
            UnifiedCostComponents(repair_ms=10), 60,
        )
        state = self.orchestrator.freeze_and_predict(
            request_id="r", request_generation=1, decisions={"c": self.decision()},
            options=[option], shared_sunk=UnifiedCostComponents(probe_ms=1),
            dense_reference_ms=100, predicted_remaining_s=1,
        )
        self.assertEqual(state.predicted_plan.decisions[0].state, SegmentPlanState.PROVISIONAL_REUSE)
        refined = self.orchestrator.refine(
            state,
            {"c": RefinedSegmentMeasurement(
                "c", self.source, 4, UnifiedCostComponents(repair_ms=90),
                60, True, transferred_bytes=100,
            )},
            actual_shared_sunk_ms=2,
        )
        self.assertEqual(refined.decisions[0].state, SegmentPlanState.REFINED_DENSE)
        self.assertEqual(refined.decisions[0].source_variant_id, self.source)
        self.orchestrator.release_request(state, reason="refined_dense")

    def test_selector_abstention_cannot_enter_planning_or_load(self):
        with self.assertRaises(ValueError):
            self.orchestrator.freeze_and_predict(
                request_id="r", request_generation=1, decisions={},
                options=[PredictedSegmentOption(
                    "c", self.source, self.artifact, self.replica, 1, 1, 2, 100,
                    UnifiedCostComponents(repair_ms=10), 60,
                )],
                shared_sunk=UnifiedCostComponents(), dense_reference_ms=100,
                predicted_remaining_s=1,
            )
        self.assertEqual(self.manager.sources[self.source].logical_lease_refcount, 0)

    def test_refinement_cannot_change_selected_source(self):
        option = PredictedSegmentOption(
            "c", self.source, self.artifact, self.replica, 1, 1, 2, 100,
            UnifiedCostComponents(repair_ms=10), 60,
        )
        state = self.orchestrator.freeze_and_predict(
            request_id="r", request_generation=1, decisions={"c": self.decision()},
            options=[option], shared_sunk=UnifiedCostComponents(),
            dense_reference_ms=100, predicted_remaining_s=1,
        )
        with self.assertRaises(RuntimeError):
            self.orchestrator.refine(
                state,
                {"c": RefinedSegmentMeasurement(
                    "c", "source-other", 2, UnifiedCostComponents(repair_ms=1),
                    60, True,
                )},
                actual_shared_sunk_ms=1,
            )
        self.orchestrator.release_request(state, reason="test")


if __name__ == "__main__":
    unittest.main()
