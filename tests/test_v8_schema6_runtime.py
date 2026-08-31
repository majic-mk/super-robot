import unittest

from probekv.contracts import KVLocation
from probekv.v8_leases import (
    LeasePurpose,
    V8LeaseManager,
    V8ReplicaResource,
)
from probekv.v8_schema6_contracts import (
    CommitAxisState,
    Gate2AxisState,
    Gate3SubsetDecision,
    PlannerSnapshot,
    PreparationAxisState,
    SelectionAxisState,
    evaluate_speculative_waste_admission,
)
from probekv.v8_schema6_hbm import GIB, UnifiedHBMReservationManager
from probekv.v8_schema6_planner import (
    DeterministicJointTimelineEstimator,
    FrozenSegmentCandidate,
    Gate2Disposition,
    PredictedJointPlannerV6,
    RefinedJointPlannerV6,
)
from probekv.v8_schema6_runtime import Schema6RequestController
from probekv.v8_schema6_workspace import acquire_elastic_selection_workspace
from probekv.v8_schema6_transfer import Schema6FullKVTransferAuthorizer
from probekv.v7_contracts import CanonicalKVArtifact, PhysicalReplica, ReplicaLocator


def snapshot(epoch=1):
    return PlannerSnapshot(1, 1, "scheduler-1", epoch, "profile-sha")


def manager_with_source():
    manager = V8LeaseManager()
    manager.register_source("source-c1", "artifact-c1", "model")
    manager.register_replica(
        V8ReplicaResource(
            "cpu-c1", "source-c1", "artifact-c1", KVLocation.PINNED_CPU,
            1, 1, 1024, True,
        )
    )
    return manager


class Schema6PlannerTests(unittest.TestCase):
    def test_gate2_complete_inventory_uses_dense_fallback(self):
        estimator = DeterministicJointTimelineEstimator(
            base_future_ms=0,
            dense_cost_ms_by_segment={"c1": 40, "c2": 40},
            reuse_cost_ms_by_segment={"c1": 10, "c2": 90},
        )
        planner = PredictedJointPlannerV6(estimator, gamma=0.8)
        snap = snapshot()
        decision = planner.plan_incremental(
            inventory_segment_ids=("c1", "c2"),
            frozen_candidates=(FrozenSegmentCandidate("c1", "s1", 5, 10),),
            existing_provisional_segment_ids=(),
            existing_deferred_segment_ids=(),
            predicted_dense_segment_ids=(),
            committed_segment_ids=(),
            actual_sunk_ms=1,
            dense_reference_total_ms=100,
            selection_closed=False,
            snapshot=snap,
            current_snapshot=snap,
            union_mask_digest="mask",
        )
        self.assertEqual(
            decision.disposition_by_segment["c1"],
            Gate2Disposition.PROVISIONAL_REUSE,
        )
        # c2 is unresolved and therefore still costs its 40 ms dense fallback.
        self.assertEqual(decision.predicted_request_total_ms, 51)

    def test_gate2_stale_snapshot_is_rejected(self):
        planner = PredictedJointPlannerV6(
            DeterministicJointTimelineEstimator(
                base_future_ms=0,
                dense_cost_ms_by_segment={"c1": 40},
                reuse_cost_ms_by_segment={"c1": 10},
            )
        )
        with self.assertRaises(RuntimeError):
            planner.plan_incremental(
                inventory_segment_ids=("c1",),
                frozen_candidates=(FrozenSegmentCandidate("c1", "s1", 2, 1),),
                existing_provisional_segment_ids=(), existing_deferred_segment_ids=(),
                predicted_dense_segment_ids=(), committed_segment_ids=(),
                actual_sunk_ms=0, dense_reference_total_ms=100,
                selection_closed=False, snapshot=snapshot(1), current_snapshot=snapshot(2),
                union_mask_digest="mask",
            )

    def test_gate3_returns_partial_subset_by_joint_marginal(self):
        estimator = DeterministicJointTimelineEstimator(
            base_future_ms=10,
            dense_cost_ms_by_segment={"c1": 50, "c2": 50},
            reuse_cost_ms_by_segment={"c1": 10, "c2": 70},
        )
        snap = snapshot()
        decision = RefinedJointPlannerV6(estimator, gamma=0.8).plan_subset(
            inventory_segment_ids=("c1", "c2"),
            eligible_ready_segment_ids=("c1", "c2"),
            committed_segment_ids=(),
            actual_boundary_by_segment={"c1": 3, "c2": 5},
            actual_sunk_ms=0,
            dense_reference_total_ms=100,
            snapshot=snap,
            current_snapshot=snap,
            union_mask_digest="mask",
        )
        self.assertEqual(decision.accepted_ready_segment_ids, ("c1",))
        self.assertEqual(decision.rejected_ready_segment_ids, ("c2",))
        self.assertLessEqual(decision.request_total_ms, 80)


class Schema6ResourceAndStateTests(unittest.TestCase):
    def test_full_kv_transfer_authorizer_enforces_freeze_lease_and_hbm(self):
        hbm = UnifiedHBMReservationManager(
            allocator_capacity_bytes=5 * GIB, safety_bytes=4 * GIB
        )
        authorizer = Schema6FullKVTransferAuthorizer(
            hbm_manager_provider=lambda: hbm
        )
        artifact = CanonicalKVArtifact(
            "artifact", "source", 1, "parent", "logical", "bytes",
            32, 8, 128,
        )
        replica = PhysicalReplica(
            "cpu", "artifact", 1, KVLocation.PINNED_CPU,
            "logical", "bytes", 4096,
            ReplicaLocator("cpu-buffer", "contiguous-bf16"),
        )
        authorization = authorizer.authorize(
            segment_id="c1", source_variant_id="source", artifact=artifact,
            replica=replica, predicted_remaining_s=1,
        )
        authorization.assert_valid_for(
            source_variant_id="source", artifact_id="artifact", replica_id="cpu"
        )
        with self.assertRaisesRegex(RuntimeError, "binding differs"):
            authorization.assert_valid_for(
                source_variant_id="other", artifact_id="artifact", replica_id="cpu"
            )
        row = authorization.controller.records[authorization.segment_id]
        self.assertTrue(row.logical_lease_id)
        self.assertTrue(row.physical_lease_id)
        self.assertTrue(row.hbm_reservation_id)
        authorization.mark_ready(actual_reuse_boundary=5)
        authorization.release()
        self.assertTrue(authorizer.audit()["passed"])

    def test_deferred_ready_then_atomic_promotion_and_commit(self):
        leases = manager_with_source()
        hbm = UnifiedHBMReservationManager(
            allocator_capacity_bytes=8 * GIB, safety_bytes=4 * GIB
        )
        controller = Schema6RequestController(
            request_id="r", request_generation=1, ordered_segment_ids=("c1",),
            policy="immediate_staggered_closed_loop",
            lease_manager=leases, hbm_manager=hbm,
        )
        controller.decision_ready("c1", "source-c1", 4)
        controller.gate1("c1", passed=True, at_lmax=False, predicted_remaining_s=1)
        snap = snapshot(hbm.epoch)
        controller.apply_gate2(
            {"c1": Gate2AxisState.DEFERRED.value}, snapshot=snap, current_snapshot=snap
        )
        controller.begin_winner_prefetch(
            "c1", artifact_id="artifact-c1", replica_id="cpu-c1",
            replica_generation=1, placement_epoch=1, target_hbm_bytes=1024,
            predicted_remaining_s=1, speculative_resource_admitted=True,
        )
        controller.mark_winner_ready("c1", actual_reuse_boundary=5)
        row = controller.records["c1"]
        self.assertEqual(row.preparation_state, PreparationAxisState.READY)
        self.assertEqual(row.gate2_state, Gate2AxisState.DEFERRED)
        lease_id = row.physical_lease_id
        snap2 = snapshot(hbm.epoch)
        controller.apply_gate2(
            {"c1": Gate2AxisState.PROVISIONAL_REUSE.value},
            snapshot=snap2, current_snapshot=snap2,
        )
        self.assertEqual(leases.leases[lease_id].purpose, LeasePurpose.EXECUTION)
        snap3 = snapshot(hbm.epoch)
        decision = Gate3SubsetDecision(
            ("c1",), (), (), 50, 100, snap3, {"c1": "accepted"}
        )
        controller.apply_gate3_subset(decision, current_snapshot=snap3)
        self.assertEqual(row.commit_state, CommitAxisState.REUSE_COMMIT)
        controller.complete_reuse_execution("c1")
        self.assertEqual(leases.sources["source-c1"].logical_lease_refcount, 0)
        self.assertTrue(leases.leases[lease_id].state.value == "released")

    def test_full_kv_prefetch_requires_frozen_source(self):
        controller = Schema6RequestController(
            request_id="r", request_generation=1, ordered_segment_ids=("c1",),
            policy="causal_commit_wait", lease_manager=manager_with_source(),
            hbm_manager=UnifiedHBMReservationManager(
                allocator_capacity_bytes=8 * GIB, safety_bytes=4 * GIB
            ),
        )
        with self.assertRaises(RuntimeError):
            controller.begin_winner_prefetch(
                "c1", artifact_id="artifact-c1", replica_id="cpu-c1",
                replica_generation=1, placement_epoch=1, target_hbm_bytes=1024,
                predicted_remaining_s=1,
            )

    def test_speculative_budget_uses_one_times_dense(self):
        accepted = evaluate_speculative_waste_admission(
            actual_sunk_ms=10, dense_fallback_joint_future_ms=70,
            dense_reference_total_ms=100, predicted_visible_copy_ms=10,
            predicted_copy_interference_ms=5, hbm_available=True,
            preserves_existing_reservations=True,
        )
        self.assertTrue(accepted.admitted)
        self.assertEqual(accepted.waste_safety_budget_ms, 20)

    def test_elastic_workspace_is_one_shot_if_fit_and_microbatch_otherwise(self):
        hbm = UnifiedHBMReservationManager(
            allocator_capacity_bytes=5 * GIB, safety_bytes=4 * GIB
        )
        one = acquire_elastic_selection_workspace(
            hbm, owner_request_id="r1", segment_id="selection-r1", compared_k=16,
            current_state_bytes=1_000, per_source_state_bytes=1_000,
        )
        self.assertTrue(one.one_shot)
        hbm.release(one.reservation.reservation_id)
        small = UnifiedHBMReservationManager(
            allocator_capacity_bytes=4 * GIB + 5_000, safety_bytes=4 * GIB
        )
        many = acquire_elastic_selection_workspace(
            small, owner_request_id="r2", segment_id="selection-r2", compared_k=16,
            current_state_bytes=1_000, per_source_state_bytes=1_000,
        )
        self.assertFalse(many.one_shot)
        self.assertEqual(many.microbatch_k, 4)


if __name__ == "__main__":
    unittest.main()
