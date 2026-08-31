import unittest

from probekv.contracts import KVLocation
from probekv.model_adapters import (
    MISTRAL_SCHEMA8_SPEC,
    validate_schema8_checkpoint_contract,
)
from probekv.v8_leases import V8LeaseManager, V8ReplicaResource
from probekv.v8_schema6_contracts import CommitAxisState, PlannerSnapshot
from probekv.v8_schema6_hbm import GIB, UnifiedHBMReservationManager
from probekv.v8_schema7_contracts import FinalCommitDecision
from probekv.v8_schema8_barrier import close_dense_selection_barrier
from probekv.v8_schema8_contracts import RepairRatioScope
from probekv.v8_schema8_planner import Gate1LayerCost, Gate1LocalPlan
from probekv.v8_schema8_repair import (
    SegmentLayerRepairRatio,
    validate_union_repair_ratio_plan,
)
from probekv.v8_schema8_storage import Schema8TieredBackingManager
from probekv.v8_schema8_storage import Schema8TieredReplicaCoordinator
from probekv.v8_schema8_runtime import Schema8BarrierRequestController
from probekv.v8_schema8_selector import Schema8D1D2Selector
from probekv.v8_contracts import CandidateCounts, ResidualCandidate


def _runtime_resources():
    leases = V8LeaseManager()
    for suffix in ("1", "2"):
        leases.register_source(f"source-c{suffix}", f"artifact-c{suffix}", "model")
        leases.register_replica(
            V8ReplicaResource(
                f"cpu-c{suffix}", f"source-c{suffix}", f"artifact-c{suffix}",
                KVLocation.PINNED_CPU, 1, 1, 1024, True,
            )
        )
    hbm = UnifiedHBMReservationManager(
        allocator_capacity_bytes=8 * GIB, safety_bytes=4 * GIB
    )
    return leases, hbm


class Schema8GateAndBarrierTests(unittest.TestCase):
    def test_gate1_uses_overlap_critical_path_and_positive_saving(self):
        plan = Gate1LocalPlan(
            "winner",
            1,
            1,
            2,
            1.0,
            0.5,
            (Gate1LayerCost(2, 0.15, 4.0, 6.0, 1.0),),
            8.0,
        )
        self.assertEqual(plan.predicted_reuse_future_upper_ms, 8.5)
        self.assertFalse(plan.passed)
        cheaper = Gate1LocalPlan(
            "winner",
            1,
            1,
            2,
            0.5,
            0.5,
            (Gate1LayerCost(2, 0.15, 2.0, 3.0, 1.0),),
            5.0,
        )
        self.assertEqual(cheaper.predicted_reuse_future_upper_ms, 5.0)
        self.assertTrue(cheaper.passed)
        with self.assertRaisesRegex(ValueError, "gamma"):
            Gate1LocalPlan("winner", 1, 1, 2, 0, 0, (), 1, gate1_gamma=0.8)

    def test_all_d1_starts_layer2_and_any_d2_starts_layer3(self):
        all_d1 = close_dense_selection_barrier(
            segment_ids=("c1", "c2"),
            resolved_completed_depth_by_segment={"c1": 1, "c2": 1},
            source_frozen_segment_ids=("c1", "c2"),
            abstained_segment_ids=(),
        )
        self.assertEqual(all_d1.first_selective_reuse_layer, 2)
        rescue = close_dense_selection_barrier(
            segment_ids=("c1", "c2", "c3"),
            resolved_completed_depth_by_segment={"c1": 1, "c2": 2, "c3": 2},
            source_frozen_segment_ids=("c1", "c2"),
            abstained_segment_ids=("c3",),
        )
        self.assertEqual(rescue.d2_rescue_segment_ids, ("c2", "c3"))
        self.assertEqual(rescue.first_selective_reuse_layer, 3)
        self.assertEqual(rescue.reuse_segment_ids, ("c1", "c2"))
        self.assertEqual(rescue.dense_segment_ids, ("c3",))

    def test_selector_barrier_preparation_and_final_commit_are_one_chain(self):
        leases, hbm = _runtime_resources()
        controller = Schema8BarrierRequestController(
            request_id="r", request_generation=1,
            ordered_segment_ids=("c1", "c2"),
            lease_manager=leases, hbm_manager=hbm,
        )
        controller.decision_ready("c1", "source-c1", 1)
        controller.apply_gate1_plan(
            "c1",
            Gate1LocalPlan(
                "source-c1", 1, 1, 2, 0, 0,
                (Gate1LayerCost(2, 0.15, 1, 1, 0),), 2,
            ),
            predicted_remaining_s=1,
        )
        with self.assertRaisesRegex(RuntimeError, "selection closure"):
            controller.begin_winner_prefetch(
                "c1", artifact_id="artifact-c1", replica_id="cpu-c1",
                replica_generation=1, placement_epoch=1,
                target_hbm_bytes=1024, predicted_remaining_s=1,
            )
        controller.decision_ready("c2", "source-c2", 1)
        controller.apply_gate1_plan(
            "c2",
            Gate1LocalPlan(
                "source-c2", 1, 1, 2, 2, 1,
                (Gate1LayerCost(2, 0.15, 2, 2, 1),), 1,
            ),
            predicted_remaining_s=1,
        )
        controller.decision_ready("c2", "source-c2", 2)
        controller.apply_gate1_plan(
            "c2",
            Gate1LocalPlan(
                "source-c2", 2, 2, 3, 2, 1,
                (Gate1LayerCost(3, 0.15, 2, 2, 1),), 1,
            ),
            predicted_remaining_s=1,
        )
        barrier = controller.close_selection_barrier()
        self.assertEqual(barrier.reuse_segment_ids, ("c1",))
        self.assertEqual(barrier.dense_segment_ids, ("c2",))
        snap = PlannerSnapshot(1, 1, "scheduler", hbm.epoch, "profile")
        controller.apply_preparation_admission(
            ("c1",), snapshot=snap, current_snapshot=snap
        )
        controller.begin_winner_prefetch(
            "c1", artifact_id="artifact-c1", replica_id="cpu-c1",
            replica_generation=1, placement_epoch=1,
            target_hbm_bytes=1024, predicted_remaining_s=1,
        )
        controller.mark_winner_ready("c1", actual_reuse_boundary=3)
        final_snapshot = PlannerSnapshot(1, 1, "scheduler", hbm.epoch, "profile")
        decision = FinalCommitDecision(
            ("c1",), (), ("c2",), 50, 100, final_snapshot,
            {"c1": "joint_timeline_pass", "c2": "dense_fallback"},
        )
        controller.apply_final_commit_admission(
            decision,
            planner_snapshot=final_snapshot,
            current_snapshot=final_snapshot,
        )
        self.assertEqual(
            controller.records["c1"].commit_state, CommitAxisState.REUSE_COMMIT
        )

    def test_schema8_model_checkpoint_sources_are_identical(self):
        self.assertEqual(MISTRAL_SCHEMA8_SPEC.checkpoints, (1, 2))
        self.assertEqual(
            validate_schema8_checkpoint_contract(
                model_id=MISTRAL_SCHEMA8_SPEC.model_id,
                checkpoint_sources={"adapter": (1, 2), "jobs": (1, 2)},
            ),
            (1, 2),
        )
        with self.assertRaises(ValueError):
            validate_schema8_checkpoint_contract(
                model_id=MISTRAL_SCHEMA8_SPEC.model_id,
                checkpoint_sources={"stale": (1, 2, 4)},
            )

    def test_d1_gate1_failure_continues_and_d2_uses_economic_residual_band(self):
        selector = Schema8D1D2Selector(
            strong_margin=0.6, stable_margin=0.3,
            residual_band_relative_tolerance=0.10,
        )
        counts = CandidateCounts(2, 2, 2, 2, 2)
        d1_candidates = (
            ResidualCandidate("quality-best", 0.10, 9, 0),
            ResidualCandidate("economic-near", 0.30, 4, 1),
        )
        d1_plans = {
            "quality-best": Gate1LocalPlan(
                "quality-best", 1, 1, 2, 1, 1,
                (Gate1LayerCost(2, 0.15, 4, 4, 1),), 5,
            ),
            "economic-near": Gate1LocalPlan(
                "economic-near", 1, 1, 2, 0, 0,
                (Gate1LayerCost(2, 0.15, 1, 1, 0),), 5,
            ),
        }
        d1 = selector.decide(
            completed_depth=1, counts=counts, candidates=d1_candidates,
            gate1_plan_by_source=d1_plans,
        )
        self.assertEqual(d1.state, "continue_probe")
        d2_plans = {
            "quality-best": Gate1LocalPlan(
                "quality-best", 2, 2, 3, 1, 1,
                (Gate1LayerCost(3, 0.15, 4, 4, 1),), 5,
            ),
            "economic-near": Gate1LocalPlan(
                "economic-near", 2, 2, 3, 0, 0,
                (Gate1LayerCost(3, 0.15, 1, 1, 0),), 5,
            ),
        }
        d2_candidates = (
            ResidualCandidate("quality-best", 0.10, 9, 0),
            ResidualCandidate("economic-near", 0.105, 4, 1),
        )
        d2 = selector.decide(
            completed_depth=2, counts=counts, candidates=d2_candidates,
            gate1_plan_by_source=d2_plans,
        )
        self.assertEqual(d2.selected_source_variant_id, "economic-near")
        self.assertEqual(d2.best_residual_source_variant_id, "quality-best")


class Schema8RepairRatioTests(unittest.TestCase):
    def test_fixed_ratio_is_uniform_in_same_layer(self):
        plan = validate_union_repair_ratio_plan(
            scope=RepairRatioScope.UNIFORM_FIXED,
            rows=(
                SegmentLayerRepairRatio("c1", 3, 3, 0.15),
                SegmentLayerRepairRatio("c2", 3, 3, 0.15),
            ),
            certified_floor=0.10,
            profile_frozen=False,
        )
        self.assertEqual(plan.ratios_for_layer(3), {"c1": 0.15, "c2": 0.15})
        with self.assertRaisesRegex(ValueError, "fixed15"):
            validate_union_repair_ratio_plan(
                scope=RepairRatioScope.UNIFORM_FIXED,
                rows=(SegmentLayerRepairRatio("c1", 3, 3, 0.12),),
                certified_floor=0.10,
                profile_frozen=False,
            )

    def test_static_schedule_is_shared_by_relative_repair_age(self):
        plan = validate_union_repair_ratio_plan(
            scope=RepairRatioScope.SHARED_RELATIVE_SCHEDULE,
            rows=(
                SegmentLayerRepairRatio("c1", 2, 2, 0.15),
                SegmentLayerRepairRatio("c1", 3, 2, 0.12),
                SegmentLayerRepairRatio("c2", 3, 3, 0.15),
                SegmentLayerRepairRatio("c2", 4, 3, 0.12),
            ),
            certified_floor=0.12,
            profile_frozen=False,
        )
        self.assertEqual(plan.ratios_for_layer(3), {"c1": 0.12, "c2": 0.15})
        with self.assertRaisesRegex(ValueError, "relative repair age"):
            validate_union_repair_ratio_plan(
                scope=RepairRatioScope.SHARED_RELATIVE_SCHEDULE,
                rows=(
                    SegmentLayerRepairRatio("c1", 2, 2, 0.15),
                    SegmentLayerRepairRatio("c2", 3, 3, 0.12),
                ),
                certified_floor=0.10,
                profile_frozen=False,
            )

    def test_adaptive_same_layer_may_differ_only_with_frozen_profile(self):
        rows = (
            SegmentLayerRepairRatio("c1", 3, 3, 0.15),
            SegmentLayerRepairRatio("c2", 3, 3, 0.10),
        )
        with self.assertRaisesRegex(ValueError, "frozen Profile"):
            validate_union_repair_ratio_plan(
                scope=RepairRatioScope.PER_SEGMENT_LOAD_AWARE,
                rows=rows,
                certified_floor=0.10,
                profile_frozen=False,
            )
        plan = validate_union_repair_ratio_plan(
            scope=RepairRatioScope.PER_SEGMENT_LOAD_AWARE,
            rows=rows,
            certified_floor=0.10,
            profile_frozen=True,
        )
        self.assertEqual(plan.ratios_for_layer(3), {"c1": 0.15, "c2": 0.10})


class Schema8TieredBackingTests(unittest.TestCase):
    def test_cpu_lru_demotes_and_ssd_lru_deletes(self):
        manager = Schema8TieredBackingManager(
            cpu_capacity_bytes=20,
            ssd_capacity_bytes=20,
        )
        manager.register("a", size_bytes=10)
        manager.register("b", size_bytes=10)
        manager.access("a")
        actions = manager.register("c", size_bytes=10)
        self.assertIn("demote_cpu_to_ssd", [row.action for row in actions])
        self.assertEqual(manager.entry("b").tier, KVLocation.SSD)
        manager.register("d", size_bytes=10)
        actions = manager.register("e", size_bytes=10)
        self.assertIn("evict_ssd_lru_source", [row.action for row in actions])
        self.assertNotIn("b", manager.snapshot()["entries"])

    def test_ssd_hit_promotes_and_busy_victim_is_protected(self):
        manager = Schema8TieredBackingManager(
            cpu_capacity_bytes=10,
            ssd_capacity_bytes=30,
        )
        manager.register("hot", size_bytes=10)
        manager.set_busy("hot", True)
        with self.assertRaisesRegex(MemoryError, "busy"):
            manager.register("new", size_bytes=10)
        manager.set_busy("hot", False)
        manager.register("new", size_bytes=10)
        self.assertEqual(manager.entry("hot").tier, KVLocation.SSD)
        manager.access("hot")
        self.assertEqual(manager.entry("hot").tier, KVLocation.PINNED_CPU)
        self.assertEqual(manager.entry("new").tier, KVLocation.SSD)

    def test_source_has_exactly_one_backing_entry(self):
        manager = Schema8TieredBackingManager(
            cpu_capacity_bytes=10,
            ssd_capacity_bytes=10,
        )
        manager.register("a", size_bytes=10)
        with self.assertRaisesRegex(ValueError, "one backing"):
            manager.register("a", size_bytes=10)
        snapshot = manager.snapshot()
        self.assertEqual(len(snapshot["entries"]), 1)

    def test_failed_pressure_transition_rolls_back_every_prior_move(self):
        manager = Schema8TieredBackingManager(
            cpu_capacity_bytes=20,
            ssd_capacity_bytes=20,
        )
        manager.register("a", size_bytes=10)
        manager.register("b", size_bytes=10)
        manager.register("ssd-busy", size_bytes=10, initially_used=False)
        manager.set_busy("ssd-busy", True)
        manager.set_busy("b", True)
        before = manager.snapshot()
        before_actions = manager.actions
        with self.assertRaises(MemoryError):
            manager.register("c", size_bytes=20)
        self.assertEqual(manager.snapshot(), before)
        self.assertEqual(manager.actions, before_actions)

    def test_busy_state_is_derived_from_logical_and_physical_leases(self):
        leases, _ = _runtime_resources()
        manager = Schema8TieredBackingManager(
            cpu_capacity_bytes=1024, ssd_capacity_bytes=2048
        )
        coordinator = Schema8TieredReplicaCoordinator(
            backing_manager=manager, lease_manager=leases
        )
        coordinator.register("source-c1", size_bytes=1024)
        logical = leases.freeze_and_acquire_logical(
            request_id="r", request_generation=1, segment_id="c1",
            source_variant_id="source-c1", predicted_remaining_s=1,
        )
        self.assertTrue(coordinator.synchronize_busy()["source-c1"])
        with self.assertRaisesRegex(MemoryError, "busy"):
            manager.register("untracked", size_bytes=1024)
        leases.release(logical.lease_id)
        self.assertFalse(coordinator.synchronize_busy()["source-c1"])


if __name__ == "__main__":
    unittest.main()
