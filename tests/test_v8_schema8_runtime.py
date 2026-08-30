import unittest

from probekv.contracts import KVLocation
from probekv.v8_schema8_barrier import close_dense_selection_barrier
from probekv.v8_schema8_contracts import RepairRatioScope
from probekv.v8_schema8_planner import Gate1LayerCost, Gate1LocalPlan
from probekv.v8_schema8_repair import (
    SegmentLayerRepairRatio,
    validate_union_repair_ratio_plan,
)
from probekv.v8_schema8_storage import Schema8TieredBackingManager


class Schema8GateAndBarrierTests(unittest.TestCase):
    def test_gate1_uses_overlap_critical_path_and_positive_saving(self):
        plan = Gate1LocalPlan(
            "winner",
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
            2,
            0.5,
            0.5,
            (Gate1LayerCost(2, 0.15, 2.0, 3.0, 1.0),),
            5.0,
        )
        self.assertEqual(cheaper.predicted_reuse_future_upper_ms, 5.0)
        self.assertTrue(cheaper.passed)
        with self.assertRaisesRegex(ValueError, "gamma"):
            Gate1LocalPlan("winner", 1, 2, 0, 0, (), 1, gate1_gamma=0.8)

    def test_all_d1_starts_layer2_and_any_d2_starts_layer3(self):
        all_d1 = close_dense_selection_barrier(
            segment_ids=("c1", "c2"),
            resolved_completed_depth_by_segment={"c1": 1, "c2": 1},
        )
        self.assertEqual(all_d1.first_selective_reuse_layer, 2)
        rescue = close_dense_selection_barrier(
            segment_ids=("c1", "c2", "c3"),
            resolved_completed_depth_by_segment={"c1": 1},
        )
        self.assertEqual(rescue.d2_rescue_segment_ids, ("c2", "c3"))
        self.assertEqual(rescue.first_selective_reuse_layer, 3)


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


if __name__ == "__main__":
    unittest.main()
