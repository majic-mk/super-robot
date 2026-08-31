import unittest

from probekv.contracts import KVLocation
from probekv.model_adapters import (
    MISTRAL_SCHEMA8_SPEC,
    validate_schema8_checkpoint_contract,
)
from probekv.v8_leases import ReplicaLifecycle, V8LeaseManager, V8ReplicaResource
from probekv.v8_schema6_contracts import CommitAxisState, PlannerSnapshot
from probekv.v8_schema6_hbm import GIB, UnifiedHBMReservationManager
from probekv.v8_schema7_contracts import FinalCommitDecision
from probekv.v8_schema8_barrier import close_dense_selection_barrier
from probekv.v8_schema8_contracts import RepairRatioScope
from probekv.v8_schema8_planner import Gate1LocalPlan, Gate1MarginalLowerBound
from probekv.v8_schema8_repair import (
    JointLoadRecomputeAwareRepairController,
    JointRepairRatioCandidate,
    RequestLayerUniformIORepairController,
    SegmentLayerRepairRatio,
    choose_request_level_adaptive_ratio,
    validate_union_repair_ratio_plan,
)
from probekv.v8_schema8_profile import (
    RuntimeCostProfileV8,
    Schema8ProfileProvenance,
    build_repair_policy_profile_v8,
    build_runtime_cost_profile_v8,
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


def _g1(
    source: str,
    depth: int,
    *,
    support: float,
    load: float,
    repair: float,
    dense: float,
    sunk: float = 0.0,
) -> Gate1LocalPlan:
    return Gate1LocalPlan(
        source,
        depth,
        depth,
        depth + 1,
        sunk,
        Gate1MarginalLowerBound(support, load, repair),
        dense,
    )


class Schema8GateAndBarrierTests(unittest.TestCase):
    def test_gate1_uses_optimistic_marginal_lower_bound(self):
        plan = _g1(
            "winner", 1, support=1.0, load=4.0, repair=3.5,
            dense=8.0, sunk=100.0,
        )
        self.assertEqual(plan.predicted_reuse_marginal_lower_ms, 8.5)
        self.assertFalse(plan.passed)
        cheaper = _g1(
            "winner", 1, support=0.5, load=1.0, repair=1.0,
            dense=5.0, sunk=100.0,
        )
        self.assertEqual(cheaper.predicted_reuse_marginal_lower_ms, 2.5)
        self.assertTrue(cheaper.passed)
        # Common repair-check work is audited as sunk, not charged once per
        # Segment inside Gate1.
        self.assertEqual(cheaper.dense_repair_check_sunk_ms, 100.0)
        with self.assertRaisesRegex(ValueError, "gamma"):
            Gate1LocalPlan(
                "winner", 1, 1, 2, 0,
                Gate1MarginalLowerBound(0, 0, 0), 1, gate1_gamma=0.8,
            )

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
            _g1("source-c1", 1, support=0, load=0.5, repair=0.5, dense=2),
            predicted_remaining_s=1,
        )
        with self.assertRaisesRegex(RuntimeError, "detached resource admission"):
            controller.begin_winner_prefetch(
                "c1", artifact_id="artifact-c1", replica_id="cpu-c1",
                replica_generation=1, placement_epoch=1,
                target_hbm_bytes=1024, predicted_remaining_s=1,
            )
        detached_snapshot = PlannerSnapshot(1, 1, "scheduler", hbm.epoch, "profile")
        controller.apply_detached_preparation_admission(
            ("c1",), snapshot=detached_snapshot,
            current_snapshot=detached_snapshot,
        )
        controller.begin_winner_prefetch(
            "c1", artifact_id="artifact-c1", replica_id="cpu-c1",
            replica_generation=1, placement_epoch=1,
            target_hbm_bytes=1024, predicted_remaining_s=1,
        )
        controller.mark_winner_ready("c1", actual_reuse_boundary=3)
        self.assertFalse(controller.gate3_eligible("c1"))
        controller.decision_ready("c2", "source-c2", 1)
        controller.apply_gate1_plan(
            "c2",
            _g1("source-c2", 1, support=1, load=1, repair=1, dense=1),
            predicted_remaining_s=1,
        )
        controller.decision_ready("c2", "source-c2", 2)
        controller.apply_gate1_plan(
            "c2",
            _g1("source-c2", 2, support=1, load=1, repair=1, dense=1),
            predicted_remaining_s=1,
        )
        barrier = controller.close_selection_barrier()
        self.assertEqual(barrier.reuse_segment_ids, ("c1",))
        self.assertEqual(barrier.dense_segment_ids, ("c2",))
        snap = PlannerSnapshot(1, 1, "scheduler", hbm.epoch, "profile")
        controller.apply_preparation_admission(
            ("c1",), snapshot=snap, current_snapshot=snap
        )
        self.assertTrue(controller.gate3_eligible("c1"))
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
            "quality-best": _g1(
                "quality-best", 1, support=1, load=3, repair=2, dense=5,
            ),
            "economic-near": _g1(
                "economic-near", 1, support=0, load=1, repair=1, dense=5,
            ),
        }
        d1 = selector.decide(
            completed_depth=1, counts=counts, candidates=d1_candidates,
            gate1_plan_by_source=d1_plans,
        )
        self.assertEqual(d1.state, "continue_probe")
        d2_plans = {
            "quality-best": _g1(
                "quality-best", 2, support=1, load=3, repair=2, dense=5,
            ),
            "economic-near": _g1(
                "economic-near", 2, support=0, load=1, repair=1, dense=5,
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
        with self.assertRaisesRegex(ValueError, "request-level joint decision"):
            validate_union_repair_ratio_plan(
                scope=RepairRatioScope.PER_SEGMENT_LOAD_AWARE,
                rows=rows,
                certified_floor=0.10,
                profile_frozen=True,
            )
        decision = choose_request_level_adaptive_ratio(
            candidates=(
                JointRepairRatioCandidate(
                    "joint-fast", 3, (("c1", 0.15), ("c2", 0.10)),
                    2.0, 3.0, 1.0,
                ),
                JointRepairRatioCandidate(
                    "fixed15", 3, (("c1", 0.15), ("c2", 0.15)),
                    2.0, 5.0, 1.0,
                ),
            ),
            expected_segment_ids=("c1", "c2"),
            repair_policy_profile_sha256="9" * 64,
            runtime_cost_profile_sha256="a" * 64,
        )
        plan = validate_union_repair_ratio_plan(
            scope=RepairRatioScope.PER_SEGMENT_LOAD_AWARE,
            rows=rows,
            certified_floor=0.10,
            profile_frozen=True,
            adaptive_joint_decisions=(decision,),
        )
        self.assertEqual(plan.ratios_for_layer(3), {"c1": 0.15, "c2": 0.10})

    def test_adaptive_vector_is_chosen_by_union_critical_path_not_per_segment(self):
        decision = choose_request_level_adaptive_ratio(
            candidates=(
                JointRepairRatioCandidate(
                    "mixed", 4, (("c1", 0.10), ("c2", 0.15)),
                    5.0, 5.0, 1.0,
                ),
                JointRepairRatioCandidate(
                    "low-independent", 4, (("c1", 0.10), ("c2", 0.10)),
                    7.0, 3.0, 1.0,
                ),
                JointRepairRatioCandidate(
                    "high-tie", 4, (("c1", 0.15), ("c2", 0.15)),
                    5.0, 5.0, 1.0,
                ),
            ),
            expected_segment_ids=("c1", "c2"),
            repair_policy_profile_sha256="8" * 64,
            runtime_cost_profile_sha256="b" * 64,
        )
        # mixed and high-tie have the same request critical path; quality-first
        # tie breaking keeps the larger joint repair allocation.
        self.assertEqual(decision.selected_candidate_id, "high-tie")
        self.assertEqual(decision.ratio_map(), {"c1": 0.15, "c2": 0.15})

    def test_profile_bound_controller_balances_joint_io_and_repair(self):
        common = dict(
            code_commit="a" * 40,
            cacheblend_patch_sha256="b" * 64,
            model_id="mistral",
            model_revision="c" * 40,
            tokenizer_hash="d" * 64,
            gpu_uuid="GPU-real",
            measurement_sha256="e" * 64,
            frozen=True,
        )
        repair_profile = build_repair_policy_profile_v8(
            provenance=Schema8ProfileProvenance(
                profile_kind="repair_policy", **common
            ),
            policy="load_recompute_aware_gradual",
            scope="per_segment_load_aware",
            certified_floor=0.10,
            shared_ratio_by_age={0: 0.15, 1: 0.12},
            no_reentry_oracle_recall=0.99,
            minimum_no_reentry_recall=0.95,
            adaptive_candidate_templates=(
                "uniform_cap", "uniform_floor", "single_segment_load_priority"
            ),
            timing_equivalence_absolute_ms=0.05,
            timing_equivalence_relative=0.0,
        )
        runtime_profile = build_runtime_cost_profile_v8(
            provenance=Schema8ProfileProvenance(
                profile_kind="runtime_cost", **common
            ),
            category_measurements={
                name: ({"cuda_event_timing": True},)
                for name in RuntimeCostProfileV8.REQUIRED_CATEGORIES
            },
        )
        controller = JointLoadRecomputeAwareRepairController(
            repair_policy_profile=repair_profile,
            runtime_cost_profile=runtime_profile,
        )
        decision = controller.choose_layer(
            candidates=(
                JointRepairRatioCandidate(
                    "repair-10", 4, (("c1", 0.10), ("c2", 0.10)),
                    5.0, 4.96, 1.0, template_id="uniform_floor",
                ),
                JointRepairRatioCandidate(
                    "repair-15", 4, (("c1", 0.15), ("c2", 0.15)),
                    5.0, 5.04, 1.0, template_id="uniform_cap",
                ),
            ),
            expected_segment_ids=("c1", "c2"),
            previous_ratio_by_segment={"c1": 0.15, "c2": 0.15},
        )
        # 6.00 vs 6.04 ms is within the frozen 0.05-ms profile resolution,
        # so quality-first tie handling retains the larger repair support.
        self.assertEqual(decision.selected_candidate_id, "repair-15")
        self.assertEqual(decision.timing_equivalence_tolerance_ms, 0.05)
        with self.assertRaisesRegex(ValueError, "may not increase"):
            controller.choose_layer(
                candidates=(
                    JointRepairRatioCandidate(
                        "illegal", 5, (("c1", 0.15), ("c2", 0.12)),
                        1.0, 1.0, 0.0,
                        template_id="single_segment_load_priority",
                    ),
                ),
                expected_segment_ids=("c1", "c2"),
                previous_ratio_by_segment={"c1": 0.12, "c2": 0.12},
            )
        plan = controller.build_plan(
            candidates_by_layer={
                3: (
                    JointRepairRatioCandidate(
                        "l3-cap", 3, (("c1", 0.15), ("c2", 0.15)),
                        8.0, 5.0, 1.0, template_id="uniform_cap",
                    ),
                ),
                4: (
                    JointRepairRatioCandidate(
                        "l4-mixed", 4, (("c1", 0.12), ("c2", 0.10)),
                        3.0, 3.1, 1.0,
                        template_id="single_segment_load_priority",
                    ),
                ),
            },
            first_selective_reuse_layer_by_segment={"c1": 3, "c2": 3},
        )
        self.assertEqual(plan.ratios_for_layer(3), {"c1": 0.15, "c2": 0.15})
        self.assertEqual(plan.ratios_for_layer(4), {"c1": 0.12, "c2": 0.10})

    def test_uniform_io_controller_uses_hidden_maximum_above_fifteen_percent(self):
        common = dict(
            code_commit="a" * 40,
            cacheblend_patch_sha256="b" * 64,
            model_id="mistral",
            model_revision="c" * 40,
            tokenizer_hash="d" * 64,
            gpu_uuid="GPU-real",
            measurement_sha256="e" * 64,
            frozen=True,
        )
        repair = build_repair_policy_profile_v8(
            provenance=Schema8ProfileProvenance(
                profile_kind="repair_policy", **common
            ),
            policy="load_recompute_aware_uniform",
            scope="request_layer_uniform_io_balanced",
            certified_floor=0.10,
            shared_ratio_by_age={},
            no_reentry_oracle_recall=0.99,
            minimum_no_reentry_recall=0.95,
        )
        runtime = build_runtime_cost_profile_v8(
            provenance=Schema8ProfileProvenance(
                profile_kind="runtime_cost", **common
            ),
            category_measurements={
                name: ({"cuda_event_timing": True},)
                for name in RuntimeCostProfileV8.REQUIRED_CATEGORIES
            },
        )
        controller = RequestLayerUniformIORepairController(
            repair_policy_profile=repair,
            runtime_cost_profile=runtime,
        )

        def candidates(layer, load, repair_ms_by_ratio):
            return tuple(
                JointRepairRatioCandidate(
                    "l%d-r%.2f" % (layer, ratio),
                    layer,
                    (("c1", ratio), ("c2", ratio)),
                    load,
                    repair_ms,
                    1.0,
                )
                for ratio, repair_ms in repair_ms_by_ratio
            )

        first = controller.choose_layer(
            candidates=candidates(
                3,
                10.0,
                ((0.10, 2.0), (0.12, 2.4), (0.15, 3.0), (0.20, 4.0),
                 (0.30, 6.0), (0.50, 9.0), (0.75, 13.0), (1.0, 18.0)),
            ),
            expected_segment_ids=("c1", "c2"),
        )
        self.assertEqual(first.io_balance_ratio, 0.50)
        self.assertEqual(first.selected_ratio, 0.50)
        self.assertEqual(first.ratio_map(), {"c1": 0.50, "c2": 0.50})

        second = controller.choose_layer(
            candidates=candidates(
                4,
                1.0,
                ((0.10, 2.0), (0.12, 2.4), (0.15, 3.0), (0.20, 4.0),
                 (0.30, 6.0), (0.50, 9.0), (0.75, 13.0), (1.0, 18.0)),
            ),
            expected_segment_ids=("c1", "c2"),
            previous_uniform_ratio=first.selected_ratio,
        )
        # The target is the certified 10% floor, but no-reentry gradual
        # filtering takes one frozen-grid step: 50% -> 30%.
        self.assertEqual(second.io_balance_ratio, 0.0)
        self.assertEqual(second.selected_ratio, 0.30)

    def test_uniform_io_plan_rejects_per_segment_ratio_vector(self):
        with self.assertRaisesRegex(ValueError, "uniform I/O rows"):
            from probekv.v8_schema8_repair import UniformIOBalanceDecision

            validate_union_repair_ratio_plan(
                scope=RepairRatioScope.REQUEST_LAYER_UNIFORM_IO_BALANCED,
                rows=(
                    SegmentLayerRepairRatio("c1", 3, 3, 0.30),
                    SegmentLayerRepairRatio("c2", 3, 3, 0.20),
                ),
                certified_floor=0.10,
                profile_frozen=True,
                certified_ratio_candidates=(0.10, 0.15, 0.20, 0.30),
                uniform_io_decisions=(
                    UniformIOBalanceDecision(
                        3, ("c1", "c2"), 0.30, 0.10, 0.15, 0.30,
                        10.0, 8.0, 1.0, "a" * 64, "b" * 64, "c" * 64,
                    ),
                ),
            )


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

    def test_backing_migration_publishes_only_after_verified_copy(self):
        leases = V8LeaseManager()
        leases.register_source("source", "artifact", "model")
        old = V8ReplicaResource(
            "cpu", "source", "artifact", KVLocation.PINNED_CPU,
            1, 1, 10, True,
        )
        leases.register_replica(old)
        manager = Schema8TieredBackingManager(
            cpu_capacity_bytes=10, ssd_capacity_bytes=10
        )
        coordinator = Schema8TieredReplicaCoordinator(
            backing_manager=manager, lease_manager=leases
        )
        coordinator.register("source", size_bytes=10)
        request_epoch = manager.record_request_use("source")
        destination = V8ReplicaResource(
            "ssd", "source", "artifact", KVLocation.SSD,
            1, 1, 10, False, lifecycle=ReplicaLifecycle.ALLOCATING,
        )
        ticket = coordinator.begin_backing_migration(destination)
        self.assertTrue(old.is_backing)
        self.assertFalse(destination.is_backing)
        coordinator.finish_backing_migration(
            ticket, copy_completed=True,
            source_logical_digest="digest", destination_logical_digest="digest",
        )
        self.assertEqual(old.lifecycle, ReplicaLifecycle.DELETED)
        self.assertTrue(destination.is_backing)
        self.assertEqual(destination.lifecycle, ReplicaLifecycle.READY)
        self.assertEqual(manager.entry("source").tier, KVLocation.SSD)
        self.assertEqual(
            manager.entry("source").last_request_use_epoch,
            request_epoch,
        )

    def test_migration_churn_does_not_refresh_request_lru(self):
        manager = Schema8TieredBackingManager(
            cpu_capacity_bytes=20, ssd_capacity_bytes=20
        )
        manager.register("cold", size_bytes=10)
        manager.register("hot", size_bytes=10)
        cold_epoch = manager.entry("cold").last_request_use_epoch
        manager.record_request_use("hot")
        # A background policy transition keeps the cold Source's epoch.  It is
        # therefore still the deterministic LRU victim under later pressure.
        manager._demote_cpu_victim(manager.entry("cold"), ())
        self.assertEqual(manager.entry("cold").last_request_use_epoch, cold_epoch)
        manager.register("new", size_bytes=10)
        self.assertEqual(manager.entry("cold").tier, KVLocation.SSD)

    def test_failed_backing_migration_keeps_original_authoritative(self):
        leases = V8LeaseManager()
        leases.register_source("source", "artifact", "model")
        old = V8ReplicaResource(
            "cpu", "source", "artifact", KVLocation.PINNED_CPU,
            1, 1, 10, True,
        )
        leases.register_replica(old)
        manager = Schema8TieredBackingManager(
            cpu_capacity_bytes=10, ssd_capacity_bytes=10
        )
        coordinator = Schema8TieredReplicaCoordinator(
            backing_manager=manager, lease_manager=leases
        )
        coordinator.register("source", size_bytes=10)
        destination = V8ReplicaResource(
            "ssd", "source", "artifact", KVLocation.SSD,
            1, 1, 10, False, lifecycle=ReplicaLifecycle.ALLOCATING,
        )
        ticket = coordinator.begin_backing_migration(destination)
        coordinator.finish_backing_migration(
            ticket, copy_completed=True,
            source_logical_digest="expected", destination_logical_digest="corrupt",
        )
        self.assertTrue(old.is_backing)
        self.assertEqual(old.lifecycle, ReplicaLifecycle.READY)
        self.assertFalse(destination.is_backing)
        self.assertEqual(destination.lifecycle, ReplicaLifecycle.DELETED)
        self.assertEqual(manager.entry("source").tier, KVLocation.PINNED_CPU)


if __name__ == "__main__":
    unittest.main()
