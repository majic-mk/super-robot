import unittest

from probekv.candidate_budget import (
    VariantComparisonCandidate,
    allocate_variant_comparisons,
)
from probekv.contracts import KVLocation
from probekv.global_source_pool import ModelServingMode
from probekv.v7_contracts import (
    CanonicalKVArtifact,
    PredictedAccessPlan,
    SourceVariantIdentity,
)
from probekv.v7_eligibility import (
    CalibrationEnvelope,
    ReplicaTierProfile,
    bind_or_replan_same_source,
    evaluate_calibration,
    evaluate_source_variant,
    preview_replica_access,
)
from probekv.v7_planner import (
    JointRequestPlanner,
    LockedSegmentOption,
    SourceCostCandidate,
    SharedSunkCost,
    choose_source_variant,
    repair_token_count,
)
from probekv.v7_source_pool import V7SourcePool


class V7EligibilityPlannerTests(unittest.TestCase):
    def make_pool(self):
        pool = V7SourcePool(serving_mode=ModelServingMode.MULTI)
        pool.activate_namespace("model")
        identity = SourceVariantIdentity(
            reuse_content_key="content",
            historical_prefix_digest="prefix",
            position_ids_digest="positions",
            occurrence_id="occ",
            model_math_signature="model",
        )
        variant = pool.register_variant(
            identity,
            canonical_source_state_digest="state",
            summary_digest="summary",
        )
        pool.register_artifact(
            "model",
            "content",
            variant.source_variant_id,
            CanonicalKVArtifact(
                artifact_id="artifact",
                source_variant_id=variant.source_variant_id,
                generation=1,
                parent_source_state_digest="state",
                artifact_logical_digest="logical",
                artifact_bytes_digest="bytes",
                num_layers=32,
                num_kv_heads=8,
                head_dim=128,
            ),
        )
        replica = pool.attach_replica(
            "model",
            "content",
            variant.source_variant_id,
            tier=KVLocation.GPU,
            locator_value="gpu:1",
            layout_signature="paged",
            bytes_digest="gpu-bytes",
            size_bytes=100,
        )
        return pool, variant, replica

    def test_layered_eligibility_has_distinct_failures(self):
        _, variant, _ = self.make_pool()
        source = evaluate_source_variant(
            variant,
            expected_content_key="content",
            expected_model_math_signature="model",
        )
        self.assertTrue(source.eligible)
        envelope = CalibrationEnvelope(
            namespace="chunker-a",
            min_segment_tokens=128,
            max_segment_tokens=640,
            max_sources=16,
            max_probe_layer=8,
            summary_formats=("exact_bf16",),
            max_feature_norm=10.0,
        )
        calibration = evaluate_calibration(
            canonicalizer_namespace="chunker-b",
            segment_tokens=512,
            source_count=4,
            probe_layer=2,
            summary_format="exact_bf16",
            feature_norm=1.0,
            envelope=envelope,
        )
        self.assertFalse(calibration.eligible)
        self.assertIn("namespace_mismatch", calibration.reasons)

    def test_stale_preview_replans_only_same_source(self):
        pool, variant, replica = self.make_pool()
        profile = {KVLocation.GPU: ReplicaTierProfile(0.0, 0.0, 0.1)}
        preview = preview_replica_access(
            pool,
            variant,
            scheduler_snapshot_id=1,
            profile_version="p1",
            tier_profiles=profile,
            repair_selection_upper_ms=0.1,
            repair_upper_ms=2.0,
            remaining_upper_ms=3.0,
            expected_geometry=(32, 8, 128),
        )
        pool.relocate_replica(
            "model", "content", variant.source_variant_id, replica.replica_id,
            locator_value="gpu:2",
        )
        plan, bound, replans = bind_or_replan_same_source(
            pool,
            variant,
            preview,
            model_math_signature="model",
            tier_profiles=profile,
            scheduler_snapshot_id=2,
            profile_version="p1",
            expected_geometry=(32, 8, 128),
            repair_selection_upper_ms=0.1,
            repair_upper_ms=2.0,
            remaining_upper_ms=3.0,
        )
        self.assertIsNotNone(plan)
        self.assertIsNotNone(bound)
        self.assertEqual(plan.source_variant_id, variant.source_variant_id)
        self.assertGreaterEqual(replans, 1)

    def test_repair_count_is_conservative(self):
        self.assertEqual(repair_token_count(10, 0), 0)
        self.assertEqual(repair_token_count(10, 1), 10)
        self.assertEqual(repair_token_count(10, 0.16), 2)
        self.assertEqual(repair_token_count(1, 0.05), 1)

    def test_joint_planner_supports_staggered_partial_reuse(self):
        options = (
            LockedSegmentOption(
                "c1", "s1", "r1", 3, 0.1, 512, 100, 1, 0, 0, 0, 5, 5, 30
            ),
            LockedSegmentOption(
                "c2", "s2", "r2", 7, 0.2, 512, 100, 20, 5, 3, 1, 15, 10, 30
            ),
            LockedSegmentOption(
                "c3", "s3", "r3", 5, 0.1, 512, 100, 1, 0, 0, 0, 5, 5, 30,
                source_ready=False,
            ),
        )
        plan = JointRequestPlanner(gamma=0.8, hbm_capacity_bytes=150).plan(
            "request", options, shared_sunk_ms=5, dense_reference_ms=120
        )
        accepted = [item for item in plan.decisions if item.path.value == "reuse"]
        self.assertEqual([item.segment_id for item in accepted], ["c1"])
        self.assertEqual(accepted[0].actual_reuse_boundary, 3)
        self.assertEqual(len(plan.decisions), 3)
        self.assertGreater(plan.wasted_bytes, 0)

    def test_shared_probe_cost_is_not_multiplied_by_candidates(self):
        sunk = SharedSunkCost(1.0, 0.5, 0.5, 1.0)
        self.assertEqual(sunk.total_ms, 3.0)
        self.assertTrue(sunk.within_probe_budget(100.0))

    def test_sixteenth_variant_can_be_compared_and_selected(self):
        compared = allocate_variant_comparisons(
            tuple(
                VariantComparisonCandidate(
                    "c0", "s%d" % index, float(index), 20.0, 0.01
                )
                for index in range(16)
            ),
            full_reference_ms=100.0,
            probe_ms=1.0,
            metadata_ms=0.1,
        )
        self.assertEqual(compared.audits[0].compared_k, 16)
        candidates = []
        for index in range(16):
            source_id = "s%d" % index
            future = 4.0 if index == 15 else 20.0 + index
            plan = PredictedAccessPlan(
                access_plan_id="plan-%d" % index,
                source_variant_id=source_id,
                artifact_id="artifact-%d" % index,
                artifact_generation=1,
                replica_id="replica-%d" % index,
                replica_generation=1,
                placement_epoch=1,
                pool_snapshot_id=1,
                scheduler_snapshot_id=1,
                profile_version="profile",
                visible_load_upper_ms=future,
                post_ready_blocking_upper_ms=0,
                interference_upper_ms=0,
                repair_selection_upper_ms=0,
                repair_upper_ms=0,
                remaining_upper_ms=0,
            )
            candidates.append(SourceCostCandidate(source_id, plan, True))
        selected = choose_source_variant(
            candidates,
            sunk=SharedSunkCost(1, 0.1, 0.1, compared.budget_used_ms),
            dense_reference_ms=100,
            gamma=0.8,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.source_variant_id, "s15")

    def test_budget_reduction_is_deterministic_and_never_cartesian(self):
        candidates = tuple(
            VariantComparisonCandidate(
                "c%d" % segment, "c%d-s%d" % (segment, source),
                float(source), 10.0 - segment, 0.08,
            )
            for segment in range(5)
            for source in range(16)
        )
        first = allocate_variant_comparisons(
            candidates, full_reference_ms=20.0, probe_ms=0.4, metadata_ms=0.3
        )
        second = allocate_variant_comparisons(
            candidates, full_reference_ms=20.0, probe_ms=0.4, metadata_ms=0.3
        )
        self.assertEqual(first.audits, second.audits)
        self.assertLess(sum(row.compared_k for row in first.audits), 80)
        self.assertLessEqual(first.budget_used_ms, first.budget_available_ms)


if __name__ == "__main__":
    unittest.main()
