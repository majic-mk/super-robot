import tempfile
import unittest
from dataclasses import replace, asdict
from pathlib import Path
from unittest.mock import patch

import torch

from probekv.contracts import KVLocation
from probekv.v7_contracts import SourceVariantIdentity
from probekv.v8_schema10_contracts import AbsoluteResidualThreshold
from probekv.v8_schema10_pool import Schema10SourcePool
from probekv.v8_schema10_profile import VariantAdmissionProfileV10
from probekv.v8_schema10_storage import TensorFileSourceStore
from probekv.v8_schema10_cost_provider import (
    RequestExecutionShape, ProfiledJointTimelineEstimator, UnsupportedTimelineCost,
)
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema6_planner import JointTimelineContext, RefinedJointPlannerV6
from probekv.v8_schema6_contracts import PlannerSnapshot
from probekv.v8_cfo import build_source_cfo_metadata


def profile(k=16):
    return VariantAdmissionProfileV10(code_commit="commit", cacheblend_patch_sha256="a" * 64,
        model_id="model", model_revision="revision", tokenizer_hash="b" * 64,
        source_residual_trim_ratio=.15, thresholds=(AbsoluteResidualThreshold(1, .2), AbsoluteResidualThreshold(2, .25)),
        max_variants_per_content=k, exploration_quota_per_content=1)


def identity(i, content="c"):
    return SourceVariantIdentity(content, "prefix" + str(i), "positions", str(i), "model")


class StorageTransactions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pool = Schema10SourcePool(profile=profile(2))
        self.pool.activate_namespace("model")
        self.store = TensorFileSourceStore(self.pool, self.tmp.name, cpu_bytes=100000, ssd_bytes=100000, pin_cpu=False)
        self.layers = tuple((torch.full((4, 2, 2), float(i), dtype=torch.bfloat16),
                             torch.ones((4, 2, 2), dtype=torch.bfloat16)) for i in range(3))

    def publish(self, i, content="c", **kwargs):
        cfo = build_source_cfo_metadata(historical_prefix_chunk_occurrences=(),
            inter_mass_by_layer_and_occurrence=[{}, {}, {}], intra_mass_by_layer=[1., 1., 1.], target_token_count=4)
        return self.store.publish_exact_dense(identity(i, content), layers=self.layers,
            selection_states={1: self.layers[1][0].clone(), 2: self.layers[2][0].clone()},
            metadata={"token_ids": [1, 2, 3, 4], "cfo": asdict(cfo), "tokenizer_hash": "tokenizer", "runtime_compatibility": "runtime"}, request_epoch=i,
            whole_request_origin="exact_dense_full_prefill", materialization_reason="content_miss", **kwargs)

    def expire(self, content="c"):
        for _ in range(2):
            self.pool.finish_content_lookup("model", content)

    def test_lru_commit_does_not_revert_to_value_density(self):
        first, second = self.publish(1), self.publish(2)
        self.expire()
        first = self.pool._get("model", "c", first.source_variant_id)
        first.stats.realized_saved_ms_sum = 100000
        first.stats.admissions = 1
        plan = self.pool.replacement_transaction("model", "c")
        self.assertEqual(plan.victim_source_variant_id, first.source_variant_id)
        self.publish(3, replacement_transaction=plan)
        self.assertNotIn(first.source_variant_id, self.store.objects)
        self.assertIn(second.source_variant_id, self.store.objects)

    def test_replacement_rejects_victim_used_since_preview(self):
        first = self.publish(1)
        self.publish(2)
        self.expire()
        plan = self.pool.replacement_transaction("model", "c")
        self.store.promote_request_use(first.source_variant_id)
        before = self.store.describe()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.publish(3, replacement_transaction=plan)
        self.assertEqual(self.store.describe(), before)

    def test_logical_freeze_protects_last_backing(self):
        row = self.publish(1)
        self.expire()
        generation = self.pool.content_generation("model", "c")
        with self.store.leased_winner("model", "c", row.source_variant_id, expected_generation=generation) as layers:
            self.assertTrue(torch.equal(layers[0][0], self.layers[0][0]))
            with self.assertRaises(RuntimeError):
                self.pool._evict_variant(row, "test")
            with self.assertRaises(RuntimeError):
                self.store.snapshot()
            with self.assertRaises(RuntimeError):
                self.pool.purge_namespace("model")
        self.assertEqual(self.pool.logical_lease_counts[row.source_variant_id], 0)

    def test_logical_digest_matches_existing_gpu_loader_encoding(self):
        from probekv.cacheblend_v6_online_engine import TorchLayerwiseSourceLoader
        row = self.publish(1)
        self.assertEqual(row.canonical_source_state_digest, TorchLayerwiseSourceLoader._digest(torch, self.layers))

    def test_invalid_metadata_cannot_publish_half_visible_source(self):
        before = self.store.describe()
        with self.assertRaises(ValueError):
            self.store.publish_exact_dense(identity(1), layers=self.layers, selection_states={1: self.layers[1][0]},
                metadata={"token_ids": [1, 2, 3, 4]}, request_epoch=1,
                whole_request_origin="exact_dense_full_prefill", materialization_reason="content_miss")
        self.assertEqual(before, self.store.describe())

    def test_snapshot_release_reclaims_replaced_ssd_files(self):
        self.store.cpu_bytes = 1
        first = self.publish(1)
        snapshot = self.store.snapshot()
        path = Path(self.store.objects[first.source_variant_id].kv_path)
        self.publish(2)
        self.expire()
        self.publish(3)
        self.assertTrue(path.exists(), "explicit A/B snapshot owns the old backing")
        self.store.release_snapshot(snapshot)
        self.assertFalse(path.exists(), "released snapshots must not defeat SSD capacity")

    def test_no_partial_variant_on_backing_failure(self):
        self.store.cpu_bytes = 1
        before = self.store.describe()
        with patch.object(self.store, "_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.publish(1)
        self.assertEqual(self.store.describe(), before)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_selective_and_r1_are_not_canonical(self):
        for origin in ("selective_reuse", "r1_reuse", "segment_dense_in_partial_request"):
            with self.assertRaises(ValueError):
                self.store.publish_exact_dense(identity(1), layers=self.layers, selection_states={1: self.layers[1][0]},
                    metadata={"valid": True}, request_epoch=1, whole_request_origin=origin, materialization_reason="content_miss")

    def test_selection_is_separate_and_missing_state_never_reads_kv(self):
        self.store.cpu_bytes = 1
        row = self.publish(1)
        obj = self.store.objects[row.source_variant_id]
        Path(obj.kv_path).write_bytes(b"damaged full KV")
        self.assertTrue(torch.equal(self.store.read_selection(row.source_variant_id, 1), self.layers[1][0]))
        with self.assertRaises(KeyError):
            self.store.read_selection(row.source_variant_id, 9)

    def test_ssd_corruption_blocks_qualification_restore(self):
        self.store.cpu_bytes = 1
        row = self.publish(1)
        snapshot = self.store.snapshot()
        Path(self.store.objects[row.source_variant_id].kv_path).write_bytes(b"corrupt")
        with self.assertRaises(RuntimeError):
            self.store.restore(snapshot)

    def test_cpu_pressure_demotes_then_ssd_pressure_deletes_lru(self):
        first = self.publish(1, "a")
        self.expire("a")
        first_epoch = self.pool._get("model", "a", first.source_variant_id).last_request_use_epoch
        self.store.cpu_bytes = self.store.objects[first.source_variant_id].size_bytes + 10
        second = self.publish(2, "b")
        self.assertIs(self.store.objects[first.source_variant_id].tier, KVLocation.SSD)
        self.assertEqual(self.pool._get("model", "a", first.source_variant_id).last_request_use_epoch, first_epoch)
        self.expire("b")
        self.store.ssd_bytes = self.store.objects[first.source_variant_id].size_bytes + 100
        self.publish(3, "c")
        self.assertNotIn(first.source_variant_id, self.store.objects)
        self.assertIs(self.store.objects[second.source_variant_id].tier, KVLocation.SSD)

    def test_snapshot_restores_grace_lru_and_physical_placement(self):
        row = self.publish(1)
        snapshot = self.store.snapshot()
        self.expire()
        self.store.promote_request_use(row.source_variant_id)
        self.store.restore(snapshot)
        self.assertEqual(self.store.describe(), snapshot["state"])

    def test_unrelated_content_does_not_invalidate_replacement(self):
        self.publish(1)
        self.publish(2)
        self.expire()
        plan = self.pool.replacement_transaction("model", "c")
        self.publish(4, "unrelated")
        self.publish(3, replacement_transaction=plan)


class MeasuredCostLookup(unittest.TestCase):
    def setUp(self):
        self.provenance = {k: k for k in ("model", "code", "patch", "gpu", "config", "runtime_profile", "timing_scope")}
        self.shape = RequestExecutionShape(8, 2, 3, 1, {"a": (2, 3), "b": (4, 5)},
                    {"a": {2: (2,), 3: (2,)}, "b": {2: (4,), 3: (4,)}}, {}, {}, {"request": "matched"})

    def context(self, active):
        return JointTimelineContext(("a", "b"), tuple(active), tuple(s for s in ("a", "b") if s not in active),
                                    (), {s: 2 for s in active}, "caller-stale-mask", "scheduler")

    def estimator(self, values):
        empty = ProfiledJointTimelineEstimator(provenance=self.provenance, shape=self.shape,
                    measurements=[], measurement_digest="e" * 64, allow_test_measurements=True)
        rows = []
        for active, value in values:
            row = {"query": empty.query(self.context(active)), "provenance": self.provenance,
                   "origin": "cpu_test_only", "fake_timing": True, "warmup_excluded": True,
                   "outlier_policy": "none", "joint_future_wall_ms_samples": [value]}
            rows.append({**row, "row_sha256": digest_json(row)})
        return ProfiledJointTimelineEstimator(provenance=self.provenance, shape=self.shape,
                    measurements=rows, measurement_digest="e" * 64, allow_test_measurements=True)

    def test_unsupported_is_not_zero(self):
        estimator = self.estimator([])
        result = estimator.lookup(self.context(("a",)))
        self.assertEqual(result.status, "UNSUPPORTED")
        self.assertIsNone(result.estimate)
        with self.assertRaises(UnsupportedTimelineCost):
            estimator.estimate(self.context(("a",)))

    def test_masks_rebuilt_and_unresolved_dense_remains(self):
        estimator = self.estimator([])
        both, partial = estimator.query(self.context(("a", "b"))), estimator.query(self.context(("a",)))
        self.assertNotEqual(both["union_mask_digest"], partial["union_mask_digest"])
        self.assertEqual(partial["layer_active_rows"]["2"], 5)
        self.assertEqual(partial["dense"], ["b"])

    def test_partial_pruning_uses_new_joint_measurements(self):
        estimator = self.estimator([(("a", "b"), 90), (("a",), 60), (("b",), 85), ((), 100)])
        snapshot = PlannerSnapshot(1, 1, "scheduler", 1, "profile")
        result = RefinedJointPlannerV6(estimator).plan_subset(inventory_segment_ids=("a", "b"),
            eligible_ready_segment_ids=("a", "b"), committed_segment_ids=(), actual_boundary_by_segment={"a": 2, "b": 2},
            actual_sunk_ms=10, dense_reference_total_ms=100, snapshot=snapshot, current_snapshot=snapshot, union_mask_digest="stale")
        self.assertEqual(result.accepted_ready_segment_ids, ("a",))
        self.assertEqual(result.request_total_ms, 70)

    def test_pruning_memo_is_limited_to_one_snapshot_call(self):
        estimator = self.estimator([(("a", "b"), 90), (("a",), 60), (("b",), 85), ((), 100)])
        planner = RefinedJointPlannerV6(estimator)
        snapshot = PlannerSnapshot(1, 1, "scheduler", 1, "profile")
        kwargs = dict(inventory_segment_ids=("a", "b"), eligible_ready_segment_ids=("a", "b"),
            committed_segment_ids=(), actual_boundary_by_segment={"a": 2, "b": 2}, actual_sunk_ms=10,
            dense_reference_total_ms=100, snapshot=snapshot, current_snapshot=snapshot, union_mask_digest="mask")
        with patch.object(estimator, "estimate", wraps=estimator.estimate) as estimate:
            self.assertEqual(planner.plan_subset(**kwargs).accepted_ready_segment_ids, ("a",))
            self.assertEqual(estimate.call_count, 3)
            planner.plan_subset(**kwargs)
            self.assertEqual(estimate.call_count, 6)

    def test_cpu_samples_cannot_supply_real_admission(self):
        estimator = self.estimator([(("a",), 10)])
        row = next(iter(estimator.rows.values()))
        with self.assertRaises(ValueError):
            ProfiledJointTimelineEstimator(provenance=self.provenance, shape=self.shape,
                measurements=[{**row, "row_sha256": digest_json(row)}], measurement_digest="e" * 64)


if __name__ == "__main__":
    unittest.main()
