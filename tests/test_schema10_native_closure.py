from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from probekv.v8_schema10_inventory import native_segment_inventory
from probekv.v8_schema10_layer_storage import LayerFile, write_layer_replica
from probekv.v8_schema10_staging import PhysicalPinnedStagingPool
from probekv.v8_schema10_cost_provider import MeasurementKey, EXECUTION_SHAPE_KEY, ProfiledJointTimelineEstimator, RequestExecutionShape
from probekv.v8_schema10_native_adapter import dispatch_depths
from probekv.v8_schema10_prefix_shadow import PrefixShadowStore
from probekv.v8_schema10_canonical import request_occurrences
from probekv.v8_schema10_qa import answer_evidence, validate_answer_evidence
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema6_planner import JointTimelineContext
from probekv.model_adapters import SCHEMA6_MODEL_SPECS


class NativeOwnershipTests(unittest.TestCase):
    def segments(self):
        return {str(i): {"positions": list(range(4*i, 4*i+4)), "token_ids": list(range(4*i, 4*i+4))} for i in range(3)}

    def test_partial_prefix_tail_stays_dense_without_rechunking(self):
        rows = native_segment_inventory(self.segments(), prompt_tokens=13, cached_prefix_tokens=6)
        self.assertEqual([r.disposition for r in rows.values()], ["PREFIX_EXACT", "DENSE_PREFIX_TAIL", "NONPREFIX_CANDIDATE"])
        self.assertEqual(rows["1"].canonical_positions, (4, 5, 6, 7))
        self.assertEqual(rows["1"].remaining_positions, (6, 7))
        self.assertFalse(rows["1"].comparison_eligible)

    def test_all_prefix_and_empty_inventory(self):
        rows = native_segment_inventory(self.segments(), prompt_tokens=13, cached_prefix_tokens=12)
        self.assertFalse(any(r.comparison_eligible for r in rows.values()))
        self.assertEqual(native_segment_inventory({}, prompt_tokens=13, cached_prefix_tokens=12), {})

    def test_overlap_and_out_of_range_fail(self):
        rows = self.segments()
        rows["1"] = rows["0"]
        with self.assertRaises(ValueError):
            native_segment_inventory(rows, prompt_tokens=13, cached_prefix_tokens=0)

    def test_fast_and_legacy_depths_remain_independent(self):
        for spec in SCHEMA6_MODEL_SPECS.values():
            self.assertEqual(dispatch_depths("d1_only", spec), (1,))
            self.assertEqual(dispatch_depths("d1_d2_rescue", spec), (1, 2))
            self.assertEqual(dispatch_depths("legacy_multicheckpoint", spec), spec.checkpoints)

    def test_cfo_occurrences_cover_gaps_and_duplicates(self):
        q = {"token_ids": [1,2,3,1,2,4], "segments": [
            {"segment_id": "a", "positions": [0,1], "token_ids": [1,2], "content_key": "c"},
            {"segment_id": "b", "positions": [3,4], "token_ids": [1,2], "content_key": "c"}]}
        occurrences, targets, ids = request_occurrences(q)
        self.assertEqual(len(ids), 6)
        self.assertNotEqual(targets["a"].match_id, targets["b"].match_id)
        self.assertEqual(sum(o.token_count for o in occurrences), 6)


class PhysicalStorageTests(unittest.TestCase):
    def test_layer_read_does_not_deserialize_full_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "layers.pt"
            layers = tuple((torch.full((4, 2, 3), i, dtype=torch.bfloat16),
                            torch.full((4, 2, 3), -i, dtype=torch.bfloat16)) for i in range(4))
            with path.open("wb") as stream:
                write_layer_replica(stream, layers)
            reader = LayerFile(path)
            with patch.object(torch, "load", side_effect=AssertionError("full read")):
                pair = reader[2]
                self.assertTrue(torch.equal(pair[0], layers[2][0]))
                buffers = (torch.empty_like(pair[0]), torch.empty_like(pair[1]))
                reader.read_into(3, buffers)
                self.assertTrue(torch.equal(buffers[1], layers[3][1]))
            self.assertEqual(reader.full_kv_bytes, 4 * 2 * 4 * 2 * 3 * 2)
            with path.open("ab") as stream:
                stream.write(b"bad")
            with self.assertRaises(RuntimeError):
                LayerFile(path)

    def test_pinned_slot_waits_for_copy_and_budget_is_global(self):
        from types import SimpleNamespace
        pool = PhysicalPinnedStagingPool(192, pin_memory=False)
        a, b = pool.acquire((4, 2, 3)), pool.acquire((4, 2, 3))
        event = SimpleNamespace(query=lambda: False)
        pool.release_after(a, event)
        with self.assertRaises(MemoryError):
            pool.acquire((4, 2, 3))
        with self.assertRaises(RuntimeError):
            pool.clear()
        event.query = lambda: True
        c = pool.acquire((4, 2, 3))
        self.assertIs(c, a)
        pool.release_after(b, None)
        pool.release_after(c, None)
        self.assertEqual(pool.peak_bytes, 192)
        pool.clear()

    def test_shadow_strict_origin_geometry_and_bounded_admission(self):
        store = PrefixShadowStore(model_signature="m", num_layers=2, kv_heads=1, head_dim=2, capacity_bytes=64)
        kv = tuple((torch.zeros((4,1,2), dtype=torch.bfloat16), torch.ones((4,1,2), dtype=torch.bfloat16)) for _ in range(2))
        with self.assertRaises(ValueError):
            store.publish(range(4), kv, origin="native_prefix_dense_remaining")
        self.assertTrue(store.publish(range(4), kv, origin="exact_dense_full_prefill"))
        self.assertEqual(store.resident_bytes, 64)
        with self.assertRaises(ValueError):
            store.publish(range(3), kv, origin="exact_dense_full_prefill")

    def test_retained_prefix_shadows_remain_in_the_same_host_budget(self):
        store = PrefixShadowStore(model_signature="m", num_layers=1, kv_heads=1, head_dim=1, capacity_bytes=16)
        kv = ((torch.zeros((4,1,1), dtype=torch.bfloat16), torch.ones((4,1,1), dtype=torch.bfloat16)),)
        store.publish([1,2,3,4], kv, origin="exact_dense_full_prefill")
        store.retain("paired-snapshot")
        store.clear()
        self.assertEqual(store.resident_bytes, 16)
        self.assertFalse(store.publish([4,3,2,1], kv, origin="exact_dense_full_prefill"))
        store.restore_retained("paired-snapshot")
        self.assertEqual(store.descriptor()["rows"][0]["tokens"], [1,2,3,4])
        store.release_retained("paired-snapshot")
        store.clear()
        self.assertEqual(store.resident_bytes, 0)

    def test_pinned_shadow_allocation_failure_is_not_silent_pageable_fallback(self):
        store = PrefixShadowStore(model_signature="m", num_layers=1, kv_heads=1,
                                  head_dim=1, capacity_bytes=16, pin_memory=True)
        kv = ((torch.zeros((4,1,1), dtype=torch.bfloat16), torch.ones((4,1,1), dtype=torch.bfloat16)),)
        with patch.object(torch, "empty", side_effect=RuntimeError("pinned allocation failed")) as allocate:
            with self.assertRaisesRegex(RuntimeError, "pinned allocation"):
                store.publish([1,2,3,4], kv, origin="exact_dense_full_prefill")
        self.assertTrue(allocate.call_args.kwargs["pin_memory"])
        self.assertFalse(store.entries)

    def test_real_pinned_copy_is_not_required_for_cpu_storage_tests(self):
        pool = PhysicalPinnedStagingPool(32, pin_memory=False)
        slot = pool.acquire((4,1,1))
        self.assertEqual(slot.pair[0].device.type, "cpu")
        self.assertFalse(slot.pair[0].is_pinned())
        pool.release_after(slot, None)


class NativeBoundaryAndCaptureTests(unittest.TestCase):
    def test_current_full_prefill_export_does_not_run_another_forward(self):
        from types import SimpleNamespace as NS
        from unittest.mock import Mock
        from probekv.v8_schema10_canonical import export_original_full_prefill
        from probekv.v8_cfo import build_source_cfo_metadata
        raw = tuple((torch.full((4,1,1), i, dtype=torch.bfloat16),
                     torch.full((4,1,1), i + 1, dtype=torch.bfloat16)) for i in range(3))
        class Collector:
            def finalize(self, *, prefix_occurrences, target_occurrence):
                metadata = build_source_cfo_metadata(historical_prefix_chunk_occurrences=prefix_occurrences,
                    inter_mass_by_layer_and_occurrence=[{o.match_id: 0. for o in prefix_occurrences} for _ in range(3)],
                    intra_mass_by_layer=[1.,1.,1.], target_token_count=target_occurrence.token_count)
                return metadata, {"captured_layers": 3}
        adapter = NS(inner=NS(layers=[NS(self_attn=NS(hack_kv=p, num_kv_heads=1, head_dim=1)) for p in raw]),
            spec=NS(checkpoints=(1,2)), provenance={"tokenizer_hash": "t", "runtime_compatibility": "r"},
            shadows=PrefixShadowStore(model_signature="m", num_layers=3, kv_heads=1, head_dim=1, capacity_bytes=48),
            outer=Mock(side_effect=AssertionError("extra forward")))
        q = {"token_ids": [3,4,5,6], "segments": [{"segment_id": "s", "positions": [1,2],
             "token_ids": [4,5], "content_key": "c"}]}
        exported = export_original_full_prefill(adapter, q, Collector())["s"]
        adapter.outer.assert_not_called()
        self.assertEqual(exported["capture_audit"]["extra_full_prefill_count"], 0)
        self.assertEqual(exported["source_metadata"]["token_ids"], [4,5])
        self.assertTrue(torch.equal(exported["selection_states"][1], raw[1][0][1:3]))
        raw[1][0].zero_()
        self.assertEqual(exported["selection_states"][1].sum().item(), 2.)

    def test_native_sampling_suffix_is_never_a_repair_candidate(self):
        from probekv.v8_schema10_inventory import mandatory_suffix_positions
        q = {"token_ids": [1,2,3], "segments": [{"positions": [0,1]}]}
        self.assertEqual(mandatory_suffix_positions(q), (2,))
        q["segments"][0]["positions"] = [1,2]
        with self.assertRaises(ValueError):
            mandatory_suffix_positions(q)

    def test_prefix_or_selective_origin_cannot_export_a_capture(self):
        from probekv.v8_schema10_native_adapter import NativeRequestContext
        context = NativeRequestContext.__new__(NativeRequestContext)
        context.canonical_exports = {"s": "captured-only-if-exact"}
        context.committed, context.cached_prefix_tokens = {}, 16
        self.assertEqual(context.export_exact_dense(), {})
        context.committed, context.cached_prefix_tokens = {"s": 2}, 0
        self.assertEqual(context.export_exact_dense(), {})
        context.committed = {}
        self.assertEqual(context.export_exact_dense(), context.canonical_exports)

    def test_fast_and_legacy_constructors_reject_substitution_before_model_access(self):
        from probekv.v8_schema10_native_adapter import FastNativeOnlineAdapter, LegacyNativeOnlineAdapter
        with self.assertRaises(ValueError):
            FastNativeOnlineAdapter(selection_path="legacy_multicheckpoint")
        with self.assertRaises(ValueError):
            LegacyNativeOnlineAdapter(selection_path="d1_only")

    def test_oracle_rejects_missing_qa_before_accessing_backend(self):
        from probekv.v8_schema10_native_oracle import run_native_source_oracle
        with self.assertRaises(ValueError):
            run_native_source_oracle(None, request={}, dispatch={}, segment_id="s", source_ids=["v"],
                first_reuse_layer=2, repair_ratio=.15, chosen_source_id="v", max_answer_f1_drop=.01)

    def test_installed_runtime_audit_requires_the_actual_complete_files(self):
        from probekv.v8_schema10_native_factory import verify_installed_runtime_sources
        with self.assertRaises(ValueError):
            verify_installed_runtime_sources({"installed_runtime_source_files_sha256": {}}, ".")


class StableCostTests(unittest.TestCase):
    def test_identity_not_permitted_in_shape_key(self):
        with self.assertRaises(ValueError):
            MeasurementKey("future", {"nested": {"source_id": "a"}}).query()

    def test_new_source_and_snapshot_same_shape_can_reuse_measurement(self):
        shape = RequestExecutionShape(12, 0, 3, 1, {"s": (2,3,4)},
            {"s": {2: (2,), 3: (2,)}}, {},
            {"s": {"source_id": "old", "tier": "pinned_cpu", "bytes": 36, "ready_layers": [2,3], "layout": "bf16"}},
            {"request": "old", "sampling": {"temperature": 0}})
        provenance = {k: k for k in ("model", "code", "patch", "gpu", "config", "runtime_profile", "timing_scope")}
        ctx = JointTimelineContext(("s",), ("s",), (), (), {"s": 2}, "a", "snapshot-old")
        estimator = ProfiledJointTimelineEstimator(provenance=provenance, shape=shape, measurements=[],
            measurement_digest="d", allow_test_measurements=True, key_contract=EXECUTION_SHAPE_KEY)
        first = estimator.query(ctx)
        physical = {"s": {**shape.source_state_by_segment["s"], "source_id": "new"}}
        estimator.shape = replace(shape, source_state_by_segment=physical, dense_reference_identity={"request": "new", "sampling": {"temperature": 0}})
        self.assertEqual(first, estimator.query(replace(ctx, scheduler_state_id="snapshot-new")))
        self.assertNotEqual(first, estimator.query(replace(ctx, reuse_segment_ids=(), dense_fallback_segment_ids=("s",), boundary_by_segment={})))

    def test_joint_query_audit_preserves_supported_and_unsupported_queries(self):
        shape = RequestExecutionShape(12, 0, 3, 1, {"s": (2,3,4)},
            {"s": {2: (2,), 3: (2,)}}, {},
            {"s": {"tier": "pinned_cpu", "bytes": 36, "ready_layers": [2,3], "layout": "bf16"}},
            {"sampling": {"temperature": 0}})
        provenance = {k: k for k in ("model", "code", "patch", "gpu", "config", "runtime_profile", "timing_scope")}
        ctx = JointTimelineContext(("s",), ("s",), (), (), {"s": 2}, "a", "snapshot")
        query = ProfiledJointTimelineEstimator(provenance=provenance, shape=shape, measurements=[],
            measurement_digest="d", allow_test_measurements=True,
            key_contract=EXECUTION_SHAPE_KEY).query(ctx)
        row = {"query": query, "provenance": provenance, "origin": "real_cuda_execution",
               "fake_timing": False, "warmup_excluded": True, "outlier_policy": "none",
               "joint_future_wall_ms_samples": [1.0]}
        row["row_sha256"] = digest_json(row)
        audit = []
        estimator = ProfiledJointTimelineEstimator(provenance=provenance, shape=shape, measurements=[row],
            measurement_digest="d", allow_test_measurements=True, key_contract=EXECUTION_SHAPE_KEY,
            query_audit=audit)
        self.assertEqual(estimator.lookup(ctx).status, "SUPPORTED")
        self.assertEqual(audit[-1]["query"], query)
        self.assertEqual(audit[-1]["status"], "SUPPORTED")
        unsupported = replace(ctx, reuse_segment_ids=(), dense_fallback_segment_ids=("s",), boundary_by_segment={})
        self.assertEqual(estimator.lookup(unsupported).status, "UNSUPPORTED")
        self.assertEqual(audit[-1]["query"], estimator.query(unsupported))
        self.assertEqual(audit[-1]["reason"], "no_exact_joint_measurement")
        # Copy progress remains a measured dimension: neither faster nor
        # slower readiness is silently treated as the same measured cell.
        for layers in ([1, 2, 3], [2]):
            estimator.shape = replace(shape, source_state_by_segment={
                "s": {**shape.source_state_by_segment["s"], "ready_layers": layers}})
            self.assertEqual(estimator.lookup(ctx).status, "UNSUPPORTED")
        rebound = estimator.for_shape(shape)
        self.assertEqual(rebound.lookup(ctx).status, "SUPPORTED")
        self.assertIs(rebound.rows, estimator.rows)
        self.assertIsNot(rebound.queries, estimator.queries)
        self.assertNotEqual(estimator.shape, rebound.shape)


class QAClosureTests(unittest.TestCase):
    def test_missing_reference_is_null_not_zero_violation(self):
        from types import SimpleNamespace
        request = {"token_ids": [1,2]}
        row = answer_evidence([3], tokenizer=SimpleNamespace(decode=lambda *a, **k: "answer"), request=request)
        self.assertIsNone(row["quality_passed"])
        self.assertIsNone(validate_answer_evidence(row["qa_evidence"], request=request))

    def test_scores_recomputed_and_corrupt_evidence_rejected(self):
        from types import SimpleNamespace
        request = {"token_ids": [1,2], "answers": ["Paris"]}
        row = answer_evidence([3], tokenizer=SimpleNamespace(decode=lambda *a, **k: "Paris"), request=request)["qa_evidence"]
        self.assertEqual(validate_answer_evidence(row, request=request), 1)
        row["answer"] = "Berlin"
        with self.assertRaises(ValueError):
            validate_answer_evidence(row, request=request)

    def test_resigned_manual_quality_pass_without_dense_evidence_fails(self):
        from types import SimpleNamespace
        request = {"token_ids": [1], "answers": ["yes"]}
        row = answer_evidence([2], tokenizer=SimpleNamespace(decode=lambda *a, **k: "yes"), request=request)["qa_evidence"]
        row["quality_passed"] = True
        row["evidence_sha256"] = digest_json({k:v for k,v in row.items() if k != "evidence_sha256"})
        with self.assertRaises(ValueError):
            validate_answer_evidence(row, request=request)


class FailClosedNativeGates(unittest.TestCase):
    def test_factory_import_does_not_start_cuda_and_missing_inputs_fail(self):
        from probekv.v8_schema10_native_factory import create_native_backend
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("must validate input first")):
            with self.assertRaises(ValueError):
                create_native_backend({"manifest_sha256": "wrong"})

    def test_unmeasured_runtime_cannot_start_online_trace(self):
        from probekv.v8_schema10_native_factory import NativeExperimentBackend, UnmeasuredCosts
        backend = object.__new__(NativeExperimentBackend)
        backend.costs = UnmeasuredCosts()
        with self.assertRaisesRegex(RuntimeError, "measurement-only"):
            backend.execute({}, {}, arrival_ns=time.perf_counter_ns())

    def test_collector_refuses_cpu_fake_timings(self):
        from probekv.v8_schema10_cost_collection import CudaCostCollector
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                CudaCostCollector(provenance={})

    def test_pass_flag_does_not_unlock_correctness(self):
        from probekv.v8_schema10_native_validation import validate_correctness_observation
        for category in ("native_prefix", "k_hook", "r1", "source_digest", "absolute_mask"):
            with self.assertRaises(ValueError):
                validate_correctness_observation(category, {"passed": True, "origin": "real_cuda_execution", "fake_timing": False})

    def test_raw_r1_and_mask_values_are_validated(self):
        from probekv.v8_schema10_native_validation import validate_correctness_observation
        row = {"origin": "real_cuda_execution", "fake_timing": False, "dense_token_ids": [1]*32,
               "reuse_token_ids": [1]*32, "logit_relative_l2": 1e-5, "logit_token_count": 32}
        self.assertTrue(validate_correctness_observation("r1", row))
        row["logit_relative_l2"] = float("nan")
        with self.assertRaises(ValueError):
            validate_correctness_observation("r1", row)


if __name__ == "__main__":
    unittest.main()
