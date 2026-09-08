"""CPU integration: real tensor/file operations, never GPU evidence."""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from test_schema10_transactional_backend import profile
from probekv.v7_contracts import SourceVariantIdentity
from probekv.v8_cfo import build_source_cfo_metadata
from probekv.v8_schema10_pool import Schema10SourcePool
from probekv.v8_schema10_storage import TensorFileSourceStore, file_digest
from probekv.v8_schema10_profile import PreparationPolicyProfile
from probekv.v8_schema10_selector import Schema10CheckpointSelector
from probekv.v8_schema10_online_backend import Schema10OnlineExperimentBackend
from probekv.v8_schema10_event_log import OnlineEventLog, aggregate_online_events
from probekv.v8_schema10_cost_provider import ProfiledJointTimelineEstimator, RequestExecutionShape
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_experiments import run_causal_capacity_traces, run_gate1_pairs, _online, run_serving_trace
from probekv.v8_schema8_planner import Gate1LocalPlan, Gate1MarginalLowerBound
from probekv.v8_schema6_hbm import UnifiedHBMReservationManager
from probekv.v8_schema6_contracts import PlannerSnapshot
from probekv.v8_schema6_planner import JointTimelineContext


class TinyLiveContext:
    evidence_origin = "cpu_integration_test"
    cached_prefix_tokens = 0

    def __init__(self, request):
        self.request = request
        self.segments = {s["segment_id"]: s for s in request["segments"]}
        self.depth, self.committed, self.prepared = 0, (), {}
        tokens = torch.tensor(request["token_ids"], dtype=torch.float32).reshape(-1, 1, 1)
        self.layers = tuple(((tokens + l + 1).to(torch.bfloat16),
                             (tokens * .5 + l).to(torch.bfloat16)) for l in range(3))

    def assert_dispatch(self, depths):
        if depths not in {(1,), (1, 2)}:
            raise RuntimeError("tiny FAST adapter does not implement legacy")

    def advance_to_depth(self, depth):
        if depth < self.depth:
            raise RuntimeError("cannot rewind a request")
        self.depth = depth

    def synchronize(self):
        pass

    def observe_current_k(self, sid, depth):
        return self.layers[depth][0][list(self.segments[sid]["positions"])]

    def prepare_winner(self, sid, source_id, layers, reservation):
        assert not reservation.released
        self.prepared[sid] = tuple((k.clone(), v.clone()) for k, v in layers)
        return self.prepared[sid]

    def finish_selection(self, frozen, prepared):
        self.frozen = frozen

    def ready_for_final_commit(self, prepared):
        return {sid: self.depth + 1 for sid in prepared}, "recomputed-by-provider"

    def planner_snapshot(self, epoch):
        return PlannerSnapshot(1, 1, "scheduler", epoch, "cost")

    def commit_reuse(self, decision):
        assert decision.request_total_ms <= .8 * decision.dense_reference_total_ms or not decision.accepted_ready_segment_ids
        self.committed = decision.accepted_ready_segment_ids

    def finish(self, on_first_token):
        logits = self.layers[-1][0].flatten()
        token = int(logits.argmax())
        on_first_token()
        return {"token_ids": [token], "quality_passed": True,
                "whole_request_origin": "selective_reuse" if self.committed else "exact_dense_full_prefill"}

    def export_exact_dense(self):
        assert not self.committed
        return {sid: {"layers": tuple((k[list(s["positions"])], v[list(s["positions"])]) for k, v in self.layers),
                      "selection_states": {d: self.observe_current_k(sid, d).clone() for d in (1, 2)}}
                for sid, s in self.segments.items()}

    def materialization_metadata(self):
        result = {}
        for sid, s in self.segments.items():
            cfo = build_source_cfo_metadata(historical_prefix_chunk_occurrences=(),
                inter_mass_by_layer_and_occurrence=[{}, {}, {}], intra_mass_by_layer=[1., 1., 1.], target_token_count=len(s["positions"]))
            result[sid] = {"identity": SourceVariantIdentity(s["content_key"], "prefix", "positions",
                          self.request["request_id"] + sid, "model"), "estimated_materialization_ms": 1.,
                          "source_metadata": {"cfo": asdict(cfo), "token_ids": s["token_ids"],
                            "tokenizer_hash": "tokenizer", "runtime_compatibility": "runtime"}}
        return result


class TinyAdapter:
    capabilities = {"native_prefix_block_allocator": True, "production_dispatch": "d1_d2_rescue"}
    def reset(self): pass
    def snapshot(self): return {"test_allocator": "quiescent", "prefix": "cold"}
    def restore(self, value): assert value == self.snapshot()
    @contextmanager
    def open_request(self, request, *, arrival_ns):
        yield TinyLiveContext(request)


class TinyCosts:
    missing_joint = False
    def dense_reference(self, context): return 1000.
    def comparison_ms(self, *args): return 1.
    def preparation(self, *args): return {"resource_admitted": True}
    def candidate_future_ms(self, *args): return 10.
    def gate1(self, context, sid, source_id, depth):
        return Gate1LocalPlan(source_id, depth, depth, depth + 1, 0., Gate1MarginalLowerBound(1., 1., 1.), 100.)
    def joint_estimator(self, context):
        provenance = {k: k for k in ("model", "code", "patch", "gpu", "config", "runtime_profile", "timing_scope")}
        shape = RequestExecutionShape(len(context.request["token_ids"]), 0, 3, context.depth,
            {sid: tuple(s["positions"]) for sid, s in context.segments.items()},
            {sid: {l: tuple(s["positions"][:1]) for l in range(context.depth + 1, 4)} for sid, s in context.segments.items()}, {}, {}, {"request": "test"})
        empty = ProfiledJointTimelineEstimator(provenance=provenance, shape=shape, measurements=[],
                                               measurement_digest="d" * 64, allow_test_measurements=True)
        rows = []
        if not self.missing_joint:
            ids = tuple(context.segments)
            active = tuple(context.prepared)
            ctx = JointTimelineContext(ids, active, tuple(s for s in ids if s not in active), (),
                {sid: context.depth + 1 for sid in active}, "mask", "scheduler")
            row = {"query": empty.query(ctx), "provenance": provenance, "origin": "cpu_test_only",
                   "fake_timing": True, "warmup_excluded": True, "outlier_policy": "none", "joint_future_wall_ms_samples": [10.]}
            rows.append({**row, "row_sha256": digest_json(row)})
        return ProfiledJointTimelineEstimator(provenance=provenance, shape=shape, measurements=rows,
                                              measurement_digest="d" * 64, allow_test_measurements=True)


def request(i, segments=1):
    return {"request_id": "q" + str(i), "request_epoch": i, "token_ids": list(range(1, 4 * segments + 1)),
            "segments": [{"segment_id": "s" + str(s), "content_key": "c" + str(s),
                          "token_ids": list(range(4 * s + 1, 4 * s + 5)), "positions": list(range(4 * s, 4 * s + 4)),
                          "prefix_occurrences": []} for s in range(segments)]}


class OnlineIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        def factory(k, budget):
            pool = Schema10SourcePool(profile=profile(k))
            pool.activate_namespace("model")
            return TensorFileSourceStore(pool, Path(self.tmp.name) / str(time.perf_counter_ns()),
                cpu_bytes=budget // 2, ssd_bytes=budget - budget // 2, pin_cpu=False)
        def selector(dispatch, capacity):
            return Schema10CheckpointSelector(variant_profile=profile(capacity),
                preparation_profile=PreparationPolicyProfile(code_commit="commit", model_id="model", runtime_policy="dense_selection_barrier",
                    gate1_mode=dispatch.get("gate1_mode", "explicit_barrier")), strong_margin=.6, stable_margin=.3, residual_band_relative_tolerance=.1)
        self.costs = TinyCosts()
        self.backend = Schema10OnlineExperimentBackend(store_factory=factory, adapters={"d1_d2_rescue": TinyAdapter()},
            selector_factory=selector, cost_provider=self.costs, hbm_manager=UnifiedHBMReservationManager(allocator_capacity_bytes=2**30, safety_bytes=0),
            provenance={"model_signature": "model", "tokenizer_hash": "tokenizer", "runtime_compatibility": "runtime", "code_commit": "commit"})
        self.backend.reset(capacity=16, global_byte_budget=2000000)
        self.dispatch = {"selection_path": "d1_d2_rescue", "gate1_mode": "explicit_barrier"}

    def execute(self, i, segments=1):
        q = request(i, segments)
        row = self.backend.execute(q, self.dispatch, arrival_ns=time.perf_counter_ns())
        self.backend.finalize_request(q, row)
        _online(row)
        return row

    def binding(self):
        value = {k: k for k in ("code_commit", "patch_sha256", "model_signature", "config_sha256", "initial_state_sha256", "runtime_measurement_sha256", "dispatch")}
        return {**value, "code_commit": "commit", "model_signature": "model"}

    def test_dense_publishes_after_coverage_then_future_lookup_reuses(self):
        first = self.execute(1)
        self.assertEqual(first["coverage_event"]["visible_variant_creation_epochs"], {})
        self.assertEqual(len(self.backend.store.objects), 1)
        second = self.execute(2)
        self.assertTrue(second["committed_source_variant_ids"])
        self.assertEqual(len(self.backend.store.objects), 1, "selective result must not publish a new Variant")
        self.assertFalse(second["gpu_runtime_qualified"])

    def test_all_segment_counts_have_execution_ownership(self):
        for n in (1, 2, 5, 10, 37):
            self.backend.reset(capacity=16, global_byte_budget=2000000)
            self.execute(1, n)
            hit = self.execute(2, n)
            self.assertEqual(len(hit["committed_source_variant_ids"]), n)

    def test_missing_cost_falls_back_without_fabricating_total(self):
        self.execute(1)
        self.costs.missing_joint = True
        result = self.execute(2)
        self.assertEqual(result["committed_source_variant_ids"], [])
        self.assertIsNone(result["final_predicted_request_total_ms"])
        self.assertTrue(result["selected_source_variant_ids"])

    def test_legacy_is_not_silently_executed_as_fast(self):
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            self.backend.execute(request(1), {"selection_path": "legacy_multicheckpoint"}, arrival_ns=time.perf_counter_ns())

    def test_queue_time_is_included(self):
        arrival = time.perf_counter_ns() - 10_000_000
        result = self.backend.execute(request(1), self.dispatch, arrival_ns=arrival)
        self.assertGreaterEqual(result["queue_ms"], 10.)
        self.assertAlmostEqual(result["request_ttft_ms"], (result["first_token_ns"] - arrival) / 1e6)

    def test_each_k_has_its_own_actual_causal_run(self):
        result = run_causal_capacity_traces(self.backend, [request(1), request(2)], self.dispatch,
                                           global_byte_budget=2000000)
        for k, rows in result["traces"].items():
            self.assertEqual(rows[0]["visible_variant_creation_epochs"], {})
            self.assertTrue(rows[1]["committed_variant_ids"])
        self.assertEqual([r["k"] for r in result["curves"]], [1, 2, 4, 8, 16])

    def test_paired_arms_restore_runtime_lru_grace_and_hbm(self):
        self.execute(1)
        result = run_gate1_pairs(self.backend, [request(2)], self.dispatch)
        arms = result[0]["arms"]
        self.assertEqual(arms["explicit_barrier"]["initial_pool_snapshot_sha256"], arms["fused_advisory"]["initial_pool_snapshot_sha256"])
        self.assertFalse(arms["explicit_barrier"]["ssd_page_cache_controlled"])
        self.assertFalse(self.backend.store._snapshots)

    def test_missing_source_local_cost_is_normal_dense(self):
        self.execute(1)
        with patch.object(self.costs, "gate1", return_value=None):
            row = self.execute(2)
        self.assertEqual(row["final_commit_not_applicable_reason"], "selection_cost_unsupported")
        self.assertFalse(row["coverage_event"]["residual_compatibility_observed"])
        self.assertFalse(row["committed_source_variant_ids"])

    def test_missing_dense_cost_does_not_create_fake_comparison(self):
        with patch.object(self.costs, "dense_reference", return_value=None):
            row = self.execute(1)
        self.assertEqual(row["selection_events"], [])
        self.assertFalse(self.backend.store.objects)
        self.assertEqual(row["final_commit_not_applicable_reason"], "matched_dense_cost_unsupported")

    def test_partial_prefix_cannot_export_locally_dense_tail_as_canonical(self):
        with patch.object(TinyLiveContext, "cached_prefix_tokens", 2):
            row = self.execute(1)
        self.assertEqual(row["segment_ownership"]["s0"]["disposition"], "DENSE_PREFIX_TAIL")
        self.assertFalse(row["selected_source_variant_ids"])
        self.assertFalse(self.backend.store.objects)

    def test_fully_covered_prefix_never_compares_or_materializes(self):
        with patch.object(TinyLiveContext, "cached_prefix_tokens", 4):
            row = self.execute(1)
        self.assertEqual(row["segment_ownership"]["s0"]["disposition"], "PREFIX_EXACT")
        self.assertEqual(row["selection_events"], [])
        self.assertFalse(self.backend.store.objects)

    def test_stale_final_snapshot_is_dense_not_a_fake_commit(self):
        self.execute(1)
        with patch.object(TinyLiveContext, "commit_reuse", side_effect=RuntimeError("stale Planner snapshot cannot be applied")):
            row = self.execute(2)
        self.assertFalse(row["committed_source_variant_ids"])
        self.assertIsNone(row["final_predicted_request_total_ms"])
        self.assertEqual(self.backend.hbm.active_reserved_bytes, 0)

    def test_final_admission_includes_time_spent_inside_planner(self):
        from probekv.v8_schema7_planner import FinalCommitPlanner
        self.execute(1)
        real_clock = time.perf_counter_ns
        plan = FinalCommitPlanner.plan_ready_subset
        offset = [0]
        def slow_plan(planner, **kwargs):
            result = plan(planner, **kwargs)
            offset[0] += 1_000_000_000
            return result
        with patch.object(FinalCommitPlanner, "plan_ready_subset", slow_plan), \
                patch.object(time, "perf_counter_ns", side_effect=lambda: real_clock() + offset[0]):
            row = self.execute(2)
        self.assertFalse(row["committed_source_variant_ids"])
        self.assertTrue(row["selected_source_variant_ids"])
        self.assertTrue(any(e.get("reason") == "planner_elapsed_exceeds_gamma" for e in row["runtime_events"]))
        self.assertEqual(self.backend.hbm.active_reserved_bytes, 0)

    def test_transient_snapshot_progress_replans_same_winner_without_recopy(self):
        self.execute(1)
        commit = TinyLiveContext.commit_reuse
        attempts = []
        def transient(context, decision):
            attempts.append(tuple(decision.accepted_ready_segment_ids))
            if len(attempts) == 1:
                raise RuntimeError("stale Planner snapshot cannot be applied")
            return commit(context, decision)
        with patch.object(TinyLiveContext, "commit_reuse", transient):
            row = self.execute(2)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        self.assertTrue(row["committed_source_variant_ids"])
        self.assertEqual(sum(e["kind"] == "winner_preparation" for e in row["runtime_events"]), 1)
        self.assertEqual(sum(e["kind"] == "planner_snapshot_retry" for e in row["runtime_events"]), 1)

    def test_quarantined_hbm_cannot_be_reset_to_free_space(self):
        from probekv.v8_schema6_hbm import HBMReservationKind
        self.backend.hbm.reserve_batch(owner_request_id="crashed", rows=(("s", 100, HBMReservationKind.WINNER_PREFETCH),))
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            self.backend.reset(capacity=1, global_byte_budget=2000000)

    def test_freeze_lease_failure_preserves_selected_audit_but_no_prefetch(self):
        self.execute(1)
        with patch.object(self.backend.store, "leased_winner", side_effect=RuntimeError("generation changed")):
            row = self.execute(2)
        self.assertEqual(row["final_commit_not_applicable_reason"], "all_source_freezes_failed")
        self.assertTrue(row["selected_source_variant_ids"])
        self.assertFalse(row["committed_source_variant_ids"])
        self.assertFalse(any(e["kind"] == "winner_preparation" for e in row["runtime_events"]))

    def test_missing_preparation_table_retains_source_and_falls_back_dense(self):
        from probekv.v8_schema10_cost_provider import UnsupportedTimelineCost
        self.execute(1)
        with patch.object(self.costs, "preparation", side_effect=UnsupportedTimelineCost("missing prep table")):
            row = self.execute(2)
        self.assertTrue(row["selected_source_variant_ids"])
        self.assertFalse(row["committed_source_variant_ids"])
        self.assertEqual(self.backend.hbm.active_reserved_bytes, 0)

    def test_regular_trace_does_not_retain_backing_snapshots(self):
        self.execute(1)
        self.execute(2)
        self.assertFalse(self.backend.store._snapshots)

    def test_failed_transfer_fences_and_releases_before_next_request_is_blocked(self):
        self.execute(1)
        with patch.object(TinyLiveContext, "prepare_winner", side_effect=RuntimeError("copy failure")):
            with self.assertRaisesRegex(RuntimeError, "copy failure"):
                self.backend.execute(request(2), self.dispatch, arrival_ns=time.perf_counter_ns())
        self.assertEqual(self.backend.hbm.active_reserved_bytes, 0)
        self.assertFalse(any(self.backend.store.pool.logical_lease_counts.values()))
        self.assertFalse(any(p.busy for v in self.backend.store.pool._variants.values() for p in v.replicas.values()))
        with self.assertRaisesRegex(RuntimeError, "failed runtime"):
            self.backend.execute(request(3), self.dispatch, arrival_ns=time.perf_counter_ns())

    def test_throughput_includes_final_materialization_service_end(self):
        q = {**request(1), "arrival_offset_ms": 0.}
        result = run_serving_trace(self.backend, [q], self.dispatch, concurrency=1, slo_ttft_ms=1000)
        self.assertGreaterEqual(self.backend.service_completion_ns(q["request_id"]), result["rows"][0]["completion_ns"])

    def test_event_integrity_no_fake_gpu_and_resume(self):
        path = Path(self.tmp.name) / "events.jsonl"
        binding = self.binding()
        self.backend.event_log = OnlineEventLog(path, binding=binding)
        self.execute(1)
        report = aggregate_online_events(path, expected_file_sha256=file_digest(path), binding=binding,
            fit_rows=[{"case_id": "a", "content_group_id": "fit"}], validation_rows=[{"case_id": "b", "content_group_id": "validation"}])
        self.assertEqual(report["finalized"], 1)
        self.assertFalse(report["real_cuda_samples"])
        self.assertFalse(report["sentinel_evidence_complete"])
        OnlineEventLog(path, binding=binding, resume=True)
        with self.assertRaises(ValueError):
            OnlineEventLog(path, binding={**binding, "code_commit": "new"}, resume=True)

    def test_bad_digest_and_partition_leakage_fail(self):
        path = Path(self.tmp.name) / "events.jsonl"
        binding = self.binding()
        self.backend.event_log = OnlineEventLog(path, binding=binding)
        self.execute(1)
        with self.assertRaises(ValueError):
            aggregate_online_events(path, expected_file_sha256="0" * 64, binding=binding, fit_rows=[], validation_rows=[])
        with self.assertRaises(ValueError):
            aggregate_online_events(path, expected_file_sha256=file_digest(path), binding=binding,
                fit_rows=[{"case_id": "a", "content_group_id": "leak"}], validation_rows=[{"case_id": "b", "content_group_id": "leak"}])


if __name__ == "__main__": unittest.main()
