"""CPU contract/harness tests; no fake events are promoted to GPU evidence."""
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import ExitStack
from threading import RLock
import tempfile
import unittest

import torch

from probekv.v8_schema10_cost_collection import measurement_endpoint
from probekv.v8_schema10_stage_journal import StageEvidenceJournal
from probekv.v8_schema10_event_log import OnlineEventLog
from probekv.v8_schema10_native_adapter import NativeRequestContext, validate_native_sampling_request
from probekv.v8_schema10_native_factory import NativeExperimentBackend
from probekv.v8_schema10_cost_provider import validate_measurement_provenance
from probekv.v8_schema10_native_preflight import isolated_native_preflight, run_r1_equivalence_sentinel


class MeasurementEndpointTests(unittest.TestCase):
    def test_new_provisional_costs_bind_plan_not_a_fake_profile(self):
        p = {k: "test" for k in ("model", "code", "patch", "gpu", "config", "timing_scope")}
        p.update(profile_binding_kind="preregistered_measurement_plan", measurement_plan_sha256="a"*64,
                 runtime_profile=None)
        validate_measurement_provenance(p)
        for change in ({"runtime_profile": "made-up"}, {"measurement_plan_sha256": ""},
                       {"profile_binding_kind": "unknown"}):
            with self.assertRaises(ValueError):
                validate_measurement_provenance({**p, **change})

    def test_joint_and_source_future_stop_at_first_token_not_decode(self):
        for category, joint in [("dense_reference", False), ("source_future", False),
                                ("joint_future", False), ("union_mask_remaining", True)]:
            self.assertEqual(measurement_endpoint(category, {"first_token_ns": 120},
                begin=100, finish=900, joint=joint), (120, "first_token"))

    def test_missing_or_invalid_first_token_never_fills_completion(self):
        for value in (None, True, 99, 901, float("nan")):
            with self.assertRaises(ValueError):
                measurement_endpoint("joint_future", {"first_token_ns": value}, begin=100, finish=900)

    def test_transfer_uses_completion(self):
        self.assertEqual(measurement_endpoint("full_kv_tier_load", {}, begin=100, finish=900),
                         (900, "operation_completion"))


class StageJournalTests(unittest.TestCase):
    def binding(self):
        return {**{k: "cpu-test-only" for k in ("code_commit", "patch_sha256", "model_signature",
                "model_revision", "tokenizer_hash", "config_sha256", "initial_state_sha256",
                "measurement_plan_sha256", "gpu_uuid")},
                "runtime_measurement_sha256": None, "evidence_scope": "native_staged_prequalification"}

    def jobs(self):
        return [{"job_id": "env", "stage": "environment", "input_sha256": "test-input-1"},
                {"job_id": "r1", "stage": "correctness", "input_sha256": "test-input-2"}]

    def test_pre_cost_journal_works_but_online_log_remains_blocked(self):
        with tempfile.TemporaryDirectory() as root:
            StageEvidenceJournal(Path(root)/"stages.jsonl", binding=self.binding(), jobs=self.jobs())
            with self.assertRaises(ValueError):
                OnlineEventLog(Path(root)/"online.jsonl", binding={**self.binding(), "dispatch": "test"})

    def test_order_and_success_prefix_raw_file_validation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"stages.jsonl"
            j = StageEvidenceJournal(path, binding=self.binding(), jobs=self.jobs())
            with self.assertRaises(ValueError):
                j.begin("r1")
            j.begin("env")
            row = j.complete({"origin": "cpu_test_only"}, validate=lambda r: r["origin"] == "cpu_test_only")
            resumed = StageEvidenceJournal(path, binding=self.binding(), jobs=self.jobs(), resume=True)
            self.assertEqual(len(resumed.completed), 1)
            (Path(root)/row["relative_path"]).unlink()
            with self.assertRaises(ValueError):
                StageEvidenceJournal(path, binding=self.binding(), jobs=self.jobs(), resume=True)

    def test_failed_job_cannot_complete_advance_or_resume(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"stages.jsonl"
            j = StageEvidenceJournal(path, binding=self.binding(), jobs=self.jobs())
            j.begin("env")
            j.fail(RuntimeError("test failure"))
            with self.assertRaises(RuntimeError):
                j.complete({}, validate=lambda r: True)
            with self.assertRaises(RuntimeError):
                j.begin("r1")
            with self.assertRaises(ValueError):
                StageEvidenceJournal(path, binding=self.binding(), jobs=self.jobs(), resume=True)

    def test_false_validation_and_incomplete_job_cannot_pass(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"stages.jsonl"
            j = StageEvidenceJournal(path, binding=self.binding(), jobs=self.jobs())
            j.begin("env")
            with self.assertRaises(ValueError):
                j.complete({"passed": True}, validate=lambda r: False)
            with self.assertRaises(ValueError):
                StageEvidenceJournal(path, binding=self.binding(), jobs=self.jobs(), resume=True)

    def test_measured_digest_cannot_masquerade_as_plan(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                StageEvidenceJournal(Path(root)/"events", binding={**self.binding(),
                    "runtime_measurement_sha256": "pretend"}, jobs=self.jobs())


class NativeFinishHarness(unittest.TestCase):
    def context(self, request):
        # Execute actual NativeRequestContext.finish on a tiny CPU fake model.
        # This tests native orchestration, NOT CacheBlend numerical correctness.
        validate_native_sampling_request(request)
        ctx = NativeRequestContext.__new__(NativeRequestContext)
        tokenizer = NS(eos_token_id=0, decode=lambda ids, **kw: " ".join(map(str, ids)))
        class Outer:
            def __call__(self, **kw):
                return torch.ones((3, 2))
            def compute_logits(self, hidden, sampling):
                return torch.tensor([[5., 1.]])  # EOS wins on every forward
        feeds, exact, steps = [], [], []
        sampling = NS(selected_token_indices=torch.tensor([2]))
        prepared = (torch.tensor([1, 2, 3]), torch.arange(3), NS(), sampling)
        adapter = NS(torch=torch, outer=Outer(), inner=NS(cache_fuse_metadata={}), kv=[],
            llm=NS(get_tokenizer=lambda: tokenizer), warm_history=[], check_deadline=lambda: None,
            prepare=lambda metadata: prepared)
        ctx.adapter, ctx.request = adapter, request
        ctx.native = NS(finish_prefill=lambda **kw: exact.append(kw),
            append_for_decode=lambda token: feeds.append(token), finish_decode_step=lambda: steps.append(1))
        ctx.finished, ctx.engine, ctx.capture_collector = False, None, None
        ctx.committed, ctx.cached_prefix_tokens = {}, 0
        ctx.sampling_signature = {"max_new_tokens": request.get("max_new_tokens", 32)}
        ctx._prepared_inputs, ctx.attention, ctx.sampling = prepared, prepared[2], sampling
        ctx._enable_original_capture = lambda: None
        return ctx, feeds, exact, steps

    def test_teacher_forcing_ignores_predicted_eos_and_produces_no_qa(self):
        request = {"token_ids": [1, 2, 3], "max_new_tokens": 32, "capture_logits": True,
                   "teacher_token_ids": [1]*31}
        ctx, feeds, exact, steps = self.context(request)
        first = []
        result = ctx.finish(lambda: first.append(1))
        self.assertEqual(len(result["token_ids"]), 32)
        self.assertEqual(len(ctx.logit_trace), 32)
        self.assertEqual(feeds, [1]*31)
        self.assertEqual(len(steps), 31)
        self.assertEqual(first, [1])
        self.assertIsNone(result["qa_evidence"])
        self.assertIsNone(result["quality_passed"])
        self.assertEqual(ctx.adapter.warm_history, [])
        self.assertEqual(ctx.sampling.selected_token_indices.tolist(), [2])
        with self.assertRaises(RuntimeError):
            ctx.finish(lambda: None)

    def test_free_decode_keeps_eos_behavior_and_real_qa_structure(self):
        ctx, feeds, _, _ = self.context({"token_ids": [1, 2, 3], "max_new_tokens": 32, "answers": ["0"]})
        result = ctx.finish(lambda: None)
        self.assertEqual(result["token_ids"], [0])
        self.assertEqual(result["qa_evidence"]["answer_f1"], 1.0)
        self.assertEqual(feeds, [])
        self.assertEqual(len(ctx.adapter.warm_history), 1)

    def test_invalid_teacher_length_or_capture_rejected_before_execution(self):
        for request in ({"max_new_tokens": 3, "teacher_token_ids": [1], "capture_logits": True},
                        {"max_new_tokens": 2, "teacher_token_ids": [1]},
                        {"max_new_tokens": 0}, {"max_new_tokens": True}):
            with self.assertRaises(ValueError):
                validate_native_sampling_request(request)

    def test_production_backend_rejects_diagnostic_switches(self):
        backend = NativeExperimentBackend.__new__(NativeExperimentBackend)
        backend.costs = NS(sha="test-only")
        for field, value in (("capture_logits", True), ("teacher_token_ids", []), ("correctness_repair_ratio", 1)):
            with self.assertRaises(ValueError):
                backend.execute({field: value}, {}, arrival_ns=0)

    def test_native_close_fences_before_clearing_or_releasing(self):
        ctx = NativeRequestContext.__new__(NativeRequestContext)
        order = []
        ctx.closed = False
        ctx.synchronize = lambda: order.append("fence")
        ctx.hot_leases = ExitStack()
        ctx.hot_leases.callback(lambda: order.append("hot_lease_release"))
        ctx.hot_replicas = {}
        ctx.engine, ctx.prepared, ctx._observation = object(), {"s": object()}, {1: object()}
        ctx.workspace, ctx.capture_reservation = NS(reservation_id="working"), NS(reservation_id="capture")
        inner = NS(layers=[NS(self_attn=NS(hack_kv=[1]))], cache_fuse_metadata={}, old_kvs=[1])
        pool = NS(mutation_lock=RLock())
        ctx.adapter = NS(store_provider=lambda: NS(pool=pool), inner=inner, spec=NS(num_layers=1),
                         hbm=NS(release=lambda rid: order.append(rid)))
        ctx.close()
        self.assertEqual(order, ["fence", "hot_lease_release", "working", "capture"])
        self.assertIsNone(ctx.engine)
        self.assertEqual(ctx.prepared, {})
        self.assertEqual(inner.layers[0].self_attn.hack_kv, [])
        ctx.close()
        self.assertEqual(len(order), 4)

    def test_failed_fence_quarantines_native_context(self):
        ctx = NativeRequestContext.__new__(NativeRequestContext)
        ctx.closed = False
        ctx.engine = object()
        def broken():
            raise RuntimeError("simulated CUDA fence failure")
        ctx.synchronize = broken
        with self.assertRaises(RuntimeError):
            ctx.close()
        self.assertFalse(ctx.closed)
        self.assertIsNotNone(ctx.engine)  # no cleanup/release executed after failed fence


class PreflightResourceTests(unittest.TestCase):
    def test_r1_requires_real_callbacks_and_raw_traces(self):
        with self.assertRaises(TypeError):
            run_r1_equivalence_sentinel(request={}, dense_executor=None, reuse_executor=lambda _: {})
        def row():
            return {"token_ids": [1, 2], "logits": [[1.0] * 32],
                    "origin": "real_cuda_execution", "fake_timing": False}
        result = run_r1_equivalence_sentinel(request={}, dense_executor=lambda _: row(),
            reuse_executor=lambda _: row())
        self.assertEqual(result["dense_token_ids"], result["reuse_token_ids"])
        self.assertLessEqual(result["logit_relative_l2"], 1e-4)

    def test_r1_rejects_claim_without_raw_logits(self):
        def row():
            return {"token_ids": [1], "origin": "real_cuda_execution", "fake_timing": False}
        with self.assertRaises(ValueError):
            run_r1_equivalence_sentinel(request={}, dense_executor=lambda _: row(), reuse_executor=lambda _: row())

    def test_cpu_cannot_produce_real_preflight_rows(self):
        adapter = NS(active=None, hbm=NS(active_reserved_bytes=0), torch=torch)
        if torch.cuda.is_available():
            self.skipTest("CPU fail-closed test")
        with self.assertRaises(RuntimeError):
            with isolated_native_preflight(adapter):
                self.fail("CPU entered native GPU preflight")

    def test_quarantined_resources_cannot_start_preflight(self):
        adapter = NS(active=None, hbm=NS(active_reserved_bytes=1))
        with self.assertRaises(RuntimeError):
            with isolated_native_preflight(adapter):
                self.fail("preflight ignored quarantine")
