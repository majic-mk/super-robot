"""No GPU execution: manifest and measured-table rejection contract tests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_event_log import atomic_json
from probekv.v8_schema10_storage import file_digest
from probekv.v8_schema10_single_request_sentinel import prepare_manifest, validate_manifest, run_manifest
from probekv.v8_schema10_measured_costs import MeasuredRequestCostProvider


class SentinelContract(unittest.TestCase):
    def make(self):
        traces = {}
        for n in (1, 2, 5, 37):
            traces[str(n)] = [{"request_id": "q1", "request_epoch": 1,
                "token_ids": list(range(2 * n + 1)), "locked_test_accessed": False,
                "content_group_id": "development-group", "partition_id": "development",
                "segments": [{"segment_id": str(s), "content_key": "content" + str(s),
                    "positions": [2 * s, 2 * s + 1], "token_ids": [2 * s, 2 * s + 1]}
                    for s in range(n)]}]
        binding = {k: "a" * 64 for k in ("patch_sha256", "config_sha256", "runtime_measurement_sha256", "tokenizer_hash")}
        binding.update(code_commit="b" * 40, model_signature="model", model_revision="revision", global_byte_budget=2**30)
        dispatches = {name: {"selection_path": path, "selection_budget_policy": "end_to_end_aware"}
                      for name, path in (("fast", "d1_d2_rescue"), ("legacy", "legacy_multicheckpoint"))}
        return prepare_manifest(trace_set=traces, dispatches=dispatches, binding=binding)

    def resign(self, manifest):
        manifest["manifest_sha256"] = digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})

    def test_twenty_jobs_separate_fast_legacy_and_no_readiness_claim(self):
        manifest = self.make()
        validate_manifest(manifest)
        self.assertEqual(len(manifest["jobs"]), 20)
        self.assertFalse(manifest["readiness"]["ready_for_single_request_gpu_sentinel"])
        self.assertIsNone(manifest["backend_factory"])

    def test_premeasurement_blueprint_has_no_fake_cost_sha_and_cannot_run(self):
        manifest = self.make()
        traces = {str(n): next(j["requests"] for j in manifest["jobs"] if j["segments"] == n) for n in (1,2,5,37)}
        dispatches = {name: next(j["dispatch"] for j in manifest["jobs"] if j["job_id"].startswith(name)) for name in ("fast", "legacy")}
        binding = {**manifest["binding"], "runtime_measurement_sha256": None}
        blueprint = prepare_manifest(trace_set=traces, dispatches=dispatches, binding=binding, measurement_pending=True)
        self.assertIsNone(blueprint["binding"]["runtime_measurement_sha256"])
        self.assertFalse(blueprint["online_trace_execution_allowed"])
        self.assertEqual(blueprint["jobs"], manifest["jobs"])
        with self.assertRaises(ValueError):
            validate_manifest(blueprint)
        with self.assertRaises(ValueError):
            prepare_manifest(trace_set=traces, dispatches=dispatches, binding=manifest["binding"], measurement_pending=True)

    def test_deleting_job_cannot_pass_by_resigning_hash(self):
        manifest = self.make()
        manifest["jobs"].pop()
        self.resign(manifest)
        with self.assertRaises(ValueError): validate_manifest(manifest)

    def test_all_k_runs_require_same_causal_trace(self):
        manifest = deepcopy(self.make())
        manifest["jobs"][3]["requests"] = deepcopy(manifest["jobs"][3]["requests"])
        manifest["jobs"][3]["requests"][0]["request_epoch"] = 4
        self.resign(manifest)
        with self.assertRaises(ValueError): validate_manifest(manifest)

    def test_no_instance_confirmation_means_no_factory_or_gpu_calls(self):
        with patch("probekv.v8_schema10_single_request_sentinel.importlib.import_module") as factory:
            with self.assertRaises(RuntimeError):
                run_manifest(self.make(), output_dir="unused", hourly_yuan=7., instance_confirmed=False, repository=".")
            factory.assert_not_called()

    def test_wrong_code_and_missing_factory_stop_before_gpu(self):
        for head in ("c" * 40, "b" * 40):
            with patch("probekv.v8_schema10_single_request_sentinel.subprocess.check_output", side_effect=[head, ""]):
                with self.assertRaises(RuntimeError):
                    run_manifest(self.make(), output_dir="unused", hourly_yuan=7., instance_confirmed=True, repository=".")


class MeasuredCostContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "cost.json"
        self.provenance = {k: k for k in ("model", "code", "patch", "gpu", "config", "runtime_profile", "timing_scope")}
        self.ctx = NS(request={"token_ids": [1, 2, 3]}, cached_prefix_tokens=0,
                      prefix_cache_mode="cold", sampling_signature="greedy", actual_repair_check_sunk_ms=1.)

    def row(self, category, query, samples, **extra):
        # Synthetic validator inputs, not emitted or aggregated as GPU evidence.
        row = {"category": category, "query": query, "samples_ms": samples,
            "provenance": self.provenance, "origin": "real_cuda_execution", "fake_timing": False,
            "warmup_excluded": True, "outlier_policy": "none", **extra}
        return {**row, "row_sha256": digest_json(row)}

    def load(self, rows):
        atomic_json(self.path, {"provenance": self.provenance, "formal_profile_frozen": False, "rows": rows, "joint_rows": []})
        return MeasuredRequestCostProvider(self.path, expected_sha256=file_digest(self.path), provenance=self.provenance)

    def test_missing_and_mismatched_prefix_dense_is_none(self):
        identity = MeasuredRequestCostProvider.identity(self.ctx)
        provider = self.load([self.row("dense_reference", identity, [100.])])
        self.assertEqual(provider.dense_reference(self.ctx), 100.)
        self.ctx.cached_prefix_tokens = 1
        self.assertIsNone(provider.dense_reference(self.ctx))

    def test_future_upper_is_not_gate1_marginal_lower(self):
        query = {"request": MeasuredRequestCostProvider.identity(self.ctx), "segment_id": "s", "source_id": "v",
                 "completed_depth": 1, "first_reuse_layer": 2}
        provider = self.load([self.row("source_local_marginal", query, [7., 9.],
                                      component_lower_ms={"support_build": 1., "visible_load": 1., "repair": 1.}),
                              self.row("source_local_dense", query, [10.]),
                              self.row("source_future", query, [40., 45.])])
        self.assertEqual(provider.gate1(self.ctx, "s", "v", 1).predicted_reuse_marginal_lower_ms, 3.)
        self.assertEqual(provider.candidate_future_ms(self.ctx, "s", "v", 1), 45.)
        self.assertIsNone(provider.candidate_future_ms(self.ctx, "s", "unknown", 1))

    def test_explicit_gate1_preparation_is_not_blocked_by_speculative_waste_budget(self):
        provider = self.load([])
        provider.sha = "measured"
        provider._lookup = lambda category, query: ({"samples_ms": [2.5]}
            if category == "winner_visible_preparation" else None)
        provider.joint_estimator = lambda context: self.fail("explicit Gate1 must not query speculative waste")
        self.ctx.current_completed_depth = 1
        self.ctx.selector = NS(preparation_profile=NS(gate1_mode="explicit_barrier"))
        result = provider.preparation(self.ctx, "s", "v")
        self.assertTrue(result["resource_admitted"])
        self.assertIsNone(result["speculative_waste_budget_ms"])
        self.assertEqual(result["admission_basis"], "explicit_gate1_plus_measured_copy")

    def test_fake_timing_and_other_gpu_provenance_rejected(self):
        for overrides in ({"fake_timing": True}, {"provenance": {**self.provenance, "gpu": "another"}}):
            with self.assertRaises(ValueError):
                self.load([self.row("dense_reference", {}, [10.], **overrides)])


if __name__ == "__main__": unittest.main()
