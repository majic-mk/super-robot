from pathlib import Path
import runpy
import unittest


validate = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    "scripts/server/run_schema10_native_online_closure.py"))["validate_native_dense_reference"]
ready_sample = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    "scripts/server/run_schema10_native_online_closure.py"))["ready_joint_sample"]
source_options = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    "scripts/server/run_schema10_native_correctness.py"))["cost_probe_source_options"]
pair_specs = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    "scripts/server/run_schema10_native_correctness.py"))["position_validation_pair_specs"]


class NativeCostBaselineTests(unittest.TestCase):
    def test_position_pairs_preregister_warmup_and_alternate_order(self):
        rows = pair_specs(20)
        self.assertEqual(len(rows), 22)
        self.assertEqual(sum(r["warmup"] for r in rows), 2)
        self.assertEqual(rows[0]["arm_order"], [False, True])
        self.assertEqual(rows[1]["arm_order"], [True, False])
        with self.assertRaises(ValueError):
            pair_specs(0)

    def test_gpu_resident_control_never_relabels_streaming_measurement(self):
        for hot in (False, True):
            options = source_options(hot)
            self.assertEqual(options["streaming"], {"wait_all_source_layers": False,
                "use_gpu_hot_cache": False, "retain_gpu_hot_cache": False})
            self.assertTrue(options["all_ready"]["wait_all_source_layers"])
            self.assertEqual(options["all_ready"]["retain_gpu_hot_cache"], hot)

    def test_streaming_future_excludes_already_elapsed_preparation(self):
        row = dict(selection_boundary_ready_ns=10_000_000,
                   winner_source_ready_ns=12_000_000, first_token_ns=55_000_000,
                   boundary_to_first_token_ms=45., ready_to_first_token_ms=43.,
                   ready_to_first_token_cuda_ms=42.)
        sample, interval = ready_sample(row)
        self.assertEqual(sample, 43.)
        self.assertEqual(interval["host_start_ns"], 12_000_000)
        self.assertEqual(interval["wall_endpoint_kind"], "ready_to_first_token")
        with self.assertRaisesRegex(ValueError, "differs"):
            ready_sample({**row, "ready_to_first_token_ms": 45.})

    def reference(self):
        return dict(resumable_engine_used=False, source_id=None, diagnostic_completed_depth=0,
            committed_segments={}, whole_request_origin="native_prefix_dense_remaining",
            request_tokens_sha256="prompt", cached_prefix_tokens=256,
            prefix_cache_mode="native_exact_blocks", sampling_signature={"temperature": 0})

    def test_resumable_control_cannot_replace_direct_native_baseline(self):
        reference = self.reference()
        validate(reference, reference)
        for updates in ({"resumable_engine_used": True}, {"diagnostic_completed_depth": 1},
                        {"source_id": "source"}, {"committed_segments": {"C": 2}}):
            with self.assertRaises(ValueError):
                validate({**reference, **updates}, reference)

    def test_primary_baseline_requires_matched_request_prefix_and_sampling(self):
        reference = self.reference()
        for key, value in (("request_tokens_sha256", "other"), ("cached_prefix_tokens", 0),
                           ("sampling_signature", {"temperature": 1})):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate(reference, {**reference, key: value})
