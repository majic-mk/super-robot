from pathlib import Path
import runpy
import unittest


validate = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    "scripts/server/run_schema10_native_online_closure.py"))["validate_native_dense_reference"]
ready_sample = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    "scripts/server/run_schema10_native_online_closure.py"))["ready_joint_sample"]


class NativeCostBaselineTests(unittest.TestCase):
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
