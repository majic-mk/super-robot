from pathlib import Path
import runpy
import unittest


validate = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    "scripts/server/run_schema10_native_online_closure.py"))["validate_native_dense_reference"]


class NativeCostBaselineTests(unittest.TestCase):
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
