import unittest
from probekv.prefill_phase_diagnostic import summarize_prefill_phase


class PrefillPhaseDiagnosticTests(unittest.TestCase):
    def test_decode_exclusion_external_correlation_and_union(self):
        def e(cat, name, ts, dur, external=None):
            return dict(ph="X", cat=cat, name=name, ts=ts, dur=dur, pid=1, tid=2,
                        args={} if external is None else {"External id": external})
        events = [e("user_annotation", "cacheblend.native_prefill", 0, 100),
                  e("cuda_runtime", "cudaLaunchKernel", 1, 1, 4),
                  e("kernel", "compute", 110, 20, 4),  # asynchronous tail still belongs to prefill
                  e("gpu_memcpy", "copy", 115, 20, 4),
                  e("cuda_runtime", "cudaStreamSynchronize", 50, 5, 5),
                  e("cuda_runtime", "cudaLaunchKernel", 160, 1, 9),
                  e("kernel", "decode", 170, 99, 9)]
        d = summarize_prefill_phase(dict(traceEvents=events), "cacheblend_loop")
        self.assertEqual(d["kernel_count"], 1)
        self.assertEqual(d["kernel_union_ms"], .020)
        self.assertEqual(d["gpu_activity_union_ms"], .025)
        self.assertEqual(d["runtime_calls"]["cudaStreamSynchronize"]["count"], 1)

    def test_missing_marker_cannot_claim_zero_overhead(self):
        with self.assertRaises(ValueError):
            summarize_prefill_phase(dict(traceEvents=[]), "probekv")
