import unittest
from pathlib import Path

from probekv.v6_a800_executor import (
    RealCacheBlendA800Executor,
    aggregate_relative_l2,
    infer_canonical_kv_geometry,
    per_position_relative_l2,
)
from probekv.v6_qualification_worker import (
    QualificationJobResult,
    validate_qualification_results,
)
from probekv.v6_a800_jobs import build_v6_a800_jobs


class V6A800ExecutorContractTests(unittest.TestCase):
    def test_canonical_kv_geometry_accepts_flattened_cacheblend_k(self):
        class TensorShape:
            shape = (512, 1024)

        self.assertEqual(
            infer_canonical_kv_geometry(TensorShape(), configured_kv_heads=8),
            (8, 128),
        )

    def test_canonical_kv_geometry_accepts_explicit_heads(self):
        class TensorShape:
            shape = (512, 8, 128)

        self.assertEqual(
            infer_canonical_kv_geometry(TensorShape(), configured_kv_heads=8),
            (8, 128),
        )

    def test_canonical_kv_geometry_rejects_incompatible_width(self):
        class TensorShape:
            shape = (512, 1025)

        with self.assertRaises(ValueError):
            infer_canonical_kv_geometry(TensorShape(), configured_kv_heads=8)

    def test_aggregate_relative_l2(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        reference = (torch.tensor([[3.0, 4.0]]),)
        observed = (torch.tensor([[0.0, 4.0]]),)
        self.assertAlmostEqual(
            aggregate_relative_l2(observed, reference), 3.0 / 5.0
        )
        with self.assertRaises(ValueError):
            aggregate_relative_l2((), ())

    def test_per_position_relative_l2(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        reference = (
            torch.tensor([[3.0, 4.0]]),
            torch.tensor([[0.0, 2.0]]),
        )
        observed = (
            torch.tensor([[0.0, 4.0]]),
            torch.tensor([[0.0, 1.0]]),
        )
        self.assertEqual(
            per_position_relative_l2(observed, reference), (0.6, 0.5)
        )

    def test_result_round_trip_and_complete_validation(self):
        import json

        jobs = build_v6_a800_jobs(json.loads(
            Path("configs/v6_a800_microbench.json").read_text(encoding="utf-8")
        ))
        results = tuple(
            QualificationJobResult(
                job_id=job.job_id,
                passed=True,
                cuda_event_timing=True,
                gpu_ms=1.0,
                host_ms=1.1,
            )
            for job in jobs
        )
        restored = tuple(
            QualificationJobResult.from_row(row.to_row()) for row in results
        )
        self.assertEqual(restored, results)
        validate_qualification_results(jobs, restored)
        with self.assertRaisesRegex(RuntimeError, "immutable job order"):
            validate_qualification_results(jobs, restored[:-1])

    def test_real_runner_has_no_fake_executor_escape_hatch(self):
        text = Path(
            "scripts/server/run_v6_a800_qualification.py"
        ).read_text(encoding="utf-8")
        self.assertIn("RealCacheBlendA800Executor", text)
        self.assertIn("len(jobs) != 140", text)
        self.assertIn("git\", \"status\", \"--porcelain", text)
        self.assertIn("native_prefix_cache_audit.json", text)
        self.assertIn("run_native_prefix_cache_sentinel", text)
        self.assertNotIn("FakeQualification", text)

    def test_prefix_hit_uses_scheduler_metadata_not_timing(self):
        text = Path("src/probekv/v6_a800_executor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("row.computed_block_nums", text)
        self.assertIn('"timing_inference_used": False', text)
        self.assertIn("enable_prefix_caching=True", text)
        self.assertIn(
            '"dense_reference_scope": "same_native_prefix_cache_hit"', text
        )
        self.assertIn(
            "native_prefix_kernel_vs_monolithic_logit_relative_l2", text
        )

    def test_prefix_matched_reference_is_scoped_only_to_prefix_sentinel(self):
        import inspect

        ordinary = inspect.getsource(RealCacheBlendA800Executor._run_sentinel)
        native_prefix = inspect.getsource(
            RealCacheBlendA800Executor.run_native_prefix_cache_sentinel
        )
        self.assertNotIn("cached_tokens", ordinary)
        self.assertNotIn("prefix_layers", ordinary)
        self.assertIn("same native Prefix Cache hit", native_prefix)
        self.assertIn("exact_prefix_layers=prefix_layers", native_prefix)

    def test_cfo_sentinel_cannot_reuse_native_prefix_prompt(self):
        import inspect

        source = inspect.getsource(
            RealCacheBlendA800Executor.run_cfo_eager_streaming_sentinel
        )
        self.assertIn("[PKV-CFO-FULL-%d]", source)
        self.assertNotIn("self._exact_prefix_ids", source)

    def test_dense_path_restores_one_cacheblend_slot_per_layer(self):
        text = Path("src/probekv/v6_a800_executor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "[None, None] for _ in range(self.model_spec.num_layers)", text
        )
        self.assertNotIn("self.inner_model.old_kvs = []", text)

    def test_canonical_source_fixture_forces_and_validates_full_prefill(self):
        text = Path("src/probekv/v6_a800_executor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("canonical_fixture_nonce", text)
        self.assertIn('"[PKV-P%d-S%d-K%d-V%d] "', text)
        self.assertIn(
            "canonical Source requires a complete full-prefill", text
        )
        self.assertIn(
            "canonical Source KV rows differ from Segment tokens", text
        )


if __name__ == "__main__":
    unittest.main()
