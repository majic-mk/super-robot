import unittest
import inspect
from pathlib import Path
from types import SimpleNamespace

from probekv.experiment_jobs import E1Job
from probekv.v6_a800_executor import GenerationTrace
from probekv.v6_a800_executor import RealCacheBlendA800Executor
from probekv.v6_h1_runtime import (
    V6H1CaseRuntime,
    V6H1CorrectnessError,
    stable_cacheblend_repair_positions,
)


class V6H1RepairPolicyTests(unittest.TestCase):
    def test_cacheblend_v_drift_ranking_is_stable_nested_and_segment_only(self):
        positions = (10, 11, 12, 13, 14)
        scores = (1.0, 9.0, 9.0, 4.0, 0.0)
        expected = {
            0.0: (),
            0.2: (11,),
            0.4: (11, 12),
            0.8: (10, 11, 12, 13),
            1.0: positions,
        }
        previous = set()
        for ratio, indices in expected.items():
            observed = stable_cacheblend_repair_positions(
                positions, scores, ratio
            )
            self.assertEqual(observed, indices)
            self.assertTrue(previous.issubset(observed))
            self.assertTrue(set(observed).issubset(positions))
            previous = set(observed)

    def test_invalid_drift_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete Segment"):
            stable_cacheblend_repair_positions((3, 4), (1.0,), 0.5)

    def test_r1_token_mismatch_is_a_hard_gate_before_grid_rows(self):
        runtime = object.__new__(V6H1CaseRuntime)
        runtime.source_index = {"s0": 0}
        runtime.case = SimpleNamespace(
            sources=(SimpleNamespace(source_id="s0"),)
        )
        runtime.fixture = SimpleNamespace(source_ids=("s0",))
        runtime.full = GenerationTrace(token_ids=(7,), logits=(), gpu_ms=0, host_ms=0)
        runtime._validate_job = lambda job, source: None
        runtime._generate = lambda *args, **kwargs: GenerationTrace(
            token_ids=(8,), logits=(), gpu_ms=0, host_ms=0
        )
        job = E1Job(
            job_id="job-r1",
            case_id="case",
            dataset="dataset",
            split="pilot",
            construction="controlled",
            case_digest="a" * 64,
            content_hash="b" * 64,
            model_signature="model@revision",
            source_id="s0",
            source_context_id="ctx",
            segment_tokens=8,
            reuse_layer=5,
            repair_ratio=1.0,
            seed=20260726,
        )
        with self.assertRaisesRegex(V6H1CorrectnessError, "token IDs"):
            runtime.run_group("s0", (job,), {})


class V6H1ServerContractTests(unittest.TestCase):
    def test_executor_accepts_explicit_boundary_mask_and_model_namespace(self):
        parameters = inspect.signature(
            RealCacheBlendA800Executor._reuse_generate
        ).parameters
        self.assertIn("boundary_by_segment", parameters)
        self.assertIn("repair_positions_by_segment", parameters)
        self.assertIn("model_signature", parameters)
        self.assertIn("stop_token_ids", parameters)
        self.assertIn("force_nonpaper_measurement_admission", parameters)

    def test_worker_uses_resumable_runtime_and_hard_failure_gate(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts" / "server" / "run_v6_h1_pilot.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("V6H1CaseRuntime", text)
        self.assertIn("V6H1CorrectnessError", text)
        self.assertIn('"r1_dense_equivalence_passed"', text)
        self.assertNotIn("CacheBlendCaseRuntime", text)
        self.assertIn("round(spec.num_layers * 0.15)", text)
        self.assertNotIn("job.reuse_layer == 5", text)
        self.assertIn('parser.add_argument("--handoff", required=True)', text)
        self.assertIn('handoff.get("code_commit") != code_commit', text)
        self.assertIn('handoff.get("patch_audit_sha256")', text)
        self.assertIn(
            'parser.add_argument("--qualification-gate", required=True)', text
        )
        self.assertIn("validate_h1_qualification_gate", text)
        self.assertLess(
            text.index("validate_h1_qualification_gate("),
            text.index("RealCacheBlendA800Executor("),
        )

    def test_model_data_handoff_retokenizes_and_blocks_locked_test(self):
        root = Path(__file__).resolve().parents[1]
        text = (
            root / "scripts" / "server" / "prepare_v6_h1_model_data.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--tokenizer", str(snapshot)', text)
        self.assertIn('"locked_test_accessed": False', text)
        self.assertIn('"ready_for_v6_h1_gpu_sentinel": True', text)
        self.assertIn('"code_commit": code_commit', text)
        self.assertIn('"patch_audit_sha256": sha256_file(patch_audit_path)', text)


if __name__ == "__main__":
    unittest.main()
