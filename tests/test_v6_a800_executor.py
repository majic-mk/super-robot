import unittest
from pathlib import Path

from probekv.v6_a800_executor import aggregate_relative_l2
from probekv.v6_qualification_worker import (
    QualificationJobResult,
    validate_qualification_results,
)
from probekv.v6_a800_jobs import build_v6_a800_jobs


class V6A800ExecutorContractTests(unittest.TestCase):
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
        self.assertNotIn("FakeQualification", text)


if __name__ == "__main__":
    unittest.main()
