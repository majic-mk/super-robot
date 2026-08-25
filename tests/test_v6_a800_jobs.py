import json
import unittest
from pathlib import Path

from probekv.v6_a800_jobs import (
    V6A800Job,
    V6A800JobKind,
    build_v6_a800_job_manifest,
    build_v6_a800_jobs,
)


class V6A800JobMatrixTests(unittest.TestCase):
    def test_matrix_covers_frozen_correctness_and_profile_dimensions(self):
        raw = json.loads(
            Path("configs/v6_a800_microbench.json").read_text(encoding="utf-8")
        )
        self.assertIs(raw["segment_count_samples_are_not_runtime_caps"], True)
        jobs = build_v6_a800_jobs(raw)
        self.assertTrue(all(not job.paper_evidence for job in jobs))
        correctness = {
            (job.segment_count, job.stored_variants)
            for job in jobs
            if job.kind is V6A800JobKind.CORRECTNESS
        }
        self.assertEqual(
            correctness,
            {(s, k) for s in (1, 2, 5, 10) for k in (1, 4, 16)},
        )
        compared = {
            job.compared_variants
            for job in jobs
            if job.kind is V6A800JobKind.CANDIDATE_COMPARE
        }
        self.assertEqual(compared, {1, 2, 4, 8, 16})
        profile_segments = {
            job.segment_count
            for job in jobs
            if job.kind is V6A800JobKind.UNION_REPAIR
        }
        self.assertEqual(profile_segments, {1, 5, 10, 15})
        self.assertEqual(len({job.job_id for job in jobs}), len(jobs))
        self.assertEqual(len(jobs), 140)
        self.assertEqual(
            [V6A800Job.from_row(job.to_row()) for job in jobs], list(jobs)
        )

    def test_manifest_binds_jobs_to_code_model_and_cacheblend(self):
        raw = json.loads(
            Path("configs/v6_a800_microbench.json").read_text(encoding="utf-8")
        )
        jobs = build_v6_a800_jobs(raw)
        manifest = build_v6_a800_job_manifest(
            jobs,
            jobs_sha256="jobs",
            code_commit="a" * 40,
            git_clean=True,
            config_sha256="config",
            contract_sha256="contract",
            server_lock_sha256="lock",
            model_id="model",
            model_revision="b" * 40,
            runtime_backend="cacheblend_multisegment_closed_loop",
            runtime_implementation_status="engine_pending",
            cacheblend_commit="c" * 40,
            cacheblend_patch_mode="probekv_v6_multiregion",
            cacheblend_patch_sha256="patch",
        )
        self.assertEqual(manifest["jobs"], 140)
        self.assertEqual(manifest["code_commit"], "a" * 40)
        self.assertEqual(manifest["model"]["revision"], "b" * 40)
        self.assertEqual(
            manifest["cacheblend"]["patch_mode"],
            "probekv_v6_multiregion",
        )
        self.assertFalse(manifest["runtime"]["qualified"])


if __name__ == "__main__":
    unittest.main()
