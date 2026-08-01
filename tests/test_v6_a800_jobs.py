import json
import unittest
from pathlib import Path

from probekv.v6_a800_jobs import V6A800JobKind, build_v6_a800_jobs


class V6A800JobMatrixTests(unittest.TestCase):
    def test_matrix_covers_frozen_correctness_and_profile_dimensions(self):
        raw = json.loads(
            Path("configs/v6_a800_microbench.json").read_text(encoding="utf-8")
        )
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


if __name__ == "__main__":
    unittest.main()
