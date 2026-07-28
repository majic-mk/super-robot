import unittest

from scripts.server.build_stage2_readiness import validate_stage2_audits


class Stage2ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "cases": 150,
            "datasets": {
                name: {"cases": 50, "natural": 25, "controlled": 25}
                for name in ("MuSiQue", "2WikiMultiHopQA", "HotPotQA")
            },
            "all_split_pilot": True,
            "locked_test_accessed": False,
            "paper_evidence": False,
        }
        self.cb_jobs = {"jobs": 252}
        self.h1_jobs = {
            "jobs": 9720,
            "layer_job_counts": {
                "5": 5400,
                "3": 1080,
                "7": 1080,
                "10": 1080,
                "13": 1080,
            },
        }
        self.cb0 = {"passed": True, "cached_outputs": 10, "full_outputs": 10}

    def test_frozen_stage2_geometry_passes(self):
        validate_stage2_audits(
            self.manifest, self.cb_jobs, self.h1_jobs, self.cb0
        )

    def test_locked_test_or_partial_jobs_are_rejected(self):
        self.manifest["locked_test_accessed"] = True
        with self.assertRaises(ValueError):
            validate_stage2_audits(
                self.manifest, self.cb_jobs, self.h1_jobs, self.cb0
            )
        self.manifest["locked_test_accessed"] = False
        self.h1_jobs["jobs"] = 9719
        with self.assertRaises(ValueError):
            validate_stage2_audits(
                self.manifest, self.cb_jobs, self.h1_jobs, self.cb0
            )


if __name__ == "__main__":
    unittest.main()
