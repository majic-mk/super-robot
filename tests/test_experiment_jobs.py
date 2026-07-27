import unittest
from dataclasses import replace

from probekv.e1_analysis import analyze_e1
from probekv.experiment_jobs import (
    E1Result,
    ResultStatus,
    generate_e1_jobs,
    merge_e1_results,
    resumable_e1_jobs,
    select_job_shard,
    simulate_e1_results,
)
from probekv.manifest import synthetic_manifest


class JobGenerationTests(unittest.TestCase):
    def setUp(self):
        self.cases = synthetic_manifest(20, 20260726)
        self.jobs = generate_e1_jobs(
            self.cases,
            32,
            [0.0, 0.2, 0.5, 1.0],
            include_splits=("train", "calibration"),
        )

    def test_jobs_are_unique_and_test_is_excluded(self):
        self.assertEqual(len({job.job_id for job in self.jobs}), len(self.jobs))
        self.assertNotIn("test", {job.split for job in self.jobs})
        self.assertTrue(all(job.reuse_layer in {3, 5, 7, 10, 13} for job in self.jobs))

    def test_job_identity_changes_with_manifest_semantics(self):
        original = self.cases[0]
        changed = replace(original, current_context=original.current_context + " changed")
        original_jobs = generate_e1_jobs(
            [original], 32, [0.0, 1.0], include_splits=(original.split,)
        )
        changed_jobs = generate_e1_jobs(
            [changed], 32, [0.0, 1.0], include_splits=(changed.split,)
        )
        self.assertNotEqual(
            {job.job_id for job in original_jobs},
            {job.job_id for job in changed_jobs},
        )

    def test_ratio_grid_requires_full_endpoints(self):
        with self.assertRaisesRegex(ValueError, "endpoints"):
            generate_e1_jobs(
                self.cases,
                32,
                [0.1, 0.5],
                include_splits=("train",),
            )

    def test_shards_partition_jobs_exactly(self):
        shards = [select_job_shard(self.jobs, index, 4) for index in range(4)]
        identifiers = [job.job_id for shard in shards for job in shard]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(identifiers), {job.job_id for job in self.jobs})


class ResultAuditTests(unittest.TestCase):
    def setUp(self):
        cases = synthetic_manifest(10, 20260726)
        self.jobs = generate_e1_jobs(
            cases,
            32,
            [0.0, 0.2, 0.5, 1.0],
            include_splits=("train",),
        )

    def test_complete_simulation_merges_and_analyzes(self):
        results = simulate_e1_results(self.jobs)
        latest, audit = merge_e1_results(self.jobs, results)
        self.assertTrue(audit["all_completed"])
        self.assertFalse(audit["paper_evidence"])
        analysis = analyze_e1(self.jobs, latest, total_layers=32)
        self.assertGreater(analysis["safe_labels"], 0)
        self.assertFalse(analysis["paper_gate_claimable"])

    def test_missing_and_retryable_failures_are_never_hidden(self):
        results = simulate_e1_results(self.jobs)
        failed = E1Result(
            job_id=results[0].job_id,
            attempt=1,
            status=ResultStatus.GPU_RESET,
            error_type="GPU_RESET",
            error_message="fixture",
        )
        latest, audit = merge_e1_results(self.jobs, results[1:-1] + [failed])
        self.assertFalse(audit["all_accounted"])
        self.assertIn(results[-1].job_id, audit["missing_job_ids"])
        self.assertIn(results[0].job_id, audit["retryable_job_ids"])

    def test_mutated_source_result_is_rejected(self):
        valid = simulate_e1_results(self.jobs[:1])[0]
        with self.assertRaisesRegex(ValueError, "mutated"):
            replace(valid, source_digest_after="different").validate()

    def test_duplicate_attempt_prevents_complete_audit(self):
        results = simulate_e1_results(self.jobs)
        latest, audit = merge_e1_results(self.jobs, results + [results[0]])
        self.assertFalse(audit["all_accounted"])
        self.assertFalse(audit["all_completed"])
        self.assertEqual(audit["duplicate_attempt_rows"], 1)

    def test_resume_retries_only_retryable_failures_with_next_attempt(self):
        first, second, third = self.jobs[:3]
        completed = simulate_e1_results([first])[0]
        retryable = E1Result(
            job_id=second.job_id,
            attempt=2,
            status=ResultStatus.TRANSIENT_IO,
            error_type="TRANSIENT_IO",
        )
        terminal = E1Result(
            job_id=third.job_id,
            attempt=0,
            status=ResultStatus.OOM,
            error_type="OOM",
        )
        pending, attempts = resumable_e1_jobs(
            [first, second, third], [completed, retryable, terminal]
        )
        self.assertEqual([job.job_id for job in pending], [second.job_id])
        self.assertEqual(attempts[second.job_id], 3)

    def test_resume_rejects_cross_shard_rows(self):
        outside = simulate_e1_results([self.jobs[1]])[0]
        with self.assertRaisesRegex(ValueError, "outside this shard"):
            resumable_e1_jobs([self.jobs[0]], [outside])

    def test_locked_test_requires_explicit_permission(self):
        test_case = next(case for case in synthetic_manifest(10, 20260726) if case.split == "test")
        jobs = generate_e1_jobs(
            [test_case], 32, [0.0, 1.0], include_splits=("test",)
        )
        with self.assertRaisesRegex(ValueError, "locked test"):
            analyze_e1(jobs, simulate_e1_results(jobs), total_layers=32)


if __name__ == "__main__":
    unittest.main()
