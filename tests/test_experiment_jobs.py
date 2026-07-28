import unittest
from dataclasses import replace

from probekv.e1_analysis import analyze_e1
from probekv.experiment_jobs import (
    E1Result,
    ResultStatus,
    generate_e1_jobs,
    group_e1_jobs,
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

    def test_grouping_orders_ratios_and_preserves_all_jobs(self):
        groups = group_e1_jobs(list(reversed(self.jobs)))
        flattened = [job for _, members in groups for job in members]
        self.assertEqual(
            {job.job_id for job in flattened},
            {job.job_id for job in self.jobs},
        )
        for _, members in groups:
            self.assertEqual(
                [job.repair_ratio for job in members],
                sorted(job.repair_ratio for job in members),
            )


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
        self.assertIn(results[0].job_id, audit["duplicate_job_ids"])
        self.assertTrue(audit["duplicate_conflicts"])
        self.assertFalse(audit["publication_ready"])

    def test_incomplete_paper_rows_can_never_be_publication_ready(self):
        paper_rows = [
            replace(
                result,
                evidence_class="paper_measurement",
                paper_evidence=True,
            )
            for result in simulate_e1_results(self.jobs[:-1])
        ]
        _, audit = merge_e1_results(self.jobs, paper_rows)
        self.assertTrue(audit["run_environment_valid"])
        self.assertFalse(audit["result_set_complete"])
        self.assertFalse(audit["publication_ready"])
        self.assertFalse(audit["paper_evidence"])
        self.assertEqual(audit["missing_job_ids"], [self.jobs[-1].job_id])

    def test_complete_paper_rows_are_publication_ready(self):
        paper_rows = [
            replace(
                result,
                evidence_class="paper_measurement",
                paper_evidence=True,
            )
            for result in simulate_e1_results(self.jobs)
        ]
        latest, audit = merge_e1_results(self.jobs, paper_rows)
        self.assertTrue(audit["run_environment_valid"])
        self.assertTrue(audit["result_set_complete"])
        self.assertTrue(audit["publication_ready"])
        analysis = analyze_e1(
            self.jobs,
            latest,
            total_layers=32,
            result_set_audit=audit,
        )
        self.assertTrue(analysis["paper_evidence"])

    def test_failed_or_unexpected_result_blocks_publication(self):
        paper_rows = [
            replace(
                result,
                evidence_class="paper_measurement",
                paper_evidence=True,
            )
            for result in simulate_e1_results(self.jobs)
        ]
        failed = E1Result(
            job_id=self.jobs[0].job_id,
            attempt=1,
            status=ResultStatus.OOM,
            error_type="OOM",
            code_commit="local-simulation",
            environment_hash="local-simulation",
            finished_at_utc="deterministic-2",
            evidence_class="paper_measurement",
            paper_evidence=False,
        )
        foreign = replace(paper_rows[1], job_id="unexpected-job")
        _, audit = merge_e1_results(
            self.jobs,
            paper_rows[1:] + [failed, foreign],
        )
        self.assertIn(self.jobs[0].job_id, audit["failed_job_ids"])
        self.assertEqual(audit["unexpected_job_ids"], ["unexpected-job"])
        self.assertFalse(audit["result_set_complete"])
        self.assertFalse(audit["paper_evidence"])

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

    def test_server_pilot_can_never_be_paper_evidence(self):
        result = replace(
            simulate_e1_results(self.jobs[:1])[0],
            evidence_class="server_pilot",
            paper_evidence=True,
        )
        with self.assertRaisesRegex(ValueError, "never"):
            result.validate()


if __name__ == "__main__":
    unittest.main()
