"""CPU-only proposal tests; simulated samples are never GPU evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
import math
import unittest
from unittest.mock import patch

from probekv.v8_schema6_contracts import PlannerSnapshot
from probekv.v8_schema6_planner import JointTimelineContext
from probekv.v8_schema7_repair import SourceScoreTrimIndices
from probekv.v8_schema10_cost_provider import (
    EXECUTION_SHAPE_KEY, ProfiledJointTimelineEstimator, RequestExecutionShape,
)
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_slack_repair import propose_single_segment_slack_repair


class SlackRepairTests(unittest.TestCase):
    def setUp(self):
        self.positions = tuple(range(8, 28))
        self.ranking = tuple(reversed(self.positions))
        self.shape = RequestExecutionShape(30, 4, 3, 1, {"s": self.positions}, {}, {},
            {"s": {"tier": "pinned_cpu", "bytes": 4096,
                   "ready_layers": [1, 2, 3], "copy_in_flight": False, "layout": "bf16"}},
            {"request": "q", "prefix_tokens": 4, "sampling": "greedy"})
        self.context = JointTimelineContext(("s",), ("s",), (), (), {"s": 2}, "old-mask", "scheduler")
        self.snapshot = PlannerSnapshot(1, 1, "scheduler", 1, "d" * 64)
        self.provenance = {k: k for k in ("model", "code", "patch", "gpu", "config", "timing_scope", "runtime_profile")}
        self.dense = {"identity": self.shape.dense_reference_identity,
                      "origin": "real_cuda_execution", "fake_timing": False, "ttft_ms": 100.}
        # Explicitly synthetic cost rows, allowed ONLY by the estimator's CPU
        # test switch. The proposal cannot produce a production PASS regardless.
        self.costs = {.15: 60., .2: 65., .3: 69., .5: 83., .75: 90., 1.: 100.}

    def estimator(self, costs=None):
        values = self.costs if costs is None else costs
        query_audit = []
        estimator = ProfiledJointTimelineEstimator(provenance=self.provenance, shape=self.shape,
            measurements=[], measurement_digest="d" * 64, allow_test_measurements=True,
            key_contract=EXECUTION_SHAPE_KEY, query_audit=query_audit)
        rows = []
        for ratio, future in values.items():
            support = tuple(sorted(self.ranking[:math.ceil(ratio * len(self.positions))]))
            trial = replace(self.shape, repair_by_segment_by_layer={"s": {2: support, 3: support}})
            row = {"provenance": self.provenance, "origin": "cpu_test", "fake_timing": True,
                   "query": estimator.for_shape(trial).query(self.context),
                   "joint_future_wall_ms_samples": [future], "warmup_excluded": True,
                   "outlier_policy": "none"}
            row["row_sha256"] = digest_json(row)
            rows.append(row)
        return ProfiledJointTimelineEstimator(provenance=self.provenance, shape=self.shape,
            measurements=rows, measurement_digest="d" * 64, allow_test_measurements=True,
            key_contract=EXECUTION_SHAPE_KEY, query_audit=query_audit)

    def propose(self, estimator=None, **kwargs):
        args = dict(estimator=estimator or self.estimator(), context=self.context,
            snapshot=self.snapshot, current_snapshot=lambda: self.snapshot,
            frozen_source_variant_id="winner-A", absolute_reuse_eligible=True,
            ranked_winner_repair_positions=self.ranking, base_ratio=.15,
            quality_supported_ratios=(.15, .2, .3, .5, .75, 1.),
            quality_evidence_sha256="e" * 64, dense_reference=self.dense, actual_sunk_ms=10.)
        args.update(kwargs)
        with patch("probekv.v8_schema10_slack_repair.time.perf_counter_ns", return_value=0):
            return propose_single_segment_slack_repair(**args)

    def test_max_ratio_under_point_eight_not_point_nine_five(self):
        result = self.propose()
        self.assertEqual(result["selected_ratio"], .3)
        self.assertEqual(result["predicted_request_total_ms"], 79.)
        self.assertEqual(result["remaining_gamma_slack_ms"], 1.)
        larger = next(r for r in result["candidate_observations"] if r["ratio"] == .5)
        self.assertEqual(larger["predicted_request_total_ms"], 93.)
        self.assertFalse(larger["within_gamma_budget"])

    def test_never_authorizes_commit_or_quality_certification(self):
        result = self.propose()
        self.assertFalse(result["production_admission_allowed"])
        self.assertTrue(result["fresh_final_commit_required"])
        self.assertFalse(result["gpu_runtime_qualified"])
        self.assertFalse(result["paper_evidence"])
        self.assertIn("not_certification", result["quality_support_status"])
        claimed = result.pop("report_sha256")
        self.assertEqual(claimed, digest_json(result))

    def test_larger_ratios_require_explicit_quality_evidence(self):
        result = self.propose(quality_supported_ratios=(.15, .2))
        self.assertEqual(result["selected_ratio"], .2)
        self.assertEqual([r["ratio"] for r in result["candidate_observations"]], [.15, .2])

    def test_missing_high_ratio_cost_is_not_interpolated(self):
        costs = dict(self.costs)
        del costs[.3]
        result = self.propose(self.estimator(costs))
        self.assertEqual(result["selected_ratio"], .2)
        missing = next(r for r in result["candidate_observations"] if r["ratio"] == .3)
        self.assertEqual(missing["cost_status"], "UNSUPPORTED")
        self.assertIsNone(missing["joint_future_ms"])

    def test_missing_base_dense_even_if_large_ratio_has_cost(self):
        costs = dict(self.costs)
        del costs[.15]
        result = self.propose(self.estimator(costs))
        self.assertEqual(result["recommended_action"], "dense")
        self.assertEqual(result["reason"], "base_cost_unsupported")
        self.assertEqual(result["selected_source_variant_id"], "winner-A")

    def test_uneconomic_base_dense_even_if_noisy_large_ratio_cheaper(self):
        costs = dict(self.costs)
        costs[.15] = 81.
        result = self.propose(self.estimator(costs))
        self.assertEqual(result["reason"], "base_exceeds_gamma_budget")
        self.assertIsNone(result["selected_ratio"])

    def test_exact_budget_boundary_is_allowed(self):
        costs = {.15: 70.}
        self.assertEqual(self.propose(self.estimator(costs))["selected_ratio"], .15)

    def test_source_absolute_rejection_does_not_query_or_upgrade(self):
        result = self.propose(absolute_reuse_eligible=False)
        self.assertEqual(result["reason"], "absolute_reuse_ineligible")
        self.assertEqual(result["query_audit"], [])

    def test_candidate_masks_are_nested_prefix_and_suffix_remain_dense(self):
        result = self.propose()
        previous = set()
        digests = []
        for row in result["query_audit"]:
            query = row["query"]["geometry"]
            active = set(query["layer_active_positions"]["2"])
            support = active & set(self.positions)
            self.assertLessEqual(previous, support)
            self.assertFalse(active & {0, 1, 2, 3})
            self.assertLessEqual({4, 5, 6, 7, 28, 29}, active)
            previous = support
            digests.append(row["query_digest"])
        self.assertEqual(len(digests), len(set(digests)))

    def test_source_score_trim_is_not_runtime_repair(self):
        with self.assertRaisesRegex(TypeError, "trim indices"):
            self.propose(ranked_winner_repair_positions=SourceScoreTrimIndices(self.positions))

    def test_partial_ranking_cannot_reenter_filtered_rows(self):
        with self.assertRaisesRegex(ValueError, "full non-prefix"):
            self.propose(ranked_winner_repair_positions=self.ranking[:3])

    def test_prefix_or_duplicate_ranking_rejected(self):
        for ranking in ((0,) + self.ranking[1:], self.ranking[:-1] + (self.ranking[0],)):
            with self.assertRaises(ValueError):
                self.propose(ranked_winner_repair_positions=ranking)

    def test_no_state_or_production_query_audit_mutation(self):
        estimator = self.estimator()
        before = deepcopy(asdict(estimator.shape))
        self.propose(estimator)
        self.assertEqual(asdict(estimator.shape), before)
        self.assertEqual(estimator.queries, [])
        self.assertEqual(estimator.query_audit, [])

    def test_stale_before_or_after_queries_rejected(self):
        stale = replace(self.snapshot, request_generation=2)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.propose(current_snapshot=lambda: stale)
        snapshots = iter((self.snapshot, stale))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.propose(current_snapshot=lambda: next(snapshots))

    def test_runtime_binding_mismatch_rejected(self):
        snapshot = replace(self.snapshot, runtime_cost_profile_sha="f" * 64)
        with self.assertRaisesRegex(ValueError, "measurement digest"):
            self.propose(snapshot=snapshot, current_snapshot=lambda: snapshot)

    def test_dense_identity_or_fake_timing_rejected(self):
        for dense in ({**self.dense, "fake_timing": True},
                      {**self.dense, "identity": {"request": "other"}}):
            with self.assertRaises((ValueError, RuntimeError)):
                self.propose(dense_reference=dense)

    def test_grid_and_missing_quality_base_rejected(self):
        for params in ({"base_ratio": .25}, {"quality_supported_ratios": (.2, .3)},
                       {"quality_supported_ratios": (.15, .15)}, {"quality_evidence_sha256": ""}):
            with self.assertRaises(ValueError):
                self.propose(**params)

    def test_base_can_be_above_fifteen_percent_without_old_floor_alias(self):
        result = self.propose(base_ratio=.2)
        self.assertEqual(result["base_ratio"], .2)
        self.assertEqual(result["selected_ratio"], .3)

    def test_invalid_sunk_or_layer_boundary_rejected(self):
        for value in (float("nan"), float("inf"), -1., True):
            with self.assertRaises(ValueError):
                self.propose(actual_sunk_ms=value)
        with self.assertRaisesRegex(ValueError, "dense check"):
            self.propose(context=replace(self.context, boundary_by_segment={"s": 3}))

    def test_committed_and_multisegment_paths_are_not_supported(self):
        for context in (
            JointTimelineContext(("s",), (), (), ("s",), {}, "mask", "scheduler"),
            JointTimelineContext(("s", "t"), ("s",), ("t",), (), {"s": 2}, "mask", "scheduler"),
        ):
            with self.assertRaisesRegex(ValueError, "single-Segment and pre-commit"):
                self.propose(context=context)

    def test_planning_time_is_not_free_slack(self):
        args = dict(estimator=self.estimator(), context=self.context,
            snapshot=self.snapshot, current_snapshot=lambda: self.snapshot,
            frozen_source_variant_id="winner-A", absolute_reuse_eligible=True,
            ranked_winner_repair_positions=self.ranking, base_ratio=.15,
            quality_supported_ratios=(.15, .2, .3), quality_evidence_sha256="e" * 64,
            dense_reference=self.dense, actual_sunk_ms=10.)
        with patch("probekv.v8_schema10_slack_repair.time.perf_counter_ns", side_effect=[0, 2_000_000]):
            result = propose_single_segment_slack_repair(**args)
        self.assertEqual(result["accounted_sunk_ms"], 12.)
        self.assertEqual(result["selected_ratio"], .2)


if __name__ == "__main__":
    unittest.main()
