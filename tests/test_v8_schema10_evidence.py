import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

from probekv.v8_contracts import CandidateCounts, ResidualCandidate
from probekv.v8_schema8_planner import Gate1LocalPlan, Gate1MarginalLowerBound
from probekv.v8_schema10_contracts import AbsoluteResidualThreshold, Gate1Mode
from probekv.v8_schema10_evidence import (
    EvidenceOrigin, MetricEvidence, RequestTimingEvidence,
    require_profile_freeze_evidence, summarize_repair_evidence,
    validate_disjoint_case_groups,
    summarize_measured_coverage, validate_paired_gate1_executions,
)
from probekv.v8_schema10_profile import (
    PreparationPolicyProfile, VariantAdmissionProfileV10, SCHEMA10_TRIM_GRID,
)
from probekv.v8_schema10_profile_analysis import (
    build_selection_candidates, build_threshold_table, replay_production_d1d2,
    select_dispatch,
)
from probekv.v8_schema10_selector import Schema10D1D2Selector

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "schema10_aggregate_test", ROOT / "scripts/server/aggregate_v8_schema10_profile.py"
)
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


def repair_row(case="q", source="s", ratio=0.15, drop=0.0):
    return dict(case_id=case, source_id=source, repair_ratio=ratio,
                answer_f1_drop=drop, source_digest_unchanged=True,
                artifact_digest_unchanged=True, absolute_union_mask_verified=True,
                token_ids_equal_full=True, logit_relative_l2=0.0)


def selector():
    profile = VariantAdmissionProfileV10(
        "code", "patch", "model", "rev", "tok", 0.15,
        (AbsoluteResidualThreshold(1, 0.5), AbsoluteResidualThreshold(2, 0.5)),
    )
    return Schema10D1D2Selector(
        variant_profile=profile,
        preparation_profile=PreparationPolicyProfile("code", "model", "policy", Gate1Mode.EXPLICIT_BARRIER),
        strong_margin=0.6, stable_margin=0.3, residual_band_relative_tolerance=0.05,
    )


def plan(source, depth, passed=True):
    return Gate1LocalPlan(source, depth, depth, depth + 1, 0,
                         Gate1MarginalLowerBound(0, 0, 1 if passed else 20), 10)


class EvidenceTests(unittest.TestCase):
    def test_measured_coverage_does_not_equate_compatible_selected_commit(self):
        row = dict(request_id="q", request_epoch=2, capacity=4, execution_kind="online",
                   visible_variant_creation_epochs={"s": 1}, compatible_variant_ids=["s"],
                   selected_variant_ids=[], committed_variant_ids=[])
        result = summarize_measured_coverage({4: [row]})[0]
        self.assertEqual(result["residual_compatible_coverage"], 1)
        self.assertEqual(result["selected_coverage"], 0)
        self.assertEqual(result["commit_coverage"], 0)

    def test_operational_coverage_has_no_future_variant_visibility(self):
        row = dict(request_id="q", request_epoch=2, capacity=4, execution_kind="online",
                   visible_variant_creation_epochs={"future": 3})
        with self.assertRaisesRegex(ValueError, "future Variant"):
            summarize_measured_coverage({4: [row]})

    def test_k16_trace_cannot_masquerade_as_k4_online_run(self):
        with self.assertRaisesRegex(ValueError, "own online capacity"):
            summarize_measured_coverage({4: [dict(request_id="q", capacity=16, execution_kind="online")]})

    def test_dense_vs_forced_reuse_cannot_pass_gate1_ab(self):
        with self.assertRaisesRegex(ValueError, "not forced reuse"):
            validate_paired_gate1_executions({"execution_kind": "diagnostic"}, {})

    def test_gate1_ab_requires_same_dispatch_and_measures_signed_difference(self):
        row = dict(execution_kind="online_policy", forced_source=False, final_commit_executed=True,
                   request_id="q", initial_pool_snapshot_sha256="snapshot", code_commit="code",
                   model_signature="m", request_ttft_ms=70,
                   dispatch_config={"gate1_mode": "explicit_barrier", "repair": .15})
        bypass = dict(row, request_ttft_ms=72, dispatch_config={"gate1_mode": "fused_advisory", "repair": .15})
        self.assertEqual(validate_paired_gate1_executions(row, bypass)["ttft_delta_ms_without_gate1"], 2)
        bypass["dispatch_config"]["repair"] = .3
        with self.assertRaisesRegex(ValueError, "another dispatch"):
            validate_paired_gate1_executions(row, bypass)

    def test_stale_gate1_depth_and_duplicate_sources_are_rejected(self):
        step = dict(completed_depth=2, counts=CandidateCounts(1, 1, 1, 1, 1),
                    candidates=[ResidualCandidate("a", .1, 1, 0)], gate1_plan_by_source={"a": plan("a", 1)})
        with self.assertRaisesRegex(ValueError, "Source/depth"):
            selector().decide(**step)

    def test_unknown_metrics_cannot_be_assigned_success(self):
        for value in (True, 0.0, 1.0):
            with self.assertRaises(ValueError):
                MetricEvidence(value, EvidenceOrigin.UNAVAILABLE)
        self.assertIsNone(MetricEvidence(None, EvidenceOrigin.UNAVAILABLE).value)

    def test_derived_metric_requires_events_and_formula(self):
        with self.assertRaises(ValueError):
            MetricEvidence(1.0, EvidenceOrigin.DERIVED, ("e1",))
        with self.assertRaises(ValueError):
            MetricEvidence(float("nan"), EvidenceOrigin.DIAGNOSTIC)
        MetricEvidence(1.0, EvidenceOrigin.DERIVED, ("e1",), "union-v1")

    def test_ttft_excludes_decode_and_overlaps_are_counted_once(self):
        timing = RequestTimingEvidence("q", 0, 80_000_000, 500_000_000, 100,
                                       ((10_000_000, 20_000_000), (15_000_000, 25_000_000)))
        self.assertEqual(timing.ttft_ms, 80)
        self.assertEqual(timing.selection_active_ms, 15)
        self.assertEqual(timing.selection_dense_fraction, 0.15)

    def test_selection_after_first_token_is_not_ttft_evidence(self):
        with self.assertRaises(ValueError):
            RequestTimingEvidence("q", 0, 10, 30, 1, ((5, 20),))

    def test_integrity_success_does_not_certify_quality(self):
        rows = [repair_row(drop=0.8), repair_row(ratio=1)]
        result = summarize_repair_evidence(rows)
        self.assertEqual(result["integrity_violations"], 0)
        self.assertIsNone(result["observed_quality_violations"])
        result = summarize_repair_evidence(rows, per_request_answer_f1_drop_max=0.02)
        self.assertEqual(result["observed_quality_violations"], 1)

    def test_dense_self_reference_is_not_reuse_r1_success(self):
        result = summarize_repair_evidence([repair_row(source=None, ratio=1)])
        self.assertFalse(result["r1_coverage_complete"])
        self.assertEqual(result["r1_measured_rows"], 0)

    def test_r1_requires_every_actual_source(self):
        rows = [repair_row(source="a"), repair_row(source="b"), repair_row(source="a", ratio=1)]
        self.assertFalse(summarize_repair_evidence(rows)["r1_coverage_complete"])

    def test_duplicate_rows_cannot_inflate_request_units(self):
        with self.assertRaises(ValueError):
            summarize_repair_evidence([repair_row(), repair_row()])

    def test_unique_request_quality_units_not_ratio_or_source_rows(self):
        result = summarize_repair_evidence(
            [repair_row(source="a", drop=.5), repair_row(source="b", drop=.8)],
            per_request_answer_f1_drop_max=.1,
        )
        self.assertEqual(result["actual_reuse_request_units"], 1)
        self.assertEqual(result["observed_quality_violations"], 1)

    def test_disjoint_fit_validation_groups(self):
        fit = [{"case_id": "a", "content_group_id": "doc1"}]
        with self.assertRaisesRegex(ValueError, "content-group leakage"):
            validate_disjoint_case_groups(fit, [{"case_id": "b", "content_group_id": "doc1"}])
        validate_disjoint_case_groups(fit, [{"case_id": "b", "content_group_id": "doc2"}])

    def test_old_success_flags_and_real_cuda_do_not_unlock_freeze(self):
        with self.assertRaisesRegex(ValueError, "contract v2"):
            require_profile_freeze_evidence({"real_gpu_measurements": True, "final_consistency": {"passed": True}})
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            require_profile_freeze_evidence({"evidence_contract_version": 2, "evidence_scope": "diagnostic_measurements"})

    def test_missing_report_cannot_unlock_freeze(self):
        with self.assertRaisesRegex(ValueError, "missing validated"):
            require_profile_freeze_evidence({"evidence_contract_version": 2, "evidence_scope": "production_policy_validation"})

    def test_aggregator_preserves_unknown_commit_and_quality(self):
        rows = [{"kind": "repair_policy_development_sweep", "measurement": {
            "observations": [repair_row(), repair_row(ratio=1)]}}]
        result = aggregate.aggregate_diagnostics(rows, {
            "real_gpu_measurements": True, "fake_timing": False,
            "failed": 0, "planned": 1, "completed": 1,
        })
        self.assertFalse(result["profile_freeze_allowed"])
        self.assertIsNone(result["coverage_curves"])
        self.assertIsNone(result["final_consistency"]["final_commit_gamma_violations"])
        self.assertIsNone(result["repair_evidence"]["observed_quality_violations"])

    def test_proxy_dispatch_is_explicit_and_never_qualified(self):
        rows = [dict(case_id="q", source_id=s, completed_depth=d,
                     source_residual_trim_ratio=r, residual_score=j)
                for r in SCHEMA10_TRIM_GRID for d in (1, 2)
                for s, j in (("a", .1), ("b", .2))]
        _, thresholds = build_threshold_table(rows, (1, 2))
        candidates = build_selection_candidates(rows, (1, 2), thresholds, 0.001)
        self.assertIsNone(candidates[0]["metrics"]["illegal_lock_count"])
        self.assertIsNone(candidates[0]["metrics"]["selection_p95_dense_fraction"])
        with self.assertRaises(ValueError):
            select_dispatch(candidates)
        self.assertFalse(select_dispatch(candidates, diagnostic_only=True)["profile_freeze_eligible"])

    def test_replay_uses_online_margin_and_gate1(self):
        online = selector()
        rows = [ResidualCandidate("a", .1, 1, 0), ResidualCandidate("b", .11, 1, 1)]
        first = dict(completed_depth=1, counts=CandidateCounts(2, 2, 2, 2, 2),
                     candidates=rows, gate1_plan_by_source={s: plan(s, 1) for s in ("a", "b")})
        second = dict(first, completed_depth=2,
                      gate1_plan_by_source={s: plan(s, 2) for s in ("a", "b")})
        decisions = replay_production_d1d2(online, (first, second))
        self.assertEqual(decisions[0].state, "continue_probe")
        self.assertEqual(decisions[1].state, "decision_ready")
        self.assertEqual(decisions[0], online.decide(**first))

    def test_missing_gate1_cannot_be_replaced_by_residual_threshold(self):
        step = dict(completed_depth=2, counts=CandidateCounts(1, 1, 1, 1, 1),
                    candidates=[ResidualCandidate("a", .001, 1, 0)], gate1_plan_by_source={})
        self.assertEqual(replay_production_d1d2(selector(), (step,))[0].state, "abstained")

    def test_freeze_stops_replay_before_second_source(self):
        step = dict(completed_depth=1, counts=CandidateCounts(1, 1, 1, 1, 1),
                    candidates=[ResidualCandidate("a", .1, 1, 0)],
                    gate1_plan_by_source={"a": plan("a", 1)})
        self.assertEqual(len(replay_production_d1d2(selector(), (step, {"completed_depth": 2}))), 1)


if __name__ == "__main__":
    unittest.main()
