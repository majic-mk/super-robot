from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from probekv.config import load_config
from probekv.global_source_pool import ModelServingMode
from probekv.runtime_source_audit import audit_v8_schema10_runtime_sources
from probekv.v7_contracts import (
    CanonicalVariantProvenance,
    SourceVariantIdentity,
    SourceVariantMaturity,
    SourceVariantState,
)
from probekv.v7_source_pool import V7SourcePool
from probekv.v8_contracts import CandidateCounts, ResidualCandidate
from probekv.v8_schema8_planner import Gate1LocalPlan, Gate1MarginalLowerBound
from probekv.v8_schema10_contracts import (
    AbsoluteResidualThreshold,
    DenseKVProvenance,
    Gate1Mode,
    VariantMaterializationReasonV10,
    VariantMaterializationStateV10,
)
from probekv.v8_schema10_jobs import (
    build_schema10_h1_h5_manifests,
    build_schema10_profile_jobs,
    build_schema10_qualification_jobs,
)
from probekv.v8_schema10_materialization import (
    VariantMaterializationControllerV10,
    VariantMaterializationRequestV10,
    materialize_exact_dense_variant_v10,
)
from probekv.v8_schema10_metrics import (
    MaterializationOutcome,
    VariantGrowthPoint,
    materialization_quality_metrics,
    summarize_variant_growth,
)
from probekv.v8_schema10_preparation import (
    PreparationCostObservation,
    assert_preparation_contract,
    evaluate_gate1_counterfactual,
)
from probekv.v8_schema10_pool import Schema10SourcePool
from probekv.v8_schema10_profile import (
    PreparationPolicyProfile,
    VariantAdmissionProfileV10,
    build_preparation_policy_profile,
    build_variant_admission_profile_v10,
)
from probekv.v8_schema10_qualification import (
    evaluate_schema10_runtime_qualification,
    validate_schema10_h1_gate,
)
from probekv.v8_schema10_selector import Schema10D1D2Selector


ROOT = Path(__file__).resolve().parents[1]


def _variant_profile(**updates: object) -> VariantAdmissionProfileV10:
    values = {
        "code_commit": "commit",
        "cacheblend_patch_sha256": "a" * 64,
        "model_id": "model",
        "model_revision": "revision",
        "tokenizer_hash": "b" * 64,
        "source_residual_trim_ratio": 0.15,
        "thresholds": (
            AbsoluteResidualThreshold(1, 0.20),
            AbsoluteResidualThreshold(2, 0.25),
        ),
    }
    values.update(updates)
    return VariantAdmissionProfileV10(**values)


def _preparation_profile(mode: Gate1Mode = Gate1Mode.EXPLICIT_BARRIER) -> PreparationPolicyProfile:
    return PreparationPolicyProfile(
        code_commit="commit",
        model_id="model",
        runtime_policy="dense_selection_barrier",
        gate1_mode=mode,
    )


def _identity(index: int, content: str = "content") -> SourceVariantIdentity:
    return SourceVariantIdentity(
        reuse_content_key=content,
        historical_prefix_digest=f"prefix-{index}",
        position_ids_digest=f"position-{index}",
        occurrence_id=f"occurrence-{index}",
        model_math_signature="model",
    )


def _gate1(source: str, *, passed: bool = True) -> Gate1LocalPlan:
    return Gate1LocalPlan(
        source_variant_id=source,
        selection_completed_depth=2,
        repair_check_completed_depth=2,
        first_selective_reuse_layer=3,
        dense_repair_check_sunk_ms=1.0,
        marginal_lower_bound=Gate1MarginalLowerBound(0.2, 0.3, 0.4),
        dense_marginal_same_origin_ms=4.0 if passed else 0.5,
        gate1_gamma=1.0,
    )


class Schema10VariantStateTests(unittest.TestCase):
    def _pool(self) -> V7SourcePool:
        pool = V7SourcePool(
            serving_mode=ModelServingMode.SINGLE,
            max_variants_per_content=16,
            bounded_probation=True,
            probation_observations=2,
            max_protected_probation_per_content=2,
            probation_lookup_opportunities=2,
        )
        pool.activate_namespace("model")
        return pool

    def test_correctness_maturity_and_lifecycle_are_orthogonal(self) -> None:
        pool = self._pool()
        row = pool.register_variant(
            _identity(0),
            canonical_source_state_digest="source",
            summary_digest="summary",
        )
        self.assertIs(row.canonical_provenance, CanonicalVariantProvenance.DENSE_EXACT_CANONICAL)
        self.assertIs(row.maturity, SourceVariantMaturity.PROBATION)
        self.assertIs(row.state, SourceVariantState.ACTIVE)

    def test_probation_promotes_after_two_comparisons(self) -> None:
        pool = self._pool()
        row = pool.register_variant(_identity(0), canonical_source_state_digest="s", summary_digest="m")
        for _ in range(2):
            pool.record_observation("model", "content", row.source_variant_id, lookup_hit=True, compared=True)
        self.assertIs(row.maturity, SourceVariantMaturity.VERIFIED)

    def test_probation_expires_after_two_content_lookup_opportunities(self) -> None:
        pool = self._pool()
        row = pool.register_variant(_identity(0), canonical_source_state_digest="s", summary_digest="m")
        pool.record_content_lookup_opportunity("model", "content")
        self.assertIs(row.maturity, SourceVariantMaturity.PROBATION)
        pool.record_content_lookup_opportunity("model", "content")
        self.assertIs(row.maturity, SourceVariantMaturity.EXPIRED)

    def test_only_two_probation_variants_remain_protected(self) -> None:
        pool = self._pool()
        rows = [
            pool.register_variant(_identity(i), canonical_source_state_digest=f"s{i}", summary_digest=f"m{i}")
            for i in range(3)
        ]
        self.assertTrue(all(row.maturity is SourceVariantMaturity.PROBATION for row in rows))
        self.assertFalse(rows[0].probation_protected)
        self.assertEqual(sum(row.probation_protected for row in rows), 2)

    def test_schema10_pool_binds_probation_to_variant_profile(self) -> None:
        pool = Schema10SourcePool(profile=_variant_profile())
        pool.activate_namespace("model")
        row = pool.register_variant(_identity(0), canonical_source_state_digest="s", summary_digest="m")
        pool.finish_content_lookup("model", "content")
        pool.finish_content_lookup("model", "content")
        self.assertIs(row.maturity, SourceVariantMaturity.EXPIRED)

    def test_second_comparison_in_second_lookup_verifies_before_expiry(self) -> None:
        pool = Schema10SourcePool(profile=_variant_profile())
        pool.activate_namespace("model")
        row = pool.register_variant(
            _identity(0), canonical_source_state_digest="s", summary_digest="m"
        )
        for _ in range(2):
            pool.record_observation(
                "model", "content", row.source_variant_id,
                lookup_hit=True, compared=True,
            )
            pool.finish_content_lookup("model", "content")
        self.assertIs(row.maturity, SourceVariantMaturity.VERIFIED)
        self.assertFalse(row.probation_protected)


class Schema10MaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = VariantMaterializationControllerV10(_variant_profile())

    def _request(self, **updates: object) -> VariantMaterializationRequestV10:
        values = {
            "reason": VariantMaterializationReasonV10.CONTENT_MISS,
            "correctness_eligible_k": 0,
            "compared_k": 0,
            "best_residual": None,
            "absolute_threshold": None,
            "dense_kv_provenance": DenseKVProvenance.DENSE_EXACT,
            "existing_variant_count": 0,
            "dense_reference_total_ms": 100.0,
            "estimated_materialization_ms": 1.0,
        }
        values.update(updates)
        return VariantMaterializationRequestV10(**values)

    def test_only_exact_dense_can_materialize(self) -> None:
        for provenance in (DenseKVProvenance.SELECTIVE_REPAIR, DenseKVProvenance.R1_REPAIR_EQUIVALENT):
            result = self.controller.decide(self._request(dense_kv_provenance=provenance))
            self.assertIs(result.state, VariantMaterializationStateV10.REJECTED)

    def test_complete_mismatch_proves_novelty(self) -> None:
        result = self.controller.decide(
            self._request(
                reason=VariantMaterializationReasonV10.COMPLETE_SCOPE_ABSOLUTE_MISMATCH,
                correctness_eligible_k=4,
                compared_k=4,
                best_residual=0.40,
                absolute_threshold=0.25,
                existing_variant_count=4,
            )
        )
        self.assertIs(result.state, VariantMaterializationStateV10.ADMITTED)
        self.assertTrue(result.context_novelty_proven)

    def test_budget_truncated_exploration_is_canonical_but_not_novelty(self) -> None:
        result = self.controller.decide(
            self._request(
                reason=VariantMaterializationReasonV10.BUDGET_TRUNCATED_EXPLORATION,
                correctness_eligible_k=16,
                compared_k=4,
                best_residual=0.40,
                absolute_threshold=0.25,
                existing_variant_count=15,
                exploration_authorized=True,
            )
        )
        self.assertIs(result.state, VariantMaterializationStateV10.ADMITTED)
        self.assertFalse(result.context_novelty_proven)

    def test_content_miss_cannot_implicitly_replace_at_variant_limit(self) -> None:
        result = self.controller.decide(
            self._request(existing_variant_count=16)
        )
        self.assertIs(result.state, VariantMaterializationStateV10.REJECTED)
        self.assertEqual(
            result.rejection_reason,
            "content_miss_cannot_implicitly_replace_at_variant_limit",
        )

    def test_replacement_requires_complete_scope_and_its_own_budget(self) -> None:
        request = self._request(
            reason=VariantMaterializationReasonV10.COMPLETE_SCOPE_ABSOLUTE_MISMATCH,
            correctness_eligible_k=16,
            compared_k=16,
            best_residual=0.40,
            absolute_threshold=0.25,
            existing_variant_count=16,
            estimated_replacement_ms=2.0,
        )
        result = self.controller.decide(
            request, replacement_source_variant_id="victim"
        )
        self.assertIs(result.state, VariantMaterializationStateV10.REJECTED)
        self.assertEqual(result.rejection_reason, "replacement_budget_exceeded")

    def test_admitted_exact_dense_publishes_probation_variant(self) -> None:
        pool = Schema10SourcePool(profile=_variant_profile())
        pool.activate_namespace("model")
        decision = self.controller.decide(self._request())
        row = materialize_exact_dense_variant_v10(
            decision=decision,
            pool=pool,
            identity=_identity(0),
            canonical_source_state_digest="state",
            summary_digest="summary",
        )
        self.assertIs(row.maturity, SourceVariantMaturity.PROBATION)
        self.assertIs(
            row.canonical_provenance,
            CanonicalVariantProvenance.DENSE_EXACT_CANONICAL,
        )
        self.assertEqual(row.materialization_reason, "content_miss")

    def test_exploration_quota_is_recounted_from_pool_metadata(self) -> None:
        pool = Schema10SourcePool(profile=_variant_profile())
        pool.activate_namespace("model")
        pool.register_variant(
            _identity(0),
            canonical_source_state_digest="state",
            summary_digest="summary",
            materialization_reason="budget_truncated_exploration",
        )
        request = self._request(
            reason=VariantMaterializationReasonV10.BUDGET_TRUNCATED_EXPLORATION,
            correctness_eligible_k=4,
            compared_k=2,
            existing_variant_count=1,
            exploration_materializations_for_content=0,
            exploration_authorized=True,
        )
        with self.assertRaisesRegex(RuntimeError, "stale exploration quota"):
            self.controller.decide_with_pool(
                request,
                pool=pool,
                model_math_signature="model",
                reuse_content_key="content",
            )

    def test_publication_never_implicitly_replaces_after_capacity_race(self) -> None:
        profile = _variant_profile(
            max_variants_per_content=1,
            exploration_quota_per_content=1,
            max_protected_probation_per_content=1,
        )
        pool = Schema10SourcePool(profile=profile)
        pool.activate_namespace("model")
        pool.register_variant(
            _identity(0), canonical_source_state_digest="old", summary_digest="old-summary"
        )
        controller = VariantMaterializationControllerV10(profile)
        decision = controller.decide(self._request(existing_variant_count=0))
        with self.assertRaisesRegex(RuntimeError, "capacity changed"):
            materialize_exact_dense_variant_v10(
                decision=decision,
                pool=pool,
                identity=_identity(1),
                canonical_source_state_digest="new",
                summary_digest="new-summary",
            )

    def test_exploration_cannot_replace_at_k16(self) -> None:
        result = self.controller.decide(
            self._request(
                reason=VariantMaterializationReasonV10.BUDGET_TRUNCATED_EXPLORATION,
                correctness_eligible_k=16,
                compared_k=4,
                existing_variant_count=16,
                exploration_authorized=True,
            ),
            replacement_source_variant_id="victim",
        )
        self.assertIs(result.state, VariantMaterializationStateV10.REJECTED)
        self.assertIn("cannot_replace", result.rejection_reason)


class Schema10SelectorAndCounterfactualTests(unittest.TestCase):
    def test_partial_scope_mismatch_proposes_exploration(self) -> None:
        selector = Schema10D1D2Selector(
            variant_profile=_variant_profile(),
            preparation_profile=_preparation_profile(),
            strong_margin=0.6,
            stable_margin=0.3,
            residual_band_relative_tolerance=0.05,
        )
        result = selector.decide(
            completed_depth=2,
            counts=CandidateCounts(16, 16, 16, 4, 4),
            candidates=tuple(ResidualCandidate(f"s{i}", 0.3 + i / 10, 1.0, i) for i in range(4)),
            gate1_plan_by_source={},
        )
        self.assertIs(result.materialization_reason, VariantMaterializationReasonV10.BUDGET_TRUNCATED_EXPLORATION)

    def test_fused_advisory_can_select_failed_gate1_but_final_contract_remains(self) -> None:
        selector = Schema10D1D2Selector(
            variant_profile=_variant_profile(),
            preparation_profile=_preparation_profile(Gate1Mode.FUSED_ADVISORY),
            strong_margin=0.6,
            stable_margin=0.3,
            residual_band_relative_tolerance=0.05,
        )
        candidate = ResidualCandidate("s", 0.1, 1.0, 0)
        result = selector.decide(
            completed_depth=2,
            counts=CandidateCounts(1, 1, 1, 1, 1),
            candidates=(candidate,),
            gate1_plan_by_source={"s": _gate1("s", passed=False)},
        )
        self.assertEqual(result.state, "decision_ready")
        self.assertTrue(result.gate1_was_advisory_failure)
        with self.assertRaises(RuntimeError):
            assert_preparation_contract(atomic_reservation_acquired=False, final_commit_admitted=True, selective_reuse_started=True)
        with self.assertRaises(RuntimeError):
            assert_preparation_contract(atomic_reservation_acquired=True, final_commit_admitted=False, selective_reuse_started=True)

    def test_gate1_counterfactual_uses_waste_not_pass_rate(self) -> None:
        rows = [
            PreparationCostObservation(
                request_id=f"r{i}", dense_reference_total_ms=100.0, gate1_passed=True,
                additional_winner_full_kv_bytes_without_gate1=0,
                additional_visible_copy_ms_without_gate1=0,
                additional_pinned_staging_ms_without_gate1=0,
                additional_copy_interference_ms_without_gate1=0,
                additional_hbm_reservation_byte_ms_without_gate1=0,
                additional_wasted_preparation_ms_without_gate1=0.1,
                ttft_delta_ms_without_gate1=0.1,
                counterfactual_path_economically_invalid=False,
                counterfactual_final_commit_admitted=True,
            ) for i in range(10)
        ]
        summary = evaluate_gate1_counterfactual(rows, total_winner_full_kv_bytes_with_gate1=1, profile=_preparation_profile())
        self.assertIs(summary.recommended_gate1_mode, Gate1Mode.FUSED_ADVISORY)


class Schema10MetricsProfileJobsTests(unittest.TestCase):
    def test_materialization_metric_denominators_are_separate(self) -> None:
        rows = (
            MaterializationOutcome("a", VariantMaterializationReasonV10.COMPLETE_SCOPE_ABSOLUTE_MISMATCH, True, True, True, True, True, True, 10),
            MaterializationOutcome("b", VariantMaterializationReasonV10.BUDGET_TRUNCATED_EXPLORATION, False, None, True, False, False, False, 20),
        )
        metrics = materialization_quality_metrics(rows)
        self.assertEqual(metrics["novelty_precision"], 1.0)
        self.assertEqual(metrics["exploration_yield_at_32"], 1.0)
        self.assertEqual(metrics["useful_materialization_precision_at_32"], 0.5)

    def test_growth_summary_reports_saturation(self) -> None:
        rows = (
            VariantGrowthPoint(
                0, 1, None, None, 10, 20, 1, 0, 0, 0,
                materialization_write_bytes=100,
                ttft_ms=100.0,
            ),
            VariantGrowthPoint(
                1, 4, None, None, 10, 20, 1, 0, 0, 0,
                materialization_write_bytes=100,
                ttft_ms=90.0,
            ),
            VariantGrowthPoint(
                2, 16, 1, 3, 20, 30, 0, 15, 1, 2,
                materialization_write_bytes=100,
                replacements_this_request=1,
                miss_to_reuse_conversion=True,
                marginal_reuse_admission_improvement_ms=9.0,
                first_selection_delay_requests=3,
                ttft_ms=80.0,
                steady_state=True,
            ),
        )
        summary = summarize_variant_growth(rows)
        self.assertEqual(summary["final_variant_count"], 16)
        self.assertAlmostEqual(summary["saturation_probability_k16"], 1 / 3)
        self.assertEqual(summary["materialization_write_amplification_bytes"], 300)
        self.assertAlmostEqual(summary["replacement_frequency"], 1 / 3)
        self.assertAlmostEqual(summary["miss_to_reuse_conversion_rate"], 1 / 3)
        self.assertEqual(summary["mean_first_selection_delay_requests"], 3)
        self.assertEqual(summary["warmup_mean_ttft_ms"], 95.0)
        self.assertEqual(summary["steady_state_mean_ttft_ms"], 80.0)

    def test_schema10_profiles_have_independent_shas(self) -> None:
        variant = build_variant_admission_profile_v10(
            code_commit="commit", cacheblend_patch_sha256="a" * 64,
            model_id="model", model_revision="rev", tokenizer_hash="b" * 64,
            source_residual_trim_ratio=0.15,
            thresholds=(AbsoluteResidualThreshold(1, 0.2), AbsoluteResidualThreshold(2, 0.25)),
            development_partition_sha256="c" * 64, frozen=True,
        )
        preparation = build_preparation_policy_profile(
            code_commit="commit", model_id="model", runtime_policy="dense_selection_barrier",
            gate1_mode=Gate1Mode.EXPLICIT_BARRIER,
            development_partition_sha256="d" * 64, frozen=True,
        )
        self.assertEqual(len(variant.profile_sha256), 64)
        self.assertEqual(len(preparation.profile_sha256), 64)
        self.assertNotEqual(variant.profile_sha256, preparation.profile_sha256)

    def test_config_jobs_audit_and_hypotheses(self) -> None:
        for config_name in ("local_system_v8_schema10_gate1_barrier.json", "local_system_v8_schema10_gate1_counterfactual.json"):
            config = load_config(str(ROOT / "configs" / config_name))
            self.assertEqual(config.v8_schema_version, 10)
        self.assertEqual(len(build_schema10_profile_jobs("mistral")), 16)
        manifests = build_schema10_h1_h5_manifests("qwen")
        self.assertEqual(set(manifests), {"H1", "H2", "H3", "H4", "H5"})
        self.assertEqual(manifests["H2"]["source_residual_trim_ratio_grid"], [0.10, 0.15, 0.20, 0.25, 0.30])
        audit = audit_v8_schema10_runtime_sources(ROOT)
        self.assertTrue(audit["runtime_source_ready"], audit["failures"])

    def test_schema10_rejects_schema9_trim_field_alias(self) -> None:
        source = json.loads(
            (ROOT / "configs" / "local_system_v8_schema10_gate1_barrier.json").read_text(
                encoding="utf-8"
            )
        )
        source["source_score_trim_ratio"] = 0.15
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rejects schema9"):
                load_config(str(path))

    def test_qualification_is_140_jobs_and_schema_isolated(self) -> None:
        shas = tuple(str(i) * 64 for i in range(1, 6))
        jobs = build_schema10_qualification_jobs(
            model_key="mistral",
            selection_depth_profile_sha256=shas[0],
            variant_admission_profile_sha256=shas[1],
            preparation_policy_profile_sha256=shas[2],
            repair_policy_profile_sha256=shas[3],
            runtime_cost_profile_sha256=shas[4],
        )
        self.assertEqual(len(jobs), 140)
        audit = {
            "protocol_version": 8, "schema_version": 10,
            "planned": 140, "completed": 140, "failed": 0,
            "cuda_event_timing": True, "fake_timing": False,
            "dense_exact_materialization_verified": True,
            "bounded_probation_verified": True,
            "exploration_novelty_separation_verified": True,
            "gate1_counterfactual_verified": True,
            "atomic_preparation_reservation_verified": True,
            "final_commit_admission_verified": True,
            "r1_dense_equivalence": True,
            "code_commit": "commit",
            "model_id": "model",
            "model_revision": "rev",
            "cacheblend_patch_sha256": "0" * 64,
            "selection_depth_profile_sha256": shas[0],
            "variant_admission_profile_sha256": shas[1],
            "preparation_policy_profile_sha256": shas[2],
            "repair_policy_profile_sha256": shas[3],
            "runtime_cost_profile_sha256": shas[4],
            "manifest_sha256": "6" * 64,
            "gpu_uuid": "GPU-test",
            "runtime_environment_hash": "7" * 64,
            "profiles_frozen": True,
        }
        gate = evaluate_schema10_runtime_qualification(
            runtime_audit=audit, code_commit="commit", model_id="model", model_revision="rev",
            cacheblend_patch_sha256="0" * 64,
            selection_depth_profile_sha256=shas[0], variant_admission_profile_sha256=shas[1],
            preparation_policy_profile_sha256=shas[2], repair_policy_profile_sha256=shas[3],
            runtime_cost_profile_sha256=shas[4], manifest_sha256="6" * 64,
        )
        validate_schema10_h1_gate(
            gate,
            expected_code_commit="commit",
            expected_model_id="model",
            expected_model_revision="rev",
            expected_cacheblend_patch_sha256="0" * 64,
            expected_manifest_sha256="6" * 64,
            expected_profile_shas=shas,
        )
        with self.assertRaises(ValueError):
            validate_schema10_h1_gate(
                {**gate, "schema_version": 9},
                expected_code_commit="commit",
                expected_model_id="model",
                expected_model_revision="rev",
                expected_cacheblend_patch_sha256="0" * 64,
                expected_manifest_sha256="6" * 64,
                expected_profile_shas=shas,
            )


if __name__ == "__main__":
    unittest.main()
