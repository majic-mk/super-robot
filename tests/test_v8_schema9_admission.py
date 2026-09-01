from __future__ import annotations

import unittest
from pathlib import Path

from probekv.config import load_config
from probekv.global_source_pool import ModelServingMode
from probekv.runtime_source_audit import audit_v8_schema9_runtime_sources
from probekv.v7_contracts import SourceVariantIdentity
from probekv.v7_source_pool import V7SourcePool
from probekv.v8_contracts import CandidateCounts, ResidualCandidate
from probekv.v8_schema8_planner import Gate1LocalPlan, Gate1MarginalLowerBound
from probekv.v8_schema9_contracts import (
    AbsoluteResidualThreshold,
    DenseKVProvenance,
    VariantMaterializationReason,
    VariantMaterializationState,
)
from probekv.v8_schema9_jobs import (
    build_schema9_h1_h5_manifests,
    build_schema9_profile_jobs,
    build_schema9_qualification_jobs,
)
from probekv.v8_schema9_materialization import (
    VariantMaterializationController,
    VariantMaterializationRequest,
)
from probekv.v8_schema9_profile import (
    VariantAdmissionProfile,
    build_variant_admission_profile,
)
from probekv.v8_schema9_selector import Schema9D1D2Selector
from probekv.v8_schema9_qualification import (
    evaluate_schema9_runtime_qualification,
    validate_schema9_h1_gate,
)


ROOT = Path(__file__).resolve().parents[1]


def _profile(**updates: object) -> VariantAdmissionProfile:
    values = {
        "code_commit": "commit",
        "cacheblend_patch_sha256": "a" * 64,
        "model_id": "model",
        "model_revision": "revision",
        "tokenizer_hash": "b" * 64,
        "source_score_trim_ratio": 0.15,
        "thresholds": (
            AbsoluteResidualThreshold(1, 0.20),
            AbsoluteResidualThreshold(2, 0.25),
        ),
        "materialization_budget_fraction": 0.02,
    }
    values.update(updates)
    return VariantAdmissionProfile(**values)


def _plan(source: str, *, passed: bool = True) -> Gate1LocalPlan:
    return Gate1LocalPlan(
        source_variant_id=source,
        selection_completed_depth=2,
        repair_check_completed_depth=2,
        first_selective_reuse_layer=3,
        dense_repair_check_sunk_ms=1.0,
        marginal_lower_bound=Gate1MarginalLowerBound(0.2, 0.3, 0.4),
        dense_marginal_same_origin_ms=(4.0 if passed else 0.5),
        gate1_gamma=1.0,
    )


class Schema9SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = Schema9D1D2Selector(
            profile=_profile(),
            strong_margin=0.60,
            stable_margin=0.30,
            residual_band_relative_tolerance=0.05,
        )

    def test_d1_absolute_failure_rescues_to_d2(self) -> None:
        candidates = (
            ResidualCandidate("a", 0.30, 2.0, 0),
            ResidualCandidate("b", 0.80, 3.0, 1),
        )
        result = self.selector.decide(
            completed_depth=1,
            counts=CandidateCounts(2, 2, 2, 2, 2),
            candidates=candidates,
            gate1_plan_by_source={"a": _plan("a")},
        )
        self.assertEqual(result.state, "continue_probe")
        self.assertEqual(result.reason, "d1_absolute_residual_failed_rescue")

    def test_d2_complete_mismatch_is_materialization_candidate(self) -> None:
        candidates = (
            ResidualCandidate("a", 0.30, 2.0, 0),
            ResidualCandidate("b", 0.40, 3.0, 1),
        )
        result = self.selector.decide(
            completed_depth=2,
            counts=CandidateCounts(2, 2, 2, 2, 2),
            candidates=candidates,
            gate1_plan_by_source={},
        )
        self.assertEqual(result.state, "abstained")
        self.assertTrue(result.materialization_candidate)
        self.assertTrue(result.selection_scope_complete)

    def test_budget_truncated_mismatch_never_materializes(self) -> None:
        result = self.selector.decide(
            completed_depth=2,
            counts=CandidateCounts(16, 16, 16, 1, 1),
            candidates=(ResidualCandidate("a", 0.30, 2.0, 0),),
            gate1_plan_by_source={},
        )
        self.assertEqual(result.reason, "insufficient_ranking_coverage")
        self.assertFalse(result.selection_scope_complete)
        self.assertFalse(result.materialization_candidate)

    def test_gate1_failure_does_not_materialize(self) -> None:
        candidates = (
            ResidualCandidate("a", 0.10, 2.0, 0),
            ResidualCandidate("b", 0.11, 3.0, 1),
        )
        result = self.selector.decide(
            completed_depth=2,
            counts=CandidateCounts(2, 2, 2, 2, 2),
            candidates=candidates,
            gate1_plan_by_source={"a": _plan("a", passed=False)},
        )
        self.assertEqual(result.reason, "d2_no_economic_source_in_compatible_band")
        self.assertFalse(result.materialization_candidate)

    def test_d2_selects_lowest_cost_inside_compatible_band(self) -> None:
        candidates = (
            ResidualCandidate("a", 0.100, 4.0, 0),
            ResidualCandidate("b", 0.103, 2.0, 1),
        )
        result = self.selector.decide(
            completed_depth=2,
            counts=CandidateCounts(2, 2, 2, 2, 2),
            candidates=candidates,
            gate1_plan_by_source={"a": _plan("a"), "b": _plan("b")},
        )
        self.assertEqual(result.state, "decision_ready")
        # Both helper plans have equal cost; residual/source-id tie breaking is deterministic.
        self.assertEqual(result.selected_source_variant_id, "a")


class Schema9MaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = VariantMaterializationController(_profile())

    def _request(self, **updates: object) -> VariantMaterializationRequest:
        values = {
            "reason": VariantMaterializationReason.CONTENT_MISS,
            "selection_scope_complete": True,
            "best_residual": None,
            "absolute_threshold": None,
            "dense_kv_provenance": DenseKVProvenance.DENSE_EXACT,
            "existing_variant_count": 0,
            "dense_reference_total_ms": 100.0,
            "estimated_materialization_ms": 1.0,
        }
        values.update(updates)
        return VariantMaterializationRequest(**values)

    def test_content_miss_exact_dense_is_admitted(self) -> None:
        decision = self.controller.decide(self._request())
        self.assertEqual(decision.state, VariantMaterializationState.ADMITTED)

    def test_partial_and_r1_repair_are_not_canonical(self) -> None:
        for provenance in (
            DenseKVProvenance.SELECTIVE_REPAIR,
            DenseKVProvenance.R1_REPAIR_EQUIVALENT,
        ):
            decision = self.controller.decide(
                self._request(dense_kv_provenance=provenance)
            )
            self.assertEqual(decision.state, VariantMaterializationState.REJECTED)

    def test_runtime_failure_never_creates_variant(self) -> None:
        decision = self.controller.decide(
            self._request(reason=VariantMaterializationReason.RUNTIME_REJECTION)
        )
        self.assertIn("not_a_new_context", decision.rejection_reason)

    def test_mismatch_requires_complete_scope_and_evidence(self) -> None:
        decision = self.controller.decide(
            self._request(
                reason=VariantMaterializationReason.ABSOLUTE_RESIDUAL_MISMATCH,
                selection_scope_complete=False,
                best_residual=0.40,
                absolute_threshold=0.25,
                existing_variant_count=4,
            )
        )
        self.assertEqual(decision.state, VariantMaterializationState.REJECTED)

    def test_materialization_budget_is_enforced(self) -> None:
        decision = self.controller.decide(
            self._request(estimated_materialization_ms=2.1)
        )
        self.assertEqual(decision.state, VariantMaterializationState.REJECTED)


class Schema9PoolAndProfileTests(unittest.TestCase):
    @staticmethod
    def _identity(index: int) -> SourceVariantIdentity:
        return SourceVariantIdentity(
            reuse_content_key="content",
            historical_prefix_digest=f"prefix-{index}",
            position_ids_digest=f"position-{index}",
            occurrence_id=f"occurrence-{index}",
            model_math_signature="model",
        )

    def test_replacement_planning_is_explicit_and_snapshot_checked(self) -> None:
        pool = V7SourcePool(
            serving_mode=ModelServingMode.SINGLE,
            max_variants_per_content=2,
            probation_observations=2,
        )
        pool.activate_namespace("model")
        variants = []
        for index in range(2):
            identity = self._identity(index)
            variants.append(
                pool.register_variant(
                    identity,
                    canonical_source_state_digest=f"source-{index}",
                    summary_digest=f"summary-{index}",
                )
            )
            for _ in range(2):
                pool.record_observation(
                    "model", "content", identity.source_variant_id,
                    lookup_hit=True, compared=True,
                )
        victim = pool.plan_variant_replacement("model", "content")
        self.assertIsNotNone(victim)
        identity = self._identity(3)
        with self.assertRaises(RuntimeError):
            pool.register_variant(
                identity,
                canonical_source_state_digest="source-3",
                summary_digest="summary-3",
                expected_replacement_source_variant_id="stale-victim",
            )
        self.assertEqual(len(pool.variants_for_content("model", "content", include_unavailable=True)), 2)

    def test_frozen_variant_profile_has_own_sha(self) -> None:
        profile = build_variant_admission_profile(
            code_commit="commit",
            cacheblend_patch_sha256="a" * 64,
            model_id="model",
            model_revision="revision",
            tokenizer_hash="b" * 64,
            source_score_trim_ratio=0.15,
            thresholds=(
                AbsoluteResidualThreshold(1, 0.2),
                AbsoluteResidualThreshold(2, 0.25),
            ),
            development_partition_sha256="c" * 64,
            frozen=True,
        )
        self.assertEqual(len(profile.profile_sha256), 64)
        self.assertEqual(profile.threshold_for_depth(2), 0.25)

    def test_config_and_jobs_are_schema9_isolated(self) -> None:
        config = load_config(
            str(ROOT / "configs" / "local_system_v8_schema9_absolute_variant_admission.json")
        )
        self.assertEqual(config.v8_schema_version, 9)
        self.assertTrue(config.absolute_residual_admission_enabled)
        self.assertEqual(len(build_schema9_profile_jobs("mistral")), 13)
        manifests = build_schema9_h1_h5_manifests("qwen")
        self.assertEqual(set(manifests), {"H1", "H2", "H3", "H4", "H5"})
        self.assertFalse(manifests["H5"]["locked_test_accessed"])
        audit = audit_v8_schema9_runtime_sources(ROOT)
        self.assertTrue(audit["runtime_source_ready"], audit["failures"])

    def test_schema9_qualification_is_140_jobs_and_version_isolated(self) -> None:
        shas = ("1" * 64, "2" * 64, "3" * 64, "4" * 64)
        jobs = build_schema9_qualification_jobs(
            model_key="mistral",
            selection_depth_profile_sha256=shas[0],
            variant_admission_profile_sha256=shas[1],
            repair_policy_profile_sha256=shas[2],
            runtime_cost_profile_sha256=shas[3],
        )
        self.assertEqual(len(jobs), 140)
        audit = {
            "protocol_version": 8,
            "schema_version": 9,
            "planned": 140,
            "completed": 140,
            "failed": 0,
            "cuda_event_timing": True,
            "fake_timing": False,
            "absolute_residual_admission_verified": True,
            "dense_exact_materialization_verified": True,
            "partial_repair_promotion_forbidden": True,
            "r1_dense_equivalence": True,
            "source_digest_unchanged": True,
        }
        gate = evaluate_schema9_runtime_qualification(
            runtime_audit=audit,
            code_commit="commit",
            model_id="model",
            model_revision="revision",
            cacheblend_patch_sha256="0" * 64,
            selection_depth_profile_sha256=shas[0],
            variant_admission_profile_sha256=shas[1],
            repair_policy_profile_sha256=shas[2],
            runtime_cost_profile_sha256=shas[3],
            manifest_sha256="5" * 64,
        )
        validate_schema9_h1_gate(
            gate,
            expected_code_commit="commit",
            expected_model_id="model",
            expected_profile_shas=shas,
        )
        with self.assertRaises(ValueError):
            validate_schema9_h1_gate(
                {**gate, "schema_version": 8},
                expected_code_commit="commit",
                expected_model_id="model",
                expected_profile_shas=shas,
            )


if __name__ == "__main__":
    unittest.main()
