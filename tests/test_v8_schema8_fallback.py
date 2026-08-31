import unittest

from probekv.gates import gate_h2_fast_selection, gate_h5
from probekv.v8_schema8_fallback import (
    FastSelectionQualification,
    SelectionRuntimePath,
    h2_selection_candidate_audit,
    resolve_h2_selection_candidate,
    resolve_selection_runtime_path,
    selection_dispatch_audit,
)
from probekv.v8_schema8_profile import (
    Schema8ProfileProvenance,
    build_selection_depth_profile_v8,
)


def frozen_selection_profile(depths):
    provenance = Schema8ProfileProvenance(
        profile_kind="selection_depth",
        code_commit="a" * 40,
        cacheblend_patch_sha256="b" * 64,
        model_id="mistral",
        model_revision="c" * 40,
        tokenizer_hash="d" * 64,
        gpu_uuid="GPU-real",
        measurement_sha256="e" * 64,
        frozen=True,
    )
    return build_selection_depth_profile_v8(
        provenance=provenance,
        allowed_completed_depths=depths,
        source_score_trim_ratio=0.15,
    )


def qualification(**overrides):
    values = {
        "state_availability": 0.995,
        "selection_coverage": 0.85,
        "early_resolution_rate_at_completed_depth5": 0.85,
        "wrong_early_lock_rate": 0.04,
        "mean_stable_normalized_oracle_regret": 0.08,
        "selection_critical_path_p95_fraction": 0.04,
        "selection_budget_realized_overrun_rate": 0.04,
        "illegal_lock_count": 0,
        "budget_admission_violation_count": 0,
    }
    values.update(overrides)
    return FastSelectionQualification(**values)


class Schema8FallbackTests(unittest.TestCase):
    def dispatch(self, *, profile, metrics, legacy=False, model="mistral"):
        return resolve_selection_runtime_path(
            model_family=model,
            selection_profile=profile,
            fast_selection_qualification=metrics,
            repair_policy_profile_frozen=True,
            runtime_cost_profile_frozen=True,
            schema8_runtime_qualified=True,
            legacy_three_gate_runtime_qualified=legacy,
        )

    def test_d1_fast_path_requires_every_gate(self):
        row = self.dispatch(
            profile=frozen_selection_profile((1,)),
            metrics=qualification(),
            legacy=True,
        )
        self.assertIs(row.path, SelectionRuntimePath.D1_ONLY_FAST)
        self.assertEqual(row.checkpoint_depths, (1,))
        self.assertTrue(row.fast_feature_set_enabled)
        self.assertIn(
            "request_layer_uniform_io_balanced_repair", row.enabled_feature_set
        )

    def test_h2_candidate_does_not_require_later_repair_profile(self):
        profile = frozen_selection_profile((1, 2))
        candidate = resolve_h2_selection_candidate(
            model_family="mistral",
            selection_profile=profile,
            fast_selection_qualification=qualification(),
            legacy_three_gate_runtime_qualified=True,
        )
        self.assertIs(candidate.path, SelectionRuntimePath.D1_D2_RESCUE_FAST)
        self.assertEqual(
            len(h2_selection_candidate_audit(candidate)["h2_selection_candidate_sha256"]),
            64,
        )
        final = resolve_selection_runtime_path(
            model_family="mistral",
            selection_profile=profile,
            fast_selection_qualification=qualification(),
            repair_policy_profile_frozen=False,
            runtime_cost_profile_frozen=True,
            schema8_runtime_qualified=True,
            legacy_three_gate_runtime_qualified=True,
            h2_candidate=candidate,
        )
        self.assertIs(
            final.path, SelectionRuntimePath.LEGACY_MULTICHECKPOINT_THREE_GATE
        )

    def test_h2_legacy_candidate_cannot_later_upgrade_to_fast(self):
        profile = frozen_selection_profile((1, 2))
        candidate = resolve_h2_selection_candidate(
            model_family="mistral",
            selection_profile=profile,
            fast_selection_qualification=qualification(selection_coverage=0.79),
            legacy_three_gate_runtime_qualified=True,
        )
        self.assertIs(
            candidate.path, SelectionRuntimePath.LEGACY_MULTICHECKPOINT_THREE_GATE
        )
        final = resolve_selection_runtime_path(
            model_family="mistral",
            selection_profile=profile,
            fast_selection_qualification=qualification(),
            repair_policy_profile_frozen=True,
            runtime_cost_profile_frozen=True,
            schema8_runtime_qualified=True,
            legacy_three_gate_runtime_qualified=True,
            h2_candidate=candidate,
        )
        self.assertIs(
            final.path, SelectionRuntimePath.LEGACY_MULTICHECKPOINT_THREE_GATE
        )

    def test_d1d2_fast_path(self):
        row = self.dispatch(
            profile=frozen_selection_profile((1, 2)),
            metrics=qualification(),
            legacy=True,
        )
        self.assertIs(row.path, SelectionRuntimePath.D1_D2_RESCUE_FAST)
        self.assertEqual(row.checkpoint_depths, (1, 2))

    def test_failed_fast_gate_selects_independently_qualified_legacy(self):
        row = self.dispatch(
            profile=frozen_selection_profile((1, 2)),
            metrics=qualification(wrong_early_lock_rate=0.06),
            legacy=True,
        )
        self.assertIs(
            row.path, SelectionRuntimePath.LEGACY_MULTICHECKPOINT_THREE_GATE
        )
        self.assertEqual(row.checkpoint_depths, (1, 2, 4, 5, 8))
        self.assertFalse(row.fast_feature_set_enabled)
        self.assertEqual(row.enabled_feature_set, ())
        self.assertEqual(
            selection_dispatch_audit(row)["runtime_contract"],
            "legacy_predicted_gate2_refined_gate3",
        )
        self.assertEqual(
            len(
                selection_dispatch_audit(row)[
                    "selection_runtime_dispatch_sha256"
                ]
            ),
            64,
        )

    def test_missing_fast_profiles_still_allow_qwen_legacy_fallback(self):
        row = self.dispatch(
            profile=None,
            metrics=None,
            legacy=True,
            model="qwen2.5",
        )
        self.assertIs(
            row.path, SelectionRuntimePath.LEGACY_MULTICHECKPOINT_THREE_GATE
        )
        self.assertEqual(row.checkpoint_depths, (1, 2, 4, 5, 7))

    def test_model_mismatched_fast_profile_cannot_unlock_qwen(self):
        row = self.dispatch(
            profile=frozen_selection_profile((1, 2)),
            metrics=qualification(),
            legacy=True,
            model="qwen2.5",
        )
        self.assertIs(
            row.path, SelectionRuntimePath.LEGACY_MULTICHECKPOINT_THREE_GATE
        )
        self.assertEqual(row.runtime_schema_version, 7)

    def test_unqualified_fast_and_legacy_are_dense(self):
        row = self.dispatch(
            profile=None,
            metrics=None,
            legacy=False,
        )
        self.assertIs(row.path, SelectionRuntimePath.FULL_DENSE_SAFE_FALLBACK)
        self.assertEqual(row.checkpoint_depths, ())

    def test_fast_gate_and_h5_prerequisites(self):
        self.assertTrue(gate_h2_fast_selection(qualification()).passed)
        self.assertFalse(
            gate_h2_fast_selection(
                qualification(selection_coverage=0.79)
            ).passed
        )
        self.assertTrue(
            gate_h5(
                h1_passed=True,
                final_runtime_dispatch_frozen=True,
                h3_passed=True,
                h4_passed=True,
                ttft_improvement=0.12,
                throughput_improvement=0.11,
                p95_improvement=0.06,
            ).passed
        )


if __name__ == "__main__":
    unittest.main()
