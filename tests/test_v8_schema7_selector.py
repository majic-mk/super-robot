import unittest

from probekv.v8_contracts import CandidateCounts, ResidualCandidate, SelectorPolicyProfile
from probekv.v8_schema7_contracts import SourceSelectionDepthPolicy
from probekv.v8_schema7_selector import (
    Schema7DepthSelector,
    evaluate_wrong_early_lock_shadow,
    schema7_checkpoint_depths,
)


class Schema7SelectorTests(unittest.TestCase):
    def profile(self):
        return SelectorPolicyProfile(
            "p", "m", "causal_commit_wait", (1, 2), 2,
            0.3, 0.6, 0.05,
        )

    def test_d1_gate1_failure_rescues_at_d2(self):
        selector = Schema7DepthSelector(
            policy=SourceSelectionDepthPolicy.D1_D2_RESCUE,
            profile=self.profile(),
        )
        counts = CandidateCounts(2, 2, 2, 2, 2)
        trace = selector.evaluate_trace(
            counts_by_depth={1: counts, 2: counts},
            candidates_by_depth={
                1: (ResidualCandidate("a", 0.01, 90, 0), ResidualCandidate("b", 0.2, 91, 1)),
                2: (ResidualCandidate("a", 0.01, 10, 0), ResidualCandidate("b", 0.2, 11, 1)),
            },
            shared_sunk_ms_by_depth={1: 1, 2: 2},
            dense_reference_ms=100,
            gate1_dense_remaining_ms_by_depth={1: 100, 2: 100},
        )
        self.assertEqual(len(trace.decisions), 2)
        self.assertEqual(trace.locked_completed_depth, 2)

    def test_shadow_wrong_lock_is_audit_only(self):
        shadow = evaluate_wrong_early_lock_shadow(
            chosen_source_variant_id="a",
            deep_candidates=(
                ResidualCandidate("a", 0.2, 1, 0),
                ResidualCandidate("b", 0.1, 1, 1),
            ),
        )
        self.assertTrue(shadow.wrong_early_lock)
        self.assertEqual(shadow.oracle_source_variant_id, "b")

    def test_frozen_depth_sets(self):
        self.assertEqual(
            schema7_checkpoint_depths(
                policy=SourceSelectionDepthPolicy.LEGACY_MULTICHECKPOINT,
                model_family="mistral",
            ),
            (1, 2, 4, 5, 8),
        )

    def test_deep_oracle_waits_for_last_depth_and_uses_full_set(self):
        selector = Schema7DepthSelector(
            policy=SourceSelectionDepthPolicy.DEEP_FULL_CANDIDATE_ORACLE,
            profile=self.profile(),
        )
        counts = CandidateCounts(2, 2, 2, 2, 2)
        trace = selector.evaluate_trace(
            counts_by_depth={1: counts, 2: counts},
            candidates_by_depth={
                1: (ResidualCandidate("a", 0.01, 1, 0), ResidualCandidate("b", 0.2, 1, 1)),
                2: (ResidualCandidate("a", 0.2, 1, 0), ResidualCandidate("b", 0.01, 1, 1)),
            },
            shared_sunk_ms_by_depth={1: 1, 2: 2},
            dense_reference_ms=100,
            gate1_dense_remaining_ms_by_depth={1: 100, 2: 100},
        )
        self.assertEqual(trace.locked_completed_depth, 2)
        self.assertEqual(trace.final_decision.selected_source_variant_id, "b")


if __name__ == "__main__":
    unittest.main()
