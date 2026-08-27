import unittest

from probekv.v8_contracts import (
    CandidateCounts,
    InsufficientRankingPolicy,
    ResidualCandidate,
    ResidualLockReason,
    ResidualSelectionState,
    SelectorPolicyProfile,
)
from probekv.v8_selector import (
    ComparisonBudget,
    TrainingFreeResidualKSelector,
    cacheblend_repair_token_count,
    normalized_token_k_drifts,
    plan_selection_scratch,
    residual_k_score,
    score_repair_token_count,
)


def profile():
    return SelectorPolicyProfile(
        "profile", "model", "causal_commit_wait", (1, 2, 4), 4,
        eta=0.3, eta_strong=0.6, residual_band_relative_tolerance=0.05,
    )


def candidate(source, score, future=10.0, rank=0):
    return ResidualCandidate(source, score, future, rank)


class V8ResidualMathTests(unittest.TestCase):
    def test_score_and_repair_counts_are_distinct_and_conservative(self):
        self.assertEqual(score_repair_token_count(2, 1.0), 1)
        self.assertEqual(cacheblend_repair_token_count(2, 1.0), 2)
        self.assertEqual(cacheblend_repair_token_count(512, 0.15), 77)

    def test_tied_topk_uses_absolute_position(self):
        score, repair = residual_k_score(
            [1.0, 1.0, 0.1, 0.1], ratio=0.25,
            absolute_positions=[20, 10, 30, 40],
        )
        self.assertEqual(repair, (1,))
        self.assertAlmostEqual(score, 0.4)

    def test_token_drift_is_normalized_per_token(self):
        values = normalized_token_k_drifts([[1.0, 0.0], [0.0, 2.0]], [[0.0, 0.0], [0.0, 1.0]])
        self.assertEqual(values, (1.0, 0.5))

    def test_batch_budget_uses_vectorized_curve(self):
        budget = ComparisonBudget(100.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(budget.available_ms, 2.0)
        self.assertEqual(budget.largest_batch({1: 0.5, 4: 1.2, 8: 2.1}, 16), 4)

    def test_sixteen_states_use_bounded_scratch_microbatches(self):
        plan = plan_selection_scratch(
            compared_k=16,
            source_state_bytes=8,
            current_state_bytes=8,
            capacity_bytes=40,
        )
        self.assertEqual(plan.microbatch_k, 4)
        self.assertEqual(len(plan.batches), 4)
        self.assertEqual(tuple(index for batch in plan.batches for index in batch), tuple(range(16)))
        self.assertLessEqual(plan.peak_bytes, plan.capacity_bytes)


class V8ResidualSelectorTests(unittest.TestCase):
    def setUp(self):
        self.selector = TrainingFreeResidualKSelector(profile())

    def test_depth_zero_is_negative_control(self):
        counts = CandidateCounts(1, 1, 1, 1, 1)
        result = self.selector.evaluate_checkpoint(
            completed_depth=0, counts=counts,
            candidates=[candidate("a", 0.1)], shared_sunk_ms=1,
            dense_reference_ms=100,
        )
        self.assertEqual(result.state, ResidualSelectionState.PENDING)
        self.assertIsNone(result.selected_source_variant_id)

    def test_single_correctness_eligible_locks_without_margin(self):
        result = self.selector.evaluate_checkpoint(
            completed_depth=1, counts=CandidateCounts(5, 1, 1, 1, 1),
            candidates=[candidate("a", 0.1)], shared_sunk_ms=1,
            dense_reference_ms=100,
        )
        self.assertEqual(result.state, ResidualSelectionState.LOCKED)
        self.assertEqual(result.lock_reason, ResidualLockReason.SINGLE_CORRECTNESS_ELIGIBLE_SOURCE)
        self.assertFalse(result.margin_defined)
        self.assertIsNone(result.runner_up_source_variant_id)

    def test_budget_truncated_one_of_sixteen_abstains(self):
        result = self.selector.evaluate_checkpoint(
            completed_depth=1, counts=CandidateCounts(16, 16, 16, 16, 1),
            candidates=[candidate("cfo-top", 0.1)], shared_sunk_ms=1,
            dense_reference_ms=100,
        )
        self.assertEqual(result.state, ResidualSelectionState.ABSTAINED)
        self.assertFalse(result.margin_defined)

    def test_cfo_top1_fallback_is_explicit_and_max_depth_only(self):
        selector = TrainingFreeResidualKSelector(
            profile(), insufficient_ranking_policy=InsufficientRankingPolicy.CFO_TOP1_FALLBACK
        )
        counts = CandidateCounts(16, 16, 16, 16, 1)
        early = selector.evaluate_checkpoint(
            completed_depth=1, counts=counts,
            candidates=[candidate("cfo-top", 0.1)], shared_sunk_ms=1,
            dense_reference_ms=100,
        )
        final = selector.evaluate_checkpoint(
            completed_depth=4, counts=counts,
            candidates=[candidate("cfo-top", 0.1)], shared_sunk_ms=1,
            dense_reference_ms=100,
        )
        self.assertEqual(early.state, ResidualSelectionState.ABSTAINED)
        self.assertEqual(final.lock_reason, ResidualLockReason.CFO_TOP1_FALLBACK)
        self.assertFalse(final.current_state_ranking_performed)

    def test_strong_and_stable_early_exit(self):
        counts = CandidateCounts(2, 2, 2, 2, 2)
        strong = self.selector.evaluate_checkpoint(
            completed_depth=1, counts=counts,
            candidates=[candidate("a", 0.04), candidate("b", 0.2)],
            shared_sunk_ms=1, dense_reference_ms=100,
        )
        stable = self.selector.evaluate_checkpoint(
            completed_depth=2, counts=counts,
            candidates=[candidate("a", 0.1), candidate("b", 0.15)],
            shared_sunk_ms=1, dense_reference_ms=100,
            previous_winner_source_id="a",
        )
        self.assertEqual(strong.lock_reason, ResidualLockReason.STRONG_MARGIN_EARLY_EXIT)
        self.assertEqual(stable.lock_reason, ResidualLockReason.STABLE_MARGIN_EARLY_EXIT)

    def test_max_depth_uses_residual_band_then_cost(self):
        result = self.selector.evaluate_checkpoint(
            completed_depth=4, counts=CandidateCounts(3, 3, 3, 3, 3),
            candidates=[
                candidate("a", 0.100, 20),
                candidate("b", 0.104, 10),
                candidate("c", 0.2, 1),
            ],
            shared_sunk_ms=1, dense_reference_ms=100,
        )
        self.assertEqual(result.selected_source_variant_id, "b")
        self.assertEqual(result.lock_reason, ResidualLockReason.MAX_DEPTH_RESIDUAL_COST)

    def test_sixteenth_source_can_win_when_all_are_compared(self):
        candidates = [candidate("s%d" % index, 0.30 - 0.01 * index, 10, index) for index in range(16)]
        result = self.selector.evaluate_checkpoint(
            completed_depth=4,
            counts=CandidateCounts(16, 16, 16, 16, 16),
            candidates=candidates,
            shared_sunk_ms=1,
            dense_reference_ms=100,
        )
        self.assertEqual(result.selected_source_variant_id, "s15")


if __name__ == "__main__":
    unittest.main()
