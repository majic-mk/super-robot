import unittest

from probekv.v8_schema7_contracts import RepairCheckBoundary, RepairMetric
from probekv.v8_schema7_repair import (
    LoadRecomputeAwareRepairController,
    SourceScoreTrimIndices,
    Schema7RepairPolicyExecutor,
    build_initial_repair_support,
    repair_support_overlap_metrics,
    repair_support_oracle_metrics,
    shrink_repair_support,
    source_score_from_k_drifts,
    winner_repair_drifts,
)
from probekv.v8_schema7_contracts import RepairPolicy


class Schema7RepairTests(unittest.TestCase):
    def test_source_score_trim_is_not_runtime_repair_support(self):
        score, trimmed = source_score_from_k_drifts(
            (0.1, 0.9, 0.2, 0.8), trim_ratio=0.25,
            absolute_positions=(10, 11, 12, 13),
        )
        self.assertIsInstance(trimmed, SourceScoreTrimIndices)
        self.assertEqual(trimmed.absolute_positions, (11,))
        self.assertAlmostEqual(score, (0.1 + 0.2 + 0.8) / 3)

    def test_k_v_and_kv_metrics_are_distinct_and_deterministic(self):
        kwargs = dict(
            current_k=((1.0, 0.0), (1.0, 0.0)),
            source_k=((0.0, 0.0), (1.0, 0.0)),
            current_v=((1.0, 0.0), (1.0, 0.0)),
            source_v=((1.0, 0.0), (0.0, 0.0)),
        )
        self.assertEqual(
            winner_repair_drifts(metric=RepairMetric.WINNER_K_ONLY, **kwargs),
            (1.0, 0.0),
        )
        self.assertEqual(
            winner_repair_drifts(metric=RepairMetric.WINNER_V_ONLY, **kwargs),
            (0.0, 1.0),
        )
        self.assertEqual(
            winner_repair_drifts(metric=RepairMetric.WINNER_KV_NORMALIZED, **kwargs),
            (1.0, 1.0),
        )

    def test_repair_boundary_and_gradual_no_reentry(self):
        RepairCheckBoundary(2, 3, 4)
        with self.assertRaises(ValueError):
            RepairCheckBoundary(2, 3, 3)
        positions = tuple(range(100, 120))
        parent = build_initial_repair_support(
            segment_id="c", source_variant_id="s",
            metric=RepairMetric.WINNER_V_ONLY,
            repair_check_completed_depth=2,
            segment_absolute_positions=positions,
            drift_scores=tuple(float(value) for value in range(20)),
            initial_cap=0.15, repair_floor=0.10,
        )
        self.assertEqual(parent.candidate_count, 3)
        child = shrink_repair_support(
            parent,
            drift_score_by_absolute_position={
                position: float(position) for position in parent.candidate_absolute_positions
            },
            next_ratio=0.10,
        )
        self.assertEqual(child.candidate_count, 2)
        self.assertTrue(set(child.candidate_absolute_positions).issubset(
            parent.candidate_absolute_positions
        ))
        with self.assertRaises(ValueError):
            shrink_repair_support(
                child,
                drift_score_by_absolute_position={
                    position: 1.0 for position in child.candidate_absolute_positions
                },
                next_ratio=0.15,
            )

    def test_controller_keeps_higher_ratio_when_io_hides_repair(self):
        plan = LoadRecomputeAwareRepairController().choose(
            parent_ratio=0.15,
            certified_floor=0.10,
            repair_ms_by_ratio={0.10: 2.0, 0.15: 4.0},
            load_ms_by_path={"cpu": 5.0},
            nonoverlap_ms=1.0,
        )
        self.assertEqual(plan.ratio, 0.15)
        self.assertEqual(plan.predicted_layer_ms, 6.0)

    def test_fixed_static_and_load_aware_paths_are_explicit(self):
        parent = build_initial_repair_support(
            segment_id="c", source_variant_id="s", metric=RepairMetric.WINNER_V_ONLY,
            repair_check_completed_depth=1,
            segment_absolute_positions=tuple(range(20)),
            drift_scores=tuple(float(value) for value in range(20)),
            initial_cap=0.15, repair_floor=0.10,
        )
        drifts = {position: float(position) for position in parent.candidate_absolute_positions}
        fixed = Schema7RepairPolicyExecutor(RepairPolicy.FIXED_15).next_support(
            parent=parent, drift_score_by_absolute_position=drifts,
        )
        static = Schema7RepairPolicyExecutor(RepairPolicy.STATIC_GRADUAL).next_support(
            parent=parent, drift_score_by_absolute_position=drifts,
            static_next_ratio=0.10,
        )
        adaptive = Schema7RepairPolicyExecutor(
            RepairPolicy.LOAD_RECOMPUTE_AWARE_GRADUAL
        ).next_support(
            parent=parent, drift_score_by_absolute_position=drifts,
            repair_ms_by_ratio={0.10: 1.0, 0.15: 2.0},
            load_ms_by_path={"cpu": 3.0},
        )
        self.assertEqual(fixed.support.candidate_count, 3)
        self.assertEqual(static.support.candidate_count, 2)
        self.assertEqual(adaptive.support.candidate_count, 3)
        self.assertIsNotNone(adaptive.layer_plan)

    def test_oracle_overlap_metrics(self):
        metrics = repair_support_overlap_metrics((1, 2), (2, 3))
        self.assertAlmostEqual(metrics["jaccard"], 1 / 3)
        self.assertEqual(metrics["oracle_recall"], 0.5)
        ranked = repair_support_oracle_metrics((1, 2, 3), (1, 2, 3))
        self.assertAlmostEqual(ranked["spearman"], 1.0)
        reversed_ranked = repair_support_oracle_metrics((1, 2, 3), (3, 2, 1))
        self.assertAlmostEqual(reversed_ranked["spearman"], -1.0)


if __name__ == "__main__":
    unittest.main()
