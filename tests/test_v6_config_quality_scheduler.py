import json
import tempfile
import unittest
from pathlib import Path

from probekv.calibration import (
    CalibrationSupportEnvelope,
    GroupedSimultaneousConformal,
    RequestQualityGuard,
)
from probekv.config import ExperimentConfig, load_config
from probekv.cli import main as cli_main
from probekv.features import (
    CacheCraftMetadata,
    cache_craft_cfo,
    cache_craft_cfo_score,
)
from probekv.prefetch import LockedSegmentWinner, choose_locked_winner_prefetch
from probekv.scheduler import (
    MultiSourceLoad,
    SchedulerPolicy,
    SchedulerScenario,
    simulate_multisource_schedule,
)
from probekv.v6_contracts import SelectionExecutionPolicy


class V6ConfigTests(unittest.TestCase):
    def test_local_v6_config_is_valid_and_has_no_legacy_kmax(self):
        config = load_config("configs/local_system_v6.json")
        self.assertEqual(config.protocol_version, 6)
        self.assertFalse(config.legacy_online_kmax_present)
        self.assertEqual(config.max_stored_variants_per_content, 16)
        self.assertIsNone(config.max_detected_segments)
        self.assertEqual(
            config.selection_execution_policy,
            SelectionExecutionPolicy.CAUSAL_COMMIT_WAIT,
        )

    def test_immediate_staggered_config_is_explicit_and_policy_matched(self):
        config = load_config("configs/local_system_v6_immediate_staggered.json")
        self.assertEqual(config.boundary_policy, "immediate_staggered")
        self.assertEqual(
            config.selection_execution_policy,
            SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP,
        )
        self.assertTrue(config.calibration_policy_match_required)

    def test_shadow_policy_is_not_part_of_the_v6_protocol(self):
        raw = json.loads(
            Path("configs/local_system_v6.json").read_text(encoding="utf-8")
        )
        raw["selection_execution_policy"] = "shadow_dense_probe"
        with self.assertRaises(ValueError):
            ExperimentConfig.from_mapping(raw)

    def test_a_c_boundary_and_execution_policy_cannot_be_mixed(self):
        raw = json.loads(
            Path("configs/local_system_v6.json").read_text(encoding="utf-8")
        )
        raw["boundary_policy"] = "immediate_staggered"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            ExperimentConfig.from_mapping(raw)

    def test_a800_v6_uses_explicit_multisegment_runtime(self):
        config = load_config("configs/a800_closed_loop_v6.json")
        self.assertEqual(
            config.runtime_backend,
            "cacheblend_multisegment_closed_loop",
        )

    def test_v6_rejects_legacy_online_kmax(self):
        raw = json.loads(Path("configs/local_system_v6.json").read_text(encoding="utf-8"))
        raw["online_kmax"] = 4
        with self.assertRaisesRegex(ValueError, "forbids legacy online_kmax"):
            ExperimentConfig.from_mapping(raw)

    def test_v6_freezes_one_variant_for_a_retained_content(self):
        raw = json.loads(Path("configs/local_system_v6.json").read_text(encoding="utf-8"))
        raw["min_variants_per_retained_content"] = 2
        with self.assertRaisesRegex(ValueError, "exactly one variant"):
            ExperimentConfig.from_mapping(raw)

    def test_v5_config_remains_loadable(self):
        config = load_config("configs/local_system_v5.json")
        self.assertNotEqual(config.protocol_version, 6)
        self.assertEqual(config.online_kmax, 4)

    def test_direct_v6_cli_shorthand_writes_rich_nonpaper_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            code = cli_main(
                [
                    "--config",
                    "configs/local_system_v6.json",
                    "--output",
                    temporary,
                ]
            )
            self.assertEqual(code, 0)
            row = json.loads(
                (Path(temporary) / "cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertFalse(row["paper_evidence"])
            for key in (
                "compared_segment_count",
                "loaded_segment_count",
                "actual_reuse_boundary",
                "a_resume_ms",
                "post_ready_blocking_ms",
                "refined_cost_components",
                "interference_accounting_mode",
            ):
                self.assertIn(key, row)


class V6QualityAndBaselineTests(unittest.TestCase):
    def test_simultaneous_conformal_and_support_envelope(self):
        calibrator = GroupedSimultaneousConformal.fit(
            {"r0": (0.01, 0.02), "r1": (0.03, 0.04)},
            miscoverage=0.5,
        )
        support = CalibrationSupportEnvelope.fit(
            ({"segments": 1.0, "length": 100.0}, {"segments": 10.0, "length": 1000.0}),
            ("segments", "length"),
        )
        guard = RequestQualityGuard(calibrator, support, max_degradation=0.10)
        accepted = guard.evaluate(0.05, {"segments": 5.0, "length": 500.0})
        unsupported = guard.evaluate(0.01, {"segments": 15.0, "length": 500.0})
        self.assertAlmostEqual(accepted.conservative_degradation_upper, 0.09)
        self.assertTrue(accepted.accepted)
        self.assertFalse(unsupported.support_covered)
        self.assertFalse(unsupported.accepted)

    def test_cachecraft_cfo_uses_paper_equation(self):
        value = cache_craft_cfo(
            prefix_overlap=0.5,
            order_penalty=0.2,
            cci=0.8,
            alpha=0.5,
        )
        self.assertAlmostEqual(value, 0.24)
        metadata = CacheCraftMetadata.from_cachecraft_components(0.5, 0.2, 0.8, 0.5)
        self.assertAlmostEqual(cache_craft_cfo_score(metadata), value)


class V6SchedulingTests(unittest.TestCase):
    def test_scheduler_reports_each_source_and_common_boundary_candidates(self):
        feedback = simulate_multisource_schedule(
            "request",
            SchedulerPolicy.A_ONLY,
            (
                MultiSourceLoad("c0", "c0-s", 0.0, 2.0, 100, 3, 8, 0.1),
                MultiSourceLoad("c1", "c1-s", 1.0, 5.0, 200, 5, 8, 0.2),
            ),
            SchedulerScenario(2.0, 1.0, 4, 2.0, 1.0, 0.0),
            (),
            a_layer=1,
            total_layers=8,
        )
        by_id = {item.segment_id: item for item in feedback.segments}
        self.assertAlmostEqual(by_id["c0"].source_ready_ms, 2.1)
        self.assertAlmostEqual(by_id["c1"].source_ready_ms, 6.2)
        self.assertAlmostEqual(by_id["c0"].source_load_finish_ms, 2.1)
        self.assertEqual(
            by_id["c1"].layer_ready_ms,
            ((5, 6.2), (8, 6.2)),
        )
        self.assertEqual(feedback.candidate_boundaries, tuple(range(3, 9)))
        self.assertAlmostEqual(feedback.load_interference_ms, 0.3)

    def test_prefetch_loads_only_one_locked_winner_per_segment(self):
        winners = (
            LockedSegmentWinner("c0", "c0-s", 1, 100, 5.0, 20.0),
            LockedSegmentWinner("c1", "c1-s", 2, 100, 3.0, 30.0),
        )
        decision = choose_locked_winner_prefetch(winners, 100, overlap_ms=1.0)
        self.assertEqual(decision.source_id_by_segment, {"c0": "c0-s"})
        self.assertEqual(decision.dropped_segment_ids, ("c1",))
        self.assertEqual(decision.transferred_bytes, 100)

    def test_prefetch_rejects_two_locked_sources_for_same_segment(self):
        winners = (
            LockedSegmentWinner("c0", "s0", 1, 10, 1.0, 5.0),
            LockedSegmentWinner("c0", "s1", 2, 10, 1.0, 5.0),
        )
        with self.assertRaisesRegex(ValueError, "one locked winner"):
            choose_locked_winner_prefetch(winners, 100, overlap_ms=0.0)


if __name__ == "__main__":
    unittest.main()
