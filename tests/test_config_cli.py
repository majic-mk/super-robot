import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from probekv.config import ExperimentConfig, load_config
from probekv.contracts import CostAccountingPolicy
from probekv.orchestration import ClosedLoopPolicy
from probekv.scheduler import SchedulerPolicy
from probekv.selector import SelectorPolicy


ROOT = Path(__file__).resolve().parents[1]


class ConfigAndCliTests(unittest.TestCase):
    def test_local_config_is_valid(self):
        config = load_config(str(ROOT / "configs" / "local_smoke.json"))
        self.assertEqual(config.online_kmax, 4)
        self.assertEqual(config.probe_checkpoints[-1], 8)
        self.assertEqual(
            config.selector_policy, SelectorPolicy.STRICT_INTERVAL
        )
        self.assertEqual(
            config.scheduler_policy, SchedulerPolicy.HYBRID_STRICT
        )

    def test_v3_policy_config_is_explicit_and_valid(self):
        config = load_config(
            str(ROOT / "configs" / "local_system_v3.json")
        )
        self.assertEqual(
            config.selector_policy,
            SelectorPolicy.FINAL_ECONOMIC_MIN_COST,
        )
        self.assertEqual(
            config.scheduler_policy,
            SchedulerPolicy.HYBRID_BOUNDED_OVERRUN,
        )
        self.assertEqual(config.max_post_ready_overrun_ms, 1.0)

    def test_server_pilot_config_is_valid_and_non_paper(self):
        config = load_config(str(ROOT / "configs" / "a800_h1_pilot.json"))
        self.assertEqual(config.evidence_class, "server_pilot")
        self.assertEqual(config.cases, 150)
        self.assertEqual(
            config.runtime_backend, "cacheblend_case_runner"
        )
        self.assertFalse(config.require_layer_resumable_prefill)

    def test_a800_closed_loop_requires_real_runtime_capabilities(self):
        config = load_config(
            str(ROOT / "configs" / "a800_closed_loop_v5.json")
        )
        self.assertEqual(config.runtime_backend, "cacheblend_closed_loop")
        self.assertTrue(config.require_async_source_loading)
        self.assertTrue(config.require_layer_resumable_prefill)
        self.assertTrue(config.require_cuda_event_timing)
        self.assertEqual(
            config.cost_accounting_policy,
            CostAccountingPolicy.UNIFIED_COMPONENTS_V1,
        )

    def test_closed_loop_backend_cannot_silently_disable_resumption(self):
        raw = {
            "name": "bad-cacheblend-runtime",
            "evidence_class": "server_pilot",
            "cases": 1,
            "total_layers": 32,
            "online_kmax": 4,
            "gamma": 0.8,
            "probe_checkpoints": [1, 2],
            "max_selection_layer": 2,
            "selector_policy": "final_economic_min_cost",
            "preliminary_economic_filter": True,
            "closed_loop_policy": "two_stage_refined_admission",
            "cost_accounting_policy": "unified_components_v1",
            "runtime_backend": "cacheblend_closed_loop",
            "require_async_source_loading": True,
            "require_layer_resumable_prefill": False,
            "require_cuda_event_timing": True,
            "repair_ratios": [0, 1],
        }
        with self.assertRaisesRegex(ValueError, "resumable prefill"):
            ExperimentConfig.from_mapping(raw)

    def test_v4_enables_refined_two_stage_closure_explicitly(self):
        config = load_config(
            str(ROOT / "configs" / "local_system_v4.json")
        )
        self.assertEqual(
            config.closed_loop_policy,
            ClosedLoopPolicy.TWO_STAGE_REFINED_ADMISSION,
        )
        self.assertTrue(config.preliminary_economic_filter)
        legacy = load_config(
            str(ROOT / "configs" / "local_system_v3.json")
        )
        self.assertEqual(
            legacy.closed_loop_policy,
            ClosedLoopPolicy.LEGACY_PRE_SCHEDULE_ADMISSION,
        )

    def test_v5_uses_one_component_cost_identity_for_both_stages(self):
        config = load_config(
            str(ROOT / "configs" / "local_system_v5.json")
        )
        self.assertEqual(
            config.cost_accounting_policy,
            CostAccountingPolicy.UNIFIED_COMPONENTS_V1,
        )
        self.assertEqual(
            config.closed_loop_policy,
            ClosedLoopPolicy.TWO_STAGE_REFINED_ADMISSION,
        )
        self.assertTrue(config.preliminary_economic_filter)

    def test_probe_max_above_25_percent_is_rejected(self):
        raw = {
            "name": "bad",
            "cases": 1,
            "total_layers": 32,
            "online_kmax": 4,
            "gamma": 0.8,
            "probe_checkpoints": [1, 9],
            "repair_ratios": [0, 1],
        }
        with self.assertRaisesRegex(ValueError, "25%"):
            ExperimentConfig.from_mapping(raw)

    def test_positive_overrun_requires_bounded_policy(self):
        raw = {
            "name": "bad-overrun",
            "cases": 1,
            "total_layers": 32,
            "online_kmax": 4,
            "gamma": 0.8,
            "probe_checkpoints": [1, 2],
            "scheduler_policy": "hybrid_strict",
            "max_post_ready_overrun_ms": 1.0,
            "repair_ratios": [0, 1],
        }
        with self.assertRaisesRegex(ValueError, "bounded"):
            ExperimentConfig.from_mapping(raw)

    def test_cli_writes_auditable_non_paper_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "probekv",
                    "simulate",
                    "--config",
                    str(ROOT / "configs" / "local_smoke.json"),
                    "--output",
                    temporary,
                ],
                cwd=str(ROOT),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            manifest = json.loads(
                (Path(temporary) / "manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (Path(temporary) / "summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["paper_evidence"])
            self.assertFalse(summary["paper_evidence"])
            self.assertTrue((Path(temporary) / "cases.jsonl").exists())
            first_row = json.loads(
                (Path(temporary) / "cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            for key in (
                "git_commit",
                "environment_hash",
                "model_signature",
                "data_manifest_hash",
                "config_hash",
                "seed",
                "timestamp_utc",
                "dynamic_prefetch_sources",
                "hybrid_a_ttft_ms",
                "hybrid_queue_fairness",
            ):
                self.assertIn(key, first_row)


if __name__ == "__main__":
    unittest.main()
