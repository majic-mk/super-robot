import json
import unittest
from pathlib import Path

from probekv.config import load_config
from probekv.model_adapters import (
    MISTRAL_SCHEMA6_SPEC,
    validate_schema6_checkpoint_contract,
)
from probekv.v8_schema6_jobs import (
    build_mistral_schema6_sentinel_jobs,
    schema6_no_gpu_gate,
)


class Schema6ConfigAndJobTests(unittest.TestCase):
    def test_schema6_configs_are_explicit(self):
        for path in (
            "configs/local_system_v8_schema6_causal_wait.json",
            "configs/local_system_v8_schema6_immediate_staggered.json",
        ):
            config = load_config(path)
            self.assertEqual(config.protocol_version, 8)
            self.assertEqual(config.v8_schema_version, 6)

    def test_mistral_schema6_checkpoint_contract(self):
        self.assertEqual(MISTRAL_SCHEMA6_SPEC.checkpoints, (1, 2, 4, 5, 8))
        with self.assertRaises(ValueError):
            validate_schema6_checkpoint_contract(
                model_id=MISTRAL_SCHEMA6_SPEC.model_id,
                checkpoint_sources={"stale": (1, 2, 4, 6, 8)},
            )

    def test_sparse_jobs_do_not_freeze_profiles_or_create_paper_evidence(self):
        jobs = build_mistral_schema6_sentinel_jobs()
        self.assertTrue(jobs)
        self.assertTrue(all(row["paper_evidence"] is False for row in jobs))
        anchors = [row for row in jobs if row["warmups"] == 20]
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["repeats"], 100)
        gate = schema6_no_gpu_gate()
        self.assertFalse(gate["runtime_cost_profile_frozen"])
        self.assertFalse(gate["h1_h2_execution_allowed"])

    def test_schema6_server_lock_and_runner_are_separate_from_schema5(self):
        root = Path(__file__).resolve().parents[1]
        lock = json.loads(
            (root / "configs/a800_server_lock_v8_schema6.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(lock["schema_version"], 6)
        self.assertEqual(
            lock["stack"]["cacheblend_patch_mode"],
            "probekv_v8_schema6_joint_cfo",
        )
        self.assertEqual(lock["models"]["mistral"]["completed_depths"], [1, 2, 4, 5, 8])
        runner = (
            root / "scripts/server/run_v8_schema6_mistral_sentinel.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--hourly-price-cny", runner)
        self.assertIn("--cacheblend", runner)
        self.assertIn("require_cacheblend_runtime_source", runner)
        self.assertIn("installed vLLM does not resolve", runner)
        relocator = (
            root / "scripts/server/relocate_vllm_editable.py"
        ).read_text(encoding="utf-8")
        self.assertIn("compiled_extension_sha256", relocator)
        self.assertIn("cacheblend_tree", relocator)
        sentinel = json.loads(
            (root / "configs/v8_schema6_a800_sentinel.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(sentinel["run_qualification"])
        self.assertFalse(sentinel["run_h1"])


if __name__ == "__main__":
    unittest.main()
