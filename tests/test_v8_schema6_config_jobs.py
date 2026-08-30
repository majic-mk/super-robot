import json
import unittest
from pathlib import Path

from probekv.config import load_config
from probekv.model_adapters import (
    MISTRAL_SCHEMA6_SPEC,
    validate_schema6_checkpoint_contract,
)
from probekv.v8_schema6_jobs import (
    build_mistral_schema6_runtime_profile_jobs,
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

    def test_full_runtime_profile_matrix_covers_every_category(self):
        expected = {
            "comparison_batch",
            "selection_state_transfer",
            "full_kv_tier_load",
            "dense_remaining_joint",
            "repair",
            "union_mask_remaining",
            "interference",
            "scheduler_blocking",
        }
        for policy in (
            "causal_commit_wait",
            "immediate_staggered_closed_loop",
        ):
            jobs = build_mistral_schema6_runtime_profile_jobs(policy)
            self.assertEqual(len(jobs), 155)
            self.assertEqual({row["kind"] for row in jobs}, expected)
            self.assertEqual(len({row["job_id"] for row in jobs}), len(jobs))
            self.assertTrue(all(row["warmups"] == 20 for row in jobs))
            self.assertTrue(all(row["repeats"] == 100 for row in jobs))
            self.assertTrue(all(row["paper_evidence"] is False for row in jobs))

    def test_profile_runner_cannot_run_qualification_or_h1(self):
        root = Path(__file__).resolve().parents[1]
        lock = json.loads(
            (root / "configs/a800_server_lock_v8_schema6_profile.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(lock["runtime"]["freeze_runtime_cost_profile"])
        self.assertFalse(lock["runtime"]["run_140_job_qualification"])
        self.assertFalse(lock["runtime"]["run_h1"])
        text = (
            root / "scripts/server/run_v8_schema6_mistral_runtime_profile.py"
        ).read_text(encoding="utf-8")
        self.assertIn("RealCacheBlendA800Executor", text)
        self.assertNotIn("FakeQualification", text)

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
