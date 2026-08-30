import unittest

from probekv.config import load_config
from probekv.v6_contracts import SelectionExecutionPolicy
from probekv.v8_schema8_jobs import (
    build_schema8_no_gpu_handoff,
    build_schema8_runtime_measurement_jobs,
    build_schema8_sentinel_jobs,
)


class Schema8ConfigTests(unittest.TestCase):
    def test_schema8_config_freezes_new_contract(self):
        config = load_config(
            "configs/local_system_v8_schema8_d1d2_gradual_barrier.json"
        )
        self.assertEqual(config.v8_schema_version, 8)
        self.assertEqual(config.gate1_gamma, 1.0)
        self.assertEqual(config.gamma, 0.8)
        self.assertIs(
            config.selection_execution_policy,
            SelectionExecutionPolicy.DENSE_SELECTION_BARRIER,
        )
        self.assertEqual(config.backing_tier_policy, "cpu_preferred_single_backing")

    def test_schema7_remains_legacy_ac_policy(self):
        config = load_config(
            "configs/local_system_v8_schema7_d1d2_gradual_causal_wait.json"
        )
        self.assertEqual(config.v8_schema_version, 7)
        self.assertIs(
            config.selection_execution_policy,
            SelectionExecutionPolicy.CAUSAL_COMMIT_WAIT,
        )
        self.assertEqual(config.gate1_gamma, 0.8)

    def test_schema8_handoff_is_unfrozen_and_schema_bound(self):
        sentinel = build_schema8_sentinel_jobs("mistral")
        runtime = build_schema8_runtime_measurement_jobs("mistral")
        self.assertTrue(sentinel)
        self.assertTrue(runtime)
        self.assertTrue(all(row["job_id"].startswith("schema8-") for row in sentinel))
        handoff = build_schema8_no_gpu_handoff(
            code_commit="a" * 40,
            model_key="mistral",
            model_revision="b" * 40,
            tokenizer_hash="c" * 64,
            cacheblend_patch_sha256="d" * 64,
            cacheblend_tree="e" * 40,
            config_sha256="f" * 64,
            contract_sha256="1" * 64,
        )
        self.assertEqual(handoff["schema_version"], 8)
        self.assertFalse(handoff["profiles_frozen"])
        self.assertFalse(handoff["gpu_runtime_qualified"])
        self.assertFalse(handoff["h1_h2_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
