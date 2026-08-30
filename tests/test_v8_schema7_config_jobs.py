import unittest
from pathlib import Path

from probekv.cacheblend_patch import patch_files_for_mode
from probekv.config import load_config
from probekv.v8_schema7_jobs import (
    build_schema7_development_jobs,
    build_schema7_no_gpu_handoff,
    build_schema7_runtime_measurement_jobs,
    build_schema7_sentinel_jobs,
)
from probekv.v8_schema7_contracts import Schema7ProfileProvenance
from probekv.v8_schema7_profile import SelectionDepthProfile
from probekv.v8_schema7_contracts import SourceSelectionDepthPolicy
from probekv.runtime_source_audit import audit_v8_schema7_runtime_sources


class Schema7ConfigJobsTests(unittest.TestCase):
    def test_three_schema7_configs_load_and_remain_unfrozen(self):
        names = (
            "configs/local_system_v8_schema7_legacy_fixed15.json",
            "configs/local_system_v8_schema7_qwen_legacy_fixed15.json",
            "configs/local_system_v8_schema7_d1d2_gradual_causal_wait.json",
            "configs/local_system_v8_schema7_d1d2_gradual_immediate.json",
        )
        for name in names:
            config = load_config(name)
            self.assertEqual(config.v8_schema_version, 7)
            self.assertEqual(config.selector_profile_status, "unfrozen")

    def test_schema6_config_is_not_silently_schema7(self):
        config = load_config("configs/local_system_v8_schema6_causal_wait.json")
        self.assertEqual(config.v8_schema_version, 6)
        with self.assertRaises(ValueError):
            Schema7ProfileProvenance(
                "id", "selection_depth", "code", "patch", "model", "revision",
                "tokenizer", protocol_version=8, schema_version=6,
            )

    def test_patch_mode_is_auditable_and_uses_fixed_tree(self):
        paths = patch_files_for_mode(
            Path("patches/cacheblend/manifest.json"),
            "probekv_v8_winner_gradual_streaming",
        )
        self.assertEqual(len(paths), 8)
        audit = audit_v8_schema7_runtime_sources(Path(".").resolve())
        self.assertTrue(audit["runtime_source_ready"], audit["failures"])

    def test_dual_model_jobs_cover_depth_repair_and_transfer(self):
        for model in ("mistral", "qwen"):
            sentinel = build_schema7_sentinel_jobs(model)
            development = build_schema7_development_jobs(model)
            kinds = {row["kind"] for row in sentinel}
            self.assertIn("integrity_qualification_full", kinds)
            self.assertIn("winner_repair_policy", kinds)
            self.assertIn("layerwise_transfer", kinds)
            self.assertTrue(all(not row["paper_evidence"] for row in sentinel))
            self.assertTrue(all(
                row["coordinates"].get("partition") == "profile_freeze"
                for row in development
            ))
            runtime = build_schema7_runtime_measurement_jobs(model)
            runtime_kinds = {row["kind"] for row in runtime}
            self.assertEqual(
                runtime_kinds,
                {
                    "comparison_batch", "selection_state_transfer",
                    "full_kv_tier_load", "repair", "dense_remaining_joint",
                    "union_mask_remaining", "scheduler_blocking", "interference",
                },
            )
            self.assertTrue(all(row["requires_real_gpu"] for row in runtime))

    def test_handoff_cannot_claim_frozen_profiles_or_gpu_qualification(self):
        handoff = build_schema7_no_gpu_handoff(
            code_commit="c", model_key="mistral", model_revision="r",
            tokenizer_hash="t", cacheblend_patch_sha256="p",
            cacheblend_tree="tree", config_sha256="cfg", contract_sha256="contract",
        )
        self.assertFalse(handoff["profiles_frozen"])
        self.assertFalse(handoff["gpu_runtime_qualified"])
        self.assertFalse(handoff["h1_h2_execution_allowed"])
        self.assertFalse(handoff["locked_test_accessed"])

    def test_no_gpu_profile_is_unfrozen_and_frozen_requires_measurements(self):
        provenance = Schema7ProfileProvenance(
            "p", "selection_depth", "code", "patch", "model", "revision",
            "tokenizer",
        )
        profile = SelectionDepthProfile(
            provenance, SourceSelectionDepthPolicy.D1_D2_RESCUE,
            (1, 2), 0.15, 0.3, 0.6,
        )
        self.assertFalse(profile.provenance.frozen)
        with self.assertRaises(ValueError):
            Schema7ProfileProvenance(
                "p", "selection_depth", "code", "patch", "model", "revision",
                "tokenizer", frozen=True,
            )


if __name__ == "__main__":
    unittest.main()
