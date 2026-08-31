import unittest
from dataclasses import replace

from probekv.config import load_config
from probekv.v6_contracts import SelectionExecutionPolicy
from probekv.v8_schema8_jobs import (
    build_schema8_no_gpu_handoff,
    build_schema8_qualification_jobs,
    build_schema8_runtime_measurement_jobs,
    build_schema8_sentinel_jobs,
)
from probekv.v8_schema8_qualification import (
    evaluate_schema8_runtime_qualification,
    validate_schema8_h1_gate,
)
from probekv.v8_schema8_profile import (
    RuntimeCostProfileV8,
    Schema8ProfileProvenance,
    SelectionDepthProfileV8,
    build_runtime_cost_profile_v8,
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
        self.assertTrue(handoff["runtime_source_audit"]["runtime_source_ready"])
        self.assertTrue(handoff["gpu_rental_ready_for_schema8_sentinel"])

    def test_schema8_profiles_are_independent_and_adaptive_requires_repair_profile(self):
        config = load_config(
            "configs/local_system_v8_schema8_d1d2_gradual_barrier.json"
        )
        self.assertEqual(config.selection_depth_profile_status, "unfrozen")
        self.assertEqual(config.repair_policy_profile_status, "unfrozen")
        self.assertEqual(config.runtime_cost_profile_status, "unfrozen")
        with self.assertRaisesRegex(ValueError, "RepairPolicyProfile"):
            replace(
                config,
                repair_policy="load_recompute_aware_gradual",
                repair_ratio_scope="per_segment_load_aware",
            ).validate()

    def test_schema8_gate_rejects_older_schema_and_accepts_exact_binding(self):
        binding = {
            "code_commit": "a" * 40,
            "model_id": "mistral",
            "model_revision": "b" * 40,
            "cacheblend_patch_sha256": "c" * 64,
            "cacheblend_tree": "d" * 40,
            "job_manifest_sha256": "e" * 64,
            "selection_depth_profile_sha256": "f" * 64,
            "repair_policy_profile_sha256": "1" * 64,
            "runtime_cost_profile_sha256": "2" * 64,
            "gpu_uuid": "GPU-test",
        }
        audit = {
            "protocol_version": 8, "schema_version": 8,
            "planned": 140, "completed": 140, "failed": 0,
            "cuda_event_timing": True,
            "native_prefix_cache_qualified": True,
            "dense_d1_d2_barrier_verified": True,
            "gate1_positive_saving_verified": True,
            "final_commit_joint_timeline_verified": True,
            "cpu_ssd_lru_verified": True,
            "repair_ratio_scope_verified": True,
            "r1_dense_equivalence": True,
            "source_digest_unchanged": True,
            "fake_timing": False,
        }
        gate = evaluate_schema8_runtime_qualification(
            binding=binding, runtime_audit=audit
        )
        self.assertTrue(gate["gpu_runtime_qualified"])
        validate_schema8_h1_gate(gate, expected_binding=binding)
        old = dict(gate, schema_version=7)
        with self.assertRaisesRegex(ValueError, "schema-v8"):
            validate_schema8_h1_gate(old, expected_binding=binding)

    def test_final_qualification_matrix_is_profile_bound_and_stratified(self):
        jobs = build_schema8_qualification_jobs(
            model_key="mistral",
            selection_depth_profile_sha256="a" * 64,
            repair_policy_profile_sha256="b" * 64,
            runtime_cost_profile_sha256="c" * 64,
        )
        self.assertEqual(len(jobs), 140)
        self.assertEqual(len({row["job_id"] for row in jobs}), 140)
        self.assertEqual(
            {row["coordinates"]["segment_count"] for row in jobs},
            {1, 2, 5, 10, 37},
        )
        self.assertTrue(all(row["paper_evidence"] is False for row in jobs))

    def test_schema8_profile_types_do_not_accept_fake_frozen_runtime(self):
        common = dict(
            code_commit="a" * 40,
            cacheblend_patch_sha256="b" * 64,
            model_id="mistral",
            model_revision="c" * 40,
            tokenizer_hash="d" * 64,
        )
        selection = SelectionDepthProfileV8(
            Schema8ProfileProvenance(profile_kind="selection_depth", **common),
            (1, 2), 0.15,
        )
        self.assertEqual(selection.allowed_completed_depths, (1, 2))
        with self.assertRaisesRegex(ValueError, "real GPU"):
            Schema8ProfileProvenance(
                profile_kind="runtime_cost", frozen=True, **common
            )
        categories = {
            name: ({"cuda_event_timing": False},)
            for name in RuntimeCostProfileV8.REQUIRED_CATEGORIES
        }
        profile = RuntimeCostProfileV8(
            Schema8ProfileProvenance(profile_kind="runtime_cost", **common),
            categories,
        )
        self.assertFalse(profile.provenance.frozen)
        frozen_provenance = Schema8ProfileProvenance(
            profile_kind="runtime_cost", frozen=True,
            gpu_uuid="GPU-real", measurement_sha256="e" * 64, **common,
        )
        real_categories = {
            name: ({"cuda_event_timing": True},)
            for name in RuntimeCostProfileV8.REQUIRED_CATEGORIES
        }
        frozen = build_runtime_cost_profile_v8(
            provenance=frozen_provenance,
            category_measurements=real_categories,
        )
        self.assertEqual(len(frozen.profile_sha256), 64)


if __name__ == "__main__":
    unittest.main()
