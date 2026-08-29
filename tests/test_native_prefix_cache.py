import unittest

from probekv.h1_qualification import (
    H1QualificationError,
    validate_h1_qualification_gate,
)
from probekv.native_prefix_cache import evaluate_native_prefix_cache_audit
from probekv.model_adapters import MISTRAL_SPEC, QWEN_SPEC
from scripts.server.build_dual_model_h1_sentinel_gate import build_joint_gate


def valid_prefix():
    return {
        "paper_evidence": False,
        "locked_test_accessed": False,
        "hit_evidence_source": "vllm_scheduler_computed_block_nums",
        "timing_inference_used": False,
        "requested_prefix_tokens": 192,
        "native_prefix_cache_hit": True,
        "cached_prefix_blocks": 12,
        "cached_prefix_tokens": 192,
        "block_size": 16,
        "prefix_shadow_layers": 32,
        "prefix_shadow_rows": 192,
        "prefix_shadow_dtype": "torch.bfloat16",
        "prefix_shadow_device": "cuda",
        "prefix_shadow_geometry_valid": True,
        "prefix_shadow_digest_before": "same",
        "prefix_shadow_digest_after": "same",
        "active_positions_start_after_prefix": True,
        "prefix_rows_excluded_from_repair": 192,
        "prefix_rows_in_repair_mask": 0,
        "prefix_rows_in_source_comparison": 0,
            "combined_prefix_r1_reuse_exercised": True,
            "dense_reference_scope": "same_native_prefix_cache_hit",
        "dense_token_ids_equal": True,
        "logit_relative_l2": 1e-5,
        "cuda_event_timing": True,
    }


def valid_gate():
    return {
        "schema_version": 2,
        "stage": "v6_a800_runtime_qualification",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "code_commit": "code",
        "model_id": "model",
        "model_revision": "revision",
        "adapter_name": "adapter",
        "cacheblend_patch_sha256": "patch",
        "cacheblend_tree": "tree",
        "job_manifest_sha256": "manifest",
        "runtime_audit_sha256": "runtime",
        "native_prefix_cache_audit_sha256": "prefix",
        "gpu_uuid": "GPU-1",
        "qualified_jobs_planned": 140,
        "qualified_jobs_completed": 140,
        "qualified_jobs_failed": 0,
        "cuda_event_timing": True,
        "native_prefix_cache_qualified": True,
        "gpu_runtime_qualified": True,
        "h1_h2_execution_allowed": True,
        "failures": [],
    }


class NativePrefixAuditTests(unittest.TestCase):
    def test_complete_block_metadata_and_shadow_pass(self):
        self.assertTrue(
            evaluate_native_prefix_cache_audit(
                valid_prefix(), expected_layers=32
            )["passed"]
        )

    def test_schema6_requires_prefix_matched_dense_reference(self):
        legacy = dict(valid_prefix())
        legacy.pop("dense_reference_scope")
        self.assertTrue(
            evaluate_native_prefix_cache_audit(
                legacy, expected_layers=32
            )["passed"]
        )
        self.assertFalse(
            evaluate_native_prefix_cache_audit(
                legacy,
                expected_layers=32,
                require_matched_dense_reference=True,
            )["passed"]
        )

    def test_zero_hit_or_timing_only_evidence_fails(self):
        for changes in (
            {"native_prefix_cache_hit": False, "cached_prefix_blocks": 0,
             "cached_prefix_tokens": 0, "prefix_shadow_rows": 0,
             "prefix_rows_excluded_from_repair": 0},
            {"hit_evidence_source": "ttft", "timing_inference_used": True},
        ):
            row = {**valid_prefix(), **changes}
            self.assertFalse(
                evaluate_native_prefix_cache_audit(
                    row, expected_layers=32
                )["passed"]
            )

    def test_missing_layer_shape_dtype_or_digest_mutation_fails(self):
        for changes in (
            {"prefix_shadow_layers": 31},
            {"prefix_shadow_rows": 191},
            {"prefix_shadow_dtype": "torch.float16"},
            {"prefix_shadow_geometry_valid": False},
            {"prefix_shadow_digest_after": "changed"},
        ):
            result = evaluate_native_prefix_cache_audit(
                {**valid_prefix(), **changes}, expected_layers=32
            )
            self.assertFalse(result["passed"])

    def test_prefix_repair_overlap_or_output_mismatch_fails(self):
        for changes in (
            {"prefix_rows_in_repair_mask": 1},
            {"prefix_rows_in_source_comparison": 1},
            {"active_positions_start_after_prefix": False},
            {"dense_token_ids_equal": False},
            {"logit_relative_l2": 0.001},
        ):
            result = evaluate_native_prefix_cache_audit(
                {**valid_prefix(), **changes}, expected_layers=32
            )
            self.assertFalse(result["passed"])


class H1QualificationGateTests(unittest.TestCase):
    def validate(self, gate):
        validate_h1_qualification_gate(
            gate,
            code_commit="code",
            model_id="model",
            model_revision="revision",
            adapter_name="adapter",
            cacheblend_patch_sha256="patch",
            cacheblend_tree="tree",
            gpu_uuid="GPU-1",
        )

    def test_valid_schema_v2_gate_unlocks_h1(self):
        self.validate(valid_gate())

    def test_old_or_partial_or_wrong_model_gate_is_rejected(self):
        for changes in (
            {"schema_version": 1},
            {"model_id": "other"},
            {"qualified_jobs_completed": 139},
            {"native_prefix_cache_qualified": False},
            {"cuda_event_timing": False},
            {"code_commit": "old"},
            {"cacheblend_patch_sha256": "old-patch"},
        ):
            with self.assertRaises(H1QualificationError):
                self.validate({**valid_gate(), **changes})

    def test_dual_model_gate_stops_after_two_h1_sentinels(self):
        def model_rows(spec):
            qualification = {
                **valid_gate(),
                "model_id": spec.model_id,
                "model_revision": spec.revision,
                "adapter_name": spec.adapter_name,
            }
            h1 = {
            "paper_evidence": False,
            "code_commit": "code",
            "model_id": spec.model_id,
            "completed_cases_this_run": 1,
            "completed_groups_this_run": 4,
            "appended_rows_this_run": 36,
            "elapsed_seconds_this_run": 10.0,
            "r1_dense_equivalence_passed": True,
            "h1_scan_allowed": True,
            "failure": None,
            }
            return qualification, h1
        mistral_q, mistral_h1 = model_rows(MISTRAL_SPEC)
        qwen_q, qwen_h1 = model_rows(QWEN_SPEC)
        result = build_joint_gate({
            "mistral_qualification": mistral_q,
            "mistral_h1": mistral_h1,
            "qwen_qualification": qwen_q,
            "qwen_h1": qwen_h1,
        }, hourly_price=6.0)
        self.assertTrue(result["ready_for_full_h1_pilot"])
        self.assertFalse(result["full_h1_started"])
        self.assertFalse(result["paper_evidence"])

    def test_dual_model_gate_rejects_partial_h1(self):
        mistral_q = {
            **valid_gate(),
            "model_id": MISTRAL_SPEC.model_id,
            "model_revision": MISTRAL_SPEC.revision,
            "adapter_name": MISTRAL_SPEC.adapter_name,
        }
        qwen_q = {
            **valid_gate(),
            "model_id": QWEN_SPEC.model_id,
            "model_revision": QWEN_SPEC.revision,
            "adapter_name": QWEN_SPEC.adapter_name,
        }
        h1 = {
            "paper_evidence": False,
            "code_commit": "code",
            "model_id": MISTRAL_SPEC.model_id,
            "completed_cases_this_run": 1,
            "completed_groups_this_run": 4,
            "appended_rows_this_run": 35,
            "elapsed_seconds_this_run": 10.0,
            "r1_dense_equivalence_passed": True,
            "h1_scan_allowed": True,
            "failure": None,
        }
        result = build_joint_gate({
            "mistral_qualification": mistral_q,
            "mistral_h1": h1,
            "qwen_qualification": qwen_q,
            "qwen_h1": {**h1, "model_id": QWEN_SPEC.model_id},
        })
        self.assertFalse(result["ready_for_full_h1_pilot"])


if __name__ == "__main__":
    unittest.main()
