import json
import unittest
from pathlib import Path

from probekv.v7_a800_jobs import (
    build_v7_a800_job_manifest,
    build_v7_a800_jobs,
)
from probekv.v7_runtime_qualification import (
    build_v7_joint_gate,
    evaluate_v7_runtime_qualification,
    validate_v7_h1_gate,
)
from probekv.v7_server_readiness import evaluate_v7_no_gpu_readiness


ROOT = Path(__file__).resolve().parents[1]


def _lock():
    return json.loads(
        (ROOT / "configs" / "a800_server_lock_v7.json").read_text(encoding="utf-8")
    )


def _manifest(model_key="mistral"):
    lock = _lock()
    model = lock["models"][model_key]
    jobs = build_v7_a800_jobs(
        json.loads(
            (ROOT / "configs" / "v7_a800_microbench.json").read_text(
                encoding="utf-8"
            )
        )
    )
    return build_v7_a800_job_manifest(
        jobs,
        jobs_sha256="jobs",
        code_commit="code",
        git_clean=True,
        config_sha256="config",
        contract_sha256="contract",
        server_lock_sha256="lock",
        model_id=model["model_id"],
        model_revision=model["revision"],
        tokenizer_hash=model_key + "-tokenizer",
        adapter_name=model["adapter_name"],
        runtime_backend=lock["runtime"]["backend"],
        cacheblend_commit=lock["stack"]["cacheblend_commit"],
        cacheblend_patch_mode=lock["stack"]["cacheblend_patch_mode"],
        cacheblend_patch_sha256="patch",
        cacheblend_tree="tree",
    )


def _prefix(manifest):
    model = manifest["model"]
    layers = 32 if model["adapter_name"].startswith("mistral") else 28
    return {
        "paper_evidence": False,
        "locked_test_accessed": False,
        "code_commit": "code",
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "adapter_name": model["adapter_name"],
        "cacheblend_patch_sha256": "patch",
        "cacheblend_tree": "tree",
        "gpu_uuid": "GPU-v7",
        "model_num_layers": layers,
        "hit_evidence_source": "vllm_scheduler_computed_block_nums",
        "timing_inference_used": False,
        "requested_prefix_tokens": 192,
        "native_prefix_cache_hit": True,
        "cached_prefix_blocks": 12,
        "cached_prefix_tokens": 192,
        "block_size": 16,
        "prefix_shadow_layers": layers,
        "prefix_shadow_rows": 192,
        "prefix_shadow_dtype": "torch.bfloat16",
        "prefix_shadow_device": "cuda",
        "prefix_shadow_geometry_valid": True,
        "prefix_shadow_digest_before": "prefix-digest",
        "prefix_shadow_digest_after": "prefix-digest",
        "active_positions_start_after_prefix": True,
        "prefix_rows_excluded_from_repair": 192,
        "prefix_rows_in_repair_mask": 0,
        "prefix_rows_in_source_comparison": 0,
        "combined_prefix_r1_reuse_exercised": True,
        "dense_token_ids_equal": True,
        "logit_relative_l2": 1e-5,
        "cuda_event_timing": True,
    }


def _audit(manifest):
    lock = _lock()
    return {
        "schema_version": 3,
        "protocol_version": 7,
        "stage": "v7_a800_runtime_qualification",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "runtime_backend": lock["runtime"]["backend"],
        "concrete_engine_hook": True,
        "capabilities": {name: True for name in lock["runtime"]["required_capabilities"]},
        "code_commit": "code",
        "job_digest": manifest["job_digest"],
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["revision"],
        "adapter_name": manifest["model"]["adapter_name"],
        "cacheblend_patch_sha256": "patch",
        "runtime_provenance": {
            "torch": "2.2.1+cu121",
            "vllm": "0.4.1",
            "xformers": "0.0.25",
            "torch_cuda": "12.1",
            "gpu_name": "NVIDIA A800-SXM4-80GB",
            "compute_capability": [8, 0],
            "cacheblend_tree": "tree",
        },
        "correctness": {
            "r1_dense_token_ids_equal": True,
            "max_teacher_forced_logit_relative_l2": 1e-5,
            "canonical_source_digests_unchanged": True,
            "artifact_digests_unchanged": True,
            "absolute_union_mask_verified": True,
        },
        "single_artifact_policy_verified": True,
        "max_artifacts_per_source_variant_observed": 1,
        "all_job_artifact_digests_unchanged": True,
        "repair_rounding_policy": "ceil",
        "alignment_quantum": 16,
        "runtime_vllm_block_size": 16,
        "jobs": {"planned": 140, "completed": 140, "failed": 0},
        "gpu_uuid": "GPU-v7",
    }


class V7QualificationTests(unittest.TestCase):
    def test_v7_matrix_has_140_unique_nonpaper_jobs(self):
        jobs = build_v7_a800_jobs(
            json.loads(
                (ROOT / "configs" / "v7_a800_microbench.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        self.assertEqual(len(jobs), 140)
        self.assertEqual(len({job.job_id for job in jobs}), 140)
        self.assertTrue(all(not job.paper_evidence for job in jobs))

    def test_complete_v7_gate_unlocks_only_matching_h1(self):
        manifest = _manifest()
        gate = evaluate_v7_runtime_qualification(
            _lock(), manifest, _audit(manifest), _prefix(manifest),
            job_manifest_sha256="manifest-hash",
            runtime_audit_sha256="runtime-hash",
            prefix_audit_sha256="prefix-hash",
        )
        self.assertTrue(gate["gpu_runtime_qualified"], gate["failures"])
        validate_v7_h1_gate(
            gate,
            code_commit="code",
            model_id=manifest["model"]["model_id"],
            model_revision=manifest["model"]["revision"],
            adapter_name=manifest["model"]["adapter_name"],
            cacheblend_patch_sha256="patch",
            cacheblend_tree="tree",
        )
        with self.assertRaises(RuntimeError):
            validate_v7_h1_gate(
                {**gate, "protocol_version": 6},
                code_commit="code",
                model_id=manifest["model"]["model_id"],
                model_revision=manifest["model"]["revision"],
                adapter_name=manifest["model"]["adapter_name"],
                cacheblend_patch_sha256="patch",
                cacheblend_tree="tree",
            )

    def test_fake_or_incomplete_artifact_evidence_cannot_qualify(self):
        manifest = _manifest()
        audit = _audit(manifest)
        audit["capabilities"]["cuda_event_timing"] = False
        audit["all_job_artifact_digests_unchanged"] = False
        gate = evaluate_v7_runtime_qualification(
            _lock(), manifest, audit, _prefix(manifest),
            job_manifest_sha256="manifest-hash",
            runtime_audit_sha256="runtime-hash",
            prefix_audit_sha256="prefix-hash",
        )
        self.assertFalse(gate["gpu_runtime_qualified"])
        self.assertTrue(any("Artifact" in item for item in gate["failures"]))

    def test_alignment_mismatch_is_contract_not_math_claim(self):
        manifest = _manifest()
        audit = _audit(manifest)
        audit["runtime_vllm_block_size"] = 32
        gate = evaluate_v7_runtime_qualification(
            _lock(), manifest, audit, _prefix(manifest),
            job_manifest_sha256="manifest-hash",
            runtime_audit_sha256="runtime-hash",
            prefix_audit_sha256="prefix-hash",
        )
        self.assertFalse(gate["experiment_contract_compatible"])
        self.assertFalse(gate["gpu_runtime_qualified"])
        self.assertNotIn("gpu_runtime_correctness", gate)

    def test_joint_gate_requires_two_qualifications_and_36_row_sentinels(self):
        manifest = _manifest()
        gate = evaluate_v7_runtime_qualification(
            _lock(), manifest, _audit(manifest), _prefix(manifest),
            job_manifest_sha256="m", runtime_audit_sha256="r",
            prefix_audit_sha256="p",
        )
        sentinel = {
            "protocol_version": 7,
            "passed": True,
            "appended_rows_this_run": 36,
            "r1_dense_equivalence_passed": True,
        }
        result = build_v7_joint_gate(
            code_commit="code", mistral_gate=gate, qwen_gate=gate,
            mistral_h1_sentinel=sentinel, qwen_h1_sentinel=sentinel,
        )
        self.assertTrue(result["ready_for_full_h1_pilot"])
        self.assertFalse(result["full_h1_started"])


class V7NoGpuReadinessTests(unittest.TestCase):
    def test_complete_cpu_handoff_allows_rental_not_h1(self):
        lock = _lock()
        manifests = {key: _manifest(key) for key in ("mistral", "qwen")}
        packages = {
            "torch": "2.2.1+cu121", "xformers": "0.0.25", "vllm": "0.4.1",
            "numpy": "1.26.4", "transformers": "4.40.2",
            "tokenizers": "0.19.1", "huggingface-hub": "0.36.2",
            "ray": "2.10.0", "cmake": "4.4.0", "ninja": "1.13.0",
        }
        host = {
            "git_commit": "code", "git_status": "", "python": "3.10.14",
            "cpu_count": 16, "host_memory_gib": 120, "nvcc_cuda": "12.1",
            "packages": packages,
        }
        patch = {
            "patch_mode": lock["stack"]["cacheblend_patch_mode"],
            "cacheblend_commit": lock["stack"]["cacheblend_commit"],
            "cacheblend_patch_sha256": "patch", "cacheblend_tree": "tree",
        }
        audits = {
            key: {
                "complete": True,
                "model_id": lock["models"][key]["model_id"],
                "revision": lock["models"][key]["revision"],
            }
            for key in ("mistral", "qwen")
        }
        hashes = {
            key: {
                "jobs_sha256": "jobs", "config_sha256": "config",
                "contract_sha256": "contract", "server_lock_sha256": "lock",
            }
            for key in ("mistral", "qwen")
        }
        result = evaluate_v7_no_gpu_readiness(
            lock, manifests, host, {"storage_ready": True}, audits, patch,
            {
                "runtime_source_ready": True,
                "patch_mode": lock["stack"]["cacheblend_patch_mode"],
                "failures": [],
            },
            expected_code_commit="code", actual_hashes_by_model=hashes,
        )
        self.assertTrue(result["gpu_rental_ready_for_runtime_qualification"], result["failures"])
        self.assertFalse(result["gpu_runtime_qualified"])
        self.assertFalse(result["h1_h2_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
