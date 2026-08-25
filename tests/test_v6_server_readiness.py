import json
import tempfile
import unittest
from pathlib import Path

import yaml

from probekv.v6_runtime_qualification import evaluate_runtime_qualification
from probekv.v6_server_readiness import (
    evaluate_dual_model_no_gpu_readiness,
    evaluate_no_gpu_readiness,
)
from scripts.server.audit_post_download_storage import audit_post_download_storage
from scripts.server.download_model_snapshot import audit_snapshot
from scripts.server.install_prebuilt_vllm_extensions import parse_sm_arches


def lock_record():
    return {
        "platform": {
            "python_major_minor": "3.10",
            "minimum_cpu_count": 16,
            "minimum_host_memory_gib": 110,
            "minimum_data_disk_free_gib": 250,
        },
        "stack": {
            "pytorch": "2.2.1",
            "pytorch_cuda": "12.1",
            "xformers": "0.0.25",
            "vllm": "0.4.1",
            "numpy": "1.26.4",
            "transformers": "4.40.2",
            "tokenizers": "0.19.1",
            "huggingface-hub": "0.36.2",
            "ray": "2.10.0",
            "cmake": "4.4.0",
            "ninja": "1.13.0",
            "cacheblend_commit": "cb",
            "cacheblend_patch_mode": "probekv_v6_multiregion",
        },
        "model": {"model_id": "model", "revision": "revision"},
        "runtime": {
            "backend": "cacheblend_multisegment_closed_loop",
            "implementation_status": "engine_pending",
            "required_capabilities": [
                "async_multisource_loading",
                "layer_resumable_prefill",
                "layer_indexed_union_repair_masks",
                "per_segment_staggered_boundaries",
                "causal_commit_wait_execution",
                "immediate_staggered_closed_loop_execution",
                "policy_conditioned_probe_state",
            ],
        },
    }


def job_manifest():
    return {
        "protocol_version": 6,
        "paper_evidence": False,
        "jobs": 140,
        "job_digest": "job-digest",
        "jobs_sha256": "jobs",
        "code_commit": "code",
        "git_clean": True,
        "config_sha256": "config",
        "contract_sha256": "contract",
        "server_lock_sha256": "lock",
        "model": {"model_id": "model", "revision": "revision"},
        "runtime": {"backend": "cacheblend_multisegment_closed_loop"},
        "cacheblend": {"patch_sha256": "patch"},
    }


def host_record():
    return {
        "python": "3.10.14",
        "cpu_count": 18,
        "host_memory_gib": 120,
        "data_disk_free_gib": 300,
        "git_commit": "code",
        "git_status": "",
        "nvcc_cuda": "12.1",
        "packages": {
            "torch": "2.2.1+cu121",
            "xformers": "0.0.25",
            "vllm": "0.4.1",
            "numpy": "1.26.4",
            "transformers": "4.40.2",
            "tokenizers": "0.19.1",
            "huggingface-hub": "0.36.2",
            "ray": "2.10.0",
            "cmake": "4.4.0",
            "ninja": "1.13.0",
        },
    }


class NoGpuReadinessTests(unittest.TestCase):
    def test_dual_model_source_gate_allows_qualification_not_h1(self):
        lock = lock_record()
        lock.pop("model", None)
        lock["models"] = {
            key: {
                "model_id": key + "-model",
                "revision": key + "-revision",
                "adapter_name": key + "-adapter",
            }
            for key in ("mistral", "qwen")
        }
        lock["runtime"]["implementation_status"] = (
            "concrete_engine_hook_complete_requires_a800_qualification"
        )
        manifests = {}
        audits = {}
        hashes = {}
        for key in ("mistral", "qwen"):
            manifest = job_manifest()
            manifest["model"] = {
                "model_id": key + "-model",
                "revision": key + "-revision",
                "adapter_name": key + "-adapter",
                "tokenizer_hash": key + "-tokenizer",
            }
            manifests[key] = manifest
            audits[key] = {
                "complete": True,
                "model_id": key + "-model",
                "revision": key + "-revision",
                "tokenizer_hash": key + "-tokenizer",
            }
            hashes[key] = {
                "jobs_sha256": "jobs", "config_sha256": "config",
                "contract_sha256": "contract", "server_lock_sha256": "lock",
            }
        result = evaluate_dual_model_no_gpu_readiness(
            lock, manifests, host_record(),
            {"storage_ready": True, "storage_mode": "dual_model_resident"},
            audits,
            {
                "cacheblend_commit": "cb",
                "patch_mode": "probekv_v6_multiregion",
                "cacheblend_patch_sha256": "patch",
                "cacheblend_tree": "tree",
            },
            {"runtime_source_ready": True, "failures": []},
            expected_code_commit="code", actual_hashes_by_model=hashes,
        )
        self.assertTrue(result["artifact_preparation_ready"], result["failures"])
        self.assertTrue(result["mistral_runtime_source_ready"])
        self.assertTrue(result["qwen_runtime_source_ready"])
        self.assertTrue(result["gpu_rental_ready_for_runtime_qualification"])
        self.assertFalse(result["gpu_runtime_qualified"])
        self.assertFalse(result["h1_h2_execution_allowed"])

    def test_complete_artifacts_allow_rental_but_not_h1(self):
        result = evaluate_no_gpu_readiness(
            lock_record(),
            job_manifest(),
            host_record(),
            {"complete": True, "model_id": "model", "revision": "revision"},
            {
                "cacheblend_commit": "cb",
                "patch_mode": "probekv_v6_multiregion",
                "cacheblend_patch_sha256": "patch",
                "cacheblend_tree": "tree",
            },
            expected_code_commit="code",
            actual_hashes={
                "jobs_sha256": "jobs",
                "config_sha256": "config",
                "contract_sha256": "contract",
                "server_lock_sha256": "lock",
            },
        )
        self.assertTrue(result["artifact_preparation_ready"])
        self.assertTrue(result["gpu_rental_ready_for_runtime_bringup"])
        self.assertFalse(result["gpu_rental_ready_for_runtime_qualification"])
        self.assertIsNotNone(result["blocking_source_implementation"])
        self.assertFalse(result["gpu_runtime_qualified"])
        self.assertFalse(result["h1_h2_execution_allowed"])

    def test_wrong_stack_dirty_tree_and_model_fail(self):
        host = host_record()
        host["git_status"] = " M source.py"
        host["packages"]["vllm"] = "0.5.0"
        result = evaluate_no_gpu_readiness(
            lock_record(),
            job_manifest(),
            host,
            {"complete": False, "model_id": "model", "revision": "other"},
            {
                "cacheblend_commit": "wrong",
                "patch_mode": "probekv_v6_multiregion",
                "cacheblend_patch_sha256": "patch",
                "cacheblend_tree": "tree",
            },
            expected_code_commit="code",
            actual_hashes={
                "jobs_sha256": "jobs",
                "config_sha256": "config",
                "contract_sha256": "contract",
                "server_lock_sha256": "lock",
            },
        )
        self.assertFalse(result["artifact_preparation_ready"])
        self.assertGreaterEqual(len(result["failures"]), 4)

    def test_model_snapshot_audit_requires_config_tokenizer_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "revision"
            root.mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            audit = audit_snapshot(root, "model", "revision")
            self.assertTrue(audit["complete"])
            self.assertEqual(audit["weight_files"], ["model.safetensors"])


class RuntimeQualificationTests(unittest.TestCase):
    def valid_audit(self):
        return {
            "paper_evidence": False,
            "runtime_backend": "cacheblend_multisegment_closed_loop",
            "concrete_engine_hook": True,
            "capabilities": {
                "async_multisource_loading": True,
                "layer_resumable_prefill": True,
                "layer_indexed_union_repair_masks": True,
                "per_segment_staggered_boundaries": True,
                "causal_commit_wait_execution": True,
                "immediate_staggered_closed_loop_execution": True,
                "policy_conditioned_probe_state": True,
            },
            "code_commit": "code",
            "job_digest": "job-digest",
            "model_revision": "revision",
            "cacheblend_patch_sha256": "patch",
            "correctness": {
                "r1_dense_token_ids_equal": True,
                "max_teacher_forced_logit_relative_l2": 0.00001,
                "canonical_source_digests_unchanged": True,
                "absolute_union_mask_verified": True,
            },
            "jobs": {"planned": 140, "completed": 140, "failed": 0},
        }

    def test_full_a800_audit_unlocks_h1_h2(self):
        result = evaluate_runtime_qualification(
            lock_record(), job_manifest(), self.valid_audit()
        )
        self.assertTrue(result["gpu_runtime_qualified"])
        self.assertTrue(result["h1_h2_execution_allowed"])

    def test_contract_only_adapter_cannot_unlock_experiments(self):
        audit = self.valid_audit()
        audit["concrete_engine_hook"] = False
        audit["jobs"]["completed"] = 0
        result = evaluate_runtime_qualification(
            lock_record(), job_manifest(), audit
        )
        self.assertFalse(result["gpu_runtime_qualified"])
        self.assertTrue(any("concrete" in item for item in result["failures"]))


class ServerScriptSafetyTests(unittest.TestCase):
    def test_setup_script_has_exact_stack_and_no_embedded_credentials(self):
        text = Path("scripts/server/setup_a800_env.sh").read_text(encoding="utf-8")
        for expected in ("torch==2.2.1", "xformers==0.0.25", "release 12\\.1"):
            self.assertIn(expected, text)
        lowered = text.lower()
        self.assertNotIn("password=", lowered)
        self.assertNotIn("hf_token=", lowered)
        self.assertIn('python_bin="$(command -v "$python_bin")"', text)
        self.assertIn("PROBEKV_NVCC_BIN", text)
        self.assertIn(
            'TORCH_CUDA_ARCH_LIST="${PROBEKV_CUDA_ARCH_LIST:-8.0}"',
            text,
        )
        self.assertIn("PROBEKV_PREBUILT_VLLM_SOURCE", text)
        self.assertIn("install_prebuilt_vllm_extensions.py", text)
        self.assertNotIn("import vllm, vllm._C", text)
        self.assertIn("dynamic loading is deferred to the A800 gate", text)
        self.assertIn("PROBEKV_ENV_DIR", text)
        self.assertIn('--repo "$repo"', text)
        self.assertIn(
            '"$python_bin" "$repo/scripts/server/plan_server_storage.py"',
            text,
        )
        self.assertEqual(
            text.count('--manifest "$repo/patches/cacheblend/manifest.json"'),
            2,
        )
        patch_setup = Path("scripts/server/prepare_cacheblend.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('python_bin="${PROBEKV_PYTHON_BIN:-python3}"', patch_setup)
        self.assertIn('"$python_bin" - <<\'PY\'', patch_setup)
        self.assertIn("PROBEKV_CACHEBLEND_SOURCE", patch_setup)
        self.assertIn('git clone --no-hardlinks "$repository" "$target"', patch_setup)
        verifier = Path("scripts/server/verify_cacheblend_patch.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Path(__file__).resolve().parents[2]", verifier)
        legacy = Path("scripts/server/run_preflight.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('mode="${1:-gpu}"', legacy)
        self.assertIn('if [[ "$mode" == "gpu" ]]', legacy)

    def test_package_metadata_matches_runtime_syntax_floor(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        setup_cfg = Path("setup.cfg").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.10"', pyproject)
        self.assertIn("python_requires = >=3.10", setup_cfg)

    def test_server_lock_matches_contract_and_exact_install_file(self):
        lock = json.loads(
            Path("configs/a800_server_lock.json").read_text(encoding="utf-8")
        )
        contract = yaml.safe_load(
            Path("configs/experiment_contract.yaml").read_text(encoding="utf-8")
        )
        primary = contract["stacks"]["primary"]
        self.assertEqual(lock["stack"]["cacheblend_commit"], primary["commit"])
        self.assertEqual(lock["stack"]["vllm"], str(primary["vllm"]))
        self.assertEqual(lock["stack"]["pytorch"], str(primary["pytorch"]))
        self.assertEqual(lock["stack"]["xformers"], str(primary["xformers"]))
        self.assertEqual(
            lock["stack"]["cacheblend_patch_mode"],
            "probekv_v6_staggered_runtime",
        )
        self.assertEqual(set(lock["models"]), {"mistral", "qwen"})
        self.assertEqual(lock["models"]["qwen"]["role"], "formal_primary")
        self.assertEqual(lock["platform"]["minimum_combined_free_gib"], 70)
        requirements = Path("requirements/server-tools.txt").read_text(
            encoding="utf-8"
        )
        for package in (
            "numpy",
            "transformers",
            "tokenizers",
            "huggingface-hub",
            "ray",
            "cmake",
            "ninja",
        ):
            self.assertIn(
                "%s==%s" % (package, lock["stack"][package]),
                requirements,
            )

    def test_no_gpu_preflight_separates_pre_and_post_download_storage(self):
        text = Path("scripts/server/run_v6_no_gpu_preflight.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'pre_download_storage="$(dirname "$mistral_audit")/storage.json"',
            text,
        )
        self.assertIn("audit_post_download_storage.py", text)

    def test_post_download_storage_enforces_system_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_post_download_storage(
                Path(directory), Path(directory)
            )
        self.assertIn("stage_filesystem_free_gib", result)
        self.assertIn("system_free_gib", result)
        self.assertEqual(result["system_reserve_gib"], 15)

    def test_prebuilt_extension_arch_parser_requires_exact_sm80(self):
        output = (
            "ELF file 1: _C.cpython-310-x86_64-linux-gnu.1.sm_80.cubin\n"
            "ELF file 2: _C.cpython-310-x86_64-linux-gnu.2.sm_80.cubin\n"
        )
        self.assertEqual(parse_sm_arches(output), {"80"})
        self.assertEqual(
            parse_sm_arches(output + "ELF file 3: x.sm_90.cubin\n"),
            {"80", "90"},
        )


if __name__ == "__main__":
    unittest.main()
