import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from probekv.cacheblend_v6_online_engine import CacheBlendV8OnlineEngine
from probekv.runtime_source_audit import audit_v8_runtime_sources
from probekv.v8_a800_jobs import (
    build_v8_a800_jobs,
    build_v8_preprofile_manifest,
    build_v8_profile_bound_qualification_manifest,
)
from probekv.v8_profile import (
    freeze_selector_profile,
    selector_profile_candidates,
    validate_frozen_selector_profile,
)
from probekv.v8_runtime_qualification import (
    evaluate_v8_runtime_qualification,
    validate_v8_h1_gate,
)
from probekv.v8_server_readiness import evaluate_v8_no_gpu_readiness


ROOT = Path(__file__).resolve().parents[1]
MISTRAL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"


def lock():
    return json.loads(
        (ROOT / "configs" / "a800_server_lock_v8.json").read_text(encoding="utf-8")
    )


def frozen_profile():
    candidate = selector_profile_candidates("mistral", "causal_commit_wait")[0]
    rows = [
        {
            "dataset": dataset,
            "paper_evidence": False,
            "locked_test_accessed": False,
            "cuda_event_timing": True,
            "fake_timing": False,
            "candidate_profile": candidate,
            "normalized_oracle_regret": 0.1,
            "completed_depth": 1,
            "selection_overhead_fraction": 0.04,
            "invalid_lock": False,
        }
        for dataset in ("musique", "2wikimultihopqa", "hotpotqa")
    ]
    return freeze_selector_profile(
        model_key="mistral",
        policy="causal_commit_wait",
        rows=rows,
        code_commit="code",
        model_revision=MISTRAL_REVISION,
        tokenizer_hash="tokenizer",
        cacheblend_patch_sha256="patch",
        microbenchmark_sha256="microbench",
    )


def qualification_manifest(profile):
    return build_v8_profile_bound_qualification_manifest(
        build_v8_a800_jobs(),
        profile=profile,
        code_commit="code",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_revision=MISTRAL_REVISION,
        tokenizer_hash="tokenizer",
        adapter_name="mistral_cacheblend_llama_v041",
        cacheblend_patch_sha256="patch",
        cacheblend_tree="tree",
        jobs_sha256="jobs-file-sha",
    )


def prefix_audit():
    return {
        "native_prefix_cache_qualified": True,
        "code_commit": "code",
        "model_revision": MISTRAL_REVISION,
        "cacheblend_patch_sha256": "patch",
    }


def runtime_audit(manifest):
    required = lock()["runtime"]["required_capabilities"]
    return {
        "schema_version": 4,
        "protocol_version": 8,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "code_commit": "code",
        "model_id": manifest["model"]["model_id"],
        "model_revision": MISTRAL_REVISION,
        "adapter_name": manifest["model"]["adapter_name"],
        "tokenizer_hash": "tokenizer",
        "cacheblend_patch_sha256": "patch",
        "cacheblend_tree": "tree",
        "profile_sha256": manifest["profile_sha256"],
        "job_digest": manifest["job_digest"],
        "cuda_event_timing": True,
        "fake_timing": False,
        "capabilities": {name: True for name in required},
        "single_artifact_policy_verified": True,
        "selection_state_k_only_verified": True,
        "selection_state_separate_backing_verified": True,
        "selection_scratch_peak_bytes": 1024,
        "selection_scratch_capacity_bytes": 2048,
        "fixed_repair_ratio": 0.15,
        "runtime_vllm_block_size": 16,
        "jobs": {"planned": 140, "completed": 140, "failed": 0},
        "correctness": {
            "r1_dense_token_ids_equal": True,
            "source_digest_unchanged": True,
            "artifact_digest_unchanged": True,
            "absolute_union_mask_verified": True,
            "completed_depth_hook_verified": True,
        },
        "selection_transfer": {
            "request_attributed_full_kv_bytes_transferred_for_selection": 0,
            "request_attributed_nonwinner_full_kv_bytes_transferred": 0,
            "request_attributed_full_kv_prefetch_before_source_freeze": 0,
        },
    }


class V8ProfileTests(unittest.TestCase):
    def test_frozen_profile_digest_is_recomputed_by_every_reader(self):
        profile = frozen_profile()
        validate_frozen_selector_profile(
            profile,
            model_key="mistral",
            code_commit="code",
            model_revision=MISTRAL_REVISION,
            tokenizer_hash="tokenizer",
            cacheblend_patch_sha256="patch",
        )
        tampered = json.loads(json.dumps(profile))
        tampered["selected_profile"]["eta"] = 0.99
        with self.assertRaises(ValueError):
            validate_frozen_selector_profile(tampered)

    def test_profile_requires_real_timing_and_all_three_datasets_per_candidate(self):
        profile = frozen_profile()
        self.assertTrue(profile["selector_profile_frozen"])
        candidate = selector_profile_candidates("mistral", "causal_commit_wait")[0]
        incomplete = [{
            "dataset": "musique", "paper_evidence": False,
            "locked_test_accessed": False, "cuda_event_timing": True,
            "fake_timing": False, "candidate_profile": candidate,
            "normalized_oracle_regret": 0.1, "completed_depth": 1,
            "selection_overhead_fraction": 0.01,
        }]
        with self.assertRaises(ValueError):
            freeze_selector_profile(
                model_key="mistral", policy="causal_commit_wait", rows=incomplete,
                code_commit="c", model_revision="m", tokenizer_hash="t",
                cacheblend_patch_sha256="p", microbenchmark_sha256="b",
            )
        fake = [dict(row, fake_timing=True) for row in incomplete]
        with self.assertRaises(ValueError):
            freeze_selector_profile(
                model_key="mistral", policy="causal_commit_wait", rows=fake,
                code_commit="c", model_revision="m", tokenizer_hash="t",
                cacheblend_patch_sha256="p", microbenchmark_sha256="b",
            )

    def test_qualification_manifest_cannot_precede_profile_or_omit_jobs_sha(self):
        profile = frozen_profile()
        with self.assertRaises(ValueError):
            build_v8_profile_bound_qualification_manifest(
                build_v8_a800_jobs(), profile={**profile, "selector_profile_frozen": False},
                code_commit="code", model_id="m", model_revision=MISTRAL_REVISION,
                tokenizer_hash="tokenizer", adapter_name="a",
                cacheblend_patch_sha256="patch", cacheblend_tree="tree",
                jobs_sha256="jobs",
            )
        with self.assertRaises(ValueError):
            build_v8_profile_bound_qualification_manifest(
                build_v8_a800_jobs(), profile=profile, code_commit="code",
                model_id="m", model_revision=MISTRAL_REVISION, tokenizer_hash="tokenizer",
                adapter_name="a", cacheblend_patch_sha256="patch",
                cacheblend_tree="tree", jobs_sha256="",
            )


class V8QualificationTests(unittest.TestCase):
    def test_140_job_matrix_is_unique_and_nonpaper(self):
        jobs = build_v8_a800_jobs()
        self.assertEqual(len(jobs), 140)
        self.assertEqual(len({item.job_id for item in jobs}), 140)
        self.assertTrue(all(not item.paper_evidence for item in jobs))

    def test_complete_profile_bound_gate_unlocks_only_exact_v8_h1(self):
        profile = frozen_profile()
        manifest = qualification_manifest(profile)
        gate = evaluate_v8_runtime_qualification(
            lock(), manifest, profile, runtime_audit(manifest), prefix_audit()
        )
        self.assertTrue(gate["gpu_runtime_qualified"], gate["failures"])
        validate_v8_h1_gate(
            gate,
            code_commit="code",
            model_id=manifest["model"]["model_id"],
            model_revision=MISTRAL_REVISION,
            adapter_name=manifest["model"]["adapter_name"],
            tokenizer_hash="tokenizer",
            cacheblend_patch_sha256="patch",
            cacheblend_tree="tree",
            profile_sha256=profile["profile_sha256"],
            job_digest=manifest["job_digest"],
        )
        with self.assertRaises(RuntimeError):
            validate_v8_h1_gate(
                {**gate, "protocol_version": 7},
                code_commit="code", model_id=manifest["model"]["model_id"],
                model_revision=MISTRAL_REVISION, adapter_name=manifest["model"]["adapter_name"],
                tokenizer_hash="tokenizer", cacheblend_patch_sha256="patch",
                cacheblend_tree="tree", profile_sha256=profile["profile_sha256"],
                job_digest=manifest["job_digest"],
            )

    def test_fake_timing_missing_capability_and_selection_transfer_fail(self):
        profile = frozen_profile()
        manifest = qualification_manifest(profile)
        audit = runtime_audit(manifest)
        audit["fake_timing"] = True
        audit["capabilities"]["selection_state_k_only"] = False
        audit["selection_transfer"]["request_attributed_nonwinner_full_kv_bytes_transferred"] = 1
        gate = evaluate_v8_runtime_qualification(
            lock(), manifest, profile, audit, prefix_audit()
        )
        self.assertFalse(gate["gpu_runtime_qualified"])
        self.assertGreaterEqual(len(gate["failures"]), 3)

    def test_runtime_source_audit_and_capabilities_cover_v8(self):
        audit = audit_v8_runtime_sources(ROOT)
        self.assertTrue(audit["runtime_source_ready"], audit["failures"])
        capabilities = CacheBlendV8OnlineEngine.capabilities()
        for name in lock()["runtime"]["required_capabilities"]:
            self.assertTrue(capabilities.get(name), name)

    def test_full_kv_prefetch_before_source_freeze_is_rejected_and_audited(self):
        engine = CacheBlendV8OnlineEngine.__new__(CacheBlendV8OnlineEngine)
        engine.frozen_source_by_segment = {}
        engine.request_attributed_full_kv_bytes_transferred_for_selection = 0
        engine.request_attributed_nonwinner_full_kv_bytes_transferred = 0
        engine.request_attributed_full_kv_prefetch_before_source_freeze = 0
        with self.assertRaises(RuntimeError):
            engine.start_artifact_replica_prefetch(
                segment_id="c", source_variant_id="s", artifact=object(),
                replica=SimpleNamespace(size_bytes=123), canonical_layers=(),
                segment_positions=(),
            )
        self.assertEqual(engine.request_attributed_full_kv_prefetch_before_source_freeze, 1)
        self.assertEqual(engine.request_attributed_nonwinner_full_kv_bytes_transferred, 0)


class V8NoGpuReadinessTests(unittest.TestCase):
    def test_complete_preprofile_handoff_allows_profile_rental_not_h1(self):
        frozen_lock = lock()
        patch = {
            "patch_mode": "probekv_v8_training_free_residual_k",
            "cacheblend_patch_sha256": "patch",
            "cacheblend_tree": "tree",
        }
        manifests = {}
        audits = {}
        for key in ("mistral", "qwen"):
            model = frozen_lock["models"][key]
            audits[key] = {"complete": True, "tokenizer_hash": key + "-tokenizer"}
            for policy_key, policy in (
                ("causal_wait", "causal_commit_wait"),
                ("immediate_staggered", "immediate_staggered_closed_loop"),
            ):
                manifests["%s_%s" % (key, policy_key)] = build_v8_preprofile_manifest(
                    code_commit="code", model_id=model["model_id"],
                    model_revision=model["revision"], tokenizer_hash=audits[key]["tokenizer_hash"],
                    adapter_name=model["adapter_name"],
                    selection_execution_policy=policy,
                    checkpoint_depths=model["completed_depths"],
                    cacheblend_patch_sha256="patch", cacheblend_tree="tree",
                )
        result = evaluate_v8_no_gpu_readiness(
            frozen_lock, manifests, audits, patch,
            expected_code_commit="code", actual_code_commit="code", git_clean=True,
            storage_ready=True, runtime_source_ready=True,
        )
        self.assertTrue(result["gpu_rental_ready_for_profile_freeze"], result["failures"])
        self.assertFalse(result["selector_profile_frozen"])
        self.assertFalse(result["h1_h2_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
