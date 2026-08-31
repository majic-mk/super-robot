from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .cacheblend_patch import patch_files_for_mode


def audit_runtime_sources(repo: Path) -> Dict[str, Any]:
    manifest = repo / "patches" / "cacheblend" / "manifest.json"
    failures = []
    try:
        paths = patch_files_for_mode(
            manifest, "probekv_v6_prefix_hardened_runtime"
        )
    except (OSError, ValueError) as error:
        return {"runtime_source_ready": False, "failures": [str(error)]}
    patch = paths[-2].read_text(encoding="utf-8")
    prefix_patch = paths[-1].read_text(encoding="utf-8")
    required_patch_markers = (
        "class Qwen2Model",
        "probekv_begin_prefill",
        "probekv_advance_prefill",
        "probekv_finish_prefill",
        "target_active_positions",
        "local_imp_indices",
        "org_seq_len\": int(absolute[-1].item()) + 1",
        "reuse_commit",
        "transition = bool(reuse_commit)",
    )
    for marker in required_patch_markers:
        if marker not in patch:
            failures.append("runtime patch lacks %s" % marker)
    for marker in ("exact_prefix_tokens", "_make_partial_bias_gqa"):
        if marker not in prefix_patch:
            failures.append("prefix runtime patch lacks %s" % marker)
    engine_path = repo / "src" / "probekv" / "cacheblend_v6_online_engine.py"
    session_path = repo / "src" / "probekv" / "resumable_prefill.py"
    worker_path = repo / "src" / "probekv" / "v6_qualification_worker.py"
    executor_path = repo / "src" / "probekv" / "v6_a800_executor.py"
    runner_path = repo / "scripts" / "server" / "run_v6_a800_qualification.py"
    for path, markers in (
        (engine_path, (
            "class CacheBlendV6OnlineEngine",
            "TorchLayerwiseSourceLoader",
            "exact_prefix_layers",
            "source_ready_observed_host_ms_by_segment_layer",
            "source_ready_gpu_ms_by_segment_layer",
            "layer_ready_gpu_ms",
        )),
        (session_path, ("class ProbeKVResumablePrefillSession", "commit_segment_reuse")),
        (worker_path, (
            "dispatch_qualification",
            "cuda_event_timing",
            "r1_dense_token_ids_equal",
            "teacher_forced_logit_relative_l2",
        )),
        (executor_path, (
            "class RealCacheBlendA800Executor",
            "canonical_variants",
            "teacher_tokens",
            "winner_variant",
            "aggregate_relative_l2",
            "expected_cacheblend_tree",
            "runtime_provenance",
            "run_native_prefix_cache_sentinel",
            "vllm_scheduler_computed_block_nums",
        )),
        (runner_path, (
            "requires the frozen 140-job matrix",
            "sentinel-only",
            "append_jsonl_fsync",
            "validate_qualification_results",
        )),
    ):
        if not path.is_file():
            failures.append("missing runtime source %s" % path.name)
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                failures.append("%s lacks %s" % (path.name, marker))
    return {
        "schema_version": 1,
        "patch_mode": "probekv_v6_prefix_hardened_runtime",
        "patch_files": [path.name for path in paths],
        "runtime_source_ready": not failures,
        "mistral_adapter": "mistral_cacheblend_llama_v041",
        "qwen_adapter": "qwen2_5_vllm041",
        "gpu_runtime_qualified": False,
        "failures": failures,
    }


def audit_v7_runtime_sources(repo: Path) -> Dict[str, Any]:
    manifest = repo / "patches" / "cacheblend" / "manifest.json"
    failures = []
    try:
        paths = patch_files_for_mode(
            manifest, "probekv_v7_single_artifact_runtime"
        )
    except (OSError, ValueError) as error:
        return {"runtime_source_ready": False, "failures": [str(error)]}
    rounding_patch = paths[-1].read_text(encoding="utf-8")
    for marker in ("repair_rounding_policy", '== "ceil"', "topk_num += 1"):
        if marker not in rounding_patch:
            failures.append("v7 patch lacks %s" % marker)
    checks = (
        (
            repo / "src" / "probekv" / "cacheblend_v6_online_engine.py",
            (
                "class CacheBlendV7OnlineEngine",
                "single_lossless_bf16_artifact",
                "start_artifact_replica_prefetch",
                '"repair_rounding_policy"',
            ),
        ),
        (
            repo / "src" / "probekv" / "v7_source_pool.py",
            ("class V7SourcePool", "exactly one full-KV Artifact", "bind_replica"),
        ),
        (
            repo / "src" / "probekv" / "v7_runtime_qualification.py",
            ("evaluate_v7_runtime_qualification", "validate_v7_h1_gate"),
        ),
        (
            repo / "scripts" / "server" / "run_v7_a800_qualification.py",
            ("protocol_version=7", "CacheBlendV7OnlineEngine", "sentinel-only"),
        ),
    )
    for path, markers in checks:
        if not path.is_file():
            failures.append("missing v7 runtime source %s" % path.name)
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                failures.append("%s lacks %s" % (path.name, marker))
    return {
        "schema_version": 2,
        "protocol_version": 7,
        "patch_mode": "probekv_v7_single_artifact_runtime",
        "patch_files": [path.name for path in paths],
        "runtime_source_ready": not failures,
        "single_artifact_policy": True,
        "multiple_replica_policy": True,
        "repair_rounding_policy": "ceil",
        "mistral_adapter": "mistral_cacheblend_llama_v041",
        "qwen_adapter": "qwen2_5_vllm041",
        "gpu_runtime_qualified": False,
        "failures": failures,
    }


def audit_v8_runtime_sources(repo: Path) -> Dict[str, Any]:
    """Static pre-rental audit for the v8 training-free runtime path."""
    manifest = repo / "patches" / "cacheblend" / "manifest.json"
    failures = []
    try:
        paths = patch_files_for_mode(
            manifest, "probekv_v8_training_free_residual_k"
        )
    except (OSError, ValueError) as error:
        return {"runtime_source_ready": False, "failures": [str(error)]}
    checks = (
        (
            repo / "src" / "probekv" / "cacheblend_v6_online_engine.py",
            (
                "class CacheBlendV8OnlineEngine",
                "training_free_residual_k_selection",
                "selection_state_k_only",
                "request_attributed_nonwinner_full_kv_bytes_transferred",
                "v8 full-KV prefetch is legal only for the frozen winner",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_selector.py",
            (
                "class TrainingFreeResidualKSelector",
                "evaluate_checkpoint_trace",
                "class RequestSelectionBudgetLedger",
                "score_repair_token_count",
                "cacheblend_repair_token_count",
                "INSUFFICIENT_RANKING_COVERAGE",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_selection_state_store.py",
            ("class SelectionStateStore", "full-KV fallback is forbidden"),
        ),
        (
            repo / "src" / "probekv" / "v8_leases.py",
            (
                "class V8LeaseManager",
                "freeze_and_acquire_logical",
                "compare_and_lease_batch",
                "same-Source Replica replan is limited to two attempts",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_planner.py",
            ("class PredictedJointPlanner", "class RefinedJointPlanner"),
        ),
        (
            repo / "src" / "probekv" / "v8_orchestration.py",
            ("class V8IncrementalCommitController", "REUSE_COMMIT"),
        ),
        (
            repo / "src" / "probekv" / "v8_profile.py",
            ("build_profile_freeze_contract", "build_runtime_cost_profile"),
        ),
        (
            repo / "src" / "probekv" / "v6_a800_executor.py",
            (
                "protocol_version == 8",
                "completed_depth_hook_verified",
                "selection_state_separate_backing_verified",
                "fixture.selection_variants",
                "residual_ratio=(0.15 if self.protocol_version == 8 else None)",
            ),
        ),
        (
            repo / "scripts" / "server" / "run_v8_a800_qualification.py",
            (
                "validate_frozen_selector_profile",
                "CacheBlendV8OnlineEngine",
                "requires the frozen 140-job matrix",
            ),
        ),
        (
            repo / "scripts" / "server" / "run_v8_h1_pilot.py",
            ("main(protocol_version=8)",),
        ),
    )
    for path, markers in checks:
        if not path.is_file():
            failures.append("missing v8 runtime source %s" % path.name)
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                failures.append("%s lacks %s" % (path.name, marker))
    return {
        "schema_version": 5,
        "protocol_version": 8,
        "patch_mode": "probekv_v8_training_free_residual_k",
        "patch_files": [path.name for path in paths],
        "runtime_source_ready": not failures,
        "training_free_selector": True,
        "selection_state_k_only": True,
        "single_artifact_policy": True,
        "winner_only_prefetch": True,
        "predicted_and_refined_planners": True,
        "gpu_runtime_qualified": False,
        "failures": failures,
    }


def audit_v8_schema7_runtime_sources(repo: Path) -> Dict[str, Any]:
    """Static no-GPU audit for the CacheBlend-aligned schema-v7 path."""
    manifest = repo / "patches" / "cacheblend" / "manifest.json"
    failures = []
    try:
        paths = patch_files_for_mode(
            manifest, "probekv_v8_winner_gradual_streaming"
        )
    except (OSError, ValueError) as error:
        return {"runtime_source_ready": False, "failures": [str(error)]}
    checks = (
        (
            repo / "src" / "probekv" / "cacheblend_v6_online_engine.py",
            (
                "class CacheBlendV8Schema7OnlineEngine",
                "online_immutable_no_full_digest",
                "qualification_destination_digest",
                "shrink_gradual_repair_support",
                "observe_winner_repair_check",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_schema7_repair.py",
            (
                "class SourceScoreTrimIndices",
                "class LoadRecomputeAwareRepairController",
                "build_initial_repair_support",
                "shrink_repair_support",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_schema7_planner.py",
            ("class PreparationAdmissionPlanner", "class FinalCommitPlanner"),
        ),
        (
            repo / "src" / "probekv" / "v8_schema7_transfer.py",
            ("class PinnedStagingPool", "SSD_GDS_TO_GPU"),
        ),
        (
            repo / "scripts" / "server" / "build_v8_schema7_no_gpu_handoff.py",
            ("build_schema7_no_gpu_handoff",),
        ),
    )
    for path, markers in checks:
        if not path.is_file():
            failures.append("missing schema-v7 runtime source %s" % path.name)
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                failures.append("%s lacks %s" % (path.name, marker))
    return {
        "protocol_version": 8,
        "schema_version": 7,
        "patch_mode": "probekv_v8_winner_gradual_streaming",
        "patch_files": [path.name for path in paths],
        "runtime_source_ready": not failures,
        "source_selection_repair_separated": True,
        "fixed15_fallback": True,
        "formal_online_full_digest": False,
        "gpu_runtime_qualified": False,
        "failures": failures,
    }


def audit_v8_schema8_runtime_sources(repo: Path) -> Dict[str, Any]:
    """Static proof that schema-v8 has a real server-to-engine call chain.

    This audit deliberately checks executable entry points, not just contracts
    or local simulations.  It is a no-GPU rental prerequisite, not a GPU
    qualification result.
    """

    manifest = repo / "patches" / "cacheblend" / "manifest.json"
    failures = []
    try:
        paths = patch_files_for_mode(
            manifest, "probekv_v8_gradual_barrier_tiered_lru"
        )
    except (OSError, ValueError) as error:
        return {"runtime_source_ready": False, "failures": [str(error)]}
    checks = (
        (
            repo / "src" / "probekv" / "cacheblend_v6_online_engine.py",
            (
                "class CacheBlendV8Schema8OnlineEngine",
                "configure_dense_selection_barrier",
                "admit_detached_preparation",
                "admit_preparation",
                "authorize_final_commit",
                "selective reuse before FinalCommitAdmission is forbidden",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_schema8_runtime.py",
            (
                "class Schema8BarrierRequestController",
                "close_selection_barrier",
                "apply_detached_preparation_admission",
                "apply_final_commit_admission",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_schema8_selector.py",
            (
                "class Schema8D1D2Selector",
                "d1_gate1_failed_continue",
                "d2_residual_band_min_cost",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_schema8_storage.py",
            (
                "class Schema8TieredBackingManager",
                "class Schema8TieredReplicaCoordinator",
                "begin_backing_migration",
                "finish_backing_migration",
                "last_request_use_epoch",
                "evict_ssd_lru_source",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_schema8_repair.py",
            (
                "class MultiSegmentRepairRatioPlan",
                "class JointRepairRatioDecision",
                "class JointLoadRecomputeAwareRepairController",
                "class RequestLayerUniformIORepairController",
                "class UniformIOBalanceDecision",
                "choose_request_level_adaptive_ratio",
                "validate_union_repair_ratio_plan",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_schema8_planner.py",
            (
                "class Gate1MarginalLowerBound",
                "predicted_reuse_marginal_lower_ms",
                "shared costs belong to request-level FinalCommitAdmission",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_schema8_profile.py",
            (
                "class SelectionDepthProfileV8",
                "class RepairPolicyProfileV8",
                "class RuntimeCostProfileV8",
            ),
        ),
        (
            repo / "src" / "probekv" / "v8_schema8_qualification.py",
            (
                "evaluate_schema8_runtime_qualification",
                "validate_schema8_h1_gate",
            ),
        ),
        (
            repo / "src" / "probekv" / "v6_a800_executor.py",
            (
                "CacheBlendV8Schema8OnlineEngine",
                "configure_dense_selection_barrier",
                "measured_repair_positions",
                "authorize_final_commit",
                "runtime_schema_version == 8",
                "online immutable execution requires the Artifact-creation digest",
            ),
        ),
        (
            repo / "scripts" / "server" / "run_v8_schema8_a800_sentinel.py",
            (
                "CacheBlendV8Schema8OnlineEngine",
                "runtime_schema_version=8",
                "qualification_full",
                "schema8_runtime_contract_passed",
            ),
        ),
        (
            repo / "scripts" / "server" / "freeze_v8_schema8_profiles.py",
            (
                "build_selection_depth_profile_v8",
                "build_repair_policy_profile_v8",
                "build_runtime_cost_profile_v8",
            ),
        ),
    )
    for path, markers in checks:
        if not path.is_file():
            failures.append("missing schema-v8 runtime source %s" % path.name)
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                failures.append("%s lacks %s" % (path.name, marker))
    return {
        "protocol_version": 8,
        "schema_version": 8,
        "patch_mode": "probekv_v8_gradual_barrier_tiered_lru",
        "patch_files": [path.name for path in paths],
        "runtime_source_ready": not failures,
        "dense_d1_d2_barrier": True,
        "d1_detached_winner_prefetch": True,
        "gate1_optimistic_marginal_feasibility": True,
        "final_commit_joint_timeline": True,
        "cpu_ssd_lru": True,
        "verified_backing_migration": True,
        "repair_ratio_scope_explicit": True,
        "request_level_joint_adaptive_ratio": True,
        "request_layer_uniform_io_balanced_ratio": True,
        "runtime_joint_ratio_plan_execution": True,
        "request_lru_migration_preserves_epoch": True,
        "online_artifact_digest_creation_time_only": True,
        "gpu_runtime_qualified": False,
        "failures": failures,
    }
