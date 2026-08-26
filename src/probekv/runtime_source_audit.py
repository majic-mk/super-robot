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
