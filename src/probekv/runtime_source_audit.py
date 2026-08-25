from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .cacheblend_patch import patch_files_for_mode


def audit_runtime_sources(repo: Path) -> Dict[str, Any]:
    manifest = repo / "patches" / "cacheblend" / "manifest.json"
    failures = []
    try:
        paths = patch_files_for_mode(manifest, "probekv_v6_staggered_runtime")
    except (OSError, ValueError) as error:
        return {"runtime_source_ready": False, "failures": [str(error)]}
    patch = paths[-1].read_text(encoding="utf-8")
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
        "patch_mode": "probekv_v6_staggered_runtime",
        "patch_files": [path.name for path in paths],
        "runtime_source_ready": not failures,
        "mistral_adapter": "mistral_cacheblend_llama_v041",
        "qwen_adapter": "qwen2_5_vllm041",
        "gpu_runtime_qualified": False,
        "failures": failures,
    }
