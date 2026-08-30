from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


_HUNK_HEADER = re.compile(
    r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@"
)


def validate_unified_diff(path: Path) -> None:
    """Reject malformed hunk counts before a patch reaches the server."""
    lines = path.read_text(encoding="utf-8").splitlines()
    hunk_count = 0
    index = 0
    while index < len(lines):
        match = _HUNK_HEADER.match(lines[index])
        if match is None:
            index += 1
            continue
        hunk_count += 1
        expected_old = int(match.group(1) or 1)
        expected_new = int(match.group(2) or 1)
        actual_old = 0
        actual_new = 0
        index += 1
        while index < len(lines):
            line = lines[index]
            if line.startswith("@@ ") or line.startswith("diff --git "):
                break
            if line.startswith("\\"):
                index += 1
                continue
            if not line or line[0] not in (" ", "+", "-"):
                raise ValueError(
                    "invalid unified diff line in %s: %r" % (path, line)
                )
            if line[0] in (" ", "-"):
                actual_old += 1
            if line[0] in (" ", "+"):
                actual_new += 1
            index += 1
        if (actual_old, actual_new) != (expected_old, expected_new):
            raise ValueError(
                "malformed hunk in %s: expected %d/%d old/new lines, "
                "observed %d/%d"
                % (
                    path,
                    expected_old,
                    expected_new,
                    actual_old,
                    actual_new,
                )
            )
    if hunk_count == 0:
        raise ValueError("patch contains no unified diff hunks: %s" % path)


def load_patch_manifest(path: Path) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("base_commit") != (
        "b72d7945e6d6306f12be66520196e0f081fa2b0c"
    ):
        raise ValueError("unexpected CacheBlend base commit")
    modes = manifest.get("patches")
    required_modes = {
        "cb0",
        "probekv",
        "probekv_closed_loop",
        "probekv_v6_multiregion",
        "probekv_v6_staggered_runtime",
        "probekv_v6_prefix_hardened_runtime",
        "probekv_v7_single_artifact_runtime",
        "probekv_v8_training_free_residual_k",
        "probekv_v8_schema6_joint_cfo",
        "probekv_v8_winner_gradual_streaming",
    }
    if not isinstance(modes, dict) or not required_modes.issubset(modes):
        raise ValueError(
            "patch manifest must define cb0, probekv, "
            "probekv_closed_loop, probekv_v6_multiregion and "
            "probekv_v6_staggered_runtime and "
            "probekv_v6_prefix_hardened_runtime modes"
        )
    runtime_modes = manifest.get("runtime_modes")
    if (
        not isinstance(runtime_modes, dict)
        or "h1_case_runner" not in runtime_modes
        or "closed_loop_v5" not in runtime_modes
        or "closed_loop_v6" not in runtime_modes
    ):
        raise ValueError("patch manifest must define CacheBlend runtime modes")
    closed_loop = runtime_modes["closed_loop_v5"]
    for key in (
        "layer_resumable_prefill",
        "async_source_loading",
        "scheduler_feedback_required",
        "cuda_event_timing_required",
    ):
        if closed_loop.get(key) is not True:
            raise ValueError(
                "closed_loop_v5 must require %s" % key
            )
    closed_loop_v6 = runtime_modes["closed_loop_v6"]
    for key in (
        "ordered_repair_regions",
        "absolute_union_mask",
        "per_segment_ratio",
        "layer_resumable_prefill",
        "async_multisource_loading",
        "scheduler_feedback_required",
        "cuda_event_timing_required",
    ):
        if closed_loop_v6.get(key) is not True:
            raise ValueError("closed_loop_v6 must require %s" % key)
    if closed_loop_v6.get("patch_mode") != "probekv_v6_multiregion":
        raise ValueError("closed_loop_v6 must use the v6 multi-region patch mode")
    staggered = runtime_modes.get("staggered_runtime_v6")
    if not isinstance(staggered, dict):
        raise ValueError("patch manifest must define staggered_runtime_v6")
    for key in (
        "ordered_repair_regions",
        "absolute_union_mask",
        "per_segment_ratio",
        "per_segment_staggered_boundaries",
        "layer_resumable_prefill",
        "async_multisource_loading",
        "scheduler_feedback_required",
        "cuda_event_timing_required",
    ):
        if staggered.get(key) is not True:
            raise ValueError("staggered_runtime_v6 must require %s" % key)
    if staggered.get("patch_mode") != "probekv_v6_staggered_runtime":
        raise ValueError("staggered runtime must use its explicit patch mode")
    if staggered.get("status") != (
        "concrete_engine_hook_complete_requires_a800_qualification"
    ):
        raise ValueError("staggered runtime source status is not qualification-ready")
    hardened = runtime_modes.get("prefix_hardened_runtime_v6")
    if not isinstance(hardened, dict):
        raise ValueError("patch manifest must define prefix_hardened_runtime_v6")
    if hardened.get("patch_mode") != "probekv_v6_prefix_hardened_runtime":
        raise ValueError("prefix hardening must use its explicit patch mode")
    for key in (
        "native_prefix_cache",
        "exact_prefix_pre_rope_shadow",
        "prefix_rows_excluded_from_active_queries",
    ):
        if hardened.get(key) is not True:
            raise ValueError("prefix hardening must require %s" % key)
    v7 = runtime_modes.get("single_artifact_runtime_v7")
    if not isinstance(v7, dict):
        raise ValueError("patch manifest must define single_artifact_runtime_v7")
    if v7.get("patch_mode") != "probekv_v7_single_artifact_runtime":
        raise ValueError("v7 runtime must use its explicit patch mode")
    for key in (
        "single_lossless_bf16_artifact",
        "multiple_physical_replicas",
        "per_segment_staggered_boundaries",
    ):
        if v7.get(key) is not True:
            raise ValueError("v7 runtime must require %s" % key)
    if v7.get("repair_rounding_policy") != "ceil":
        raise ValueError("v7 runtime must use conservative ceil repair rounding")
    v8 = runtime_modes.get("training_free_residual_k_runtime_v8")
    if not isinstance(v8, dict):
        raise ValueError("patch manifest must define training_free_residual_k_runtime_v8")
    if v8.get("patch_mode") != "probekv_v8_training_free_residual_k":
        raise ValueError("v8 runtime must use its explicit patch mode")
    for key in (
        "training_free_residual_k_selection",
        "selection_state_k_only",
        "winner_only_prefetch",
        "predicted_and_refined_joint_planners",
    ):
        if v8.get(key) is not True:
            raise ValueError("v8 runtime must require %s" % key)
    if v8.get("full_kv_transfer_for_selection") is not False:
        raise ValueError("v8 selection must not transfer full KV")
    if v8.get("repair_ratio") != 0.15:
        raise ValueError("v8 runtime must freeze repair ratio 0.15")
    schema6 = runtime_modes.get("schema6_joint_cfo_runtime_v8")
    if not isinstance(schema6, dict):
        raise ValueError("patch manifest must define schema6_joint_cfo_runtime_v8")
    if schema6.get("patch_mode") != "probekv_v8_schema6_joint_cfo":
        raise ValueError("schema-v6 runtime must use its explicit patch mode")
    for key in (
        "post_rope_cfo_hook",
        "streaming_attention_aggregation",
        "joint_timeline_gate2_gate3",
        "active_repair_row_writeback_each_layer",
    ):
        if schema6.get(key) is not True:
            raise ValueError("schema-v6 runtime must require %s" % key)
    schema7 = runtime_modes.get("winner_gradual_streaming_runtime_v8")
    if not isinstance(schema7, dict):
        raise ValueError("patch manifest must define schema-v7 gradual runtime")
    if schema7.get("patch_mode") != "probekv_v8_winner_gradual_streaming":
        raise ValueError("schema-v7 runtime must use its explicit patch mode")
    for key in (
        "source_score_repair_support_separated",
        "winner_specific_repair_metric",
        "gradual_no_reentry_support",
        "load_recompute_overlap_controller",
        "final_commit_admission",
        "integrity_modes_split",
    ):
        if schema7.get(key) is not True:
            raise ValueError("schema-v7 runtime must require %s" % key)
    if schema7.get("formal_online_full_digest") is not False:
        raise ValueError("schema-v7 online path must not perform full digest")
    return manifest


def patch_files_for_mode(manifest_path: Path, mode: str) -> Tuple[Path, ...]:
    manifest = load_patch_manifest(manifest_path)
    if mode not in manifest["patches"]:
        raise ValueError("unknown CacheBlend patch mode: %s" % mode)
    root = manifest_path.parent
    paths = tuple(root / str(name) for name in manifest["patches"][mode])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("missing CacheBlend patches: %s" % ", ".join(missing))
    for path in paths:
        validate_unified_diff(path)
    return paths


def combined_patch_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()
