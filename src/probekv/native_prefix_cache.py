from __future__ import annotations

from typing import Any, Dict, List, Mapping


SCHEMA_VERSION = 1
MIN_PREFIX_TOKENS = 192
MIN_CACHED_PREFIX_TOKENS = 128
MAX_LOGIT_RELATIVE_L2 = 1e-4
BLOCK_METADATA_EVIDENCE = "vllm_scheduler_computed_block_nums"


def evaluate_native_prefix_cache_audit(
    observed: Mapping[str, Any],
    *,
    expected_layers: int,
) -> Dict[str, Any]:
    """Validate a real native-prefix plus ProbeKV r=1 sentinel.

    The evaluator is intentionally independent from CUDA/vLLM so malformed or
    timing-only evidence can be rejected in no-GPU tests.  The GPU runner owns
    measurement; this function owns the frozen pass/fail semantics.
    """

    failures: List[str] = []
    if observed.get("paper_evidence") is not False:
        failures.append("native Prefix Cache qualification must remain non-paper")
    if observed.get("locked_test_accessed") is not False:
        failures.append("native Prefix Cache sentinel accessed a locked test")
    if observed.get("hit_evidence_source") != BLOCK_METADATA_EVIDENCE:
        failures.append("Prefix Cache hit must come from vLLM block metadata")
    if observed.get("timing_inference_used") is not False:
        failures.append("TTFT timing cannot be used as Prefix Cache hit evidence")

    block_size = int(observed.get("block_size", 0) or 0)
    cached_blocks = int(observed.get("cached_prefix_blocks", 0) or 0)
    cached_tokens = int(observed.get("cached_prefix_tokens", 0) or 0)
    requested_tokens = int(observed.get("requested_prefix_tokens", 0) or 0)
    if block_size <= 0:
        failures.append("vLLM block size is missing")
    if requested_tokens < MIN_PREFIX_TOKENS:
        failures.append("sentinel exact prefix is shorter than 192 tokens")
    if block_size and requested_tokens % block_size:
        failures.append("sentinel exact prefix is not block aligned")
    if observed.get("native_prefix_cache_hit") is not True or cached_blocks < 1:
        failures.append("native Prefix Cache exposed no cached block hit")
    if cached_tokens < MIN_CACHED_PREFIX_TOKENS:
        failures.append("native Prefix Cache exposed fewer than 128 cached tokens")
    if block_size and cached_tokens != cached_blocks * block_size:
        failures.append("cached token count does not match cached block metadata")
    if cached_tokens > requested_tokens:
        failures.append("cached prefix exceeds the exact prefix")

    if int(observed.get("prefix_shadow_layers", -1)) != int(expected_layers):
        failures.append("pre-RoPE prefix shadow does not cover every model layer")
    if int(observed.get("prefix_shadow_rows", -1)) != cached_tokens:
        failures.append("pre-RoPE prefix shadow row count differs from cached tokens")
    if observed.get("prefix_shadow_dtype") != "torch.bfloat16":
        failures.append("pre-RoPE prefix shadow dtype is not torch.bfloat16")
    if observed.get("prefix_shadow_device") != "cuda":
        failures.append("pre-RoPE prefix shadow is not resident on CUDA")
    if observed.get("prefix_shadow_geometry_valid") is not True:
        failures.append("pre-RoPE prefix shadow KV geometry is invalid")
    before = str(observed.get("prefix_shadow_digest_before", ""))
    after = str(observed.get("prefix_shadow_digest_after", ""))
    if not before or before != after:
        failures.append("pre-RoPE prefix shadow changed during use")

    if observed.get("active_positions_start_after_prefix") is not True:
        failures.append("ProbeKV active positions include cached prefix rows")
    if int(observed.get("prefix_rows_excluded_from_repair", -1)) != cached_tokens:
        failures.append("not every cached prefix row was excluded from repair")
    if int(observed.get("prefix_rows_in_repair_mask", -1)) != 0:
        failures.append("cached prefix rows entered the repair mask")
    if int(observed.get("prefix_rows_in_source_comparison", -1)) != 0:
        failures.append("cached prefix rows entered Source comparison")
    if observed.get("combined_prefix_r1_reuse_exercised") is not True:
        failures.append("native Prefix Cache and ProbeKV r=1 were not exercised together")
    if observed.get("dense_token_ids_equal") is not True:
        failures.append("Prefix Cache plus r=1 token IDs differ from dense")
    try:
        relative_l2 = float(observed.get("logit_relative_l2", float("inf")))
    except (TypeError, ValueError):
        relative_l2 = float("inf")
    if relative_l2 > MAX_LOGIT_RELATIVE_L2:
        failures.append("Prefix Cache plus r=1 logit relative-L2 exceeds 1e-4")
    if observed.get("cuda_event_timing") is not True:
        failures.append("native Prefix Cache sentinel lacks CUDA Event timing")

    result = dict(observed)
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "native_prefix_cache_sentinel",
            "paper_evidence": False,
            "locked_test_accessed": False,
            "expected_prefix_shadow_layers": int(expected_layers),
            "passed": not failures,
            "failures": failures,
        }
    )
    return result

