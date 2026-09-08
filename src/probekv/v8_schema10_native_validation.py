"""Validate raw native sentinel observations, not a supplied `passed` flag."""
import math


def validate_correctness_observation(category, row):
    if row.get("origin") != "real_cuda_execution" or row.get("fake_timing") is not False:
        raise ValueError("native prerequisite is not real CUDA evidence")
    if category == "native_prefix":
        if (row.get("cached_prefix_blocks", 0) < 1 or row.get("cached_prefix_tokens", 0) < 128
                or row.get("cached_prefix_tokens") != row.get("cached_prefix_blocks") * row.get("block_size", 0)
                or row.get("prefix_rows_in_repair") != [] or row.get("prefix_rows_in_comparison") != []
                or not isinstance(row.get("model_layers"), int) or row["model_layers"] < 1
                or row.get("shadow_layers") != row.get("model_layers")
                or not row.get("prefix_shadow_digest_before")
                or row.get("prefix_shadow_digest_before") != row.get("prefix_shadow_digest_after")):
            raise ValueError("native Prefix block/shadow/row evidence failed")
    elif category == "k_hook":
        if (not row.get("observations") or any(r["k_observation_layer_1based"] != r["completed_depth"] + 1
                or r["completed_depth"] < 1 or r["shape"] != r["expected_shape"]
                or r["dtype"] != "bfloat16" for r in row["observations"])):
            raise ValueError("native K hook depth/geometry failed")
    elif category == "r1":
        l2 = row.get("logit_relative_l2")
        if (not row.get("dense_token_ids") or row.get("dense_token_ids") != row.get("reuse_token_ids")
                or not isinstance(l2, (int, float)) or not math.isfinite(l2) or not 0 <= l2 <= 1e-4
                or row.get("logit_token_count", 0) < 32):
            raise ValueError("native r=1 equivalence failed")
    elif category == "cfo":
        errors = row.get("eager_layer_errors")
        if (row.get("eager_reference") is not True
                or type(row.get("expected_layers")) is not int or row["expected_layers"] < 1
                or row.get("captured_layers") != row["expected_layers"]
                or not isinstance(errors, list) or len(errors) != row["expected_layers"]
                or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 2e-5 for x in errors)
                or row.get("eager_tolerance") != 2e-5
                or any(row.get(k) is not True for k in
                       ("post_rope_qk", "causal_mask", "gqa_mapping", "fp32_accumulation", "streaming_logsumexp"))
                or not row.get("metadata_digest")):
            raise ValueError("native CFO eager/streaming layer evidence failed")
    elif category == "source_digest":
        if (not row.get("source_digest_before") or not row["source_digest_before"] ==
                row.get("source_digest_after") == row.get("destination_digest")):
            raise ValueError("native canonical/destination digest failed")
    elif category == "absolute_mask":
        if not row.get("layer_rows"):
            raise ValueError("absolute mask evidence missing")
        previous = None
        for layer in row["layer_rows"]:
            active = layer["active_positions"]
            if (active != sorted(set(active)) or active != layer["expected_positions"]
                    or any(p < row["cached_prefix_tokens"] for p in active)
                    or previous is not None and not set(active) <= previous):
                raise ValueError("native active mask is not causal/monotone")
            previous = set(active)
    else:
        raise ValueError("unknown native correctness category")
    return True
