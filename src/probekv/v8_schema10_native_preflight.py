"""Concrete pre-cost native Prefix/K-hook observations.

These functions require the real factory-created CUDA adapter. They neither
invent a cost table nor execute an online admission/selection policy. No code
in this module loads a model at import time.
"""
from contextlib import contextmanager
import math
import time

from .v8_schema10_execution import digest_json
from .v8_schema10_storage import tensor_digest
from .v8_schema10_native_validation import validate_correctness_observation


@contextmanager
def isolated_native_preflight(adapter):
    if adapter.active is not None or adapter.hbm.active_reserved_bytes:
        raise RuntimeError("native preflight requires quiescent owned resources")
    if not adapter.torch.cuda.is_available():
        raise RuntimeError("native preflight requires actual CUDA")
    snapshot = adapter.snapshot(retain=True)
    try:
        adapter.reset()
        yield
    finally:
        # Failed fences deliberately retain quarantined reservations. Never
        # reset or advertise a usable allocator over those allocations.
        if not adapter.hbm.active_reserved_bytes and adapter.active is None:
            adapter.restore(snapshot)
            adapter.release_snapshot(snapshot)


def _finish_observation(context):
    first = []
    output = context.finish(lambda: first.append(time.perf_counter_ns()))
    if len(first) != 1:
        raise RuntimeError("native preflight lacks a unique first-token event")
    return {"token_ids": output["token_ids"], "first_token_ns": first[0],
            "whole_request_origin": output["whole_request_origin"]}


def _seal(category, row, adapter, request, started):
    row = {**row, "category": category, "origin": "real_cuda_execution", "fake_timing": False,
        "request_sha256": digest_json(request), "source_provenance": adapter.provenance,
        "diagnostic_total_host_ms": (time.perf_counter_ns() - started) / 1e6,
        "production_admission_applicable": False, "paper_evidence": False}
    validate_correctness_observation(category, row)
    row["raw_observation_sha256"] = digest_json(row)
    return row


def run_native_k_hook_sentinel(adapter, *, request, segment_id, completed_depths):
    depths = tuple(completed_depths)
    if (depths != adapter.depths or not depths or min(depths) < 1
            or max(depths) >= adapter.spec.num_layers):
        raise ValueError("K-hook sentinel must cover the actual dispatch checkpoints")
    if request.get("capture_logits") or "teacher_token_ids" in request:
        raise ValueError("K-hook preflight is not a teacher-forced QA task")
    started = time.perf_counter_ns()
    with isolated_native_preflight(adapter):
        # Independent monolithic full-prefill has original tokens and no native
        # cache writes. Compare with actual resumable current K at depth d.
        reference = adapter.build_exact_dense_source(request, segment_id)
        observations = []
        with adapter.open_request(request, arrival_ns=time.perf_counter_ns()) as context:
            if context.cached_prefix_tokens or not context.execution_inventory[segment_id].comparison_eligible:
                raise RuntimeError("K-hook reference requires unmodified no-prefix execution")
            attn = adapter.inner.layers[0].self_attn
            expected = [len(context.segments[segment_id]["positions"]), attn.num_kv_heads, attn.head_dim]
            for depth in depths:
                context.advance_to_depth(depth)
                key = context.observe_current_k(segment_id, depth)
                ref = reference["selection_states"][depth]
                value = key.detach().float().cpu()
                relative_l2 = float((value - ref.float()).norm() / ref.float().norm().clamp_min(1e-12))
                if (list(key.shape) != expected or str(key.dtype) != "torch.bfloat16"
                        or not math.isfinite(relative_l2) or relative_l2 > 1e-4):
                    raise RuntimeError("native K-hook differs from monolithic pre-RoPE reference")
                event = context.engine.session.layer_audit[-1]
                if event.get("event") != "selection_k_observation" or event.get("completed_depth") != depth:
                    raise RuntimeError("K-hook observation is not backed by the current session event")
                observations.append({"completed_depth": depth,
                    "k_observation_layer_1based": event["k_observation_layer_1based"],
                    "shape": list(key.shape), "expected_shape": expected, "dtype": "bfloat16",
                    "monolithic_relative_l2": relative_l2,
                    "current_k_digest": tensor_digest((key,)), "reference_k_digest": tensor_digest((ref,))})
                del key, value, ref
            output = _finish_observation(context)
        return _seal("k_hook", {"observations": observations, "execution": output,
            "reference_capture_audit": reference["capture_audit"]}, adapter, request, started)


def run_native_prefix_sentinel(adapter, *, warm_request, request):
    if any(q.get("capture_logits") or "teacher_token_ids" in q for q in (warm_request, request)):
        raise ValueError("Prefix sentinel cannot silently change generation semantics")
    if warm_request["token_ids"] == request["token_ids"]:
        raise ValueError("native Prefix sentinel requires different suffix content")
    common = 0
    for left, right in zip(warm_request["token_ids"], request["token_ids"]):
        if left != right:
            break
        common += 1
    block = adapter.scheduler.block_manager.block_size
    if common < 192 or block < 1:
        raise ValueError("preregistered exact Prefix must cover at least 192 tokens")
    started = time.perf_counter_ns()
    with isolated_native_preflight(adapter):
        # This diagnostic explicitly pays the dense capture/CFO cost. It is not
        # a production warm-up whose overhead can be omitted from an E2E claim.
        warm = {**warm_request, "capture_original_full_prefill": True}
        with adapter.open_request(warm, arrival_ns=time.perf_counter_ns()) as context:
            if context.cached_prefix_tokens:
                raise RuntimeError("Prefix warm-up unexpectedly reused cached tokens")
            warm_output = _finish_observation(context)
        with adapter.open_request(request, arrival_ns=time.perf_counter_ns()) as context:
            n = context.cached_prefix_tokens
            shadow = context.native.prefix_shadow
            if (n < 128 or n > common or context.probe_fallback_reason
                    or shadow is None or len(shadow) != adapter.spec.num_layers):
                raise RuntimeError("real block hit and complete pre-RoPE shadow required")
            attn = adapter.inner.layers[0].self_attn
            expected = (n, attn.num_kv_heads, attn.head_dim)
            if any(tuple(t.shape) != expected or str(t.dtype) != "torch.bfloat16"
                   for pair in shadow for t in pair):
                raise RuntimeError("native Prefix shadow geometry/dtype differs")
            before = tensor_digest(t for pair in shadow for t in pair)
            context.advance_to_depth(1)
            compared = []
            for sid, owner in context.execution_inventory.items():
                if owner.comparison_eligible:
                    key = context.observe_current_k(sid, 1)
                    compared.extend(context.segments[sid]["positions"])
                    del key
            active = list(context.engine.session.active_positions)
            if active != list(range(n, len(request["token_ids"]))):
                raise RuntimeError("native Prefix rows leaked into the active query set")
            output = _finish_observation(context)
            after = tensor_digest(t for pair in shadow for t in pair)
            row = {"cached_prefix_blocks": len(context.native.cached_block_ids), "cached_prefix_tokens": n,
                "block_size": block, "model_layers": adapter.spec.num_layers, "shadow_layers": len(shadow),
                "prefix_shadow_digest_before": before, "prefix_shadow_digest_after": after,
                "prefix_rows_in_comparison": sorted(p for p in compared if p < n),
                "prefix_rows_in_repair": sorted({p for supports in context.supports.values()
                                                for positions in supports.values() for p in positions if p < n}),
                "active_positions_after_prefix": active, "warm_execution": warm_output,
                "execution": output, "combined_r1_sentinel_passed": False}
        return _seal("native_prefix", row, adapter, request, started)


def run_r1_equivalence_sentinel(*, request, dense_executor, reuse_executor,
                                expected_logit_tokens=32, max_relative_l2=1e-4):
    """Compare two real executor observations at the r=1 endpoint.

    The executors are deliberately injected by the CUDA runner: this keeps the
    preflight module independent of a particular vLLM scheduler while making
    it impossible to pass by supplying only a token-id equality flag.
    """
    if not callable(dense_executor) or not callable(reuse_executor):
        raise TypeError("r=1 sentinel requires two real executor callbacks")
    if type(expected_logit_tokens) is not int or expected_logit_tokens < 32:
        raise ValueError("r=1 logit coverage must contain at least 32 tokens")
    started = time.perf_counter_ns()
    dense = dense_executor(request)
    reuse = reuse_executor(request)
    for name, row in (("dense", dense), ("reuse", reuse)):
        if (not isinstance(row, dict) or row.get("origin") != "real_cuda_execution"
                or row.get("fake_timing") is not False):
            raise ValueError("r=1 executor did not return real CUDA evidence: " + name)
    dense_ids, reuse_ids = dense.get("token_ids"), reuse.get("token_ids")
    if not isinstance(dense_ids, list) or dense_ids != reuse_ids:
        raise ValueError("r=1 generated token IDs differ")
    dense_logits, reuse_logits = dense.get("logits"), reuse.get("logits")
    if dense_logits is None or reuse_logits is None or len(dense_logits) < expected_logit_tokens:
        raise ValueError("r=1 executor lacks the preregistered teacher-forced logit trace")
    if len(dense_logits) != len(reuse_logits):
        raise ValueError("r=1 logit traces have different lengths")
    # The CUDA runner serializes finite CPU float rows after synchronization.
    import torch
    left, right = torch.as_tensor(dense_logits, dtype=torch.float32), torch.as_tensor(reuse_logits, dtype=torch.float32)
    if left.shape != right.shape or not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError("r=1 logit traces are not finite and shape-compatible")
    denom = left.norm().clamp_min(1e-12)
    relative = float((left - right).norm() / denom)
    if not math.isfinite(relative) or relative > max_relative_l2:
        raise ValueError("r=1 logit relative-L2 exceeds the fixed sentinel bound")
    row = {"dense_token_ids": dense_ids, "reuse_token_ids": reuse_ids,
        "logit_relative_l2": relative, "logit_token_count": len(dense_logits),
        "dense_first_token_ns": dense.get("first_token_ns"), "reuse_first_token_ns": reuse.get("first_token_ns"),
        "dense_observation_sha256": digest_json(dense), "reuse_observation_sha256": digest_json(reuse),
        "origin": "real_cuda_execution", "fake_timing": False,
        "diagnostic_total_host_ms": (time.perf_counter_ns() - started) / 1e6,
        "production_admission_applicable": False, "paper_evidence": False}
    validate_correctness_observation("r1", row)
    row["raw_observation_sha256"] = digest_json(row)
    return row
