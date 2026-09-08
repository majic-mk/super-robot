"""Exact-input capture without Prefix hits, decoding, or paged-cache writes."""
from dataclasses import asdict
import time

from .v8_cfo import CanonicalChunkOccurrence, CFOFullPrefillCollector
from .v8_schema10_execution import digest_json
from .v8_schema6_hbm import HBMReservationKind


def request_occurrences(request):
    """Partition every row once, preserving canonical Segment boundaries."""
    tokens, cursor, counts, occurrences, targets = request["token_ids"], 0, {}, [], {}
    def append(content, start, end):
        ordinal = counts.get(content, 0)
        counts[content] = ordinal + 1
        row = CanonicalChunkOccurrence(content, ordinal, start, end - start)
        occurrences.append(row)
        return row
    for s in sorted(request["segments"], key=lambda s: s["positions"][0]):
        positions = tuple(s["positions"])
        start, end = positions[0], positions[-1] + 1
        if positions != tuple(range(start, end)) or start < cursor or list(tokens[start:end]) != list(s["token_ids"]):
            raise ValueError("canonical occurrence input/positions mismatch")
        if cursor < start:
            append(digest_json(["dense-context", tokens[cursor:start]]), cursor, start)
        targets[s["segment_id"]] = append(s["content_key"], start, end)
        cursor = end
    if cursor < len(tokens):
        append(digest_json(["dense-context", tokens[cursor:]]), cursor, len(tokens))
    ids = [row.match_id for row in occurrences for _ in range(row.token_count)]
    if len(ids) != len(tokens):
        raise ValueError("CFO occurrence inventory incomplete")
    return occurrences, targets, ids


def export_original_full_prefill(adapter, request, collector):
    """Export a capture already produced by this exact, no-prefix request.

    This does not execute another model forward. Caller must reject any
    selective/prefix-derived origin BEFORE calling this function.
    """
    import torch
    started = time.perf_counter_ns()
    occurrences, targets, _ = request_occurrences(request)
    n = len(request["token_ids"])
    attn = adapter.inner.layers[0].self_attn
    whole = []
    for block in adapter.inner.layers:
        key, value = block.self_attn.hack_kv
        pair = tuple(t.reshape(n, attn.num_kv_heads, attn.head_dim) for t in (key, value))
        if any(t.dtype != torch.bfloat16 for t in pair):
            raise ValueError("original prefill capture must be BF16")
        whole.append(pair)
    shadow = adapter.shadows.publish(request["token_ids"], whole, origin="exact_dense_full_prefill")
    result = {}
    for sid, target in targets.items():
        begin, end = target.provenance_position, target.provenance_position + target.token_count
        layers = tuple(tuple(t[begin:end].detach().to(device="cpu", copy=True).contiguous() for t in pair) for pair in whole)
        cfo, audit = collector.finalize(prefix_occurrences=tuple(o for o in occurrences if o.provenance_position < begin),
                                       target_occurrence=target)
        result[sid] = {"layers": layers, "selection_states": {d: layers[d][0].clone() for d in adapter.spec.checkpoints},
            "source_metadata": {"token_ids": list(request["token_ids"][begin:end]),
                "tokenizer_hash": adapter.provenance["tokenizer_hash"],
                "runtime_compatibility": adapter.provenance["runtime_compatibility"], "cfo": asdict(cfo)},
            "capture_audit": {"origin": "exact_dense_full_prefill", "capture_reused_from_current_request": True,
                "extra_full_prefill_count": 0, "cached_prefix_tokens": 0, "cfo": audit,
                "shadow_published": shadow, "original_tokens_sha256": digest_json(request["token_ids"])}}
    elapsed = (time.perf_counter_ns() - started) / 1e6
    # One shared capture interval, not a per-Segment additive TTFT component.
    for row in result.values():
        row["capture_audit"]["shared_capture_export_host_ms"] = elapsed
    return result


def capture_exact_dense_source(adapter, request, segment_id, *, eager_reference=False):
    import torch
    from vllm.sequence import SequenceData, SequenceGroupMetadata
    from vllm import SamplingParams
    token_ids = tuple(request["token_ids"])
    if type(eager_reference) is not bool or eager_reference and len(token_ids) > 512:
        raise ValueError("eager CFO reference is a bounded <=512-token diagnostic")
    occurrences, targets, occurrence_ids = request_occurrences(request)
    target = targets[segment_id]
    collector = CFOFullPrefillCollector(token_occurrence_ids=occurrence_ids,
        expected_layers=adapter.spec.num_layers, eager_reference=eager_reference)
    caches = [None] * adapter.spec.num_layers
    group = SequenceGroupMetadata(request_id=request["request_id"] + ":canonical-capture", is_prompt=True,
        seq_data={0: SequenceData(list(token_ids))}, sampling_params=SamplingParams(temperature=0, max_tokens=1),
        block_tables=None, computed_block_nums=[], token_chunk_size=len(token_ids))
    n = len(token_ids)
    attn = adapter.inner.layers[0].self_attn
    size = n * adapter.spec.num_layers * attn.num_kv_heads * attn.head_dim * 4
    reservation = adapter.hbm.reserve_batch(owner_request_id=request["request_id"],
        rows=(("canonical_capture", size, HBMReservationKind.COMMITTED_EXECUTION),))[0]
    metadata = adapter.inner.cache_fuse_metadata
    original = dict(metadata)
    started = time.perf_counter_ns()
    shadows, key, value = [], None, None
    try:
        with torch.inference_mode():
            metadata.update({"check": False, "probekv_resumable": False, "exact_prefix_tokens": 0,
                             "collect": True, "probekv_cfo_collector": collector})
            inputs, positions, attention = adapter.prepare(group, caches=caches)[:3]
            if tuple(inputs.cpu().tolist()) != token_ids or tuple(positions.cpu().tolist()) != tuple(range(n)):
                raise RuntimeError("capture changed original tokens or absolute positions")
            if attention.slot_mapping.numel() != n or not bool((attention.slot_mapping == -1).all().item()):
                raise RuntimeError("capture must not target native paged-cache slots")
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            # No sample/decode call, no native allocator mutation, no nonce.
            adapter.outer(input_ids=inputs, positions=positions, kv_caches=caches, attn_metadata=attention)
            end.record()
            end.synchronize()
            layers, shadows = [], []
            for block in adapter.inner.layers:
                key, value = block.self_attn.hack_kv
                key = key.reshape(n, attn.num_kv_heads, attn.head_dim)
                value = value.reshape(n, attn.num_kv_heads, attn.head_dim)
                if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
                    raise RuntimeError("capture is not canonical BF16")
                begin, stop = target.provenance_position, target.provenance_position + target.token_count
                layers.append((key[begin:stop].detach().cpu().clone(), value[begin:stop].detach().cpu().clone()))
                shadows.append((key.detach(), value.detach()))
            cfo, cfo_audit = collector.finalize(prefix_occurrences=tuple(o for o in occurrences
                if o.provenance_position < target.provenance_position), target_occurrence=target)
            # Bounded shadow admission may decline without invalidating Source capture.
            shadow_published = adapter.shadows.publish(token_ids, shadows, origin="exact_dense_full_prefill")
            states = {d: layers[d][0].clone() for d in adapter.spec.checkpoints}
            return {"layers": tuple(layers), "selection_states": states,
                "source_metadata": {"token_ids": list(token_ids[target.provenance_position:target.provenance_position + target.token_count]),
                    "tokenizer_hash": adapter.provenance["tokenizer_hash"],
                    "runtime_compatibility": adapter.provenance["runtime_compatibility"], "cfo": asdict(cfo)},
                "capture_audit": {"origin": "exact_dense_full_prefill", "original_tokens_sha256": digest_json(token_ids),
                    "cached_prefix_tokens": 0, "paged_cache_writes": False, "decode_executed": False,
                    "capture_gpu_ms": start.elapsed_time(end), "cfo": cfo_audit,
                    "shadow_published": shadow_published, "capture_host_ms": (time.perf_counter_ns() - started) / 1e6}}
    finally:
        torch.cuda.synchronize()
        for block in adapter.inner.layers:
            block.self_attn.hack_kv = []
        shadows.clear()
        key = value = None
        metadata.clear()
        metadata.update(original)
        adapter.hbm.release(reservation.reservation_id)
