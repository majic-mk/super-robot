"""Pinned CacheBlend native loop, NOT an unmodified-upstream claim.

Explicit adapters: exact Segment repair eligibility / mandatory dense rows,
absolute causal mask, ceil rounding, flat-head RoPE ABI and common numerical
policy. Calls the pinned inner model's normal forward, never resumable hooks.
Original loop cannot compose native Prefix with arbitrary sparse queries;
only the matched zero-Prefix stratum is supported here.
"""
from contextlib import ExitStack, nullcontext
from pathlib import Path
import math
import time

from .cacheblend_v6_online_engine import allocate_working_composite, source_row_index
from .v8_schema6_hbm import HBMReservationKind
from .v8_schema10_execution import digest_json
from .v8_schema10_event_log import atomic_json
from .v8_schema10_storage import tensor_digest, file_digest


def loop_metadata(*, positions, prompt_tokens, suffix_tokens, boundary, ratio,
                  cached_prefix_tokens):
    if cached_prefix_tokens:
        raise ValueError("CacheBlend native loop does not support matched native Prefix sparse queries")
    positions = tuple(positions)
    if (not positions or positions != tuple(range(positions[0], positions[-1] + 1))
            or positions[0] < 0 or positions[-1] >= prompt_tokens - suffix_tokens
            or not 0 < suffix_tokens < prompt_tokens or boundary < 2 or ratio not in (.15, 1.0)):
        raise ValueError("invalid CacheBlend loop diagnostic shape")
    return dict(check=True, collect=False, probekv_cfo_collector=None,
        probekv_resumable=False, exact_prefix_tokens=0, reuse_active=False,
        probekv_matched_boundary_source_kv=False,
        check_layers=[boundary - 1], recomp_ratio=ratio, repair_rounding_policy="ceil",
        prefix_len=0, suffix_len=suffix_tokens, segment_start=positions[0],
        segment_len=len(positions), repair_regions=[dict(segment_id="C",
            start=positions[0], length=len(positions), recomp_ratio=ratio)],
        segment_repair_audit=None, imp_indices=None, attn_bias=None)


def install_loop_metadata(target, **kwargs):
    values = loop_metadata(**kwargs)
    # The shared patched layer uses dict.get(local, absolute). A present None
    # does NOT fall back: residual[None] would insert a dimension. The original
    # forward owns absolute imp_indices and must not inherit resumable locals.
    target.pop("local_imp_indices", None)
    target.update(values)


def execute_cacheblend_loop_arm(backend, *, request, source_id, boundary=2,
                               ratio=.15, teacher_token_ids=None):
    import torch
    a = backend.adapters["legacy_multicheckpoint"]
    if a.active or backend.pending or backend.hbm.active_reserved_bytes != a.persistent_hot_hbm_bytes:
        raise RuntimeError("CacheBlend control requires quiescence")
    if source_id not in a.hot_layer_cache:
        raise RuntimeError("CacheBlend control requires an already resident canonical Source")
    a.reset()  # no warm request: Prefix condition is explicitly zero for BOTH arms
    q = {**request, "capture_original_full_prefill": False}
    if teacher_token_ids is not None:
        q.update(capture_logits=True, teacher_token_ids=list(teacher_token_ids),
                 max_new_tokens=len(teacher_token_ids) + 1)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    first, reservation = [], None
    started = time.perf_counter_ns()
    events[0].record()
    with ExitStack() as leases:
        with a.open_request(q, arrival_ns=started) as ctx:
            saved = dict(a.inner.cache_fuse_metadata)
            try:
                if ctx.cached_prefix_tokens or ctx.engine is not None or boundary > a.spec.num_layers:
                    raise RuntimeError("native-loop control has wrong Prefix/depth/engine")
                segment = ctx.segments["C"]
                eligible = {v.source_variant_id for v in backend._lookup(segment, q["request_epoch"])[1]}
                if source_id not in eligible:
                    raise ValueError("CacheBlend control Source is not eligible")
                model, content = backend.provenance["model_signature"], segment["content_key"]
                leases.enter_context(backend.store.leased_winner(model, content, source_id,
                    expected_generation=backend.store.pool.content_generation(model, content)))
                hot = a.hot_layer_cache[source_id]
                pairs = [hot[layer] for layer in sorted(hot)]
                n = len(q["token_ids"])
                size = n * sum(t[0].numel() * t.element_size() for t in pairs[0]) * a.spec.num_layers
                reservation = backend.hbm.reserve_batch(owner_request_id=q["request_id"],
                    rows=(("native_loop_working_kv", size, HBMReservationKind.COMMITTED_EXECUTION),))[0]
                old = allocate_working_composite(torch, pairs, n, a.runner.device, packed=True)
                index = source_row_index(segment["positions"], contiguous_copy=True)
                for out, src in zip(old, pairs):
                    out[0][index], out[1][index] = src
                a.inner.old_kvs = old
                suffix = n - segment["positions"][-1] - 1
                install_loop_metadata(a.inner.cache_fuse_metadata, positions=segment["positions"],
                    prompt_tokens=n, suffix_tokens=suffix, boundary=boundary, ratio=ratio,
                    cached_prefix_tokens=ctx.cached_prefix_tokens)
                if q.get("matched_boundary_source_kv", False):
                    import inspect
                    from vllm.attention.backends.xformers import XFormersImpl
                    if "probekv_matched_boundary_source_kv" not in inspect.getsource(XFormersImpl.forward):
                        raise RuntimeError("matched backend needs independently audited 0016 boundary patch")
                    a.inner.cache_fuse_metadata["probekv_matched_boundary_source_kv"] = True
                ids, pos = ctx._prepared_inputs[:2]
                # This is the existing pinned CacheBlend forward loop. No
                # Source observation projection or resumable session is used.
                with (torch.profiler.record_function("cacheblend.native_prefill")
                      if q.get("component_timing") else nullcontext()):
                    hidden = a.outer(input_ids=ids, positions=pos, kv_caches=a.kv,
                                     attn_metadata=ctx.attention)
                a.inner.cache_fuse_metadata["check"] = False
                ctx.committed["C"] = boundary
                def mark_first():
                    events[1].record()
                    first.append(time.perf_counter_ns())
                result = ctx.finish_from_prefill_hidden(hidden, mark_first)
                ctx.synchronize()
                active = a.inner.cache_fuse_metadata["imp_indices"].detach().cpu().tolist()
                chosen = a.inner.cache_fuse_metadata["selected_segment_indices"].detach().cpu().tolist()
                expected_count = math.ceil(ratio * len(segment["positions"]))
                expected_active = sorted((set(range(n)) - set(segment["positions"])) | set(chosen))
                if (len(first) != 1 or len(chosen) != expected_count
                        or not set(chosen) <= set(segment["positions"]) or active != expected_active):
                    raise RuntimeError("CacheBlend control has invalid repair/dense ownership")
                row = dict(token_ids=result["token_ids"], source_id=source_id,
                    request_tokens_sha256=digest_json(q["token_ids"]), cached_prefix_tokens=0,
                    sampling_signature=dict(ctx.sampling_signature), boundary=boundary,
                    diagnostic_repair_ratio=ratio, selected_segment_positions=chosen,
                    active_positions=active, first_token_host_ms=(first[0]-started)/1e6,
                    first_token_cuda_ms=float(events[0].elapsed_time(events[1])),
                    control="cacheblend_pinned_segment_adapter", unmodified_upstream=False,
                    resumable_engine_used=False, native_loop_executed=True,
                    setup_included=True, selection_cost_included=False,
                    boundary_kv_policy="source_mixed" if q.get("matched_boundary_source_kv") else "current_dense",
                    setup_component_observations=ctx.setup_observations(),
                    instrumented_timing_not_performance_evidence=bool(q.get("component_timing")),
                    origin="real_cuda_execution", fake_timing=False, paper_evidence=False)
                logits = torch.cat(ctx.logit_trace) if teacher_token_ids is not None else None
            finally:
                ctx.synchronize()
                a.inner.cache_fuse_metadata.clear()
                a.inner.cache_fuse_metadata.update(saved)
                if reservation is not None:
                    backend.hbm.release(reservation.reservation_id)
    return row, logits


def run_cacheblend_loop_comparison(backend, *, request, source_id, teacher_token_ids,
                                  output_dir, boundary=2, repeats=3, matched_mask=False):
    import torch
    from .v8_schema10_native_correctness import execute_fixed_source_arm
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    a = backend.adapters["legacy_multicheckpoint"]
    hot = a.hot_layer_cache[source_id]
    before = tensor_digest(t for l in sorted(hot) for t in hot[l])
    plans = {}
    def arm(name, *, ratio=.15, teacher=None, instrumented=False):
        q = {**request, "component_timing": instrumented,
             "matched_boundary_source_kv": matched_mask}
        if name == "cacheblend_loop":
            row, logits = execute_cacheblend_loop_arm(backend, request=q, source_id=source_id,
                boundary=boundary, ratio=ratio, teacher_token_ids=teacher)
            if ratio in plans:
                plans[ratio].assert_execution(row["selected_segment_positions"], row["active_positions"])
                row["external_repair_mask_sha256"] = plans[ratio].mask_digest
            return row, logits
        kwargs = {} if name == "dense" else dict(source_id=source_id, segment_id="C",
            boundary=boundary, repair_ratio=ratio, use_gpu_hot_cache=True, wait_all_source_layers=True)
        return execute_fixed_source_arm(backend, request=q, teacher_token_ids=teacher,
            verify_full_digests=False,
            resident_repair_plan=plans.get(ratio) if name == "probekv" else None, **kwargs)
    def save(name, row, logits=None):
        if logits is not None:
            p = root / (name + ".pt")
            torch.save(logits, p)
            row.update(logits_path=p.name, logits_sha256=file_digest(p))
        row["raw_observation_sha256"] = digest_json(row)
        atomic_json(root / (name + ".json"), row)
    if matched_mask:
        from dataclasses import asdict
        from .repair_backend_contract import ResidentRepairPlan
        # One untimed setup run supplies CacheBlend's mask. Each measured CB run
        # must reproduce it exactly; ProbeKV still pays its normal repair-check
        # cost before adopting this diagnostic mask. No live selector claim.
        bootstrap, _ = arm("cacheblend_loop")
        save("mask-bootstrap", bootstrap)
        descriptor = next(s for s in request["segments"] if s["segment_id"] == "C")
        positions = tuple(descriptor["positions"])
        for ratio in (.15, 1.0):
            plans[ratio] = ResidentRepairPlan(source_id, before, digest_json(request["token_ids"]),
                boundary, ratio, positions,
                tuple(bootstrap["selected_segment_positions"]) if ratio == .15 else positions,
                len(request["token_ids"]), len(request["token_ids"]) - positions[-1] - 1)
        atomic_json(root / "repair-plans.json", {str(r): asdict(p) for r, p in plans.items()})
    records, tensors = {}, {}
    for name in ("dense", "cacheblend_loop", "probekv"):
        for teacher in (None, teacher_token_ids):
            key = name + ("_free" if teacher is None else "_teacher")
            row, logits = arm(name, ratio=1.0, teacher=teacher)
            save(key, row, logits)
            records[key], tensors[key] = row, logits
    checks = {}
    reference = tensors["dense_teacher"]
    for name in ("cacheblend_loop", "probekv"):
        observed = tensors[name + "_teacher"]
        if observed.shape != reference.shape or reference.shape[0] < 32 or not torch.isfinite(observed).all():
            raise RuntimeError("native loop control has invalid teacher logits")
        l2 = float((observed-reference).norm()/reference.norm().clamp_min(1e-12))
        equal = records[name+"_free"]["token_ids"] == records["dense_free"]["token_ids"]
        checks[name] = dict(token_ids_equal=equal, logit_relative_l2=l2, passed=equal and l2 <= 1e-4)
    atomic_json(root / "r1-comparison.json", checks)
    if not all(c["passed"] for c in checks.values()):
        raise RuntimeError("native CacheBlend loop r1 failed; no timing comparison permitted")
    if matched_mask:
        fixed = {}
        for name in ("cacheblend_loop", "probekv"):
            for teacher in (None, teacher_token_ids):
                key = name + ("_free" if teacher is None else "_teacher")
                row, logits = arm(name, teacher=teacher)
                save("fixed15-" + key, row, logits)
                fixed[key] = (row, logits)
        ref = fixed["cacheblend_loop_teacher"][1].float()
        obs = fixed["probekv_teacher"][1].float()
        if obs.shape != ref.shape or ref.shape[0] < 32 or not torch.isfinite(obs).all() or not torch.isfinite(ref).all():
            raise RuntimeError("matched fixed15 has invalid logits")
        l2 = float((obs-ref).norm() / ref.norm().clamp_min(1e-12))
        equal = fixed["cacheblend_loop_free"][0]["token_ids"] == fixed["probekv_free"][0]["token_ids"]
        match = dict(token_ids_equal=equal, logit_relative_l2=l2, passed=equal and l2 <= 1e-4,
            mask_sha256=plans[.15].mask_digest, comparison="backend_equivalence_not_dense_quality",
            selection_cost_included=False, native_prefix_supported=False, paper_evidence=False)
        atomic_json(root / "fixed15-equivalence.json", match)
        if not match["passed"]:
            raise RuntimeError("matched fixed15 backend equivalence failed; no timing comparison permitted")
    for i in range(2 + repeats):
        order = ["dense", "cacheblend_loop", "probekv"]
        if i % 2:
            order.reverse()
        for name in order:
            row, _ = arm(name)
            row.update(repeat=i, warmup=i < 2, arm_order=order, arm=name)
            save(f"{i:02d}-{name}", row)
    from .prefill_phase_diagnostic import summarize_prefill_phase
    for name in ("cacheblend_loop", "probekv"):
        a.loader.capture_hardware_trace = True
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profiler:
                row, _ = arm(name, instrumented=True)
            p = root / ("trace-" + name + ".json")
            profiler.export_chrome_trace(str(p))
            import json
            summary = summarize_prefill_phase(json.loads(p.read_text()), name)
            summary.update(trace_sha256=file_digest(p),
                           instrumented_timing_not_performance_evidence=True)
            atomic_json(root / ("profile-" + name + ".json"), summary)
            save("instrumented-" + name, row)
        finally:
            a.loader.capture_hardware_trace = False
    after = tensor_digest(t for l in sorted(hot) for t in hot[l])
    atomic_json(root / "source-integrity.json", dict(before=before, after=after,
        unchanged=before==after, hashing_outside_timing=True))
    if before != after:
        raise RuntimeError("CacheBlend control mutated the canonical GPU Source")
