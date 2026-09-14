"""Fixed-mask, resident, boundary-to-first-token executor isolation.

Both arms execute the SAME common setup and dense blocks before timing. No
online selector, cost planner, lease acquisition or H2D is in the executor
interval. CacheBlend's native boundary repair ranking remains included;
ProbeKV receives the frozen mask. Setup is reported, not hidden from TTFT.
The CacheBlend arm uses the pinned decoder continuation, not stock forward.
"""
from contextlib import ExitStack
import time

from .cacheblend_continuation import DenseLoopContinuation
from .cacheblend_loop_diagnostic import install_loop_metadata, require_matched_boundary_patch
from .v8_schema6_hbm import HBMReservationKind


def execute_resident_boundary_arm(backend, *, request, plan, arm, teacher_token_ids=None):
    import torch
    from vllm.attention.backends.xformers import XFormersImpl
    if arm not in ("cacheblend", "probekv"):
        raise ValueError("unknown resident executor")
    require_matched_boundary_patch(XFormersImpl.forward)
    a = backend.adapters["legacy_multicheckpoint"]
    if a.active or backend.pending or backend.hbm.active_reserved_bytes != a.persistent_hot_hbm_bytes:
        raise RuntimeError("boundary executor requires quiescence")
    if plan.source_id not in a.hot_layer_cache:
        raise RuntimeError("resident Source missing")
    a.reset()
    setup_start = time.perf_counter_ns()
    q = {**request, "capture_original_full_prefill": False, "correctness_repair_ratio": plan.ratio,
         "use_gpu_hot_cache": True, "retain_gpu_hot_cache": False, "component_timing": False}
    if teacher_token_ids is not None:
        q.update(capture_logits=True, teacher_token_ids=list(teacher_token_ids),
                 max_new_tokens=len(teacher_token_ids) + 1)
    reservation = ticket = None
    first = []
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    try:
        with ExitStack() as leases:
            with a.open_request(q, arrival_ns=setup_start) as ctx:
                desc = ctx.segments["C"]
                plan.assert_binding(source_id=plan.source_id, token_ids=q["token_ids"],
                    positions=desc["positions"], boundary=plan.boundary, ratio=plan.ratio,
                    cached_prefix_tokens=ctx.cached_prefix_tokens)
                model, content = backend.provenance["model_signature"], desc["content_key"]
                layers = leases.enter_context(backend.store.leased_winner(model, content, plan.source_id,
                    expected_generation=backend.store.pool.content_generation(model, content)))
                if backend.store.pool._get(model, content, plan.source_id).canonical_source_state_digest != plan.source_digest:
                    raise RuntimeError("frozen mask Source digest mismatch")
                ctx.advance_to_depth(plan.boundary - 1)
                size = sum(t.numel() * t.element_size() for pair in layers for t in pair)
                reservation = backend.hbm.reserve_batch(owner_request_id=q["request_id"],
                    rows=(("C", size, HBMReservationKind.WINNER_PREFETCH),))[0]
                ticket = ctx.prepare_winner("C", plan.source_id, layers, reservation)
                ticket.wait_all(a.loader)
                ctx.engine.preinstall_resident_diagnostic("C", segment_count=len(ctx.segments))
                ctx.finish_selection({"C": plan.source_id}, {"C": ticket})
                ctx.supports["C"] = {l: plan.repair_positions for l in range(plan.boundary, a.spec.num_layers + 1)}
                # GPU-hot Sources are preinstalled once by the engine before
                # the timed interval. Both arms therefore receive identical
                # request-owned working KV without a redundant per-layer
                # scatter in the ProbeKV executor.
                saved = dict(a.inner.cache_fuse_metadata)
                try:
                    if arm == "cacheblend":
                        handoff = DenseLoopContinuation.detach(ctx.engine.session,
                            request_id=q["request_id"], generation=ctx.generation,
                            inner_model=a.inner, boundary=plan.boundary, positions=ctx._prepared_inputs[1])
                        install_loop_metadata(a.inner.cache_fuse_metadata, positions=desc["positions"],
                            prompt_tokens=len(q["token_ids"]), suffix_tokens=plan.suffix_tokens,
                            boundary=plan.boundary, ratio=plan.ratio, cached_prefix_tokens=0)
                        a.inner.cache_fuse_metadata["probekv_matched_boundary_source_kv"] = True
                    else:
                        ctx.engine.commit_ready_segment(segment_id="C", boundary=plan.boundary,
                            segment_positions=desc["positions"], repair_positions=plan.repair_positions,
                            scheduler_boundary=plan.boundary)
                    ctx.committed["C"] = plan.boundary
                    backend.hbm.promote(reservation.reservation_id,
                        expected=HBMReservationKind.WINNER_PREFETCH,
                        target=HBMReservationKind.COMMITTED_EXECUTION)
                    ctx.synchronize()
                    execution_start = time.perf_counter_ns()
                    events[0].record()
                    def mark_first():
                        events[1].record()
                        first.append(time.perf_counter_ns())
                    if arm == "cacheblend":
                        hidden = handoff.run(request_id=q["request_id"], generation=ctx.generation,
                            positions=ctx._prepared_inputs[1], attention_metadata=ctx.attention, working_kv=a.kv)
                        a.inner.cache_fuse_metadata["check"] = False
                        result = ctx.finish_from_prefill_hidden(hidden, mark_first)
                        chosen = a.inner.cache_fuse_metadata["selected_segment_indices"].detach().cpu().tolist()
                        active = a.inner.cache_fuse_metadata["imp_indices"].detach().cpu().tolist()
                        plan.assert_execution(chosen, active)
                        executed = list(range(1, plan.boundary)) + handoff.executed_layers
                    else:
                        result = ctx.finish(mark_first)
                        executed = [r["layer"] for r in result["layer_audit"] if "active_after" in r]
                        for row in result["layer_audit"]:
                            if row.get("layer", 0) >= plan.boundary and "active_after" in row:
                                plan.assert_execution(plan.repair_positions, row["active_after"])
                    ctx.synchronize()
                    if executed != list(range(1, a.spec.num_layers + 1)) or len(first) != 1:
                        raise RuntimeError("boundary comparison skipped/repeated layers or first token")
                    row = dict(token_ids=result["token_ids"], source_id=plan.source_id,
                        request_tokens_sha256=plan.request_tokens_sha256, cached_prefix_tokens=0,
                        boundary=plan.boundary, diagnostic_repair_ratio=plan.ratio,
                        external_repair_mask_sha256=plan.mask_digest,
                        sampling_signature=dict(ctx.sampling_signature),
                        executor_start_ns=execution_start, first_token_ns=first[0], setup_start_ns=setup_start,
                        setup_excluded_ms=(execution_start-setup_start)/1e6,
                        executor_host_ms=(first[0]-execution_start)/1e6,
                        executor_cuda_ms=float(events[0].elapsed_time(events[1])),
                        setup_plus_executor_ms=(first[0]-setup_start)/1e6,
                        timing_scope="common_dense_boundary_to_first_token_not_request_ttft",
                        execution_layer_order=executed, selection_cost_included=False,
                        planner_executed=False, lease_acquisition_in_executor_interval=False,
                        full_kv_h2d_in_executor_interval=False, origin="real_cuda_execution",
                        fake_timing=False, paper_evidence=False,
                        cacheblend_boundary_repair_ranking_included=(arm == "cacheblend"))
                    logits = torch.cat(ctx.logit_trace) if teacher_token_ids is not None else None
                finally:
                    ctx.synchronize()
                    a.inner.cache_fuse_metadata.clear()
                    a.inner.cache_fuse_metadata.update(saved)
        return row, logits
    finally:
        a.torch.cuda.synchronize()  # failed fence must NOT release live buffers
        if ticket is not None:
            ticket.layer_tensors.clear()
        if reservation is not None and not reservation.released:
            backend.hbm.release(reservation.reservation_id)
