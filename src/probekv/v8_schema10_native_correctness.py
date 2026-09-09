"""Real fixed-Source r=1 execution; never a production admission bypass.

The pool must already contain a full-prefill canonical Source.  Free-generation,
common-teacher and Prefix-matched controls are executed. Raw CPU logits are
persisted separately from the recomputable audit.
"""
from contextlib import ExitStack
from pathlib import Path
import math
import time

from .v8_schema10_event_log import atomic_json
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import tensor_digest, file_digest
from .v8_schema10_native_validation import validate_correctness_observation
from .v8_schema6_hbm import HBMReservationKind


def execute_fixed_source_arm(backend, *, request, source_id=None, segment_id=None,
                             boundary=2, teacher_token_ids=None, warm_request=None,
                             diagnostic_completed_depth=0, repair_ratio=1.0,
                             verify_full_digests=True, wait_all_source_layers=False,
                             commit_source=True, use_gpu_hot_cache=False,
                             retain_gpu_hot_cache=False, resident_repair_plan=None):
    """Actual native request action; the forced action is diagnostic only."""
    import torch
    if resident_repair_plan is not None:
        from .repair_backend_contract import ResidentRepairPlan
        if not isinstance(resident_repair_plan, ResidentRepairPlan) or source_id is None or not commit_source:
            raise ValueError("resident repair plan requires a typed fixed-Source diagnostic")
    if isinstance(repair_ratio, bool) or not isinstance(repair_ratio, (int, float)) or not 0 < repair_ratio <= 1:
        raise ValueError("diagnostic repair ratio must be in (0, 1]")
    if source_id is None and repair_ratio != 1.0:
        raise ValueError("repair ratio is meaningful only for a fixed Source arm")
    if source_id is None and (wait_all_source_layers or not commit_source):
        raise ValueError("preparation controls require a fixed Source")
    adapter = backend.adapters["legacy_multicheckpoint"]
    non_hot_reserved = backend.hbm.active_reserved_bytes - adapter.persistent_hot_hbm_bytes
    if adapter.active or backend.pending or non_hot_reserved:
        raise RuntimeError("correctness arm requires a quiescent backend")
    adapter.reset()
    if warm_request is not None:
        with adapter.open_request({**warm_request, "capture_original_full_prefill": True},
                                  arrival_ns=time.perf_counter_ns()) as context:
            context.finish(lambda: None)
    q = {**request, "correctness_repair_ratio": float(repair_ratio),
         "use_gpu_hot_cache": bool(use_gpu_hot_cache),
         "retain_gpu_hot_cache": bool(retain_gpu_hot_cache)}
    if teacher_token_ids is not None:
        q.update(capture_logits=True, teacher_token_ids=list(teacher_token_ids),
                 max_new_tokens=len(teacher_token_ids) + 1)
    first, ticket, layers, reservation = [], None, None, None
    boundary_ready_ns = source_ready_ns = None
    winner_ready_layers = []
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_boundary = torch.cuda.Event(enable_timing=True)
    cuda_source_ready = torch.cuda.Event(enable_timing=True)
    cuda_first_token = torch.cuda.Event(enable_timing=True)
    before = destination = after = None
    started = time.perf_counter_ns()
    cuda_start.record()
    with ExitStack() as leases:
        with adapter.open_request(q, arrival_ns=started) as context:
            try:
                if diagnostic_completed_depth:
                    if source_id is not None:
                        raise ValueError("Prefix control must not also force a Source")
                    context.advance_to_depth(diagnostic_completed_depth)
                    context.synchronize()
                    boundary_ready_ns = time.perf_counter_ns()
                    cuda_boundary.record()
                if source_id is not None:
                    descriptor = context.segments[segment_id]
                    if resident_repair_plan is not None:
                        resident_repair_plan.assert_binding(source_id=source_id, token_ids=q["token_ids"],
                            positions=descriptor["positions"], boundary=boundary, ratio=repair_ratio,
                            cached_prefix_tokens=context.cached_prefix_tokens)
                    if not 2 <= boundary <= adapter.spec.num_layers:
                        raise ValueError("r=1 boundary must follow a completed block")
                    eligible = {v.source_variant_id for v in backend._lookup(descriptor, q["request_epoch"])[1]}
                    if source_id not in eligible or not context.execution_inventory[segment_id].comparison_eligible:
                        raise ValueError("fixed Source is unavailable/future/prefix or wrong content")
                    if context.probe_fallback_reason:
                        raise RuntimeError("combined r1 cannot pass by missing-shadow dense fallback")
                    context.advance_to_depth(boundary - 1)
                    context.synchronize()
                    boundary_ready_ns = time.perf_counter_ns()
                    cuda_boundary.record()
                    model, content = backend.provenance["model_signature"], descriptor["content_key"]
                    layers = leases.enter_context(backend.store.leased_winner(model, content, source_id,
                        expected_generation=backend.store.pool.content_generation(model, content)))
                    row = backend.store.pool._get(model, content, source_id)
                    if resident_repair_plan is not None and row.canonical_source_state_digest != resident_repair_plan.source_digest:
                        raise ValueError("resident repair plan Source digest differs from canonical Artifact")
                    if verify_full_digests:
                        before = tensor_digest(t for pair in layers for t in pair)
                        if before != row.canonical_source_state_digest:
                            raise RuntimeError("canonical Source creation digest differs")
                    size = sum(t.numel() * t.element_size() for pair in layers for t in pair)
                    reservation = backend.hbm.reserve_batch(owner_request_id=q["request_id"],
                        rows=((segment_id, size, HBMReservationKind.WINNER_PREFETCH),))[0]
                    ticket = context.prepare_winner(segment_id, source_id, layers, reservation)
                    context.finish_selection({segment_id: source_id}, {segment_id: ticket})
                    ready, _ = context.ready_for_final_commit({segment_id: ticket})
                    if wait_all_source_layers:
                        # A separate measured cell, not a relabelled streaming
                        # sample. The wait is part of preparation/sunk TTFT.
                        ticket.wait_all(adapter.loader)
                        context.register_ready_hot_replicas()
                    source_ready_ns = time.perf_counter_ns()
                    cuda_source_ready.record()
                    winner_ready_layers = sorted(layer for layer, event in ticket.layer_events.items()
                                                 if event.query())
                    if ready != {segment_id: boundary}:
                        raise RuntimeError("fixed diagnostic boundary changed")
                    support = context.supports[segment_id][boundary]
                    if resident_repair_plan is not None:
                        # Diagnostic backend isolation: retain the ordinary repair-check
                        # computation in timing, but execute exactly the external mask.
                        # Never used by production Source selection/admission.
                        support = resident_repair_plan.repair_positions
                        context.supports[segment_id] = {l: support for l in
                            range(boundary, adapter.spec.num_layers + 1)}
                    expected_count = min(len(descriptor["positions"]),
                                         math.ceil(len(descriptor["positions"]) * repair_ratio))
                    if len(support) != expected_count or not set(support) <= set(descriptor["positions"]):
                        raise RuntimeError("fixed diagnostic repair support has wrong rows")
                    if repair_ratio == 1.0 and tuple(support) != tuple(descriptor["positions"]):
                        raise RuntimeError("r=1 did not retain every Segment row")
                    if commit_source:
                        context.engine.commit_ready_segment(segment_id=segment_id, boundary=boundary,
                            segment_positions=descriptor["positions"], repair_positions=support,
                            scheduler_boundary=boundary)
                        context.committed[segment_id] = boundary
                        backend.hbm.promote(reservation.reservation_id,
                            expected=HBMReservationKind.WINNER_PREFETCH,
                            target=HBMReservationKind.COMMITTED_EXECUTION)
                def mark_first_token():
                    cuda_first_token.record()
                    first.append(time.perf_counter_ns())
                output = context.finish(mark_first_token)
                context.synchronize()
                if len(first) != 1:
                    raise RuntimeError("native action needs exactly one first-token event")
                if source_id is not None and commit_source and output["whole_request_origin"] != "selective_reuse":
                    raise RuntimeError("r1 Source arm silently executed dense")
                if source_id is None or not commit_source:
                    expected = "native_prefix_dense_remaining" if context.cached_prefix_tokens else "exact_dense_full_prefill"
                    if output["whole_request_origin"] != expected:
                        raise RuntimeError("reference/control arm changed its prefill origin")
                if ticket is not None and verify_full_digests:
                    if not ticket.fully_ready():
                        raise RuntimeError("integrity evidence requires the complete Source, not submitted layers only")
                    ticket.finalize_integrity(layers, lambda pairs: tensor_digest(t for pair in pairs for t in pair))
                    destination = tensor_digest(t for layer in sorted(ticket.layer_tensors)
                                                for t in ticket.layer_tensors[layer])
                    after = tensor_digest(t for pair in layers for t in pair)
                    if not before == destination == after:
                        raise RuntimeError("Source/destination/source digest mismatch")
                n, prefix = len(q["token_ids"]), context.cached_prefix_tokens
                layer_rows = []
                for audit in output["layer_audit"]:
                    if "active_after" not in audit:
                        continue
                    layer = audit["layer"]
                    expected = set(range(prefix, n))
                    for sid, commit_boundary in context.committed.items():
                        if layer >= commit_boundary:
                            expected.difference_update(context.segments[sid]["positions"])
                            expected.update(context.supports[sid][layer])
                    layer_rows.append({"layer": layer, "active_positions": list(audit["active_after"]),
                                       "expected_positions": sorted(expected),
                                       "gpu_ms": audit.get("gpu_ms"),
                                       "union_mask_digest": audit["union_mask_digest"]})
                if source_id is not None:
                    validate_correctness_observation("absolute_mask", {"origin": "real_cuda_execution",
                        "fake_timing": False, "layer_rows": layer_rows, "cached_prefix_tokens": prefix})
                    if len(layer_rows) != adapter.spec.num_layers:
                        raise RuntimeError("mask audit omitted executed Transformer layers")
                if resident_repair_plan is not None:
                    for audit in layer_rows:
                        if audit["layer"] >= boundary:
                            resident_repair_plan.assert_execution(support, audit["active_positions"])
                logits = torch.cat(context.logit_trace, dim=0) if teacher_token_ids is not None else None
                result = {"token_ids": output["token_ids"], "whole_request_origin": output["whole_request_origin"],
                    "request_tokens_sha256": digest_json(q["token_ids"]),
                    "teacher_tokens_sha256": digest_json(teacher_token_ids) if teacher_token_ids is not None else None,
                    "source_id": source_id, "cached_prefix_tokens": prefix,
                    "external_repair_mask_sha256": resident_repair_plan.mask_digest if resident_repair_plan else None,
                    "selection_cost_included": False,
                    "cached_prefix_blocks": len(context.native.cached_block_ids),
                    "block_size": adapter.scheduler.block_manager.block_size,
                    "source_digest_before": before, "destination_digest": destination, "source_digest_after": after,
                    "layer_rows": layer_rows, "committed_segments": dict(context.committed),
                    "overlap_trace": list(output.get("overlap_trace", ())),
                    "resumable_engine_used": context.engine is not None,
                    "kv_layout_mode": q.get("kv_layout_mode", "legacy"),
                    "instrumented_timing_not_performance_evidence": bool(q.get("component_timing", False)),
                    "component_observations": (
                        context.engine.component_observations() if context.engine else []),
                    "setup_component_observations": context.setup_observations(),
                    "first_token_ns": first[0], "diagnostic_start_ns": started,
                    "first_token_host_ms": (first[0] - started) / 1e6,
                    "first_token_cuda_ms": float(cuda_start.elapsed_time(cuda_first_token)),
                    "prompt_token_count": len(q["token_ids"]),
                    "prefix_cache_mode": context.prefix_cache_mode,
                    "sampling_signature": dict(context.sampling_signature),
                    "diagnostic_completed_depth": diagnostic_completed_depth or (boundary - 1 if source_id else 0),
                    "diagnostic_repair_ratio": float(repair_ratio) if source_id is not None else None,
                    "executed_prefetch_window": int(q.get("prefetch_window", 0)),
                    "defer_layer_timing": bool(q.get("defer_layer_timing", False)),
                    "expected_source_layers": ticket.expected_layer_count if ticket else None,
                    "source_fully_ready_at_finish": ticket.fully_ready() if ticket else None,
                    "loader_full_digest_verified": ticket.per_request_full_digest_verified if ticket else None,
                    "integrity_verification_mode": (
                        "qualification_full" if verify_full_digests else "online_immutable"
                    ),
                    "selection_boundary_ready_ns": boundary_ready_ns,
                    "winner_source_ready_ns": source_ready_ns,
                    "diagnostic_wait_all_source_layers": wait_all_source_layers,
                    "diagnostic_commit_source": commit_source,
                    "winner_ready_layers": winner_ready_layers,
                    "winner_copy_in_flight_at_commit_check": bool(
                        ticket is not None and len(winner_ready_layers) < ticket.expected_layer_count
                    ),
                    "boundary_to_first_token_ms": (
                        (first[0] - boundary_ready_ns) / 1e6
                        if boundary_ready_ns is not None else None
                    ),
                    "boundary_to_first_token_cuda_ms": (
                        float(cuda_boundary.elapsed_time(cuda_first_token))
                        if boundary_ready_ns is not None else None
                    ),
                    "winner_preparation_ms": (
                        (source_ready_ns - boundary_ready_ns) / 1e6
                        if source_ready_ns is not None and boundary_ready_ns is not None else None
                    ),
                    "winner_preparation_cuda_ms": (
                        float(cuda_boundary.elapsed_time(cuda_source_ready))
                        if source_ready_ns is not None and boundary_ready_ns is not None else None
                    ),
                    "repair_check_ms": (
                        context.actual_repair_check_sunk_ms if source_id is not None else 0.0
                    ),
                    "ready_to_first_token_ms": (
                        (first[0] - source_ready_ns) / 1e6
                        if source_ready_ns is not None else None
                    ),
                    "ready_to_first_token_cuda_ms": (
                        float(cuda_source_ready.elapsed_time(cuda_first_token))
                        if source_ready_ns is not None else None
                    ),
                    "origin": "real_cuda_execution", "fake_timing": False,
                    "production_admission_applicable": False, "paper_evidence": False}
            finally:
                # A failed fence retains the reservation.  Never free live GPU
                # storage just because a diagnostic raised an exception.
                context.synchronize()
        if ticket is not None:
            ticket.layer_tensors.clear()
        if (reservation is not None and not reservation.released
                and retain_gpu_hot_cache and source_id is not None
                and source_id in adapter.hot_layer_cache):
            adapter.hot_reservations[source_id] = reservation
            reservation = None
        if reservation is not None and not reservation.released:
            backend.hbm.release(reservation.reservation_id)
    return result, logits


def run_combined_native_r1(backend, *, request, warm_request, source_id, segment_id,
                           teacher_token_ids, output_dir, boundary=2,
                           use_gpu_hot_cache=False):
    import torch
    if len(teacher_token_ids) < 31:
        raise ValueError("r1 requires at least 32 common-teacher logit positions")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, source, teacher, warm, depth in (
        ("dense_free", None, None, None, 0), ("reuse_free", source_id, None, warm_request, 0),
        ("dense_prefix_free", None, None, warm_request, boundary - 1),
        ("dense_teacher", None, teacher_token_ids, None, 0),
        ("reuse_teacher", source_id, teacher_token_ids, warm_request, 0),
        ("native_prefix_teacher", None, teacher_token_ids, warm_request, 0),
        ("resumable_prefix_teacher", None, teacher_token_ids, warm_request, 1)):
        try:
            row, logits = execute_fixed_source_arm(backend, request=request, source_id=source,
                segment_id=segment_id, boundary=boundary, teacher_token_ids=teacher, warm_request=warm,
                diagnostic_completed_depth=depth,
                use_gpu_hot_cache=bool(use_gpu_hot_cache and source is not None))
            if logits is not None:
                path = root / (name + ".pt")
                torch.save(logits.detach().cpu(), path)
                row.update(logits_path=path.name, logits_sha256=file_digest(path), logits_shape=list(logits.shape))
            row["raw_observation_sha256"] = digest_json(row)
            atomic_json(root / (name + ".json"), row)
            records[name] = row
        except Exception as error:
            atomic_json(root / (name + "-failed.json"), {"error_type": type(error).__name__,
                "error": str(error), "paper_evidence": False})
            raise
    for name in ("reuse_free", "reuse_teacher"):
        if records[name]["cached_prefix_tokens"] < 128 or not records[name]["committed_segments"]:
            raise RuntimeError("combined r1 lacks real Prefix hit or actual reuse commit")
    left = torch.load(root / "dense_teacher.pt", map_location="cpu", weights_only=True)
    right = torch.load(root / "reuse_teacher.pt", map_location="cpu", weights_only=True)
    if left.shape != right.shape or left.ndim != 2 or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("r1 raw logits differ in geometry or contain nonfinite values")
    l2 = float((left - right).norm() / left.norm().clamp_min(1e-12))
    # Independent controls locate error introduced by native Prefix alone,
    # resumable Prefix, or Source commit. They never replace the original
    # no-cache dense reference or relax the qualification threshold.
    controls = {}
    for name in ("native_prefix_teacher", "resumable_prefix_teacher"):
        tensor = torch.load(root / (name + ".pt"), map_location="cpu", weights_only=True)
        if tensor.shape != left.shape or not torch.isfinite(tensor).all():
            raise RuntimeError("Prefix control has invalid logit geometry/values")
        controls[name] = {"vs_dense_relative_l2": float((left - tensor).norm() / left.norm().clamp_min(1e-12)),
            "vs_reuse_relative_l2": float((right - tensor).norm() / right.norm().clamp_min(1e-12)),
            "vs_dense_per_position_relative_l2": ((left - tensor).norm(dim=1) / left.norm(dim=1).clamp_min(1e-12)).tolist()}
    row = {"category": "r1", "origin": "real_cuda_execution", "fake_timing": False,
        "dense_token_ids": records["dense_free"]["token_ids"],
        "reuse_token_ids": records["reuse_free"]["token_ids"],
        "logit_relative_l2": l2, "logit_token_count": left.shape[0],
        "prefix_controls": controls,
        "arm_digests": {name: r["raw_observation_sha256"] for name, r in records.items()},
        "production_admission_applicable": False, "paper_evidence": False}
    atomic_json(root / "comparison.json", row)
    validate_correctness_observation("r1", row)
    row["raw_observation_sha256"] = digest_json(row)
    atomic_json(root / "r1.json", row)
    return row
