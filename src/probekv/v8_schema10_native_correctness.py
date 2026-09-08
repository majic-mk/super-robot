"""Real fixed-Source r=1 execution; never a production admission bypass.

The pool must already contain a full-prefill canonical Source.  Four native
executions are made: free dense/reuse and common-teacher dense/reuse.  Raw CPU
logits are persisted separately from the recomputable audit.
"""
from contextlib import ExitStack
from pathlib import Path
import time

from .v8_schema10_event_log import atomic_json
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import tensor_digest, file_digest
from .v8_schema10_native_validation import validate_correctness_observation
from .v8_schema6_hbm import HBMReservationKind


def execute_fixed_source_arm(backend, *, request, source_id=None, segment_id=None,
                             boundary=2, teacher_token_ids=None, warm_request=None,
                             diagnostic_completed_depth=0):
    """Actual native request action; the forced action is diagnostic only."""
    import torch
    adapter = backend.adapters["legacy_multicheckpoint"]
    if adapter.active or backend.pending or backend.hbm.active_reserved_bytes:
        raise RuntimeError("correctness arm requires a quiescent backend")
    adapter.reset()
    if warm_request is not None:
        with adapter.open_request({**warm_request, "capture_original_full_prefill": True},
                                  arrival_ns=time.perf_counter_ns()) as context:
            context.finish(lambda: None)
    q = {**request, "correctness_repair_ratio": 1.0}
    if teacher_token_ids is not None:
        q.update(capture_logits=True, teacher_token_ids=list(teacher_token_ids),
                 max_new_tokens=len(teacher_token_ids) + 1)
    first, ticket, layers, reservation = [], None, None, None
    before = destination = after = None
    started = time.perf_counter_ns()
    with ExitStack() as leases:
        with adapter.open_request(q, arrival_ns=started) as context:
            try:
                if diagnostic_completed_depth:
                    if source_id is not None:
                        raise ValueError("Prefix control must not also force a Source")
                    context.advance_to_depth(diagnostic_completed_depth)
                if source_id is not None:
                    descriptor = context.segments[segment_id]
                    if not 2 <= boundary <= adapter.spec.num_layers:
                        raise ValueError("r=1 boundary must follow a completed block")
                    eligible = {v.source_variant_id for v in backend._lookup(descriptor, q["request_epoch"])[1]}
                    if source_id not in eligible or not context.execution_inventory[segment_id].comparison_eligible:
                        raise ValueError("fixed Source is unavailable/future/prefix or wrong content")
                    if context.probe_fallback_reason:
                        raise RuntimeError("combined r1 cannot pass by missing-shadow dense fallback")
                    context.advance_to_depth(boundary - 1)
                    model, content = backend.provenance["model_signature"], descriptor["content_key"]
                    layers = leases.enter_context(backend.store.leased_winner(model, content, source_id,
                        expected_generation=backend.store.pool.content_generation(model, content)))
                    before = tensor_digest(t for pair in layers for t in pair)
                    row = backend.store.pool._get(model, content, source_id)
                    if before != row.canonical_source_state_digest:
                        raise RuntimeError("canonical Source creation digest differs")
                    size = sum(t.numel() * t.element_size() for pair in layers for t in pair)
                    reservation = backend.hbm.reserve_batch(owner_request_id=q["request_id"],
                        rows=((segment_id, size, HBMReservationKind.WINNER_PREFETCH),))[0]
                    ticket = context.prepare_winner(segment_id, source_id, layers, reservation)
                    context.finish_selection({segment_id: source_id}, {segment_id: ticket})
                    ready, _ = context.ready_for_final_commit({segment_id: ticket})
                    if ready != {segment_id: boundary}:
                        raise RuntimeError("fixed diagnostic boundary changed")
                    support = context.supports[segment_id][boundary]
                    if tuple(support) != tuple(descriptor["positions"]):
                        raise RuntimeError("r=1 did not retain every Segment row")
                    context.engine.commit_ready_segment(segment_id=segment_id, boundary=boundary,
                        segment_positions=descriptor["positions"], repair_positions=support,
                        scheduler_boundary=boundary)
                    context.committed[segment_id] = boundary
                    backend.hbm.promote(reservation.reservation_id,
                        expected=HBMReservationKind.WINNER_PREFETCH,
                        target=HBMReservationKind.COMMITTED_EXECUTION)
                output = context.finish(lambda: first.append(time.perf_counter_ns()))
                context.synchronize()
                if len(first) != 1:
                    raise RuntimeError("native action needs exactly one first-token event")
                if source_id is not None and output["whole_request_origin"] != "selective_reuse":
                    raise RuntimeError("r1 Source arm silently executed dense")
                if source_id is None:
                    expected = "native_prefix_dense_remaining" if context.cached_prefix_tokens else "exact_dense_full_prefill"
                    if output["whole_request_origin"] != expected:
                        raise RuntimeError("reference/control arm changed its prefill origin")
                if ticket is not None:
                    destination = tensor_digest(t for layer in sorted(ticket.layer_tensors)
                                                for t in ticket.layer_tensors[layer])
                    after = tensor_digest(t for pair in layers for t in pair)
                    if not before == destination == after:
                        raise RuntimeError("Source/destination/source digest mismatch")
                n, prefix = len(q["token_ids"]), context.cached_prefix_tokens
                layer_rows = [{"layer": a["layer"], "active_positions": list(a["active_after"]),
                               "expected_positions": list(range(prefix, n)),
                               "union_mask_digest": a["union_mask_digest"]}
                              for a in output["layer_audit"] if "active_after" in a]
                if source_id is not None:
                    validate_correctness_observation("absolute_mask", {"origin": "real_cuda_execution",
                        "fake_timing": False, "layer_rows": layer_rows, "cached_prefix_tokens": prefix})
                    if len(layer_rows) != adapter.spec.num_layers:
                        raise RuntimeError("mask audit omitted executed Transformer layers")
                logits = torch.cat(context.logit_trace, dim=0) if teacher_token_ids is not None else None
                result = {"token_ids": output["token_ids"], "whole_request_origin": output["whole_request_origin"],
                    "request_tokens_sha256": digest_json(q["token_ids"]),
                    "teacher_tokens_sha256": digest_json(teacher_token_ids) if teacher_token_ids is not None else None,
                    "source_id": source_id, "cached_prefix_tokens": prefix,
                    "cached_prefix_blocks": len(context.native.cached_block_ids),
                    "block_size": adapter.scheduler.block_manager.block_size,
                    "source_digest_before": before, "destination_digest": destination, "source_digest_after": after,
                    "layer_rows": layer_rows, "committed_segments": dict(context.committed),
                    "first_token_ns": first[0], "diagnostic_start_ns": started,
                    "first_token_host_ms": (first[0] - started) / 1e6,
                    "origin": "real_cuda_execution", "fake_timing": False,
                    "production_admission_applicable": False, "paper_evidence": False}
            finally:
                # A failed fence retains the reservation.  Never free live GPU
                # storage just because a diagnostic raised an exception.
                context.synchronize()
        if ticket is not None:
            ticket.layer_tensors.clear()
        if reservation is not None and not reservation.released:
            backend.hbm.release(reservation.reservation_id)
    return result, logits


def run_combined_native_r1(backend, *, request, warm_request, source_id, segment_id,
                           teacher_token_ids, output_dir, boundary=2):
    import torch
    if len(teacher_token_ids) < 31:
        raise ValueError("r1 requires at least 32 common-teacher logit positions")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, source, teacher, warm, depth in (
        ("dense_free", None, None, None, 0), ("reuse_free", source_id, None, warm_request, 0),
        ("dense_teacher", None, teacher_token_ids, None, 0),
        ("reuse_teacher", source_id, teacher_token_ids, warm_request, 0),
        ("native_prefix_teacher", None, teacher_token_ids, warm_request, 0),
        ("resumable_prefix_teacher", None, teacher_token_ids, warm_request, 1)):
        try:
            row, logits = execute_fixed_source_arm(backend, request=request, source_id=source,
                segment_id=segment_id, boundary=boundary, teacher_token_ids=teacher, warm_request=warm,
                diagnostic_completed_depth=depth)
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
