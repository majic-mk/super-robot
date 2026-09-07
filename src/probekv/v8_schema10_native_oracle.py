"""Real fixed-Source QA oracle, explicitly separate from online admission.

This diagnostic does not run the selector or fabricate a passing FinalCommit.
It measures one dense/Source action at a time from the same retained pool and
deterministically rebuilt native Prefix state. It never materializes a Source.
"""
from contextlib import ExitStack
from dataclasses import asdict
import time

from .v8_schema10_execution import digest_json
from .v8_schema10_experiments import SourceOutcome, measured_quality_cost_oracle
from .v8_schema10_storage import tensor_digest
from .v8_schema6_hbm import HBMReservationKind


def run_native_source_oracle(backend, *, request, dispatch, segment_id, source_ids,
                             first_reuse_layer, repair_ratio, chosen_source_id,
                             max_answer_f1_drop):
    if (not source_ids or len(source_ids) > 16 or len(source_ids) != len(set(source_ids))
            or repair_ratio not in {.15, 1.0} or not request.get("answers")
            or request.get("capture_original_full_prefill", False)
            or "teacher_token_ids" in request or request.get("capture_logits")):
        raise ValueError("native oracle requires fixed15/r1, references and a unique fixed Source set")
    adapter = backend.adapters[dispatch["selection_path"]]
    if not 2 <= first_reuse_layer <= adapter.spec.num_layers:
        raise ValueError("oracle boundary must follow at least one completed block")
    if backend.pending or adapter.active:
        raise RuntimeError("oracle requires a quiescent backend")
    snapshot = backend.snapshot()
    records, outcomes = [], []
    provenance = digest_json({"request": request, "source_ids": source_ids, "snapshot": snapshot,
        "first_reuse_layer": first_reuse_layer, "repair_ratio": repair_ratio,
        "backend": backend.provenance, "dispatch": dispatch,
        "timing_scope": "executor_prefill_start_not_request_arrival"})
    try:
        for source_id in [None, *source_ids]:
            adapter.check_deadline()
            backend.restore(snapshot)
            q = {**request, "correctness_repair_ratio": repair_ratio}
            descriptor = next(s for s in q["segments"] if s["segment_id"] == segment_id)
            eligible = {s.source_variant_id for s in backend._lookup(descriptor, request["request_epoch"])[1]}
            if not set(source_ids) <= eligible:
                raise ValueError("oracle fixed set contains unavailable, future, or incorrect-content Sources")
            before = after = destination = None
            integrity_host_ms = 0.0
            first = []
            layers = ticket = None
            # Restoration and source creation-time validation are outside the
            # named source-only interval, but remain in the raw cost audit.
            if source_id is not None:
                with backend.store.leased_winner(backend.provenance["model_signature"],
                        descriptor["content_key"], source_id,
                        expected_generation=backend.store.pool.content_generation(
                            backend.provenance["model_signature"], descriptor["content_key"])) as check_layers:
                    begin = time.perf_counter_ns()
                    before = tensor_digest(t for pair in check_layers for t in pair)
                    expected = backend.store.pool._get(backend.provenance["model_signature"],
                        descriptor["content_key"], source_id).canonical_source_state_digest
                    if before != expected:
                        raise RuntimeError("oracle source differs from its creation digest")
                    integrity_host_ms += (time.perf_counter_ns() - begin) / 1e6
            started = time.perf_counter_ns()
            with ExitStack() as leases, adapter.open_request(q, arrival_ns=started) as context:
                if source_id is not None:
                    if not context.execution_inventory[segment_id].comparison_eligible or context.probe_fallback_reason:
                        raise RuntimeError("fixed oracle Source is not a legal non-prefix candidate")
                    context.advance_to_depth(first_reuse_layer - 1)
                    layers = leases.enter_context(backend.store.leased_winner(
                        backend.provenance["model_signature"], descriptor["content_key"], source_id,
                        expected_generation=backend.store.pool.content_generation(
                            backend.provenance["model_signature"], descriptor["content_key"])))
                    size = getattr(layers, "full_kv_bytes", None)
                    if size is None:
                        size = sum(t.numel() * t.element_size() for pair in layers for t in pair)
                    reservation = backend.hbm.reserve_batch(owner_request_id=q["request_id"],
                        rows=((segment_id, size, HBMReservationKind.WINNER_PREFETCH),))[0]
                    def release():
                        context.synchronize()
                        backend.hbm.release(reservation.reservation_id)
                    leases.callback(release)
                    ticket = context.prepare_winner(segment_id, source_id, layers, reservation)
                    context.finish_selection({segment_id: source_id}, {segment_id: ticket})
                    ready, _ = context.ready_for_final_commit({segment_id: ticket})
                    if ready[segment_id] != first_reuse_layer:
                        raise RuntimeError("oracle changed the fixed boundary")
                    # Diagnostic forced action, not an economic PASS. Its event
                    # kind cannot enter the production online evidence gate.
                    context.engine.commit_ready_segment(segment_id=segment_id, boundary=first_reuse_layer,
                        segment_positions=descriptor["positions"],
                        repair_positions=context.supports[segment_id][first_reuse_layer],
                        scheduler_boundary=first_reuse_layer)
                    context.committed[segment_id] = first_reuse_layer
                    backend.hbm.promote(reservation.reservation_id, expected=HBMReservationKind.WINNER_PREFETCH,
                                        target=HBMReservationKind.COMMITTED_EXECUTION)
                output = context.finish(lambda: first.append(time.perf_counter_ns()))
                answer_f1 = output.get("qa_evidence", {}).get("answer_f1")
                if len(first) != 1 or answer_f1 is None:
                    raise RuntimeError("oracle lacks real first-token/QA observations")
                if ticket is not None:
                    begin = time.perf_counter_ns()
                    context.synchronize()
                    destination = tensor_digest(t for l in sorted(ticket.layer_tensors) for t in ticket.layer_tensors[l])
                    after = tensor_digest(t for pair in layers for t in pair)
                    integrity_host_ms += (time.perf_counter_ns() - begin) / 1e6
                    if not before == destination == after:
                        raise RuntimeError("oracle Source/destination digest changed")
                row = {**output, "execution_kind": "source_oracle_diagnostic", "forced_source": source_id,
                    "final_commit_applicable": False, "source_id": source_id,
                    "first_reuse_layer": first_reuse_layer, "repair_ratio": repair_ratio if source_id else 1.,
                    "first_token_ns": first[0], "measurement_start_ns": started,
                    "ttft_ms": (first[0] - started) / 1e6, "timing_scope": "executor_prefill_start_not_request_arrival",
                    "integrity_host_ms_outside_ttft": integrity_host_ms,
                    "source_digest_before": before, "destination_digest": destination, "source_digest_after": after,
                    "provenance_sha256": provenance, "evidence_origin": "real_cuda_execution",
                    "ssd_page_cache_controlled": False, "paper_evidence": False}
                row["raw_observation_sha256"] = digest_json(row)
                records.append(row)
                backend._emit("source_oracle_observation", request["request_id"], row)
                outcomes.append(SourceOutcome(request["request_id"], source_id, first_reuse_layer,
                    row["repair_ratio"], row["ttft_ms"], answer_f1,
                    source_id is None or before == destination == after, row["timing_scope"], provenance))
                if ticket is not None:
                    ticket.layer_tensors.clear()  # no unreserved retained GPU ticket after close
            ticket = layers = None
    finally:
        # A failed CUDA fence leaves quarantined reservations. Do not restore
        # over those pointers and never hide the original failure as success.
        if not backend.hbm.active_reserved_bytes:
            backend.restore(snapshot)
            backend.release_snapshot(snapshot)
    return {**measured_quality_cost_oracle(outcomes, chosen_source_id=chosen_source_id,
                max_answer_f1_drop=max_answer_f1_drop),
            "outcomes": [asdict(o) for o in outcomes], "raw_observations": records,
            "production_admission_applicable": False, "diagnostic_actions_not_online_evidence": True}
