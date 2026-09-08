"""Single-active-request orchestration over live native runtime contexts.

No RuntimeFixture or historical current-state tensor is accepted here. Runtime
adapters must allocate through the engine's Prefix-aware allocator and expose a
resumable request context. Unimplemented dispatch adapters fail before loading
KV. A CPU test adapter is explicitly non-GPU evidence, never qualification.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
import math
from threading import RLock
import time

from .v8_cfo import CanonicalChunkOccurrence, compute_cachecraft_cfo
from .v8_schema10_source_metadata import read_cfo_metadata
from .v8_contracts import CandidateCounts, ResidualCandidate
from .v8_schema6_hbm import HBMReservationKind
from .v8_schema7_planner import FinalCommitPlanner
from .v8_schema10_contracts import DenseKVProvenance, VariantMaterializationStateV10
from .v8_schema10_cost_provider import UnsupportedTimelineCost
from .v8_schema10_inventory import native_segment_inventory
from .v8_schema10_execution import ProductionSelectionSession, SelectionCostLedger, SelectionCostPolicy, digest_json
from .v8_schema10_materialization import VariantMaterializationControllerV10, VariantMaterializationRequestV10


class Schema10OnlineExperimentBackend:
    capabilities = {"max_integrated_concurrency": 1, "quiescent_snapshot_restore": True}

    def __init__(self, *, store_factory, adapters, selector_factory, cost_provider,
                 hbm_manager, provenance, event_log=None):
        self.store_factory, self.adapters = store_factory, dict(adapters)
        self.selector_factory, self.costs, self.hbm = selector_factory, cost_provider, hbm_manager
        self.provenance, self.event_log = dict(provenance), event_log
        self.lock = RLock()
        self.store = None
        self.pending = None
        self.completed = {}
        self.finalized_at_ns = {}
        self.epoch = -1
        self.generation = 0
        self.poisoned_error = None

    def _emit(self, kind, request_id, payload):
        if self.event_log is not None:
            self.event_log.append(kind, request_id, payload)

    def reset(self, *, capacity, global_byte_budget):
        with self.lock:
            if self.pending:
                raise RuntimeError("finalize the current request before reset")
            if self.hbm.active_reserved_bytes:
                raise RuntimeError("cannot reset live or quarantined GPU reservations")
            if not 1 <= capacity <= 16 or global_byte_budget <= 0:
                raise ValueError("invalid capacity replay budget")
            for adapter in self.adapters.values():
                adapter.reset()
            if self.store is not None:
                self.store.close()
            self.store = self.store_factory(capacity, global_byte_budget)
            if self.store.global_byte_budget != global_byte_budget:
                raise ValueError("all K runs must retain the exact same global byte budget")
            self.completed, self.finalized_at_ns, self.epoch = {}, {}, -1
            self.poisoned_error = None
            self.generation += 1

    def snapshot(self, *, retain_backing=True):
        with self.lock:
            if self.pending:
                self.finalize_request(*self.pending[:2])
            if self.hbm.active_reserved_bytes:
                raise RuntimeError("cannot snapshot active preparation/execution reservations")
            if self.store is None:
                raise RuntimeError("backend requires reset")
            hbm = deepcopy({k: v for k, v in self.hbm.__dict__.items() if k != "reservations"})
            hbm["reservations"] = {k: asdict(v) for k, v in self.hbm.reservations.items()}
            return {"pool": self.store.snapshot() if retain_backing else self.store.snapshot_descriptor(),
                    "runtime": {k: v.snapshot(retain=retain_backing) if v.capabilities.get("snapshot_accepts_retention") else v.snapshot()
                                for k, v in self.adapters.items()},
                    "hbm": hbm, "completed": deepcopy(self.completed), "finalized_at_ns": dict(self.finalized_at_ns),
                    "request_epoch": self.epoch, "generation": self.generation,
                    "provenance": self.provenance, "ssd_page_cache_controlled": False}

    def release_snapshot(self, snapshot):
        self.store.release_snapshot(snapshot["pool"])
        for name, adapter in self.adapters.items():
            if adapter.capabilities.get("snapshot_accepts_retention"):
                adapter.release_snapshot(snapshot["runtime"][name])

    def snapshot_descriptor(self):
        return self.snapshot(retain_backing=False)

    def restore(self, snapshot):
        with self.lock:
            if self.hbm.active_reserved_bytes:
                raise RuntimeError("cannot restore over in-flight GPU work")
            if snapshot["provenance"] != self.provenance:
                raise ValueError("snapshot code/model provenance mismatch")
            self.pending = None
            self.store.restore(snapshot["pool"])
            for name, adapter in self.adapters.items():
                adapter.restore(snapshot["runtime"][name])
            from .v8_schema6_hbm import HBMReservation
            hbm = deepcopy(snapshot["hbm"])
            hbm["reservations"] = {k: HBMReservation(**{**v, "kind": HBMReservationKind(v["kind"])})
                                   for k, v in hbm["reservations"].items()}
            self.hbm.__dict__.update(hbm)
            self.completed = deepcopy(snapshot["completed"])
            self.finalized_at_ns = dict(snapshot["finalized_at_ns"])
            self.epoch, self.generation = snapshot["request_epoch"], snapshot["generation"]

    def _lookup(self, segment, request_epoch):
        model = self.provenance["model_signature"]
        rows = self.store.pool.variants_for_content(model, segment["content_key"])
        visible = tuple(v for v in rows if self.store.objects[v.source_variant_id].creation_epoch < request_epoch)
        eligible = []
        for row in visible:
            obj = self.store.objects[row.source_variant_id]
            metadata = obj.metadata
            # Exact token check is mandatory even after hash bucket lookup.
            if (tuple(metadata.get("token_ids", ())) == tuple(segment["token_ids"])
                    and metadata.get("tokenizer_hash") == self.provenance["tokenizer_hash"]
                    and metadata.get("runtime_compatibility") == self.provenance["runtime_compatibility"]):
                eligible.append(row)
        return visible, tuple(eligible)

    def _compare(self, context, sid, depth, rows, current, ledger, request_id):
        import torch
        positions = tuple(context.segments[sid]["positions"])
        if tuple(sorted(set(positions))) != positions or current.dtype != torch.bfloat16 or current.ndim != 3:
            raise ValueError("live comparison requires ordered absolute rows and BF16 K geometry")
        states = []
        state_read_start = time.perf_counter_ns()
        for row in rows:
            try:
                tensor = self.store.read_selection(row.source_variant_id, depth)
            except KeyError:
                continue
            if tensor.shape != current.shape:
                continue
            states.append((row, tensor))
        ledger.observe_shared_interval(f"{request_id}:{sid}:{depth}:state-read", state_read_start, time.perf_counter_ns())
        if hasattr(context, "selection_backing_tier") and states:
            context.selection_backing_tier = "pinned_cpu" if all(t.is_pinned() for _, t in states) else "pageable_cpu"
        per_source = current.numel() * 32 + current.shape[0] * 32
        batch_k = min(len(states), max(0, (self.hbm.selector_lease_bytes - per_source) // per_source))
        values, plans = [], {}
        offset = 0
        while batch_k and offset < len(states):
            count = min(batch_k, len(states) - offset)
            prediction = None
            if ledger.policy.mode == "legacy_fixed_fraction":
                while count:
                    prediction = self.costs.comparison_ms(context, depth, count, len(positions))
                    if prediction is not None and ledger.may_compare(prediction):
                        break
                    count -= 1
                if not count:
                    break
            event_id = f"{request_id}:{sid}:{depth}:{offset}"
            reservation = self.hbm.reserve_batch(owner_request_id=request_id,
                rows=((event_id, per_source * (count + 1), HBMReservationKind.SELECTION_WORKSPACE),))[0]
            begin = time.perf_counter_ns()
            source_tensor = drift = order = None
            try:
                if prediction is not None:
                    ledger.reserve(event_id, prediction)
                source_tensor = torch.stack([t.to(current.device, non_blocking=True) for _, t in states[offset:offset + count]])
                drift = (source_tensor.float() - current.float()).square().sum((2, 3)).sqrt()
                drift /= current.float().square().sum((1, 2)).sqrt().clamp_min(1e-12)
                order = drift.argsort(dim=1, descending=True, stable=True)
                trim = min(len(positions) - 1, math.ceil(context.selector.variant_profile.source_residual_trim_ratio * len(positions)))
                scores = drift.gather(1, order)[:, trim:].mean(1).detach().cpu().tolist()
                end = time.perf_counter_ns()
                if prediction is not None:
                    ledger.settle(event_id, begin, end)
                else:
                    ledger.observe_shared_interval(event_id, begin, end)
                for rank, ((row, _), score) in enumerate(zip(states[offset:offset + count], scores), offset):
                    source_id = row.source_variant_id
                    self.store.pool.record_observation(self.provenance["model_signature"],
                        context.segments[sid]["content_key"], source_id, lookup_hit=True, compared=True)
                    plan = self.costs.gate1(context, sid, source_id, depth)
                    if plan is None:
                        raise UnsupportedTimelineCost("source_local_measurement_missing")
                    future = self.costs.candidate_future_ms(context, sid, source_id, depth)
                    if future is None or not math.isfinite(future) or future < 0:
                        raise UnsupportedTimelineCost("source_future_measurement_missing")
                    plans[source_id] = plan
                    # Do not label Gate1's marginal LOWER bound as a future UPPER cost.
                    values.append(ResidualCandidate(source_id, score, future, rank))
            finally:
                if current.device.type == "cuda":
                    torch.cuda.current_stream(current.device).synchronize()
                source_tensor = drift = order = None
                if event_id in ledger.pending:
                    ledger.cancel(event_id)
                self.hbm.release(reservation.reservation_id)
            offset += count
        return values, plans, len(states)

    def execute(self, request, dispatch, *, arrival_ns):
        if not 0 <= arrival_ns <= time.perf_counter_ns():
            raise ValueError("arrival timestamp must precede execution")
        with self.lock:
            if self.poisoned_error is not None:
                raise RuntimeError("failed runtime requires reset and new evidence directory: " + self.poisoned_error)
            # Queue waiting and previous request materialization are not erased.
            if self.pending:
                self.finalize_request(*self.pending[:2])
            rid = request["request_id"]
            if rid in self.completed or request["request_epoch"] <= self.epoch:
                raise ValueError("request IDs/epochs must form an immutable causal trace")
            name = dispatch["selection_path"]
            if name not in self.adapters:
                raise RuntimeError("dispatch adapter is not connected: " + name)
            adapter = self.adapters[name]
            if (adapter.capabilities.get("native_prefix_block_allocator") is not True
                    or adapter.capabilities.get("production_dispatch") != name):
                raise RuntimeError("diagnostic fixed blocks or mismatched dispatch cannot serve online requests")
            if dispatch.get("force_nonpaper_measurement_admission"):
                raise ValueError("production entry forbids diagnostic admission bypass")
            started = time.perf_counter_ns()
            initial = self.snapshot(retain_backing=False)
            self._emit("request_started", rid, {"request": request, "dispatch": dispatch, "arrival_ns": arrival_ns,
                                                "initial_snapshot_sha256": digest_json(initial)})
            try:
                # Native context fences and drops GPU working tensors BEFORE
                # the outer stack releases physical leases/HBM reservations.
                with ExitStack() as leases, adapter.open_request(request, arrival_ns=arrival_ns) as context:
                    context.online_context_opened_ns = time.perf_counter_ns()
                    row, exports = self._execute_context(context, request, dispatch, arrival_ns, started, initial, leases)
                self._emit("request_completed", rid, row)
                self.pending = (deepcopy(request), row, exports)
                return row
            except Exception as exc:
                self.poisoned_error = type(exc).__name__ + ": " + str(exc)
                self._emit("request_failed", rid, {"error": str(exc), "type": type(exc).__name__, "timestamp_ns": time.perf_counter_ns()})
                raise

    def _execute_context(self, context, request, dispatch, arrival_ns, started, initial, leases):
        rid, epoch = request["request_id"], request["request_epoch"]
        selector = self.selector_factory(dispatch, self.store.pool.max_variants_per_content)
        context.selector = selector
        context.assert_dispatch(selector.checkpoint_depths)
        segments = context.segments
        if set(segments) != {s["segment_id"] for s in request["segments"]}:
            raise ValueError("native request context lost part of the Segment inventory")
        inventory = native_segment_inventory(segments, prompt_tokens=len(request["token_ids"]),
                                            cached_prefix_tokens=context.cached_prefix_tokens)
        context.execution_inventory = inventory
        segments = {sid: segment for sid, segment in segments.items() if inventory[sid].comparison_eligible}
        if not segments or getattr(context, "probe_fallback_reason", None):
            return self._unsupported_dense(context, request, dispatch, arrival_ns, started, initial,
                reason=getattr(context, "probe_fallback_reason", None) or "no_nonprefix_candidates")
        dense = self.costs.dense_reference(context)
        if dense is None or not math.isfinite(dense) or dense <= 0:
            # Still execute actual dense; no invented reference, selector or cost.
            return self._unsupported_dense(context, request, dispatch, arrival_ns, started, initial)
        ledger = SelectionCostLedger(dense, SelectionCostPolicy(mode=dispatch.get("selection_budget_policy", "end_to_end_aware")))
        selection = ProductionSelectionSession(rid, tuple(segments), selector, ledger)
        visible, eligible, compatible, frozen, prepared = {}, {}, set(), {}, {}
        selection_failures, runtime_events = {}, []
        for sid, segment in segments.items():
            v, e = self._lookup(segment, epoch)
            visible[sid], eligible[sid] = v, e
        for depth in selector.checkpoint_depths:
            t = time.perf_counter_ns()
            context.advance_to_depth(depth)
            context.synchronize()
            ledger.observe_shared_interval(f"{rid}:probe:{depth}", t, time.perf_counter_ns())
            for sid, segment in segments.items():
                if sid in selection.decisions or sid in selection_failures:
                    continue
                rows = eligible[sid]
                metadata_begin = time.perf_counter_ns()
                prefix = tuple(CanonicalChunkOccurrence(**v) for v in segment["prefix_occurrences"])
                # Invalid CFO metadata does not acquire comparison eligibility.
                ranked = []
                for row in rows:
                    try:
                        metadata = read_cfo_metadata(self.store.objects[row.source_variant_id].metadata["cfo"])
                        score = compute_cachecraft_cfo(metadata, prefix).cfo_operational
                        ranked.append((score, row.source_variant_id, row))
                    except (ValueError, KeyError, TypeError):
                        pass
                ordered = [row for _, _, row in sorted(ranked)]
                current = context.observe_current_k(sid, depth)
                context.synchronize()
                ledger.observe_shared_interval(f"{rid}:{sid}:{depth}:metadata-current-k", metadata_begin, time.perf_counter_ns())
                try:
                    values, plans, available = self._compare(context, sid, depth, ordered,
                        current, ledger, rid)
                except UnsupportedTimelineCost as exc:
                    selection_failures[sid] = str(exc)
                    runtime_events.append({"kind": "dense_fallback", "segment_id": sid, "reason": str(exc)})
                    continue
                states_available = sum(depth in self.store.objects[r.source_variant_id].metadata[
                    "selection_completed_depths"] for r in rows)
                counts = CandidateCounts(len(visible[sid]), len(rows), states_available, available, len(values))
                decision = selection.step(sid, completed_depth=depth, counts=counts,
                    candidates=values, gate1_plan_by_source=plans)
                compatible.update(v.source_variant_id for v in values if v.residual_score <= decision.absolute_threshold)
                if decision.selected_source_variant_id:
                    source_id = decision.selected_source_variant_id
                    pool = self.store.pool
                    generation = pool.content_generation(self.provenance["model_signature"], segment["content_key"])
                    try:
                        layers = leases.enter_context(self.store.leased_winner(self.provenance["model_signature"],
                            segment["content_key"], source_id, expected_generation=generation))
                    except RuntimeError as exc:
                        selection_failures[sid] = "source_freeze_lease_failed"
                        runtime_events.append({"kind": "freeze_failed", "segment_id": sid, "source_id": source_id, "error": str(exc)})
                        continue
                    frozen[sid] = source_id
                    try:
                        preparation = self.costs.preparation(context, sid, source_id)
                    except UnsupportedTimelineCost as exc:
                        preparation = None
                        runtime_events.append({"kind": "dense_fallback", "segment_id": sid,
                                               "reason": str(exc)})
                    if preparation is None or preparation.get("resource_admitted") is not True:
                        runtime_events.append({"kind": "dense_fallback", "segment_id": sid,
                                               "reason": "preparation_resource_or_waste_budget_unavailable"})
                        continue
                    try:
                        size = getattr(layers, "full_kv_bytes", None)
                        if size is None:
                            size = sum(t.numel() * t.element_size() for pair in layers for t in pair)
                        reservation = self.hbm.reserve_batch(owner_request_id=rid,
                            rows=((sid, size, HBMReservationKind.WINNER_PREFETCH),))[0]
                    except MemoryError:
                        runtime_events.append({"kind": "dense_fallback", "segment_id": sid, "reason": "hbm_reservation_unavailable"})
                        continue
                    def release_after_fence(reservation_id=reservation.reservation_id):
                        # A failed CUDA fence must not advertise these bytes
                        # as reusable. Keep the reservation for quarantine.
                        context.synchronize()
                        self.hbm.release(reservation_id)
                    leases.callback(release_after_fence)
                    prepared[sid] = context.prepare_winner(sid, source_id, layers, reservation)
                    runtime_events.append({"kind": "winner_preparation", "segment_id": sid, "source_id": source_id,
                                           "reservation_id": reservation.reservation_id, "full_kv_bytes": size})
            if len(selection.decisions) + len(set(selection_failures) - set(selection.decisions)) == len(segments):
                break
        context.finish_selection(frozen, prepared)
        # Adapter provides actual winner repair supports/ready boundaries, not selector trim rows.
        ready_boundaries, union_digest = context.ready_for_final_commit(prepared)
        runtime_events.append({"kind": "online_timing_landmarks", "arrival_ns": arrival_ns,
            "context_opened_ns": getattr(context, "online_context_opened_ns", None),
            "selection_closed_ns": time.perf_counter_ns(),
            "selection_intervals": list(ledger.intervals) if hasattr(ledger, "intervals") else None})
        accepted = ()
        final_total = None
        if frozen:
            for attempt in range(3):
                snapshot = context.planner_snapshot(self.hbm.epoch)
                try:
                    estimator = self.costs.joint_estimator(context)
                    planner_started_ns = time.perf_counter_ns()
                    result = FinalCommitPlanner(estimator).plan_ready_subset(inventory_segment_ids=tuple(inventory),
                        eligible_ready_segment_ids=tuple(ready_boundaries), committed_segment_ids=(),
                        actual_boundary_by_segment=ready_boundaries, actual_sunk_ms=(planner_started_ns - arrival_ns) / 1e6,
                        dense_reference_total_ms=dense, snapshot=snapshot,
                        current_snapshot=context.planner_snapshot(self.hbm.epoch), union_mask_digest=union_digest)
                    snapshot.assert_current(context.planner_snapshot(self.hbm.epoch))
                    planner_elapsed_ms = (time.perf_counter_ns() - planner_started_ns) / 1e6
                    result = replace(result, request_total_ms=result.request_total_ms + planner_elapsed_ms)
                    if result.accepted_ready_segment_ids and result.request_total_ms > .8 * dense:
                        # The actual planner is part of request sunk time. A slow
                        # query must not authorize a path with an obsolete budget.
                        runtime_events.append({"kind": "final_commit", "accepted_ready_segment_ids": [],
                            "rejected_ready_segment_ids": list(ready_boundaries), "request_total_ms": None,
                            "proposed_request_total_ms": result.request_total_ms,
                            "planner_elapsed_ms": planner_elapsed_ms, "reason": "planner_elapsed_exceeds_gamma"})
                    else:
                        context.commit_reuse(result)
                        accepted, final_total = result.accepted_ready_segment_ids, result.request_total_ms
                        runtime_events.append({"kind": "final_commit", "decision": asdict(result),
                                               "planner_elapsed_ms": planner_elapsed_ms})
                    break
                except UnsupportedTimelineCost as exc:
                    runtime_events.append({"kind": "final_commit", "accepted_ready_segment_ids": [],
                        "rejected_ready_segment_ids": list(ready_boundaries), "request_total_ms": None, "reason": str(exc)})
                    break
                except RuntimeError as exc:
                    if str(exc) != "stale Planner snapshot cannot be applied":
                        raise
                    if attempt < 2:
                        runtime_events.append({"kind": "planner_snapshot_retry", "attempt": attempt + 1,
                            "selected_sources": dict(frozen), "timestamp_ns": time.perf_counter_ns()})
                        continue
                    # A permanently unstable snapshot is a dense fallback.
                    runtime_events.append({"kind": "final_commit", "accepted_ready_segment_ids": [],
                        "rejected_ready_segment_ids": list(ready_boundaries), "request_total_ms": None,
                        "reason": "planner_snapshot_changed_before_commit"})
        else:
            runtime_events.append({"kind": "dense_fallback", "reason": "no_frozen_sources"})
        first = []
        output = context.finish(lambda: first.append(time.perf_counter_ns()))
        if len(first) != 1:
            raise RuntimeError("runtime did not record exactly one actual first-token endpoint")
        completion = time.perf_counter_ns()
        ttft = (first[0] - arrival_ns) / 1e6
        selected = [d.selected_source_variant_id for d in selection.decisions.values() if d.selected_source_variant_id]
        committed = [frozen[sid] for sid in accepted]
        not_applicable = None
        if not frozen:
            not_applicable = ("all_source_freezes_failed" if selected else
                              "selection_cost_unsupported" if selection_failures else "no_frozen_sources")
            runtime_events.append({"kind": "dense_fallback", "reason": not_applicable})
        coverage = {"request_id": rid, "request_epoch": epoch, "execution_kind": "online",
            "capacity": self.store.pool.max_variants_per_content,
            "global_byte_budget": self.store.global_byte_budget,
            "visible_variant_creation_epochs": {v.source_variant_id: self.store.objects[v.source_variant_id].creation_epoch
                                                 for rows in visible.values() for v in rows},
            "compatible_variant_ids": sorted(compatible), "selected_variant_ids": selected,
            "committed_variant_ids": committed, "quality_passed": output.get("quality_passed"),
            "residual_compatibility_observed": not selection_failures,
            "matched_dense_ttft_ms": dense, "actual_ttft_ms": ttft}
        exports = context.export_exact_dense() if (not accepted and context.cached_prefix_tokens == 0
            and output.get("whole_request_origin") == "exact_dense_full_prefill") else {}
        row = {**output, "request_id": rid, "arrival_ns": arrival_ns, "service_start_ns": started,
            "queue_ms": (started - arrival_ns) / 1e6, "first_token_ns": first[0], "completion_ns": completion,
            "request_ttft_ms": ttft, "execution_kind": "online_policy", "forced_source": False,
            "selection_events": selection.events, "selection_failures": selection_failures,
            "runtime_events": runtime_events, "coverage_event": coverage,
            "segment_ownership": {sid: asdict(owner) for sid, owner in inventory.items()},
            "selected_source_variant_ids": selected, "committed_source_variant_ids": committed,
            "final_commit_executed": bool(frozen), "final_commit_not_applicable_reason": not_applicable,
            "final_predicted_request_total_ms": final_total,
            "realized_overrun_ms": max(0, ttft - .8 * dense) if committed else None,
            "initial_pool_snapshot_sha256": digest_json(initial), "dispatch_config": dict(dispatch),
            "code_commit": self.provenance["code_commit"], "model_signature": self.provenance["model_signature"],
            "execution_disposition": "reuse" if committed else "dense",
            "evidence_origin": context.evidence_origin, "timing_scope": "arrival_to_first_token",
            "ssd_page_cache_controlled": False, "paper_evidence": False, "gpu_runtime_qualified": False}
        builders = context.deferred_canonical_builders() if hasattr(context, "deferred_canonical_builders") else {}
        return row, {"tensors": exports, "selection": selection, "context_metadata": context.materialization_metadata(),
                     "canonical_builders": builders}

    def _unsupported_dense(self, context, request, dispatch, arrival, started, initial,
                           reason="matched_dense_cost_unsupported"):
        first = []
        output = context.finish(lambda: first.append(time.perf_counter_ns()))
        if len(first) != 1:
            raise RuntimeError("missing actual first-token endpoint")
        visible = [v for sid, s in context.segments.items()
                   if context.execution_inventory[sid].comparison_eligible
                   for v in self._lookup(s, request["request_epoch"])[0]]
        coverage = {"request_id": request["request_id"], "request_epoch": request["request_epoch"],
            "execution_kind": "online", "capacity": self.store.pool.max_variants_per_content,
            "global_byte_budget": self.store.global_byte_budget,
            "visible_variant_creation_epochs": {v.source_variant_id: self.store.objects[v.source_variant_id].creation_epoch for v in visible},
            "compatible_variant_ids": [], "selected_variant_ids": [], "committed_variant_ids": [],
            "residual_compatibility_observed": False, "quality_passed": output.get("quality_passed"),
            "matched_dense_ttft_ms": None, "actual_ttft_ms": (first[0] - arrival) / 1e6}
        row = {**output, "request_id": request["request_id"], "arrival_ns": arrival, "service_start_ns": started,
            "first_token_ns": first[0], "completion_ns": time.perf_counter_ns(), "request_ttft_ms": (first[0] - arrival) / 1e6,
            "queue_ms": (started - arrival) / 1e6, "execution_kind": "online_policy", "forced_source": False,
            "selection_events": [], "runtime_events": [{"kind": "dense_fallback", "reason": reason}],
            "segment_ownership": {sid: asdict(owner) for sid, owner in context.execution_inventory.items()},
            "selected_source_variant_ids": [], "committed_source_variant_ids": [], "final_commit_executed": False,
            "final_commit_not_applicable_reason": reason, "quality_passed": output.get("quality_passed"),
            "coverage_event": coverage, "execution_disposition": "dense",
            "matched_dense_ttft_ms": None, "initial_pool_snapshot_sha256": digest_json(initial),
            "dispatch_config": dict(dispatch), "code_commit": self.provenance["code_commit"],
            "model_signature": self.provenance["model_signature"], "evidence_origin": context.evidence_origin,
            "paper_evidence": False, "gpu_runtime_qualified": False}
        return row, {}

    def finalize_request(self, request, outcome):
        with self.lock:
            rid = request["request_id"]
            if rid in self.completed:
                if self.completed[rid] != digest_json(outcome):
                    raise ValueError("idempotent finalize received a changed outcome")
                return
            if not self.pending or self.pending[0] != request or digest_json(self.pending[1]) != digest_json(outcome):
                raise RuntimeError("no matching completed request awaiting publication")
            exports = self.pending[2]
            selection = exports.get("selection")
            # All lookup/coverage events are already durable before the publication point.
            metadata = exports.get("context_metadata", {})
            ownership = outcome.get("segment_ownership", {})
            looked_up_contents = {segment["content_key"] for segment in request["segments"]
                if not ownership or ownership[segment["segment_id"]]["disposition"] == "NONPREFIX_CANDIDATE"}
            for content_key in sorted(looked_up_contents):
                self.store.pool.finish_content_lookup(self.provenance["model_signature"], content_key)
            builders = exports.get("canonical_builders", {})
            tensors_by_sid = exports.get("tensors", {})
            for sid in sorted(set(tensors_by_sid) | set(builders)):
                tensors = tensors_by_sid.get(sid)
                decision = selection.decisions.get(sid)
                if decision is None or decision.materialization_reason is None:
                    continue
                m = metadata[sid]
                if m.get("estimated_materialization_ms") is None:
                    self._emit("materialization_skipped", rid, {"segment_id": sid, "reason": "missing_write_cost"})
                    continue
                content = m["identity"].reuse_content_key
                pool = self.store.pool
                counts = next(e["counts"] for e in reversed(selection.events) if e["segment_id"] == sid)
                existing = pool.variants_for_content(self.provenance["model_signature"], content, include_unavailable=True)
                controller = VariantMaterializationControllerV10(selection.selector.variant_profile)
                req = VariantMaterializationRequestV10(decision.materialization_reason,
                    counts["correctness_eligible_k"], counts["compared_k"], decision.best_residual,
                    decision.absolute_threshold, DenseKVProvenance.DENSE_EXACT, len(existing),
                    selection.ledger.dense_reference_ttft_ms, m["estimated_materialization_ms"],
                    exploration_materializations_for_content=pool.exploration_materialization_count(self.provenance["model_signature"], content),
                    exploration_authorized=bool(m.get("exploration_authorized", False)))
                admission = controller.decide_with_pool(req, pool=pool,
                    model_math_signature=self.provenance["model_signature"], reuse_content_key=content)
                if admission.state is not VariantMaterializationStateV10.ADMITTED:
                    self._emit("materialization_skipped", rid, {"segment_id": sid, "reason": admission.rejection_reason})
                    continue
                begin = time.perf_counter_ns()
                try:
                    if tensors is None:
                        tensors = builders[sid]()
                        if tensors.get("capture_audit", {}).get("origin") != "exact_dense_full_prefill":
                            raise RuntimeError("independent capture is not exact dense")
                        self._emit("canonical_build_completed", rid, {"segment_id": sid, **tensors["capture_audit"]})
                    if tensors.get("source_metadata") is not None:
                        m = {**m, "source_metadata": tensors["source_metadata"]}
                    replacement = pool.replacement_transaction(self.provenance["model_signature"], content)
                    if replacement.victim_source_variant_id != admission.replacement_source_variant_id:
                        raise RuntimeError("admitted materialization victim changed before publication")
                    self.store.publish_exact_dense(m["identity"], layers=tensors["layers"],
                        selection_states=tensors["selection_states"], metadata=m["source_metadata"],
                        request_epoch=request["request_epoch"], whole_request_origin="exact_dense_full_prefill",
                        materialization_reason=admission.reason.value, replacement_transaction=replacement)
                    self._emit("materialization_published", rid, {"segment_id": sid, "source_id": m["identity"].source_variant_id,
                        "host_ms": (time.perf_counter_ns() - begin) / 1e6})
                except (OSError, MemoryError, RuntimeError) as exc:
                    self.poisoned_error = type(exc).__name__ + ": " + str(exc)
                    self._emit("materialization_failed", rid, {"segment_id": sid, "error": str(exc),
                        "host_ms": (time.perf_counter_ns() - begin) / 1e6})
                    raise
            # Exclusive CPU-preferred placement is updated after GPU fences and
            # lease release; promotion is not a new independent user access.
            for source_id in outcome.get("selected_source_variant_ids", ()):
                if source_id in self.store.objects:
                    try:
                        self.store.promote_request_use(source_id, record_request_use=False)
                    except MemoryError:
                        self._emit("promotion_deferred", rid, {"source_id": source_id, "reason": "protected_capacity"})
            self.completed[rid] = digest_json(outcome)
            self.epoch = request["request_epoch"]
            self.pending = None
            self._emit("request_finalized", rid, {"outcome_sha256": digest_json(outcome),
                                                  "pool_state_sha256": digest_json(self.store.describe()),
                                                  "materialization_end_ns": time.perf_counter_ns()})
            self.finalized_at_ns[rid] = time.perf_counter_ns()

    def service_completion_ns(self, request_id):
        return self.finalized_at_ns[request_id]
