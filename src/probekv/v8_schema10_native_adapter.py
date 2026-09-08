"""Native single-request CacheBlend bridge, shared by Mistral and Qwen.

No fixture, fixed block table, altered prompt, or qualification bypass is used.
Importing this module never loads a model or starts CUDA.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import asdict
import inspect
import math
import time

from .cacheblend_v6_online_engine import CacheBlendV6OnlineEngine
from .model_adapters import PinnedCacheBlendResumableAdapter
from .v8_schema10_native_blocks import NativeBlockRequest
from .v8_schema10_inventory import native_segment_inventory, mandatory_suffix_positions
from .v8_schema10_execution import digest_json
from .v8_schema10_cost_provider import RequestExecutionShape
from .v8_schema10_measured_costs import MeasuredRequestCostProvider
from .v8_schema6_hbm import HBMReservationKind
from .v8_schema6_contracts import PlannerSnapshot
from .v8_schema6_planner import JointTimelineContext
from .v7_contracts import SourceVariantIdentity


def dispatch_depths(selection_path, model_spec):
    if selection_path == "d1_only":
        return (1,)
    if selection_path == "d1_d2_rescue":
        return (1, 2)
    if selection_path == "legacy_multicheckpoint":
        # model_spec must be the schema6+ spec, not historical v6 MISTRAL_SPEC.
        return tuple(model_spec.checkpoints)
    raise ValueError("unconnected native dispatch")


def validate_native_sampling_request(request):
    count = request.get("max_new_tokens", 32)
    if type(count) is not int or count < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if "teacher_token_ids" in request:
        tokens = request["teacher_token_ids"]
        if (not request.get("capture_logits") or not isinstance(tokens, (list, tuple))
                or len(tokens) != count - 1
                or any(type(t) is not int or t < 0 for t in tokens)
                or request.get("capture_original_full_prefill")):
            raise ValueError("teacher-forced diagnostic requires exactly max_new_tokens-1 input tokens and logits")


class NativeOnlineAdapter:
    def __init__(self, *, llm, model_spec, selection_path, loader, hbm, shadow_store,
                 store_provider, provenance, cost_provider, shared_runtime_state=None):
        import torch
        self.torch, self.llm, self.spec = torch, llm, model_spec
        self.worker = llm.llm_engine.model_executor.driver_worker
        self.runner, self.outer = self.worker.model_runner, self.worker.model_runner.model
        self.inner, self.kv = self.outer.model, self.worker.cache_engine.gpu_cache
        self.scheduler = llm.llm_engine.scheduler[0] if isinstance(llm.llm_engine.scheduler, list) else llm.llm_engine.scheduler
        self.loader, self.hbm, self.shadows = loader, hbm, shadow_store
        self.store_provider, self.provenance, self.costs = store_provider, provenance, cost_provider
        self.path, self.depths = selection_path, dispatch_depths(selection_path, model_spec)
        self.capabilities = {"native_prefix_block_allocator": True, "production_dispatch": selection_path,
                             "snapshot_accepts_retention": True}
        self.retained_shadows = {}
        self.shared = shared_runtime_state if shared_runtime_state is not None else {"active": None, "warm_history": [], "generation": 1}
        self.deadline = math.inf
        self.projection = PinnedCacheBlendResumableAdapter(self.inner, self.spec)
        params = inspect.signature(self.runner.prepare_input_tensors).parameters
        self.prepare_accepts_cache = len(params) >= 2  # pinned CacheBlend adds kv_caches

    @property
    def active(self):
        return self.shared["active"]

    @active.setter
    def active(self, value):
        self.shared["active"] = value

    @property
    def warm_history(self):
        return self.shared["warm_history"]

    @warm_history.setter
    def warm_history(self, value):
        self.shared["warm_history"] = value

    @property
    def generation(self):
        return self.shared["generation"]

    @generation.setter
    def generation(self, value):
        self.shared["generation"] = value

    def prepare(self, metadata, *, caches=None):
        caches = self.kv if caches is None else caches
        if self.prepare_accepts_cache:
            return self.runner.prepare_input_tensors([metadata], caches)
        return self.runner.prepare_input_tensors([metadata])

    def check_deadline(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("session deadline reached before next layer")

    def reset(self):
        if self.active is not None:
            raise RuntimeError("cannot reset active native request")
        self.torch.cuda.synchronize()
        manager = self.scheduler.block_manager
        if manager.block_tables:
            raise RuntimeError("native allocator is owned by another request")
        if manager.block_sliding_window is not None:
            raise RuntimeError("frozen native Prefix stack cannot silently disable sliding window")
        self.scheduler.block_manager = type(manager)(block_size=manager.block_size,
            num_gpu_blocks=manager.num_total_gpu_blocks, num_cpu_blocks=manager.num_total_cpu_blocks,
            watermark=manager.watermark, sliding_window=None, enable_caching=True)
        self.shadows.clear()
        self.warm_history = []
        self.generation += 1

    def snapshot(self, *, retain=False):
        if self.active is not None:
            raise RuntimeError("native snapshot requires quiescence")
        row = {"native_prefix_rebuild": list(self.warm_history), "selection_path": self.path,
                "model_signature": self.provenance["model_signature"],
                "prefix_shadows": self.shadows.descriptor(),
                "gpu_pointers_serialized": False, "ssd_page_cache_controlled": False}
        if retain:
            key = self.path + ":" + digest_json(row)
            self.shadows.retain(key)
            self.retained_shadows[digest_json(row)] = key
        return row

    def release_snapshot(self, snapshot):
        self.shadows.release_retained(self.retained_shadows.pop(digest_json(snapshot)))

    def restore(self, snapshot):
        if snapshot["selection_path"] != self.path or snapshot["model_signature"] != self.provenance["model_signature"]:
            raise ValueError("native snapshot dispatch/model mismatch")
        key = digest_json(snapshot)
        if key not in self.retained_shadows:
            raise ValueError("native restoration requires an explicitly retained snapshot")
        self.reset()
        # Re-execute the same dense warm history through native allocation.
        # No CUDA pointer or block index is deserialized.
        for request in snapshot["native_prefix_rebuild"]:
            with self.open_request(request, arrival_ns=time.perf_counter_ns()) as ctx:
                ctx.finish(lambda: None)
        self.shadows.restore_retained(self.retained_shadows[key])
        if self.snapshot() != snapshot:
            raise RuntimeError("native Prefix warm reconstruction differs")

    @contextmanager
    def open_request(self, request, *, arrival_ns):
        from vllm import SamplingParams
        from .v8_schema10_numerical_policy import assert_numerical_policy
        assert_numerical_policy(self.torch, getattr(self, "expected_numerical_execution_policy", None))
        mandatory_suffix_positions(request)
        validate_native_sampling_request(request)
        if self.active is not None:
            raise RuntimeError("max_integrated_concurrency=1")
        self.check_deadline()
        self.active = request["request_id"]
        holder = {}
        try:
            native = NativeBlockRequest(block_manager=self.scheduler.block_manager,
                request_id=request["request_id"], prompt_token_ids=request["token_ids"],
                sampling_params=SamplingParams(temperature=0, max_tokens=int(request.get("max_new_tokens", 32))),
                synchronize=self.torch.cuda.synchronize,
                apply_block_copies=self.worker.cache_engine.copy,
                prefix_shadow_provider=lambda tokens, ids: self.shadows.lookup(tokens, ids, native_request=holder["native"]))
        except Exception:
            self.active = None
            raise
        holder["native"] = native
        native.allow_missing_shadow = True
        ctx = None
        try:
            with self.torch.inference_mode(), native:
                ctx = NativeRequestContext(self, native, request, arrival_ns)
                try:
                    yield ctx
                finally:
                    ctx.close()
        finally:
            self.active = None

    def build_exact_dense_source(self, request, segment_id):
        from .v8_schema10_canonical import capture_exact_dense_source
        if self.active is not None:
            raise RuntimeError("independent canonical build must run after online request")
        self.check_deadline()
        return capture_exact_dense_source(self, request, segment_id)


class NativeRequestContext:
    evidence_origin = "real_cuda_execution"
    selection_backing_tier = "pinned_cpu"
    actual_repair_check_sunk_ms = 0.0

    def __init__(self, adapter, native, request, arrival_ns):
        self.adapter, self.native, self.request = adapter, native, request
        self.arrival_ns = arrival_ns
        self.segments = {s["segment_id"]: dict(s) for s in request["segments"]}
        from .v8_schema10_canonical import request_occurrences
        occurrences, targets, _ = request_occurrences(request)
        for sid, segment in self.segments.items():
            segment["prefix_occurrences"] = [asdict(o) for o in occurrences
                if o.provenance_position < targets[sid].provenance_position]
        self.cached_prefix_tokens = native.cached_prefix_tokens
        self.prefix_cache_mode = "native_exact_blocks"
        self.sampling_signature = {"temperature": 0, "max_new_tokens": int(request.get("max_new_tokens", 32))}
        self.execution_inventory = native_segment_inventory(self.segments,
            prompt_tokens=len(request["token_ids"]), cached_prefix_tokens=self.cached_prefix_tokens)
        self.probe_fallback_reason = "native_prefix_shadow_unavailable" if native.shadow_missing else None
        self.prepared, self.replica_reservations, self.supports, self.committed = {}, {}, {}, {}
        self.hot_replicas, self.hot_leases = {}, ExitStack()
        self.frozen, self.selection_closed = {}, False
        self.repair_ratio = float(request.get("correctness_repair_ratio", .15))
        if self.repair_ratio not in {.15, 1.0}:
            raise ValueError("native fixed15 path only accepts .15 or correctness r=1")
        self.engine = None
        self.workspace = None
        self.capture_reservation = None
        self.capture_collector = None
        self.canonical_exports = {}
        self.closed = self.finished = False
        self.generation = 1
        self._observation = {}
        self._prepared_inputs = adapter.prepare(native.metadata(is_prompt=True))
        self.attention, self.sampling = self._prepared_inputs[2:4]

    @property
    def current_completed_depth(self):
        return self.engine.session.current_layer if self.engine is not None else 0

    @property
    def actual_sunk_ms(self):
        return (time.perf_counter_ns() - self.arrival_ns) / 1e6

    def assert_dispatch(self, depths):
        if tuple(depths) != self.adapter.depths:
            raise RuntimeError("native FAST/legacy checkpoint mismatch")

    def _begin(self):
        if self.engine is not None:
            return
        if self.probe_fallback_reason:
            raise RuntimeError("missing Prefix shadow must use native dense fallback")
        a = self.adapter
        n = len(self.request["token_ids"])
        # Full request working composite, not just per-winner rows. This was
        # previously an unaccounted HBM allocation inside the engine.
        layer = a.inner.layers[0].self_attn
        # The engine retains both the prefix input shadow and the full working
        # composite. They are distinct allocations, even for an exact hit.
        size = (n + self.cached_prefix_tokens) * a.spec.num_layers * layer.num_kv_heads * layer.head_dim * 4
        self.workspace = a.hbm.reserve_batch(owner_request_id=self.request["request_id"],
            rows=(("request_working_kv", size, HBMReservationKind.COMMITTED_EXECUTION),))[0]
        try:
            self._enable_original_capture()
            # Pinned immutable inputs remain owned by this request. Transfer
            # K and V as two contiguous layer batches instead of issuing one
            # H2D operation per tensor (64 small copies for Mistral).
            # Layer views preserve the engine's existing per-layer contract.
            prefix_cpu = self.native.prefix_shadow or ()
            if prefix_cpu:
                key_cpu = a.torch.empty((len(prefix_cpu),) + tuple(prefix_cpu[0][0].shape),
                                        dtype=prefix_cpu[0][0].dtype, device="cpu", pin_memory=True)
                value_cpu = a.torch.empty((len(prefix_cpu),) + tuple(prefix_cpu[0][1].shape),
                                          dtype=prefix_cpu[0][1].dtype, device="cpu", pin_memory=True)
                a.torch.stack(tuple(pair[0] for pair in prefix_cpu), dim=0, out=key_cpu)
                a.torch.stack(tuple(pair[1] for pair in prefix_cpu), dim=0, out=value_cpu)
                key_gpu = key_cpu.to(a.runner.device, non_blocking=key_cpu.is_pinned())
                value_gpu = value_cpu.to(a.runner.device, non_blocking=value_cpu.is_pinned())
                shadows = tuple((key_gpu[layer], value_gpu[layer])
                                for layer in range(len(prefix_cpu)))
            else:
                shadows = ()
            self.engine = CacheBlendV6OnlineEngine(inner_model=a.inner, model_spec=a.spec, source_loader=a.loader)
            self.engine.begin_prefill(model_signature=a.provenance["model_signature"],
                token_ids=tuple(self.request["token_ids"][self.cached_prefix_tokens:]),
                absolute_positions=tuple(range(self.cached_prefix_tokens, n)),
                exact_prefix_tokens=self.cached_prefix_tokens, exact_prefix_layers=shadows,
                attention_metadata=self.attention, working_kv=a.kv)
        except Exception:
            a.torch.cuda.synchronize()
            self.engine = None
            shadows = ()
            a.inner.old_kvs = [[None, None] for _ in range(a.spec.num_layers)]
            a.hbm.release(self.workspace.reservation_id)
            self.workspace = None
            raise

    def _enable_original_capture(self):
        # Explicit preregistered capture task only: no surprise CFO work on
        # normal TTFT requests, no claim that extra materialization is free.
        if (not self.request.get("capture_original_full_prefill", False)
                or self.cached_prefix_tokens or self.capture_collector is not None):
            return
        from .v8_cfo import CFOFullPrefillCollector
        from .v8_schema10_canonical import request_occurrences
        _, _, ids = request_occurrences(self.request)
        a = self.adapter
        attn = a.inner.layers[0].self_attn
        size = len(ids) * a.spec.num_layers * attn.num_kv_heads * attn.head_dim * 4
        self.capture_reservation = a.hbm.reserve_batch(owner_request_id=self.request["request_id"],
            rows=(("original_full_prefill_capture", size, HBMReservationKind.COMMITTED_EXECUTION),))[0]
        self.capture_collector = CFOFullPrefillCollector(token_occurrence_ids=ids, expected_layers=a.spec.num_layers,
                                                        eager_reference=False)
        a.inner.cache_fuse_metadata.update(collect=True, probekv_cfo_collector=self.capture_collector)

    def advance_to_depth(self, depth):
        self._begin()
        while self.current_completed_depth < depth:
            self.adapter.check_deadline()
            layer = self.current_completed_depth + 1
            # Native single-concurrency scheduling waits on the actual next
            # layer event. Waiting is inside request wall-clock accounting.
            for sid in self.committed:
                self.prepared[sid].layer_events[layer].synchronize()
            self.engine.advance_to_layer(layer)
            self.generation += 1

    def synchronize(self):
        self.adapter.torch.cuda.synchronize()

    def register_ready_hot_replicas(self):
        from .contracts import KVLocation
        pool = self.adapter.store_provider().pool
        model = self.adapter.provenance["model_signature"]
        for sid, ticket in self.prepared.items():
            if sid in self.hot_replicas or not all(e.query() for e in ticket.layer_events.values()):
                continue
            source = pool._get(model, self.segments[sid]["content_key"], ticket.source_id)
            backing = source.healthy_backing_replicas[0]
            with pool.mutation_lock:
                shared = next((pair for pair in self.hot_replicas.values() if pair[0].source_variant_id == ticket.source_id), None)
                if shared is not None:
                    self.hot_replicas[sid] = shared
                    continue
                replica = pool.attach_replica(model, self.segments[sid]["content_key"], ticket.source_id,
                    tier=KVLocation.GPU, locator_value="request-hot:" + self.request["request_id"] + ":" + sid,
                    layout_signature="layer-contiguous-bf16",
                    bytes_digest=ticket.destination_digest or "expected-copy:" + ticket.expected_artifact_digest,
                    size_bytes=ticket.requested_bytes, derived_from_replica_id=backing.replica_id, is_backing=False)
                self.hot_replicas[sid] = (source, replica)
                self.hot_leases.enter_context(pool.lease_replica(model, self.segments[sid]["content_key"], ticket.source_id, replica.replica_id))

    def observe_current_k(self, sid, depth):
        if not self.execution_inventory[sid].comparison_eligible:
            raise RuntimeError("Prefix/tail cannot enter Source comparison")
        if depth not in self._observation:
            self._observation.clear()
            self._observation[depth] = self.engine.session.observe_pre_rope_k(depth)
        indices = {p: i for i, p in enumerate(self.engine.session.active_positions)}
        return self._observation[depth][[indices[p] for p in self.segments[sid]["positions"]]]

    def source_measurement_shape(self, sid, source_id, depth):
        obj = self.adapter.store_provider().objects[source_id]
        a = self.adapter.inner.layers[0].self_attn
        n = len(self.segments[sid]["positions"])
        return {"prompt_tokens": len(self.request["token_ids"]), "prefix_tokens": self.cached_prefix_tokens,
            "positions": list(self.segments[sid]["positions"]), "completed_depth": depth,
            "first_reuse_layer": depth + 1, "num_layers": self.adapter.spec.num_layers,
            "dtype": "bfloat16", "kv_heads": a.num_kv_heads, "head_dim": a.head_dim,
            "tier": obj.tier.value, "bytes": n * a.num_kv_heads * a.head_dim * self.adapter.spec.num_layers * 4,
            "layout": "pre_rope_k_raw_v", "repair_ratio": self.repair_ratio,
            "timing_scope": "source_local_boundary_future"}

    def prepare_winner(self, sid, source_id, layers, reservation):
        self.frozen[sid] = source_id
        self.replica_reservations[sid] = reservation
        store = self.adapter.store_provider()
        obj = store.objects[source_id]
        row = store.pool._get(self.adapter.provenance["model_signature"], self.segments[sid]["content_key"], source_id)
        if reservation.released or not any(p.busy for p in row.healthy_backing_replicas):
            raise RuntimeError("transfer without physical backing lease and HBM reservation")
        ticket = self.engine.start_winner_prefetch(segment_id=sid, source_id=source_id,
            canonical_layers=layers, segment_positions=self.segments[sid]["positions"],
            expected_artifact_digest=row.canonical_source_state_digest,
            request_id=self.request["request_id"], replica_id=row.healthy_backing_replicas[0].replica_id)
        self.prepared[sid] = ticket
        self.generation += 1
        return ticket

    def finish_selection(self, frozen, prepared):
        self.frozen, self.selection_closed = dict(frozen), True
        self.generation += 1

    def ready_for_final_commit(self, prepared):
        if not self.selection_closed:
            raise RuntimeError("dense-clean selection closure required")
        if not prepared:
            return {}, digest_json([])
        start = time.perf_counter_ns()
        depth = self.current_completed_depth
        current_k, current_v = self.engine.session.observe_repair_check_pre_rope_kv(depth)
        local = {p: i for i, p in enumerate(self.engine.session.active_positions)}
        ready = {}
        for sid, ticket in prepared.items():
            ticket.layer_events[depth + 1].synchronize()
            positions = tuple(self.segments[sid]["positions"])
            # Winner V-only metric is independent of Source-score trim indices.
            v = current_v[[local[p] for p in positions]].float()
            old = ticket.layer_tensors[depth + 1][1].float()
            drift = (v - old).square().sum((1, 2)).sqrt() / v.square().sum((1, 2)).sqrt().clamp_min(1e-12)
            order = drift.argsort(descending=True, stable=True).cpu().tolist()
            count = min(len(positions), math.ceil(len(positions) * self.repair_ratio))
            support = tuple(sorted(positions[i] for i in order[:count]))
            self.supports[sid] = {l: support for l in range(depth + 1, self.adapter.spec.num_layers + 1)}
            ready[sid] = depth + 1
        self.actual_repair_check_sunk_ms += (time.perf_counter_ns() - start) / 1e6
        self.register_ready_hot_replicas()
        self.generation += 1
        return ready, digest_json(self.supports)

    def planner_snapshot(self, epoch):
        ready = {sid: [l for l, event in ticket.layer_events.items() if event.query()]
                 for sid, ticket in self.prepared.items()}
        return PlannerSnapshot(self.generation, 1, digest_json([self.native.sequence.seq_id,
                               self.generation, self.current_completed_depth, ready]),
                               epoch, self.adapter.costs.sha)

    def settle_preparation_for_replan(self):
        """Fence only existing winner copies after a readiness-snapshot race.

        This starts no transfer and changes no Source. The backend includes
        the wait in actual sunk time before its next admission attempt.
        """
        before = {sid: [l for l, event in ticket.layer_events.items() if event.query()]
                  for sid, ticket in self.prepared.items()}
        started = time.perf_counter_ns()
        for ticket in self.prepared.values():
            for event in ticket.layer_events.values():
                event.synchronize()
        self.register_ready_hot_replicas()
        after = {sid: [l for l, event in ticket.layer_events.items() if event.query()]
                 for sid, ticket in self.prepared.items()}
        return {"ready_layers_before": before, "ready_layers_after": after,
                "host_wait_ms": (time.perf_counter_ns() - started) / 1e6}

    def execution_shape(self):
        physical = {}
        for sid, source_id in self.frozen.items():
            shape = self.source_measurement_shape(sid, source_id, self.current_completed_depth)
            ticket = self.prepared.get(sid)
            physical[sid] = {k: shape[k] for k in ("tier", "bytes", "layout")}
            physical[sid]["ready_layers"] = [l for l in ticket.layer_events if ticket.layer_ready(l)] if ticket else []
            physical[sid]["copy_in_flight"] = bool(ticket and len(physical[sid]["ready_layers"]) < len(ticket.layer_events))
        return RequestExecutionShape(len(self.request["token_ids"]), self.cached_prefix_tokens,
            self.adapter.spec.num_layers, self.current_completed_depth,
            {sid: o.remaining_positions for sid, o in self.execution_inventory.items()}, self.supports,
            self.committed, physical, MeasuredRequestCostProvider.identity(self))

    def dense_fallback_joint_context(self):
        ids = tuple(self.segments)
        return JointTimelineContext(ids, (), tuple(s for s in ids if s not in self.committed),
            tuple(self.committed), {}, digest_json(self.supports), self.planner_snapshot(self.adapter.hbm.epoch).scheduler_snapshot_id)

    def commit_reuse(self, decision):
        decision.planner_snapshot.assert_current(self.planner_snapshot(self.adapter.hbm.epoch))
        for sid in decision.accepted_ready_segment_ids:
            # An original capture that becomes selective is ineligible. Stop
            # collecting it rather than promoting locally dense pieces.
            self.adapter.inner.cache_fuse_metadata.update(collect=False, probekv_cfo_collector=None)
            boundary = self.current_completed_depth + 1
            self.engine.commit_ready_segment(segment_id=sid, boundary=boundary,
                segment_positions=self.segments[sid]["positions"], repair_positions=self.supports[sid][boundary],
                scheduler_boundary=boundary)
            self.adapter.hbm.promote(self.replica_reservations[sid].reservation_id,
                expected=HBMReservationKind.WINNER_PREFETCH, target=HBMReservationKind.COMMITTED_EXECUTION)
            self.committed[sid] = boundary
        self.generation += 1

    def finish(self, on_first_token):
        if self.finished:
            raise RuntimeError("request finish is not repeatable")
        a, torch = self.adapter, self.adapter.torch
        if self.engine is None:
            ids, pos = self._prepared_inputs[:2]
            a.inner.cache_fuse_metadata.update({"check": False, "collect": False, "probekv_cfo_collector": None})
            self._enable_original_capture()
            hidden = a.outer(input_ids=ids, positions=pos, kv_caches=a.kv, attn_metadata=self.attention)
        else:
            self.advance_to_depth(a.spec.num_layers)
            hidden = self.engine.finish_prefill()
            if self.engine.session.active_positions[-1] != len(self.request["token_ids"]) - 1:
                raise RuntimeError("native sampling lost the mandatory suffix row")
        if self.capture_collector is not None and not self.committed and not self.cached_prefix_tokens:
            from .v8_schema10_canonical import export_original_full_prefill
            self.canonical_exports = export_original_full_prefill(a, self.request, self.capture_collector)
        # No decode call may append to the full-prefill CFO capture.
        a.inner.cache_fuse_metadata.update(collect=False, probekv_cfo_collector=None)
        self.native.finish_prefill(exact_dense=not self.committed)
        selected = self.sampling.selected_token_indices.clone()
        try:
            self.sampling.selected_token_indices[0] = hidden.shape[0] - 1
            logits = a.outer.compute_logits(hidden, self.sampling)
        finally:
            self.sampling.selected_token_indices.copy_(selected)
        predicted = [int(logits.argmax().item())]
        on_first_token()
        self.logit_trace = [logits.detach().float().cpu()] if self.request.get("capture_logits") else []
        teachers = self.request.get("teacher_token_ids", ())
        teacher_forced = "teacher_token_ids" in self.request
        eos = a.llm.get_tokenizer().eos_token_id
        for _ in range(1, self.sampling_signature["max_new_tokens"]):
            if not teacher_forced and predicted[-1] == eos:
                break
            a.check_deadline()
            feed_token = int(teachers[len(predicted) - 1]) if teacher_forced else predicted[-1]
            metadata = self.native.append_for_decode(feed_token)
            ids, pos, attention, sample = a.prepare(metadata)[:4]
            hidden = a.outer(input_ids=ids, positions=pos, kv_caches=a.kv, attn_metadata=attention)
            logits = a.outer.compute_logits(hidden, sample)
            predicted.append(int(logits.argmax().item()))
            if self.request.get("capture_logits"):
                self.logit_trace.append(logits.detach().float().cpu())
            self.native.finish_decode_step()
        self.finished = True
        origin = "selective_reuse" if self.committed else "native_prefix_dense_remaining" if self.cached_prefix_tokens else "exact_dense_full_prefill"
        # Only dense requests rebuild exact Prefix state during paired replay.
        if not self.committed and not teacher_forced:
            a.warm_history.append(self.request)
        if teacher_forced:
            return {"token_ids": predicted, "answer": None, "quality_passed": None,
                "qa_evidence": None, "generation_mode": "teacher_forced_logit_diagnostic",
                "whole_request_origin": origin, "cached_prefix_tokens": self.cached_prefix_tokens,
                "layer_audit": self.engine.session.layer_audit if self.engine else []}
        from .v8_schema10_qa import answer_evidence
        evidence = answer_evidence(predicted, tokenizer=a.llm.get_tokenizer(), request=self.request)
        return {**evidence, "whole_request_origin": origin,
                "cached_prefix_tokens": self.cached_prefix_tokens,
                "layer_audit": self.engine.session.layer_audit if self.engine else []}

    def export_exact_dense(self):
        # Ordinary requests do not run the expensive full attention/CFO capture.
        # Independent builds are admitted and measured after request completion.
        if self.committed or self.cached_prefix_tokens:
            return {}
        return self.canonical_exports

    def deferred_canonical_builders(self):
        return {sid: (lambda sid=sid: self.adapter.build_exact_dense_source(self.request, sid))
                for sid, owner in self.execution_inventory.items() if owner.comparison_eligible}

    def materialization_metadata(self):
        result = {}
        for sid, s in self.segments.items():
            positions = tuple(s["positions"])
            query = {"prompt_tokens": len(self.request["token_ids"]), "positions": list(positions),
                     "num_layers": self.adapter.spec.num_layers, "origin": "current_request_capture" if sid in self.canonical_exports
                     else "independent_exact_dense_full_prefill"}
            category = "canonical_capture_write" if sid in self.canonical_exports else "canonical_build_and_write"
            row = self.adapter.costs._lookup(category, query)
            result[sid] = {"identity": SourceVariantIdentity(s["content_key"], digest_json(self.request["token_ids"][:positions[0]]),
                digest_json(positions), self.request["request_id"] + ":" + sid, self.adapter.provenance["model_signature"]),
                "estimated_materialization_ms": max(row["samples_ms"]) if row else None,
                "source_metadata": None}
        return result

    def close(self):
        if not self.closed:
            # If a CUDA fence fails, keep reservations/replicas quarantined.
            # The enclosing backend is poisoned and cannot admit another job.
            self.synchronize()
            self.hot_leases.close()
            pool = self.adapter.store_provider().pool
            with pool.mutation_lock:
                unique = {replica.replica_id: (source, replica) for source, replica in self.hot_replicas.values()}
                for source, replica in unique.values():
                    pool._delete_replica(source, replica, "single_request_execution_complete")
            self.engine = None
            self.prepared.clear()
            self._observation.clear()
            self.adapter.inner.old_kvs = [[None, None] for _ in range(self.adapter.spec.num_layers)]
            for block in self.adapter.inner.layers:
                block.self_attn.hack_kv = []
            self.adapter.inner.cache_fuse_metadata.update({"check": False, "probekv_resumable": False,
                "collect": False, "probekv_cfo_collector": None, "exact_prefix_tokens": 0})
            if self.workspace:
                self.adapter.hbm.release(self.workspace.reservation_id)
            if self.capture_reservation:
                self.adapter.hbm.release(self.capture_reservation.reservation_id)
            self.closed = True


class FastNativeOnlineAdapter(NativeOnlineAdapter):
    """Candidate shallow dispatch. Does not certify d1/d2 policy quality."""
    def __init__(self, **kwargs):
        if kwargs.get("selection_path") not in {"d1_only", "d1_d2_rescue"}:
            raise ValueError("FAST adapter cannot impersonate legacy")
        super().__init__(**kwargs)


class LegacyNativeOnlineAdapter(NativeOnlineAdapter):
    """Dense-clean full-checkpoint dispatch; no forced admission switch.

    The shared native primitive advances every Transformer block between the
    legacy checkpoints. This is the schema10 single-request barrier adapter,
    not a claim that all historical A/C deployments have been requalified.
    """
    def __init__(self, **kwargs):
        if kwargs.get("selection_path") != "legacy_multicheckpoint":
            raise ValueError("legacy adapter requires the complete checkpoint policy")
        super().__init__(**kwargs)
        if self.depths != tuple(self.spec.checkpoints):
            raise RuntimeError("incomplete legacy depth implementation")
