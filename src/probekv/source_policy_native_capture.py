"""Isolated single-Segment dense-shadow capture through native request APIs.

SelectionState only: no Source freeze, winner preparation, full-KV read, repair
or materialization. Comparison is deliberately on CPU to avoid borrowing GPU
workspace outside the allocator. Diagnostic host/D2H costs remain visible and
must never be reported as the timing of a pruned online GPU selector.
"""
import time

from .source_policy_replay import build_observation, observe_depth_k
from .v8_schema10_execution import digest_json
from .v8_schema10_inventory import native_segment_inventory


def capture_native_source_observation(backend, *, request, selection_path, source_ids,
                                      binding, host_budget_bytes, allow_cpu_test=False,
                                      legacy_diagnostic=False):
    if (len(request.get("segments", ())) != 1 or request.get("locked_test_accessed") is not False
            or request.get("partition_role") not in {"fit", "validation", "development", "profile_freeze"}
            or not request.get("development_partition_digest")
            or request.get("capture_original_full_prefill") or request.get("capture_logits")
            or "teacher_token_ids" in request
            or type(host_budget_bytes) is not int or host_budget_bytes <= 0
            or not source_ids or len(source_ids) > 16 or len(set(source_ids)) != len(source_ids)):
        raise ValueError("capture requires bounded single-Segment development diagnostic inputs")
    required = {"model_signature", "tokenizer_hash", "code_commit", "patch_sha256", "config_sha256"}
    if not required <= set(binding):
        raise ValueError("incomplete capture execution binding")
    for key in ("model_signature", "tokenizer_hash", "code_commit"):
        if binding[key] != backend.provenance.get(key):
            raise ValueError("capture/backend identity differs: " + key)
    actual_binding = getattr(backend, "native_manifest_binding", None)
    if actual_binding is None and not allow_cpu_test:
        raise ValueError("native factory manifest binding is required")
    if actual_binding is not None and any(binding[key] != actual_binding.get(key) for key in required):
        raise ValueError("capture/native manifest binding differs")
    adapter = backend.adapters[selection_path]
    checkpoints = tuple(adapter.spec.checkpoints) if legacy_diagnostic else (1, 2)
    if legacy_diagnostic:
        from .source_policy_replay import LEGACY_DEPTHS
        if checkpoints not in LEGACY_DEPTHS:
            raise ValueError('unsupported full legacy checkpoint tuple')
    with backend.lock, backend.store.pool.mutation_lock:
        if backend.pending or getattr(adapter, "active", None) or backend.hbm.active_reserved_bytes:
            raise RuntimeError("capture requires a quiescent backend")
        if backend.poisoned_error:
            raise RuntimeError("backend is quarantined")
        descriptor = request["segments"][0]
        sid = descriptor["segment_id"]
        _, eligible = backend._lookup(descriptor, request["request_epoch"])
        eligible_ids = sorted(row.source_variant_id for row in eligible)
        if not set(source_ids) <= set(eligible_ids):
            raise ValueError("Source set includes future/unavailable/non-exact variants")
        for source in source_ids:
            if not set(checkpoints) <= set(backend.store.objects[source].metadata["selection_completed_depths"]):
                raise ValueError("full d1/d2 SelectionState is required; no full-KV fallback")
        pool_before = backend.store.snapshot_descriptor()
        runtime_snapshot = (adapter.snapshot(retain=True) if adapter.capabilities.get("snapshot_accepts_retention")
                            else adapter.snapshot())
        stages, depths = [], []
        started = time.perf_counter_ns()
        first = []
        result = None
        try:
            with adapter.open_request(request, arrival_ns=started) as context:
                inventory = native_segment_inventory(context.segments,
                    prompt_tokens=len(request["token_ids"]), cached_prefix_tokens=context.cached_prefix_tokens)
                if not inventory[sid].comparison_eligible or getattr(context, "probe_fallback_reason", None):
                    raise RuntimeError("Prefix-covered Segment or unavailable Prefix shadow cannot be compared")
                native = context.evidence_origin == "real_cuda_execution"
                if not native and not allow_cpu_test:
                    raise RuntimeError("CPU context cannot impersonate native capture")
                for depth in checkpoints:
                    if hasattr(adapter, "check_deadline"):
                        adapter.check_deadline()
                    begin = time.perf_counter_ns()
                    context.advance_to_depth(depth)
                    context.synchronize()
                    current = context.observe_current_k(sid, depth)
                    if native and current.device.type != "cuda":
                        raise RuntimeError("native hook must return real CUDA K")
                    # read_selection may decode all checkpoint states of ONE Source.
                    max_states = max(len(backend.store.objects[s].metadata["selection_completed_depths"]) for s in source_ids)
                    required_bytes = current.numel() * 32 * (max_states + 2) + len(source_ids) * len(descriptor["positions"]) * 256
                    if required_bytes > host_budget_bytes:
                        raise MemoryError("diagnostic SelectionState host budget exceeded")
                    host_begin = time.perf_counter_ns()
                    current_cpu = current.detach().cpu()
                    record = None
                    for source in sorted(source_ids):
                        state = backend.store.read_selection(source, depth)
                        if state.device.type != "cpu":
                            raise RuntimeError("diagnostic requires independent CPU SelectionState backing")
                        part = observe_depth_k(current_cpu, {source: state}, completed_depth=depth)
                        if record is None:
                            record = part
                        else:
                            record["sources"].update(part["sources"])
                        del state
                    depths.append(record)
                    end = time.perf_counter_ns()
                    stages.append({"completed_depth": depth,
                        "advance_and_observe_wall_ms": (host_begin - begin) / 1e6,
                        "diagnostic_extract_hash_compare_wall_ms": (end - host_begin) / 1e6,
                        "host_budget_required_bytes": required_bytes})
                    del current_cpu, current
                context.finish_selection({}, {})
                output = context.finish(lambda: first.append(time.perf_counter_ns()))
                context.synchronize()
                if len(first) != 1 or context.committed or context.prepared:
                    raise RuntimeError("shadow must remain dense and emit exactly one first token")
                provenance = {"request_id": request["request_id"], "segment_id": sid,
                    "model_signature": binding["model_signature"], "tokenizer_signature": binding["tokenizer_hash"],
                    "request_token_ids_sha256": digest_json(request["token_ids"]),
                    "source_inventory_digest": digest_json(pool_before),
                    "code_commit": binding["code_commit"], "patch_sha256": binding["patch_sha256"],
                    "config_sha256": binding["config_sha256"],
                    "development_partition_digest": request["development_partition_digest"]}
                result = {"observation": build_observation(provenance=provenance,
                    absolute_positions=descriptor["positions"], correctness_eligible_source_ids=eligible_ids,
                    depth_observations=depths, evidence_origin="native_hook_diagnostic" if native else "cpu_interface_test",
                    legacy_diagnostic=legacy_diagnostic),
                    "diagnostic_stages": stages, "cached_prefix_tokens": context.cached_prefix_tokens,
                    "diagnostic_first_token_wall_ms": (first[0] - started) / 1e6,
                    "output_token_ids": output.get("token_ids"), "qa_passed": None,
                    "comparison_execution_device": "cpu", "shadow_only": True,
                    "gpu_runtime_qualified": False, "paper_evidence": False, "locked_test_accessed": False}
        finally:
            # Native context exit owns CUDA fences and workspace cleanup. Never
            # force free reservations after a failed fence; quarantine instead.
            if backend.hbm.active_reserved_bytes or getattr(adapter, "active", None):
                backend.poisoned_error = "capture left active/quarantined resources"
            else:
                try:
                    adapter.restore(runtime_snapshot)
                    if backend.store.snapshot_descriptor() != pool_before:
                        raise RuntimeError("read-only capture changed Source pool state")
                    if result is not None:
                        result["pool_unchanged"] = True
                except Exception as exc:
                    backend.poisoned_error = str(exc)
                    raise
                finally:
                    if adapter.capabilities.get("snapshot_accepts_retention"):
                        adapter.release_snapshot(runtime_snapshot)
        if backend.poisoned_error:
            raise RuntimeError(backend.poisoned_error)
        result["diagnostic_total_including_restore_wall_ms"] = (time.perf_counter_ns() - started) / 1e6
        result["capture_sha256"] = digest_json(result)
        return result
