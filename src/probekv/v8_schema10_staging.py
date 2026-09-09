"""Physical bounded pinned slots and winner-only layer transfer.

One pool belongs to the backend, not to each Source. A slot cannot be recycled
until its copy event completes. CPU buffers are allocated lazily within 2 GiB.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from threading import RLock
import math
import time

from .v8_schema10_layer_storage import LayerFile
from .cacheblend_v6_online_engine import LayerwiseLoadTicket
from .v8_schema10_storage import tensor_digest


@dataclass
class StagingSlot:
    slot_id: int
    pair: tuple
    size_bytes: int
    leased: bool = False
    completion: object = None


class PhysicalPinnedStagingPool:
    def __init__(self, capacity_bytes=2_147_483_648, *, pin_memory=True):
        if capacity_bytes <= 0:
            raise ValueError("positive pinned capacity required")
        self.capacity_bytes, self.pin_memory = int(capacity_bytes), pin_memory
        self.slots, self.lock = [], RLock()
        self.peak_bytes = 0

    @property
    def allocated_bytes(self):
        return sum(slot.size_bytes for slot in self.slots)

    def acquire(self, shape):
        import torch
        size = math.prod(shape) * 4
        with self.lock:
            for slot in self.slots:
                if slot.leased or (slot.completion is not None and not slot.completion.query()):
                    continue
                if tuple(slot.pair[0].shape) == tuple(shape):
                    slot.leased, slot.completion = True, None
                    return slot
            # Only completed idle buffers can be discarded for a new geometry.
            for slot in tuple(self.slots):
                if self.allocated_bytes + size <= self.capacity_bytes:
                    break
                if not slot.leased and (slot.completion is None or slot.completion.query()):
                    self.slots.remove(slot)
            if self.allocated_bytes + size > self.capacity_bytes:
                raise MemoryError("pinned staging slots are busy or exceed capacity")
            pair = tuple(torch.empty(shape, dtype=torch.bfloat16, pin_memory=self.pin_memory) for _ in range(2))
            slot = StagingSlot(max((s.slot_id for s in self.slots), default=-1) + 1, pair, size, True)
            self.slots.append(slot)
            self.peak_bytes = max(self.peak_bytes, self.allocated_bytes)
            return slot

    def release_after(self, slot, completion):
        with self.lock:
            if not any(s is slot for s in self.slots) or not slot.leased:
                raise RuntimeError("invalid staging release")
            slot.completion, slot.leased = completion, False

    def clear(self):
        with self.lock:
            if any(s.leased or (s.completion is not None and not s.completion.query()) for s in self.slots):
                raise RuntimeError("cannot clear active staging")
            self.slots.clear()


class PhysicalLayerwiseSourceLoader:
    def __init__(self, pool, *, authorize, integrity_mode="online_immutable", device="cuda"):
        import torch
        if not torch.cuda.is_available() or not pool.pin_memory:
            raise RuntimeError("physical Source transfer requires CUDA and pinned buffers")
        if integrity_mode not in {"online_immutable", "qualification_full"}:
            raise ValueError("unsupported integrity path")
        self.torch, self.pool, self.authorize = torch, pool, authorize
        self.device, self.integrity_mode = device, integrity_mode
        self.stream = torch.cuda.Stream(device=device)
        self.events = []

    def begin(self, *, segment_id, source_id, canonical_layers, segment_positions,
              expected_artifact_digest, request_id="", replica_id="", prefetch_window=0,
              resident_layers=None):
        torch = self.torch
        size = getattr(canonical_layers, "full_kv_bytes", None)
        if size is None:
            size = sum(t.numel() * t.element_size() for pair in canonical_layers for t in pair)
        # This callback MUST check the live logical/physical lease and target
        # HBM reservation; neither speculative nor resident paths bypass it.
        self.authorize(source_id=source_id, segment_id=segment_id, bytes_required=size)
        if not expected_artifact_digest:
            raise ValueError("creation-time Artifact digest required")
        before = after = destination = ""
        hash_ms = d2h_ms = staging_ms = wait_ms = 0.0
        if self.integrity_mode == "qualification_full":
            t = time.perf_counter()
            before = tensor_digest(tensor for pair in canonical_layers for tensor in pair)
            hash_ms += (time.perf_counter() - t) * 1000
            if before != expected_artifact_digest:
                raise RuntimeError("canonical digest differs before transfer")
        started = time.perf_counter() * 1000
        start = torch.cuda.Event(enable_timing=True)
        tensors, events, layer_start_events, outstanding = {}, {}, {}, []
        if prefetch_window < 0:
            raise ValueError("prefetch_window must be non-negative")
        if prefetch_window and isinstance(canonical_layers, LayerFile):
            raise ValueError("windowed prefetch requires an in-memory CPU backing")
        if resident_layers is not None:
            expected = set(range(1, len(canonical_layers) + 1))
            if set(resident_layers) != expected:
                raise ValueError("GPU-resident Source layer inventory is incomplete")
            if any(key.device.type != "cuda" or value.device.type != "cuda"
                   for key, value in resident_layers.values()):
                raise ValueError("GPU-resident Source tensors must remain on CUDA")
            with torch.cuda.stream(self.stream):
                start.record()
                for layer in sorted(resident_layers):
                    ready = torch.cuda.Event(enable_timing=True)
                    ready.record()
                    tensors[layer] = resident_layers[layer]
                    events[layer] = ready
                    layer_start_events[layer] = ready
            self.events.append({"source_id": source_id, "staging_host_ms": 0.0,
                "staging_wait_ms": 0.0, "full_kv_bytes": size,
                "path": "GPU_RESIDENT", "resident": True})
            return LayerwiseLoadTicket(segment_id, source_id, started, size, tensors, start, events,
                "", "", tuple(segment_positions), layer_start_events=layer_start_events,
                pending_layers={}, integrity_mode=self.integrity_mode,
                expected_artifact_digest=expected_artifact_digest, destination_digest="",
                expected_layer_count=len(canonical_layers),
                per_request_full_digest_verified=False)
        copy_layers = canonical_layers if not prefetch_window else canonical_layers[:prefetch_window]
        pending_layers = {} if not prefetch_window else {
            i + 1: pair for i, pair in enumerate(canonical_layers[prefetch_window:], start=prefetch_window)
        }
        try:
            with torch.cuda.stream(self.stream):
                start.record()
                for index in range(len(copy_layers)):
                    slot = None
                    if isinstance(canonical_layers, LayerFile):
                        if len(outstanding) >= 2:
                            t = time.perf_counter()
                            outstanding.pop(0).synchronize()
                            wait_ms += (time.perf_counter() - t) * 1000
                        slot = self.pool.acquire(canonical_layers.shape)
                        t = time.perf_counter()
                        try:
                            pair = canonical_layers.read_into(index, slot.pair)
                        except Exception:
                            self.pool.release_after(slot, None)
                            raise
                        staging_ms += (time.perf_counter() - t) * 1000
                    else:
                        pair = canonical_layers[index]
                        if any(t.device.type != "cpu" or not t.is_pinned() for t in pair):
                            raise ValueError("CPU backing must be pre-pinned; no silent pin_memory")
                    try:
                        copy_start = torch.cuda.Event(enable_timing=True)
                        copy_start.record()
                        tensors[index + 1] = tuple(t.to(self.device, non_blocking=True) for t in pair)
                        done = torch.cuda.Event(enable_timing=True)
                        done.record()
                        events[index + 1] = done
                        layer_start_events[index + 1] = copy_start
                        if slot:
                            self.pool.release_after(slot, done)
                            outstanding.append(done)
                    except Exception:
                        self.stream.synchronize()
                        if slot and slot.leased:
                            self.pool.release_after(slot, None)
                        raise
            if self.integrity_mode == "qualification_full" and not pending_layers:
                self.stream.synchronize()
                t = time.perf_counter()
                destination = tensor_digest(tensor for pair in tensors.values() for tensor in pair)
                d2h_ms = (time.perf_counter() - t) * 1000
                t = time.perf_counter()
                after = tensor_digest(tensor for pair in canonical_layers for tensor in pair)
                hash_ms += (time.perf_counter() - t) * 1000
                if not before == after == destination:
                    raise RuntimeError("qualification Source/destination digest mismatch")
            self.events.append({"source_id": source_id, "staging_host_ms": staging_ms,
                "staging_wait_ms": wait_ms, "full_kv_bytes": size,
                "path": "SSD_STAGED_TO_GPU" if isinstance(canonical_layers, LayerFile) else "CPU_PINNED_TO_GPU"})
            return LayerwiseLoadTicket(segment_id, source_id, started, size, tensors, start, events,
                before, after, tuple(segment_positions), layer_start_events=layer_start_events,
                pending_layers=pending_layers,
                integrity_mode=self.integrity_mode,
                expected_artifact_digest=expected_artifact_digest, destination_digest=destination,
                hash_host_ms=hash_ms, d2h_hash_host_ms=d2h_ms,
                expected_layer_count=len(canonical_layers),
                integrity_verification_pending=self.integrity_mode == "qualification_full" and bool(pending_layers),
                per_request_full_digest_verified=self.integrity_mode == "qualification_full" and not pending_layers)
        except Exception:
            self.stream.synchronize()
            raise

    def prefetch_pending(self, ticket, through_layer):
        if ticket.transfer_failed:
            raise RuntimeError("failed transfer cannot be resumed")
        pending = [layer for layer in sorted(ticket.pending_layers) if layer <= through_layer]
        if not pending:
            return
        with self.torch.cuda.stream(self.stream):
            for layer in pending:
                key, value = ticket.pending_layers[layer]
                if key.device.type != "cpu" or value.device.type != "cpu" or not key.is_pinned() or not value.is_pinned():
                    raise ValueError("windowed CPU backing must be pinned")
                start = self.torch.cuda.Event(enable_timing=True)
                start.record(self.stream)
                gpu_key = gpu_value = None
                try:
                    with (self.torch.profiler.record_function(f"probekv.copy_layer.{layer}")
                          if getattr(self, "capture_hardware_trace", False) else nullcontext()):
                        gpu_key = key.to(self.device, non_blocking=True)
                        gpu_value = value.to(self.device, non_blocking=True)
                    done = self.torch.cuda.Event(enable_timing=True)
                    done.record(self.stream)
                except Exception:
                    ticket.transfer_failed = True
                    self.stream.synchronize()
                    raise
                ticket.layer_tensors[layer] = (gpu_key, gpu_value)
                ticket.layer_start_events[layer] = start
                ticket.layer_events[layer] = done
                del ticket.pending_layers[layer]
