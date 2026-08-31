from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .model_adapters import PinnedCacheBlendResumableAdapter, ResumableModelSpec
from .resumable_prefill import ProbeKVResumablePrefillSession
from .v7_contracts import CanonicalKVArtifact, PhysicalReplica
from .v8_schema7_contracts import IntegrityVerificationMode, RepairMetric


def integrity_mode_performs_full_digest(mode: str) -> bool:
    return mode in {
        "legacy_source_full",
        IntegrityVerificationMode.QUALIFICATION_FULL.value,
    }


@dataclass
class LayerwiseLoadTicket:
    segment_id: str
    source_id: str
    started_host_ms: float
    requested_bytes: int
    layer_tensors: Dict[int, Tuple[Any, Any]]
    start_event: Any
    layer_events: Dict[int, Any]
    source_digest_before: str
    source_digest_after: str
    segment_positions: Tuple[int, ...]
    integrity_mode: str = "legacy_source_full"
    expected_artifact_digest: str = ""
    destination_digest: str = ""
    hash_host_ms: float = 0.0
    d2h_hash_host_ms: float = 0.0
    per_request_full_digest_verified: bool = False
    sampled_digest_verified: bool = False
    pinning_copy_bytes: int = 0
    pinning_host_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.segment_id or not self.source_id:
            raise ValueError("load ticket identifiers are required")
        if self.started_host_ms < 0 or self.requested_bytes < 0:
            raise ValueError("invalid load ticket timing or size")
        if set(self.layer_tensors) != set(self.layer_events):
            raise ValueError("every loaded layer requires one ready event")
        if self.hash_host_ms < 0 or self.d2h_hash_host_ms < 0:
            raise ValueError("digest timings must be non-negative")
        if self.pinning_copy_bytes < 0 or self.pinning_host_ms < 0:
            raise ValueError("pinning accounting must be non-negative")
        if self.source_digest_before != self.source_digest_after:
            raise RuntimeError("async staging mutated a canonical Source")
        if self.integrity_mode == IntegrityVerificationMode.QUALIFICATION_FULL.value:
            if not self.expected_artifact_digest:
                raise RuntimeError("qualification requires the Artifact digest")
            if not self.per_request_full_digest_verified:
                raise RuntimeError("qualification mode requires a full destination digest")
            if not (
                self.source_digest_before
                == self.destination_digest
                == self.source_digest_after
            ):
                raise RuntimeError("qualification Source/destination digests differ")
            if (
                self.expected_artifact_digest
                and self.source_digest_before != self.expected_artifact_digest
            ):
                raise RuntimeError("qualification digest differs from Artifact identity")
        if self.integrity_mode == IntegrityVerificationMode.ONLINE_IMMUTABLE.value:
            if self.per_request_full_digest_verified:
                raise RuntimeError("online immutable mode forbids per-request full hashing")
            if any((self.source_digest_before, self.source_digest_after, self.destination_digest)):
                raise RuntimeError("online immutable mode fabricated runtime digests")
            if not self.expected_artifact_digest:
                raise RuntimeError("online immutable mode requires the creation-time digest")
        if self.integrity_mode == IntegrityVerificationMode.ONLINE_SAMPLED.value:
            if not self.expected_artifact_digest:
                raise RuntimeError("online sampled mode requires the Artifact digest")
            if not self.sampled_digest_verified or self.per_request_full_digest_verified:
                raise RuntimeError("online sampled mode requires sample-only verification")
        if tuple(sorted(set(self.segment_positions))) != self.segment_positions:
            raise ValueError("ticket Segment positions must be sorted and unique")

    def layer_ready(self, layer: int) -> bool:
        event = self.layer_events.get(int(layer))
        return bool(event is not None and event.query())

    def layer_ready_gpu_ms(self, layer: int) -> float:
        event = self.layer_events.get(int(layer))
        if event is None or not event.query():
            raise RuntimeError("Source layer has no completed CUDA ready event")
        return float(self.start_event.elapsed_time(event))


@dataclass(frozen=True)
class WinnerRepairCheckMeasurement:
    segment_id: str
    source_variant_id: str
    metric: RepairMetric
    repair_check_completed_depth: int
    first_selective_reuse_layer: int
    absolute_positions: Tuple[int, ...]
    drift_scores: Tuple[float, ...]
    gpu_ms: float
    host_ms: float

    def __post_init__(self) -> None:
        if self.first_selective_reuse_layer != self.repair_check_completed_depth + 1:
            raise ValueError("repair-check consumer layer is off by one")
        if len(self.absolute_positions) != len(self.drift_scores):
            raise ValueError("repair-check drift rows do not cover the Segment")
        if min(self.gpu_ms, self.host_ms, *self.drift_scores) < 0:
            raise ValueError("repair-check timings/drifts must be non-negative")


class TorchLayerwiseSourceLoader:
    """Pinned-CPU to GPU, winner-only, layer-wise asynchronous loader."""

    def __init__(
        self,
        torch_module: Any,
        device: Any = "cuda",
        *,
        integrity_mode: str = "legacy_source_full",
        require_pre_pinned: bool = False,
        sampling_seed: int = 20260726,
        sample_layers: int = 4,
        sample_rows_per_layer: int = 16,
    ) -> None:
        self.torch = torch_module
        self.device = device
        if not self.torch.cuda.is_available():
            raise RuntimeError("real Source loader requires CUDA")
        self.stream = self.torch.cuda.Stream(device=device)
        self.integrity_mode = integrity_mode
        self.require_pre_pinned = bool(require_pre_pinned)
        self.sampling_seed = int(sampling_seed)
        self.sample_layers = int(sample_layers)
        self.sample_rows_per_layer = int(sample_rows_per_layer)
        if integrity_mode not in {
            "legacy_source_full",
            *(mode.value for mode in IntegrityVerificationMode),
        }:
            raise ValueError("unsupported Source integrity mode")
        if min(self.sample_layers, self.sample_rows_per_layer) < 1:
            raise ValueError("integrity sample dimensions must be positive")

    @staticmethod
    def _digest(torch_module: Any, layers: Sequence[Tuple[Any, Any]]) -> str:
        digest = hashlib.sha256()
        for key, value in layers:
            for tensor in (key, value):
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(
                    tensor.detach().contiguous().view(torch_module.uint8)
                    .cpu().numpy().tobytes()
                )
        return digest.hexdigest()

    def _sample_digest(
        self,
        layers: Sequence[Tuple[Any, Any]],
        *,
        sample_key: str,
    ) -> str:
        seed = int.from_bytes(
            hashlib.sha256(
                f"{self.sampling_seed}:{sample_key}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        rng = random.Random(seed)
        selected_layers = sorted(
            rng.sample(range(len(layers)), min(self.sample_layers, len(layers)))
        )
        digest = hashlib.sha256()
        for layer_index in selected_layers:
            for tensor in layers[layer_index]:
                rows = int(tensor.shape[0])
                selected_rows = sorted(
                    rng.sample(
                        range(rows), min(self.sample_rows_per_layer, rows)
                    )
                )
                sampled = tensor.detach()[selected_rows].contiguous()
                digest.update(str(layer_index).encode("ascii"))
                digest.update(str(tuple(sampled.shape)).encode("ascii"))
                digest.update(str(sampled.dtype).encode("ascii"))
                digest.update(
                    sampled.view(self.torch.uint8).cpu().numpy().tobytes()
                )
        return digest.hexdigest()

    def begin(
        self,
        *,
        segment_id: str,
        source_id: str,
        canonical_layers: Sequence[Tuple[Any, Any]],
        segment_positions: Sequence[int],
        expected_artifact_digest: str = "",
        request_id: str = "",
        replica_id: str = "",
    ) -> LayerwiseLoadTicket:
        if not canonical_layers:
            raise ValueError("canonical Source has no KV layers")
        mode = self.integrity_mode
        full_verify = integrity_mode_performs_full_digest(mode)
        hash_started = time.perf_counter()
        before = self._digest(self.torch, canonical_layers) if full_verify else ""
        hash_host_ms = (time.perf_counter() - hash_started) * 1000.0 if full_verify else 0.0
        positions = tuple(int(value) for value in segment_positions)
        if tuple(sorted(set(positions))) != positions:
            raise ValueError("Segment positions must be sorted and unique")
        started = time.perf_counter() * 1000.0
        layer_tensors: Dict[int, Tuple[Any, Any]] = {}
        layer_events: Dict[int, Any] = {}
        requested_bytes = 0
        pinning_copy_bytes = 0
        pinning_host_ms = 0.0
        with self.torch.cuda.stream(self.stream):
            start_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record(self.stream)
            for layer, (key, value) in enumerate(canonical_layers, start=1):
                if key.device.type != "cpu" or value.device.type != "cpu":
                    raise ValueError("canonical staging input must be CPU KV")
                requested_bytes += key.numel() * key.element_size()
                requested_bytes += value.numel() * value.element_size()
                if self.require_pre_pinned and (not key.is_pinned() or not value.is_pinned()):
                    raise RuntimeError(
                        "schema-v7 formal CPU path requires pre-pinned backing/staging"
                    )
                pin_started = time.perf_counter()
                host_key = key if key.is_pinned() else key.pin_memory()
                host_value = value if value.is_pinned() else value.pin_memory()
                if host_key is not key:
                    pinning_copy_bytes += key.numel() * key.element_size()
                if host_value is not value:
                    pinning_copy_bytes += value.numel() * value.element_size()
                pinning_host_ms += (time.perf_counter() - pin_started) * 1000.0
                gpu_key = host_key.to(self.device, non_blocking=True)
                gpu_value = host_value.to(self.device, non_blocking=True)
                event = self.torch.cuda.Event(enable_timing=True)
                event.record(self.stream)
                layer_tensors[layer] = (gpu_key, gpu_value)
                layer_events[layer] = event
        destination_digest = ""
        sampled_verified = False
        d2h_hash_host_ms = 0.0
        if mode == IntegrityVerificationMode.QUALIFICATION_FULL.value:
            for event in layer_events.values():
                event.synchronize()
            destination_started = time.perf_counter()
            destination_digest = self._digest(
                self.torch, tuple(layer_tensors[layer] for layer in sorted(layer_tensors))
            )
            d2h_hash_host_ms = (time.perf_counter() - destination_started) * 1000.0
        elif mode == IntegrityVerificationMode.ONLINE_SAMPLED.value:
            for event in layer_events.values():
                event.synchronize()
            key = f"{request_id}:{replica_id}:{source_id}"
            source_sample = self._sample_digest(canonical_layers, sample_key=key)
            destination_sample = self._sample_digest(
                tuple(layer_tensors[layer] for layer in sorted(layer_tensors)),
                sample_key=key,
            )
            sampled_verified = source_sample == destination_sample
            if not sampled_verified:
                raise RuntimeError("sampled destination KV integrity mismatch")
        hash_started = time.perf_counter()
        after = self._digest(self.torch, canonical_layers) if full_verify else ""
        if full_verify:
            hash_host_ms += (time.perf_counter() - hash_started) * 1000.0
        return LayerwiseLoadTicket(
            segment_id=segment_id,
            source_id=source_id,
            started_host_ms=started,
            requested_bytes=requested_bytes,
            layer_tensors=layer_tensors,
            start_event=start_event,
            layer_events=layer_events,
            source_digest_before=before,
            source_digest_after=after,
            segment_positions=positions,
            integrity_mode=mode,
            expected_artifact_digest=expected_artifact_digest,
            destination_digest=destination_digest,
            hash_host_ms=hash_host_ms,
            d2h_hash_host_ms=d2h_hash_host_ms,
            per_request_full_digest_verified=(
                mode == IntegrityVerificationMode.QUALIFICATION_FULL.value
            ),
            sampled_digest_verified=sampled_verified,
            pinning_copy_bytes=pinning_copy_bytes,
            pinning_host_ms=pinning_host_ms,
        )


@dataclass
class OnlineRequestAudit:
    model_adapter: str
    source_ready_by_segment_layer: Dict[str, Dict[int, bool]] = field(
        default_factory=dict
    )
    source_ready_observed_host_ms_by_segment_layer: Dict[str, Dict[int, float]] = field(
        default_factory=dict
    )
    source_ready_gpu_ms_by_segment_layer: Dict[str, Dict[int, float]] = field(
        default_factory=dict
    )
    actual_boundary_by_segment: Dict[str, int] = field(default_factory=dict)
    a_resume_host_ms_by_segment: Dict[str, float] = field(default_factory=dict)
    post_ready_blocking_ms_by_segment: Dict[str, float] = field(default_factory=dict)
    transferred_bytes_by_segment: Dict[str, int] = field(default_factory=dict)
    wasted_bytes_by_segment: Dict[str, int] = field(default_factory=dict)
    useful_other_request_work_ms: float = 0.0
    load_interference_ms: float = 0.0
    artifact_id_by_segment: Dict[str, str] = field(default_factory=dict)
    replica_id_by_segment: Dict[str, str] = field(default_factory=dict)
    artifact_digest_unchanged_by_segment: Dict[str, bool] = field(
        default_factory=dict
    )
    repair_rounding_policy: str = "floor"
    integrity_mode_by_segment: Dict[str, str] = field(default_factory=dict)
    per_request_full_digest_verified_by_segment: Dict[str, bool] = field(
        default_factory=dict
    )
    hash_host_ms_by_segment: Dict[str, float] = field(default_factory=dict)
    d2h_hash_host_ms_by_segment: Dict[str, float] = field(default_factory=dict)
    pinning_copy_bytes_by_segment: Dict[str, int] = field(default_factory=dict)
    pinning_host_ms_by_segment: Dict[str, float] = field(default_factory=dict)


class CacheBlendV6OnlineEngine:
    """Concrete dual-model data plane used after v6 Source selection.

    The orchestration layer still owns source locking and refined admission.
    This engine owns the real layer state, asynchronous winner loading and
    per-layer readiness observations. It never chooses a default Source.
    """

    patch_mode = "probekv_v6_prefix_hardened_runtime"
    implementation_status = (
        "concrete_engine_hook_complete_requires_a800_qualification"
    )

    def __init__(
        self,
        *,
        inner_model: Any,
        model_spec: ResumableModelSpec,
        source_loader: TorchLayerwiseSourceLoader,
    ) -> None:
        self.model_spec = model_spec
        self.adapter = PinnedCacheBlendResumableAdapter(inner_model, model_spec)
        self.source_loader = source_loader
        self.session: Optional[ProbeKVResumablePrefillSession] = None
        self.tickets: Dict[str, LayerwiseLoadTicket] = {}
        self.audit = OnlineRequestAudit(model_adapter=model_spec.adapter_name)
        self._composite_old_kvs: list[list[Any]] = []
        self._exact_prefix_layers: Tuple[Tuple[Any, Any], ...] = ()

    @staticmethod
    def capabilities() -> Mapping[str, bool]:
        return {
            "async_multisource_loading": True,
            "layer_resumable_prefill": True,
            "scheduler_feedback": True,
            "boundary_conditioned_profiles": True,
            "canonical_sources_read_only": True,
            "absolute_union_repair_mask": True,
            "layer_indexed_union_repair_masks": True,
            "per_segment_staggered_boundaries": True,
            "causal_commit_wait_execution": True,
            "immediate_staggered_closed_loop_execution": True,
            "policy_conditioned_probe_state": True,
            "cuda_event_timing": True,
            "native_prefix_cache_block_metadata": True,
        }

    def begin_prefill(
        self,
        *,
        exact_prefix_layers: Sequence[Tuple[Any, Any]] = (),
        **kwargs: Any,
    ) -> ProbeKVResumablePrefillSession:
        if self.session is not None:
            raise RuntimeError("engine already owns an active request")
        session = ProbeKVResumablePrefillSession(adapter=self.adapter, **kwargs)
        prefix_tokens = session.exact_prefix_tokens
        prefix_layers = tuple(exact_prefix_layers)
        if prefix_tokens:
            if len(prefix_layers) != self.model_spec.num_layers:
                raise ValueError(
                    "native Prefix Cache requires one pre-RoPE shadow per layer"
                )
            for key, value in prefix_layers:
                if key.shape[0] != prefix_tokens or value.shape[0] != prefix_tokens:
                    raise ValueError("exact-prefix shadow row count is inconsistent")
        elif prefix_layers:
            raise ValueError("exact-prefix shadows require an exact Prefix Cache hit")
        self._exact_prefix_layers = prefix_layers
        if prefix_layers:
            request_span = session.absolute_positions[-1] + 1
            prefix = session.exact_prefix_tokens
            for key, value in prefix_layers:
                composite_key = self.source_loader.torch.zeros(
                    (request_span,) + tuple(key.shape[1:]),
                    dtype=key.dtype,
                    device=key.device,
                )
                composite_value = self.source_loader.torch.zeros(
                    (request_span,) + tuple(value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                )
                composite_key[:prefix] = key
                composite_value[:prefix] = value
                self._composite_old_kvs.append(
                    [composite_key, composite_value]
                )
            self.adapter.inner_model.old_kvs = self._composite_old_kvs
            self.adapter.inner_model.cache_fuse_metadata[
                "exact_prefix_tokens"
            ] = prefix
        else:
            self.adapter.inner_model.cache_fuse_metadata[
                "exact_prefix_tokens"
            ] = 0
        session.begin_prefill()
        self.session = session
        return session

    def start_winner_prefetch(
        self,
        *,
        segment_id: str,
        source_id: str,
        canonical_layers: Sequence[Tuple[Any, Any]],
        segment_positions: Sequence[int],
        expected_artifact_digest: str = "",
        request_id: str = "",
        replica_id: str = "",
    ) -> LayerwiseLoadTicket:
        if self.session is None:
            raise RuntimeError("prefetch requires an active request")
        if segment_id in self.tickets:
            raise RuntimeError("only one locked winner may load per Segment")
        ticket = self.source_loader.begin(
            segment_id=segment_id,
            source_id=source_id,
            canonical_layers=canonical_layers,
            segment_positions=segment_positions,
            expected_artifact_digest=expected_artifact_digest,
            request_id=request_id,
            replica_id=replica_id,
        )
        if len(ticket.layer_tensors) != self.model_spec.num_layers:
            raise ValueError("canonical Source layer count differs from model")
        if len(ticket.segment_positions) != canonical_layers[0][0].shape[0]:
            raise ValueError("canonical Source rows differ from Segment length")
        if not self._composite_old_kvs:
            request_span = self.session.absolute_positions[-1] + 1
            for layer, (key, value) in ticket.layer_tensors.items():
                composite_key = self.source_loader.torch.zeros(
                    (request_span,) + tuple(key.shape[1:]),
                    dtype=key.dtype, device=key.device)
                composite_value = self.source_loader.torch.zeros(
                    (request_span,) + tuple(value.shape[1:]),
                    dtype=value.dtype, device=value.device)
                if self._exact_prefix_layers:
                    prefix_key, prefix_value = self._exact_prefix_layers[layer - 1]
                    for observed, expected in (
                        (prefix_key, composite_key),
                        (prefix_value, composite_value),
                    ):
                        if (
                            tuple(observed.shape[1:]) != tuple(expected.shape[1:])
                            or observed.dtype != expected.dtype
                            or observed.device != expected.device
                        ):
                            raise ValueError(
                                "exact-prefix shadow is incompatible with Source KV"
                            )
                    prefix = self.session.exact_prefix_tokens
                    composite_key[:prefix] = prefix_key
                    composite_value[:prefix] = prefix_value
                self._composite_old_kvs.append([composite_key, composite_value])
            self.adapter.inner_model.old_kvs = self._composite_old_kvs
        else:
            for layer, (key, value) in ticket.layer_tensors.items():
                expected_key, expected_value = self._composite_old_kvs[layer - 1]
                if (
                    tuple(key.shape[1:]) != tuple(expected_key.shape[1:])
                    or tuple(value.shape[1:]) != tuple(expected_value.shape[1:])
                    or key.dtype != expected_key.dtype
                    or value.dtype != expected_value.dtype
                    or key.device != expected_key.device
                    or value.device != expected_value.device
                ):
                    raise ValueError("locked Sources have incompatible KV geometry")
        self.tickets[segment_id] = ticket
        self.session.register_source_handle(segment_id, source_id, ticket)
        self.audit.transferred_bytes_by_segment[segment_id] = ticket.requested_bytes
        self.audit.integrity_mode_by_segment[segment_id] = ticket.integrity_mode
        self.audit.per_request_full_digest_verified_by_segment[segment_id] = (
            ticket.per_request_full_digest_verified
        )
        self.audit.hash_host_ms_by_segment[segment_id] = ticket.hash_host_ms
        self.audit.d2h_hash_host_ms_by_segment[segment_id] = ticket.d2h_hash_host_ms
        self.audit.pinning_copy_bytes_by_segment[segment_id] = ticket.pinning_copy_bytes
        self.audit.pinning_host_ms_by_segment[segment_id] = ticket.pinning_host_ms
        return ticket

    def _install_ready_source_rows(self, layer: int) -> None:
        for segment_id in self.session.commits if self.session else ():
            ticket = self.tickets[segment_id]
            if not ticket.layer_ready(layer):
                raise RuntimeError(
                    "committed Source layer is not ready; scheduler must wait"
                )
            key, value = ticket.layer_tensors[layer]
            positions = list(ticket.segment_positions)
            self._composite_old_kvs[layer - 1][0][positions] = key
            self._composite_old_kvs[layer - 1][1][positions] = value

    def ready_for_boundary(self, segment_id: str, boundary: int) -> bool:
        ticket = self.tickets.get(segment_id)
        if ticket is None:
            return False
        ready = ticket.layer_ready(boundary)
        self.audit.source_ready_by_segment_layer.setdefault(segment_id, {})[
            boundary
        ] = ready
        if ready:
            self.audit.source_ready_observed_host_ms_by_segment_layer.setdefault(
                segment_id, {}
            ).setdefault(boundary, time.perf_counter() * 1000.0)
            self.audit.source_ready_gpu_ms_by_segment_layer.setdefault(
                segment_id, {}
            ).setdefault(boundary, ticket.layer_ready_gpu_ms(boundary))
        return ready

    def commit_ready_segment(
        self,
        *,
        segment_id: str,
        boundary: int,
        segment_positions: Sequence[int],
        repair_positions: Sequence[int],
        scheduler_boundary: int,
    ) -> int:
        if self.session is None:
            raise RuntimeError("commit requires an active request")
        ticket = self.tickets.get(segment_id)
        if ticket is None:
            raise RuntimeError("selector abstention cannot reach reuse commit")
        actual = max(int(boundary), int(scheduler_boundary))
        if actual != self.session.current_layer + 1:
            raise ValueError("actual boundary must be the next executable layer")
        if not self.ready_for_boundary(segment_id, actual):
            raise RuntimeError("Source layer is not actually ready")
        self.session.commit_segment_reuse(
            segment_id=segment_id,
            source_id=ticket.source_id,
            boundary=actual,
            segment_positions=segment_positions,
            repair_positions=repair_positions,
        )
        self.audit.actual_boundary_by_segment[segment_id] = actual
        resumed = time.perf_counter() * 1000.0
        self.audit.a_resume_host_ms_by_segment[segment_id] = resumed
        ready_host = self.audit.source_ready_observed_host_ms_by_segment_layer[
            segment_id
        ][actual]
        self.audit.post_ready_blocking_ms_by_segment[segment_id] = max(
            0.0, resumed - ready_host
        )
        return actual

    def record_scheduler_feedback(
        self,
        *,
        useful_other_request_work_ms: float,
        load_interference_ms: float,
    ) -> None:
        if useful_other_request_work_ms < 0 or load_interference_ms < 0:
            raise ValueError("scheduler timings must be non-negative")
        self.audit.useful_other_request_work_ms += useful_other_request_work_ms
        self.audit.load_interference_ms += load_interference_ms

    def advance_to_layer(self, layer: int) -> None:
        if self.session is None:
            raise RuntimeError("advance requires an active request")
        while self.session.current_layer < layer:
            next_layer = self.session.current_layer + 1
            self._install_ready_source_rows(next_layer)
            self.session.advance_to_layer(next_layer)

    def finish_prefill(self) -> Any:
        if self.session is None:
            raise RuntimeError("finish requires an active request")
        return self.session.finish_prefill()

    def mark_refined_rejection(self, segment_id: str) -> None:
        ticket = self.tickets.get(segment_id)
        if ticket is not None:
            self.audit.wasted_bytes_by_segment[segment_id] = ticket.requested_bytes


class CacheBlendV7OnlineEngine(CacheBlendV6OnlineEngine):
    """v7 single-Artifact runtime over the pinned CacheBlend data plane."""

    patch_mode = "probekv_v7_single_artifact_runtime"
    implementation_status = "requires_dual_model_a800_qualification"

    @staticmethod
    def capabilities() -> Mapping[str, bool]:
        result = dict(CacheBlendV6OnlineEngine.capabilities())
        result.update(
            {
                "single_lossless_bf16_artifact": True,
                "multiple_physical_replicas": True,
                "conservative_repair_rounding": True,
                "same_source_replica_replan": True,
            }
        )
        return result

    def begin_prefill(
        self,
        *,
        exact_prefix_layers: Sequence[Tuple[Any, Any]] = (),
        **kwargs: Any,
    ) -> ProbeKVResumablePrefillSession:
        session = super().begin_prefill(
            exact_prefix_layers=exact_prefix_layers, **kwargs
        )
        self.adapter.inner_model.cache_fuse_metadata[
            "repair_rounding_policy"
        ] = "ceil"
        self.audit.repair_rounding_policy = "ceil"
        return session

    def start_artifact_replica_prefetch(
        self,
        *,
        segment_id: str,
        source_variant_id: str,
        artifact: CanonicalKVArtifact,
        replica: PhysicalReplica,
        canonical_layers: Sequence[Tuple[Any, Any]],
        segment_positions: Sequence[int],
    ) -> LayerwiseLoadTicket:
        if artifact.source_variant_id != source_variant_id:
            raise ValueError("Artifact belongs to another Source Variant")
        if replica.artifact_id != artifact.artifact_id:
            raise ValueError("Replica belongs to another Artifact")
        if (
            artifact.dtype != "bfloat16"
            or artifact.k_semantics != "pre_rope"
            or artifact.v_semantics != "raw"
        ):
            raise ValueError("v7 requires a lossless BF16 pre-RoPE Artifact")
        if replica.logical_digest != artifact.artifact_logical_digest:
            raise RuntimeError("Replica logical digest differs from Artifact")
        ticket = super().start_winner_prefetch(
            segment_id=segment_id,
            source_id=source_variant_id,
            canonical_layers=canonical_layers,
            segment_positions=segment_positions,
        )
        if ticket.source_digest_before != artifact.artifact_logical_digest:
            raise RuntimeError("loaded KV differs from canonical Artifact digest")
        self.audit.artifact_id_by_segment[segment_id] = artifact.artifact_id
        self.audit.replica_id_by_segment[segment_id] = replica.replica_id
        self.audit.artifact_digest_unchanged_by_segment[segment_id] = (
            ticket.source_digest_before
            == ticket.source_digest_after
            == artifact.artifact_logical_digest
        )
        return ticket


class CacheBlendV8OnlineEngine(CacheBlendV7OnlineEngine):
    """v8 data-plane contract; Source ranking is handled by K-only states."""

    patch_mode = "probekv_v8_training_free_residual_k"
    implementation_status = "requires_dual_model_a800_profile_freeze_and_qualification"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.request_attributed_full_kv_bytes_transferred_for_selection = 0
        self.request_attributed_nonwinner_full_kv_bytes_transferred = 0
        self.request_attributed_full_kv_prefetch_before_source_freeze = 0
        self.frozen_source_by_segment: Dict[str, str] = {}

    @staticmethod
    def capabilities() -> Mapping[str, bool]:
        result = dict(CacheBlendV7OnlineEngine.capabilities())
        result.update(
            {
                "completed_depth_k_observation": True,
                "training_free_residual_k_selection": True,
                "selection_state_k_only": True,
                "selection_state_separate_backing": True,
                "selection_state_scratch_bounded": True,
                "winner_only_prefetch": True,
                "predicted_and_refined_joint_planners": True,
                "request_level_selection_budget_ledger": True,
                "incremental_predicted_gate": True,
                "precommit_timeout_dense_fallback": True,
                "irreversible_reuse_commit": True,
                "atomic_logical_source_lease": True,
                "atomic_replica_batch_lease": True,
                "fixed_repair_ratio_015": True,
            }
        )
        return result

    def freeze_source(self, segment_id: str, source_variant_id: str) -> None:
        previous = self.frozen_source_by_segment.get(segment_id)
        if previous is not None and previous != source_variant_id:
            raise RuntimeError("v8 Source freeze forbids Source substitution")
        self.frozen_source_by_segment[segment_id] = source_variant_id

    def start_artifact_replica_prefetch(
        self,
        *,
        segment_id: str,
        source_variant_id: str,
        artifact: CanonicalKVArtifact,
        replica: PhysicalReplica,
        canonical_layers: Sequence[Tuple[Any, Any]],
        segment_positions: Sequence[int],
    ) -> LayerwiseLoadTicket:
        frozen = self.frozen_source_by_segment.get(segment_id)
        if frozen != source_variant_id:
            self.request_attributed_full_kv_prefetch_before_source_freeze += 1
            if frozen is not None:
                self.request_attributed_nonwinner_full_kv_bytes_transferred += int(
                    replica.size_bytes
                )
            raise RuntimeError(
                "v8 full-KV prefetch is legal only for the frozen winner"
            )
        return super().start_artifact_replica_prefetch(
            segment_id=segment_id,
            source_variant_id=source_variant_id,
            artifact=artifact,
            replica=replica,
            canonical_layers=canonical_layers,
            segment_positions=segment_positions,
        )

    def assert_selection_transfer_invariants(self) -> None:
        if any(
            (
                self.request_attributed_full_kv_bytes_transferred_for_selection,
                self.request_attributed_nonwinner_full_kv_bytes_transferred,
                self.request_attributed_full_kv_prefetch_before_source_freeze,
            )
        ):
            raise RuntimeError("v8 Source selection triggered forbidden full-KV transfer")


class CacheBlendV8Schema7OnlineEngine(CacheBlendV8OnlineEngine):
    """Winner-specific gradual-repair data plane for schema-v7.

    Source selection still reads K-only SelectionState objects.  This class is
    entered only after Source freeze and uses the configured integrity mode for
    the winner's single canonical Artifact.
    """

    patch_mode = "probekv_v8_winner_gradual_streaming"
    implementation_status = "no_gpu_complete_requires_schema7_a800_sentinel"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if (
            self.source_loader.integrity_mode
            in {
                IntegrityVerificationMode.ONLINE_IMMUTABLE.value,
                IntegrityVerificationMode.ONLINE_SAMPLED.value,
            }
            and not self.source_loader.require_pre_pinned
        ):
            raise RuntimeError(
                "schema-v7 online CPU transfer requires pre-pinned backing/staging"
            )

    @staticmethod
    def capabilities() -> Mapping[str, bool]:
        result = dict(CacheBlendV8OnlineEngine.capabilities())
        result.update(
            {
                "winner_specific_k_v_kv_repair_metrics": True,
                "gradual_no_reentry_support": True,
                "load_recompute_aware_repair_candidates": True,
                "final_commit_admission": True,
                "qualification_destination_digest": True,
                "online_immutable_no_full_digest": True,
                "online_sampled_integrity": True,
                "layerwise_transfer_paths": True,
            }
        )
        return result

    def start_artifact_replica_prefetch(
        self,
        *,
        segment_id: str,
        source_variant_id: str,
        artifact: CanonicalKVArtifact,
        replica: PhysicalReplica,
        canonical_layers: Sequence[Tuple[Any, Any]],
        segment_positions: Sequence[int],
        request_id: str = "",
    ) -> LayerwiseLoadTicket:
        frozen = self.frozen_source_by_segment.get(segment_id)
        if frozen != source_variant_id:
            self.request_attributed_full_kv_prefetch_before_source_freeze += 1
            if frozen is not None:
                self.request_attributed_nonwinner_full_kv_bytes_transferred += int(
                    replica.size_bytes
                )
            raise RuntimeError("schema-v7 prefetch requires the frozen winner")
        if artifact.source_variant_id != source_variant_id:
            raise ValueError("Artifact belongs to another Source Variant")
        if replica.artifact_id != artifact.artifact_id:
            raise ValueError("Replica belongs to another Artifact")
        if (
            artifact.dtype != "bfloat16"
            or artifact.k_semantics != "pre_rope"
            or artifact.v_semantics != "raw"
            or replica.logical_digest != artifact.artifact_logical_digest
        ):
            raise RuntimeError("schema-v7 requires one compatible lossless BF16 Artifact")
        ticket = CacheBlendV6OnlineEngine.start_winner_prefetch(
            self,
            segment_id=segment_id,
            source_id=source_variant_id,
            canonical_layers=canonical_layers,
            segment_positions=segment_positions,
            expected_artifact_digest=artifact.artifact_logical_digest,
            request_id=request_id,
            replica_id=replica.replica_id,
        )
        self.audit.artifact_id_by_segment[segment_id] = artifact.artifact_id
        self.audit.replica_id_by_segment[segment_id] = replica.replica_id
        self.audit.artifact_digest_unchanged_by_segment[segment_id] = (
            ticket.integrity_mode == IntegrityVerificationMode.ONLINE_IMMUTABLE.value
            or ticket.sampled_digest_verified
            or ticket.per_request_full_digest_verified
        )
        return ticket

    def shrink_gradual_repair_support(
        self,
        *,
        segment_id: str,
        consumer_layer: int,
        repair_positions: Sequence[int],
    ) -> None:
        if self.session is None:
            raise RuntimeError("gradual repair requires an active request")
        self.session.shrink_segment_repair_support(
            segment_id=segment_id,
            consumer_layer=consumer_layer,
            repair_positions=repair_positions,
        )

    def observe_winner_repair_check(
        self,
        *,
        segment_id: str,
        metric: RepairMetric,
        repair_check_completed_depth: Optional[int] = None,
        support_positions: Optional[Sequence[int]] = None,
        epsilon: float = 1e-12,
    ) -> WinnerRepairCheckMeasurement:
        if self.session is None:
            raise RuntimeError("winner repair check requires an active request")
        ticket = self.tickets.get(segment_id)
        if ticket is None:
            raise RuntimeError("winner repair check requires a prepared frozen Source")
        depth = (
            self.session.current_layer
            if repair_check_completed_depth is None
            else int(repair_check_completed_depth)
        )
        consumer = depth + 1
        if not ticket.layer_ready(consumer):
            raise RuntimeError("winner Source is not ready for the repair-check consumer")
        started_host = time.perf_counter()
        start_event = self.source_loader.torch.cuda.Event(enable_timing=True)
        end_event = self.source_loader.torch.cuda.Event(enable_timing=True)
        start_event.record()
        current_k, current_v = self.session.observe_repair_check_pre_rope_kv(depth)
        selected_positions = tuple(
            ticket.segment_positions
            if support_positions is None
            else sorted(set(int(value) for value in support_positions))
        )
        if not selected_positions or not set(selected_positions).issubset(
            ticket.segment_positions
        ):
            raise ValueError("repair-check support must lie inside the Segment")
        active_row = {
            position: index for index, position in enumerate(self.session.active_positions)
        }
        try:
            current_rows = [active_row[position] for position in selected_positions]
        except KeyError as error:
            raise RuntimeError("repair-check Segment contains an inactive row") from error
        source_row = {
            position: index for index, position in enumerate(ticket.segment_positions)
        }
        source_rows = [source_row[position] for position in selected_positions]
        current_k = current_k[current_rows]
        current_v = current_v[current_rows]
        source_k, source_v = ticket.layer_tensors[consumer]
        source_k = source_k[source_rows]
        source_v = source_v[source_rows]

        def normalized(current: Any, source: Any) -> Any:
            rows = current.shape[0]
            left = current.float().reshape(rows, -1)
            right = source.float().reshape(rows, -1)
            numerator = self.source_loader.torch.linalg.vector_norm(left - right, dim=1)
            denominator = self.source_loader.torch.clamp(
                self.source_loader.torch.linalg.vector_norm(left, dim=1),
                min=epsilon,
            )
            return numerator / denominator

        k_drift = normalized(current_k, source_k)
        v_drift = normalized(current_v, source_v)
        metric = RepairMetric(metric)
        if metric is RepairMetric.WINNER_K_ONLY:
            drift = k_drift
        elif metric is RepairMetric.WINNER_V_ONLY:
            drift = v_drift
        else:
            drift = self.source_loader.torch.sqrt(k_drift * k_drift + v_drift * v_drift)
        end_event.record()
        end_event.synchronize()
        values = tuple(float(value) for value in drift.detach().cpu().tolist())
        return WinnerRepairCheckMeasurement(
            segment_id,
            ticket.source_id,
            metric,
            depth,
            consumer,
            selected_positions,
            values,
            float(start_event.elapsed_time(end_event)),
            (time.perf_counter() - started_host) * 1000.0,
        )


class CacheBlendV8Schema8OnlineEngine(CacheBlendV8Schema7OnlineEngine):
    """Schema-v8 data plane with dense barrier before execution-visible reuse."""

    patch_mode = "probekv_v8_gradual_barrier_tiered_lru"
    implementation_status = "no_gpu_complete_requires_schema8_a800_sentinel"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.selection_barrier = None
        self.preparation_admitted_segment_ids: set[str] = set()
        self.detached_preparation_segment_ids: set[str] = set()
        self.final_commit_admitted_segment_ids: set[str] = set()
        self.repair_ratio_plan = None
        self.r1_endpoint_segment_ids: set[str] = set()
        self.r0_endpoint_segment_ids: set[str] = set()
        self.measurement_only_admission_segment_ids: set[str] = set()
        self.repair_metric = RepairMetric.WINNER_V_ONLY
        self.adaptive_ratio_decision_by_layer: Dict[int, Any] = {}

    @staticmethod
    def capabilities() -> Mapping[str, bool]:
        result = dict(CacheBlendV8Schema7OnlineEngine.capabilities())
        result.update(
            {
                "dense_d1_d2_selection_barrier": True,
                "d1_detached_winner_prefetch": True,
                "gate1_optimistic_marginal_feasibility": True,
                "final_commit_request_joint_timeline": True,
                "cpu_preferred_single_backing": True,
                "cpu_and_ssd_lru": True,
                "repair_ratio_scope_explicit": True,
                "request_level_joint_adaptive_ratio": True,
                "runtime_joint_ratio_plan_execution": True,
                "legacy_ac_policy_removed_from_main": True,
            }
        )
        return result

    def configure_dense_selection_barrier(self, decision: Any) -> None:
        from .v8_schema8_contracts import DenseSelectionBarrierDecision

        if not isinstance(decision, DenseSelectionBarrierDecision):
            raise TypeError("schema-v8 engine requires a barrier decision")
        if self.selection_barrier is not None and self.selection_barrier != decision:
            raise RuntimeError("selection barrier is immutable")
        frozen = set(self.frozen_source_by_segment)
        if frozen != set(decision.reuse_segment_ids):
            raise RuntimeError("barrier reuse set differs from frozen winners")
        if set(self.tickets) - self.detached_preparation_segment_ids:
            raise RuntimeError("pre-barrier transfer was not detached winner preparation")
        if self.detached_preparation_segment_ids - set(decision.reuse_segment_ids):
            raise RuntimeError("detached preparation differs from final frozen winners")
        self.selection_barrier = decision

    def admit_detached_preparation(self, segment_ids: Sequence[str]) -> None:
        """Resource-admit already frozen d1 winners before barrier closure."""

        if self.selection_barrier is not None:
            raise RuntimeError("detached preparation must precede barrier closure")
        requested = set(str(value) for value in segment_ids)
        if requested - set(self.frozen_source_by_segment):
            raise RuntimeError("detached preparation requires frozen winner identity")
        self.detached_preparation_segment_ids.update(requested)
        self.preparation_admitted_segment_ids.update(requested)

    def admit_preparation(self, segment_ids: Sequence[str]) -> None:
        if self.selection_barrier is None:
            raise RuntimeError("PreparationAdmission requires a closed barrier")
        requested = set(str(value) for value in segment_ids)
        if requested - set(self.selection_barrier.reuse_segment_ids):
            raise RuntimeError("dense Segment cannot enter winner preparation")
        self.preparation_admitted_segment_ids.update(requested)

    def install_repair_ratio_plan(self, plan: Any) -> None:
        from .v8_schema8_repair import MultiSegmentRepairRatioPlan

        if not isinstance(plan, MultiSegmentRepairRatioPlan):
            raise TypeError("schema-v8 engine requires a validated ratio plan")
        if self.selection_barrier is None:
            raise RuntimeError("repair ratio plan requires a closed barrier")
        plan_segments = {row.segment_id for row in plan.rows}
        if plan_segments != set(self.selection_barrier.reuse_segment_ids):
            raise RuntimeError("repair ratio plan must cover every reusable Segment")
        self.repair_ratio_plan = plan
        self.adaptive_ratio_decision_by_layer = {
            decision.layer_1based: decision
            for decision in (
                tuple(plan.adaptive_joint_decisions)
                + tuple(plan.uniform_io_decisions)
            )
        }

    def configure_repair_metric(self, metric: RepairMetric) -> None:
        if self.session is not None and self.session.commits:
            raise RuntimeError("repair metric must be frozen before reuse commit")
        self.repair_metric = RepairMetric(metric)

    @staticmethod
    def _rank_repair_positions(
        measurement: WinnerRepairCheckMeasurement,
        desired_count: int,
    ) -> Tuple[int, ...]:
        if not 0 <= desired_count <= len(measurement.absolute_positions):
            raise ValueError("repair support count is outside the measured Segment")
        ranked = sorted(
            zip(measurement.absolute_positions, measurement.drift_scores),
            key=lambda row: (-float(row[1]), int(row[0])),
        )
        return tuple(sorted(int(position) for position, _ in ranked[:desired_count]))

    def measured_repair_positions(
        self,
        *,
        segment_id: str,
        repair_check_completed_depth: int,
        desired_count: int,
        support_positions: Optional[Sequence[int]] = None,
    ) -> Tuple[int, ...]:
        measurement = self.observe_winner_repair_check(
            segment_id=segment_id,
            metric=self.repair_metric,
            repair_check_completed_depth=repair_check_completed_depth,
            support_positions=support_positions,
        )
        return self._rank_repair_positions(measurement, desired_count)

    def _apply_planned_gradual_support(self, consumer_layer: int) -> None:
        """Apply one profile-bound joint ratio vector before its layer.

        The explicit plan may be a legacy per-Segment vector or the main
        request/layer-uniform I/O decision.  In both cases the measured pool
        is restricted to the previous support, preserving CacheBlend-style
        no re-entry.
        """

        if self.session is None or self.repair_ratio_plan is None:
            return
        ratios = self.repair_ratio_plan.ratios_for_layer(consumer_layer)
        if not ratios:
            return
        for segment_id in sorted(self.session.commits):
            commit = self.session.commits[segment_id]
            if consumer_layer <= commit.boundary or segment_id not in ratios:
                continue
            previous = self.session.current_repair_positions_by_segment[segment_id]
            total_rows = len(self.tickets[segment_id].segment_positions)
            desired = min(total_rows, int(math.ceil(ratios[segment_id] * total_rows)))
            if desired > len(previous):
                raise RuntimeError("joint ratio plan attempted repair-token re-entry")
            if desired == len(previous):
                continue
            updated = self.measured_repair_positions(
                segment_id=segment_id,
                repair_check_completed_depth=consumer_layer - 1,
                desired_count=desired,
                support_positions=previous,
            )
            self.shrink_gradual_repair_support(
                segment_id=segment_id,
                consumer_layer=consumer_layer,
                repair_positions=updated,
            )

    def advance_to_layer(self, layer: int) -> None:
        if self.session is None:
            raise RuntimeError("advance requires an active request")
        while self.session.current_layer < layer:
            consumer = self.session.current_layer + 1
            self._apply_planned_gradual_support(consumer)
            super().advance_to_layer(consumer)

    def authorize_final_commit(
        self,
        accepted_segment_ids: Sequence[str],
        *,
        r1_endpoint: bool = False,
        r0_endpoint: bool = False,
        measurement_only: bool = False,
        joint_admission_decision: Any | None = None,
    ) -> None:
        if self.selection_barrier is None:
            raise RuntimeError("FinalCommitAdmission requires a closed barrier")
        accepted = set(str(value) for value in accepted_segment_ids)
        if r0_endpoint and r1_endpoint:
            raise ValueError("repair endpoint cannot be both r=0 and r=1")
        if not r1_endpoint and not r0_endpoint and not measurement_only:
            from .v8_schema7_contracts import FinalCommitDecision

            if not isinstance(joint_admission_decision, FinalCommitDecision):
                raise RuntimeError("online FinalCommitAdmission requires joint decision evidence")
            if accepted != set(joint_admission_decision.accepted_ready_segment_ids):
                raise RuntimeError("engine admission differs from Joint Planner subset")
            if (
                joint_admission_decision.request_total_ms
                > 0.8 * joint_admission_decision.dense_reference_total_ms + 1e-12
            ):
                raise RuntimeError("Joint Planner decision exceeds final gamma")
        if accepted - self.preparation_admitted_segment_ids:
            raise RuntimeError("FinalCommitAdmission cannot bypass preparation")
        if not r1_endpoint and not r0_endpoint and self.repair_ratio_plan is None:
            raise RuntimeError("online FinalCommitAdmission requires a repair ratio plan")
        self.final_commit_admitted_segment_ids.update(accepted)
        if r1_endpoint:
            self.r1_endpoint_segment_ids.update(accepted)
        if r0_endpoint:
            self.r0_endpoint_segment_ids.update(accepted)
        if measurement_only:
            self.measurement_only_admission_segment_ids.update(accepted)

    def start_artifact_replica_prefetch(
        self,
        *,
        segment_id: str,
        transfer_authorization: Any = None,
        **kwargs: Any,
    ) -> LayerwiseLoadTicket:
        if segment_id not in self.preparation_admitted_segment_ids:
            raise RuntimeError("schema-v8 full-KV transfer lacks PreparationAdmission")
        if transfer_authorization is None:
            raise RuntimeError("schema-v8 full-KV transfer lacks lease/HBM authorization")
        transfer_authorization.assert_valid_for(
            source_variant_id=str(kwargs["source_variant_id"]),
            artifact_id=kwargs["artifact"].artifact_id,
            replica_id=kwargs["replica"].replica_id,
        )
        return super().start_artifact_replica_prefetch(segment_id=segment_id, **kwargs)

    def commit_ready_segment(
        self,
        *,
        segment_id: str,
        boundary: int,
        segment_positions: Sequence[int],
        repair_positions: Sequence[int],
        scheduler_boundary: int,
    ) -> int:
        if segment_id not in self.final_commit_admitted_segment_ids:
            raise RuntimeError("selective reuse before FinalCommitAdmission is forbidden")
        positions = tuple(int(value) for value in segment_positions)
        repair = tuple(int(value) for value in repair_positions)
        if segment_id in self.r1_endpoint_segment_ids:
            expected_count = len(positions)
        elif segment_id in self.r0_endpoint_segment_ids:
            expected_count = 0
        else:
            ratios = self.repair_ratio_plan.ratios_for_layer(int(boundary))
            if segment_id not in ratios:
                raise RuntimeError("repair ratio plan misses the actual reuse layer")
            expected_count = min(
                len(positions),
                int(math.ceil(ratios[segment_id] * len(positions))),
            )
        if len(repair) != expected_count:
            raise RuntimeError("repair mask count differs from the admitted ratio")
        if len(set(repair)) != len(repair) or not set(repair).issubset(positions):
            raise ValueError("repair mask must be unique and inside its Segment")
        if self.selection_barrier is None or boundary < self.selection_barrier.first_selective_reuse_layer:
            raise RuntimeError("reuse boundary precedes the dense selection barrier")
        return super().commit_ready_segment(
            segment_id=segment_id,
            boundary=boundary,
            segment_positions=positions,
            repair_positions=repair,
            scheduler_boundary=scheduler_boundary,
        )
