from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .model_adapters import PinnedCacheBlendResumableAdapter, ResumableModelSpec
from .resumable_prefill import ProbeKVResumablePrefillSession
from .v7_contracts import CanonicalKVArtifact, PhysicalReplica


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

    def __post_init__(self) -> None:
        if not self.segment_id or not self.source_id:
            raise ValueError("load ticket identifiers are required")
        if self.started_host_ms < 0 or self.requested_bytes < 0:
            raise ValueError("invalid load ticket timing or size")
        if set(self.layer_tensors) != set(self.layer_events):
            raise ValueError("every loaded layer requires one ready event")
        if self.source_digest_before != self.source_digest_after:
            raise RuntimeError("async staging mutated a canonical Source")
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


class TorchLayerwiseSourceLoader:
    """Pinned-CPU to GPU, winner-only, layer-wise asynchronous loader."""

    def __init__(self, torch_module: Any, device: Any = "cuda") -> None:
        self.torch = torch_module
        self.device = device
        if not self.torch.cuda.is_available():
            raise RuntimeError("real Source loader requires CUDA")
        self.stream = self.torch.cuda.Stream(device=device)

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

    def begin(
        self,
        *,
        segment_id: str,
        source_id: str,
        canonical_layers: Sequence[Tuple[Any, Any]],
        segment_positions: Sequence[int],
    ) -> LayerwiseLoadTicket:
        if not canonical_layers:
            raise ValueError("canonical Source has no KV layers")
        before = self._digest(self.torch, canonical_layers)
        positions = tuple(int(value) for value in segment_positions)
        if tuple(sorted(set(positions))) != positions:
            raise ValueError("Segment positions must be sorted and unique")
        started = time.perf_counter() * 1000.0
        layer_tensors: Dict[int, Tuple[Any, Any]] = {}
        layer_events: Dict[int, Any] = {}
        requested_bytes = 0
        with self.torch.cuda.stream(self.stream):
            start_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record(self.stream)
            for layer, (key, value) in enumerate(canonical_layers, start=1):
                if key.device.type != "cpu" or value.device.type != "cpu":
                    raise ValueError("canonical staging input must be CPU KV")
                requested_bytes += key.numel() * key.element_size()
                requested_bytes += value.numel() * value.element_size()
                host_key = key if key.is_pinned() else key.pin_memory()
                host_value = value if value.is_pinned() else value.pin_memory()
                gpu_key = host_key.to(self.device, non_blocking=True)
                gpu_value = host_value.to(self.device, non_blocking=True)
                event = self.torch.cuda.Event(enable_timing=True)
                event.record(self.stream)
                layer_tensors[layer] = (gpu_key, gpu_value)
                layer_events[layer] = event
        after = self._digest(self.torch, canonical_layers)
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
