from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class LayerAdvanceResult:
    """One real Transformer-layer step returned by a model adapter."""

    hidden_states: Any
    residual: Any
    working_kv: Any
    gpu_ms: float = 0.0
    host_ms: float = 0.0
    union_mask_digest: str = ""
    runtime_debug: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gpu_ms < 0 or self.host_ms < 0:
            raise ValueError("layer timings must be non-negative")


class ResumablePrefillAdapter(Protocol):
    """Model-specific calls provided by the pinned CacheBlend tree."""

    adapter_name: str
    total_layers: int

    def begin_prefill(
        self,
        *,
        token_ids: Sequence[int],
        absolute_positions: Sequence[int],
        attention_metadata: Any,
        working_kv: Any,
        model_signature: str,
    ) -> Tuple[Any, Any]:
        ...

    def advance_layer(
        self,
        *,
        layer: int,
        hidden_states: Any,
        residual: Any,
        active_positions: Tuple[int, ...],
        target_active_positions: Tuple[int, ...],
        attention_metadata: Any,
        working_kv: Any,
        source_handles: Mapping[str, Any],
        reuse_commit: bool,
    ) -> LayerAdvanceResult:
        ...

    def finish_prefill(
        self,
        *,
        hidden_states: Any,
        residual: Any,
        active_positions: Tuple[int, ...],
        attention_metadata: Any,
        working_kv: Any,
    ) -> Any:
        ...

    def observe_pre_rope_k(
        self,
        *,
        completed_depth: int,
        hidden_states: Any,
        residual: Any,
        active_positions: Tuple[int, ...],
    ) -> Any:
        ...

    def observe_pre_rope_kv(
        self,
        *,
        completed_depth: int,
        hidden_states: Any,
        residual: Any,
        active_positions: Tuple[int, ...],
    ) -> Tuple[Any, Any]:
        ...


@dataclass(frozen=True)
class SegmentReuseCommit:
    segment_id: str
    boundary: int
    repair_positions: Tuple[int, ...]
    source_id: str

    def __post_init__(self) -> None:
        if not self.segment_id or not self.source_id:
            raise ValueError("reuse commit requires Segment and Source IDs")
        if self.boundary < 1:
            raise ValueError("reuse boundary is 1-based")
        if tuple(sorted(set(self.repair_positions))) != self.repair_positions:
            raise ValueError("repair positions must be sorted and unique")


@dataclass
class ProbeKVResumablePrefillSession:
    """Model-independent state machine for layer-resumable prefill.

    The session owns request working state only. Canonical Source tensors are
    opaque, read-only handles supplied to the model adapter. A commit is made
    immediately before its boundary layer, and the active query-token set may
    only shrink. This prevents a later layer from reintroducing a token whose
    hidden state was discarded by an earlier Segment reuse decision.
    """

    adapter: ResumablePrefillAdapter
    model_signature: str
    token_ids: Tuple[int, ...]
    attention_metadata: Any
    working_kv: Any
    absolute_positions: Tuple[int, ...] = ()
    exact_prefix_tokens: int = 0
    current_layer: int = 0
    hidden_states: Any = None
    residual: Any = None
    active_positions: Tuple[int, ...] = ()
    source_handles: Dict[str, Any] = field(default_factory=dict)
    commits: Dict[str, SegmentReuseCommit] = field(default_factory=dict)
    current_repair_positions_by_segment: Dict[str, Tuple[int, ...]] = field(
        default_factory=dict
    )
    layer_audit: list[Dict[str, Any]] = field(default_factory=list)
    _pending_target_positions: Optional[Tuple[int, ...]] = None
    _pending_reuse_commit: bool = False
    _started: bool = False
    _finished: bool = False

    def __post_init__(self) -> None:
        if not self.model_signature:
            raise ValueError("model_signature is required")
        if not self.token_ids:
            raise ValueError("prefill requires at least one token")
        if not self.absolute_positions:
            self.absolute_positions = tuple(range(len(self.token_ids)))
        if len(self.absolute_positions) != len(self.token_ids):
            raise ValueError("absolute positions must match input token rows")
        if tuple(sorted(set(self.absolute_positions))) != self.absolute_positions:
            raise ValueError("absolute positions must be sorted and unique")
        if not 0 <= self.exact_prefix_tokens <= self.absolute_positions[-1] + 1:
            raise ValueError("invalid exact-prefix length")
        if self.exact_prefix_tokens and any(
            position < self.exact_prefix_tokens
            for position in self.absolute_positions
        ):
            raise ValueError(
                "native Prefix Cache rows must not be recomputed by ProbeKV"
            )
        if self.adapter.total_layers <= 0:
            raise ValueError("adapter must expose a positive model depth")

    def begin_prefill(self) -> None:
        if self._started:
            raise RuntimeError("prefill session was already started")
        absolute = self.absolute_positions
        hidden, residual = self.adapter.begin_prefill(
            token_ids=self.token_ids,
            absolute_positions=absolute,
            attention_metadata=self.attention_metadata,
            working_kv=self.working_kv,
            model_signature=self.model_signature,
        )
        self.hidden_states = hidden
        self.residual = residual
        self.active_positions = absolute
        self._started = True

    def register_source_handle(
        self, segment_id: str, source_id: str, handle: Any
    ) -> None:
        if not self._started or self._finished:
            raise RuntimeError("Source handles require an active session")
        if not segment_id or not source_id:
            raise ValueError("Source handle identifiers are required")
        if segment_id in self.source_handles:
            raise RuntimeError("a Segment winner is already locked")
        self.source_handles[segment_id] = {
            "source_id": source_id,
            "handle": handle,
        }

    def observe_pre_rope_k(self, completed_depth: Optional[int] = None) -> Any:
        """Observe K entering the next causal self-attention block.

        ``completed_depth=0`` is available only for the frozen negative control;
        online locking starts at depth one.
        """
        if not self._started or self._finished:
            raise RuntimeError("K observation requires an active prefill session")
        depth = self.current_layer if completed_depth is None else int(completed_depth)
        if depth != self.current_layer:
            raise ValueError("K may only be observed at the actual completed depth")
        if not 0 <= depth < self.adapter.total_layers:
            raise ValueError("K observation must enter an existing next layer")
        observed = self.adapter.observe_pre_rope_k(
            completed_depth=depth,
            hidden_states=self.hidden_states,
            residual=self.residual,
            active_positions=self.active_positions,
        )
        self.layer_audit.append(
            {
                "event": "selection_k_observation",
                "completed_depth": depth,
                "k_observation_layer_1based": depth + 1,
                "k_observation_layer_index_0based": depth,
                "active_positions": self.active_positions,
            }
        )
        return observed

    def observe_repair_check_pre_rope_kv(
        self, completed_depth: Optional[int] = None
    ) -> Tuple[Any, Any]:
        """Observe dense current K/V entering the next layer for winner repair.

        This is separate from K-only Source selection.  It is legal only after
        at least one causal self-attention block has completed and the returned
        rows must be used by the next selective layer, never the check layer.
        """
        if not self._started or self._finished:
            raise RuntimeError("repair check requires an active prefill session")
        depth = self.current_layer if completed_depth is None else int(completed_depth)
        if depth != self.current_layer:
            raise ValueError("repair check must use the actual completed depth")
        if not 1 <= depth < self.adapter.total_layers:
            raise ValueError("repair check requires a completed layer and a consumer")
        observed = self.adapter.observe_pre_rope_kv(
            completed_depth=depth,
            hidden_states=self.hidden_states,
            residual=self.residual,
            active_positions=self.active_positions,
        )
        if not isinstance(observed, tuple) or len(observed) != 2:
            raise RuntimeError("repair-check adapter must return current K and V")
        self.layer_audit.append(
            {
                "event": "winner_repair_check_kv_observation",
                "repair_check_completed_depth": depth,
                "first_selective_reuse_layer": depth + 1,
                "active_positions": self.active_positions,
            }
        )
        return observed

    def commit_segment_reuse(
        self,
        *,
        segment_id: str,
        source_id: str,
        boundary: int,
        segment_positions: Sequence[int],
        repair_positions: Sequence[int],
    ) -> None:
        if not self._started or self._finished:
            raise RuntimeError("reuse commit requires an active session")
        if boundary != self.current_layer + 1:
            raise ValueError("reuse must commit immediately before its boundary")
        if segment_id in self.commits:
            raise RuntimeError("a Segment may be committed only once")
        locked = self.source_handles.get(segment_id)
        if locked is None or locked["source_id"] != source_id:
            raise RuntimeError("reuse can use only the locked Source")
        segment = tuple(sorted(set(int(v) for v in segment_positions)))
        repair = tuple(sorted(set(int(v) for v in repair_positions)))
        if not segment:
            raise ValueError("Segment positions cannot be empty")
        if any(v < self.exact_prefix_tokens for v in segment):
            raise ValueError("exact Prefix Cache tokens cannot enter ProbeKV")
        active = set(self.active_positions)
        if not set(segment).issubset(active):
            raise ValueError("Segment contains an inactive token")
        if not set(repair).issubset(segment):
            raise ValueError("repair positions must lie inside the Segment")
        target = tuple(v for v in self.active_positions if v not in set(segment) or v in set(repair))
        if self._pending_target_positions is not None:
            pending = set(self._pending_target_positions)
            target = tuple(v for v in target if v in pending)
        if not set(target).issubset(active):
            raise RuntimeError("active query set may only shrink")
        commit = SegmentReuseCommit(segment_id, boundary, repair, source_id)
        self.commits[segment_id] = commit
        self.current_repair_positions_by_segment[segment_id] = repair
        self._pending_target_positions = target
        self._pending_reuse_commit = True

    def shrink_segment_repair_support(
        self,
        *,
        segment_id: str,
        consumer_layer: int,
        repair_positions: Sequence[int],
    ) -> None:
        """Shrink a committed Segment's active repair rows for a later layer.

        Schema-v7 gradual repair is deliberately no-reentry: a row removed from
        the active query set has no current hidden state and therefore cannot be
        reintroduced at a deeper layer.  The change is staged immediately before
        ``consumer_layer`` and is applied by the normal layer advance path.
        """
        if not self._started or self._finished:
            raise RuntimeError("repair support update requires an active session")
        if consumer_layer != self.current_layer + 1:
            raise ValueError("repair support must update immediately before its consumer")
        commit = self.commits.get(segment_id)
        if commit is None:
            raise RuntimeError("gradual repair requires an existing reuse commit")
        previous = self.current_repair_positions_by_segment[segment_id]
        updated = tuple(sorted(set(int(value) for value in repair_positions)))
        if not set(updated).issubset(previous):
            raise RuntimeError("gradual repair support may only shrink")
        if not set(updated).issubset(self.active_positions):
            raise RuntimeError("repair support cannot reintroduce inactive rows")
        removed = set(previous) - set(updated)
        target = tuple(value for value in self.active_positions if value not in removed)
        if self._pending_target_positions is not None:
            target = tuple(
                value for value in target if value in set(self._pending_target_positions)
            )
        self.current_repair_positions_by_segment[segment_id] = updated
        self._pending_target_positions = target
        self._pending_reuse_commit = True
        self.layer_audit.append(
            {
                "event": "gradual_repair_support_shrink",
                "segment_id": segment_id,
                "consumer_layer_1based": consumer_layer,
                "previous_repair_positions": previous,
                "repair_positions": updated,
            }
        )

    def advance_to_layer(self, target_layer: int) -> None:
        if not self._started or self._finished:
            raise RuntimeError("advance requires an active prefill session")
        if not self.current_layer < target_layer <= self.adapter.total_layers:
            raise ValueError("target layer must advance within model depth")
        while self.current_layer < target_layer:
            layer = self.current_layer + 1
            before = self.active_positions
            target = self._pending_target_positions or before
            if not set(target).issubset(before):
                raise RuntimeError("adapter target would reintroduce active tokens")
            result = self.adapter.advance_layer(
                layer=layer,
                hidden_states=self.hidden_states,
                residual=self.residual,
                active_positions=before,
                target_active_positions=target,
                attention_metadata=self.attention_metadata,
                working_kv=self.working_kv,
                source_handles=self.source_handles,
                reuse_commit=self._pending_reuse_commit,
            )
            self.hidden_states = result.hidden_states
            self.residual = result.residual
            self.working_kv = result.working_kv
            self.active_positions = target
            self.current_layer = layer
            self.layer_audit.append(
                {
                    "layer": layer,
                    "active_before": before,
                    "active_after": target,
                    "gpu_ms": result.gpu_ms,
                    "host_ms": result.host_ms,
                    "union_mask_digest": result.union_mask_digest,
                    "runtime_debug": dict(result.runtime_debug),
                }
            )
            self._pending_target_positions = None
            self._pending_reuse_commit = False

    def finish_prefill(self) -> Any:
        if not self._started or self._finished:
            raise RuntimeError("finish requires an active unfinished session")
        if self.current_layer != self.adapter.total_layers:
            self.advance_to_layer(self.adapter.total_layers)
        output = self.adapter.finish_prefill(
            hidden_states=self.hidden_states,
            residual=self.residual,
            active_positions=self.active_positions,
            attention_metadata=self.attention_metadata,
            working_kv=self.working_kv,
        )
        self._finished = True
        return output
