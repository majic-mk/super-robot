from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple

from .contracts import KVLocation
from .v8_contracts import (
    SelectionStateReplica,
    SourceSelectionState,
    stable_v8_digest,
)


class SelectionStateUnavailable(RuntimeError):
    pass


def build_selection_state(
    *,
    source_variant_id: str,
    completed_depth: int,
    token_count: int,
    num_kv_heads: int,
    head_dim: int,
    parent_source_state_digest: str,
    logical_digest: str,
) -> SourceSelectionState:
    identity = {
        "source_variant_id": source_variant_id,
        "completed_depth": completed_depth,
        "token_count": token_count,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "parent_source_state_digest": parent_source_state_digest,
        "logical_digest": logical_digest,
        "dtype": "bfloat16",
        "k_semantics": "pre_rope",
    }
    return SourceSelectionState(
        selection_state_id=stable_v8_digest("selection-state", identity),
        source_variant_id=source_variant_id,
        completed_depth=completed_depth,
        k_observation_layer_1based=completed_depth + 1,
        token_count=token_count,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        parent_source_state_digest=parent_source_state_digest,
        logical_digest=logical_digest,
    )


class SelectionStateStore:
    """Logical SelectionState store; it never falls back to a full-KV Artifact."""

    def __init__(self) -> None:
        self._states: Dict[str, SourceSelectionState] = {}
        self._replicas: Dict[str, Dict[str, SelectionStateReplica]] = {}
        self._generation = 0
        self._placement_epoch = 0

    def register(self, state: SourceSelectionState) -> SourceSelectionState:
        existing = self._states.get(state.selection_state_id)
        if existing is not None and existing != state:
            raise ValueError("SelectionState identity collision")
        self._states[state.selection_state_id] = state
        self._replicas.setdefault(state.selection_state_id, {})
        return state

    def attach_replica(
        self,
        selection_state_id: str,
        *,
        tier: KVLocation,
        locator: str,
        layout_signature: str,
        bytes_digest: str,
        size_bytes: int,
        is_backing: Optional[bool] = None,
    ) -> SelectionStateReplica:
        try:
            state = self._states[selection_state_id]
        except KeyError as error:
            raise SelectionStateUnavailable("unknown SelectionState") from error
        tier = KVLocation(tier)
        replicas = self._replicas[selection_state_id]
        if any(item.tier is tier and item.healthy for item in replicas.values()):
            raise ValueError("only one healthy SelectionStateReplica is allowed per tier")
        backing_exists = any(item.is_backing and item.healthy for item in replicas.values())
        backing = not backing_exists if is_backing is None else bool(is_backing)
        if backing and backing_exists:
            raise ValueError("only one SelectionState backing Replica is allowed")
        self._generation += 1
        self._placement_epoch += 1
        replica_id = stable_v8_digest(
            "selection-state-replica",
            {
                "selection_state_id": selection_state_id,
                "tier": tier.value,
                "generation": self._generation,
                "placement_epoch": self._placement_epoch,
            },
        )
        replica = SelectionStateReplica(
            state_replica_id=replica_id,
            selection_state_id=selection_state_id,
            tier=tier,
            generation=self._generation,
            placement_epoch=self._placement_epoch,
            locator=locator,
            layout_signature=layout_signature,
            logical_digest=state.logical_digest,
            bytes_digest=bytes_digest,
            size_bytes=size_bytes,
            is_backing=backing,
        )
        replicas[replica_id] = replica
        return replica

    def require_state(self, selection_state_id: str) -> SourceSelectionState:
        try:
            state = self._states[selection_state_id]
        except KeyError as error:
            raise SelectionStateUnavailable(
                "SelectionState is unavailable; full-KV fallback is forbidden"
            ) from error
        healthy = [
            item
            for item in self._replicas.get(selection_state_id, {}).values()
            if item.healthy and item.logical_digest == state.logical_digest
        ]
        if not healthy or not any(item.is_backing for item in healthy):
            raise SelectionStateUnavailable(
                "SelectionState lacks a healthy backing; full-KV fallback is forbidden"
            )
        return state

    def replicas(self, selection_state_id: str) -> Tuple[SelectionStateReplica, ...]:
        self.require_state(selection_state_id)
        return tuple(
            sorted(
                (
                    item
                    for item in self._replicas[selection_state_id].values()
                    if item.healthy
                ),
                key=lambda item: (item.tier.value, item.state_replica_id),
            )
        )

    def relocate_replica(
        self,
        selection_state_id: str,
        state_replica_id: str,
        *,
        locator: str,
        layout_signature: Optional[str] = None,
    ) -> SelectionStateReplica:
        replica = self._replicas[selection_state_id][state_replica_id]
        if not replica.healthy:
            raise SelectionStateUnavailable("cannot relocate an unhealthy SelectionStateReplica")
        self._placement_epoch += 1
        moved = replace(
            replica,
            placement_epoch=self._placement_epoch,
            locator=locator,
            layout_signature=layout_signature or replica.layout_signature,
        )
        self._replicas[selection_state_id][state_replica_id] = moved
        return moved

    def mark_corrupt(
        self, selection_state_id: str, state_replica_id: str
    ) -> SelectionStateReplica:
        replica = self._replicas[selection_state_id][state_replica_id]
        corrupted = replace(replica, healthy=False)
        self._replicas[selection_state_id][state_replica_id] = corrupted
        return corrupted
