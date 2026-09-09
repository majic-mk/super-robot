"""One-shot dense-state handoff to the pinned CacheBlend decoder loop.

Diagnostic only: zero native Prefix, one resident fixed winner, no prior reuse.
This calls the same decoder layers as normal forward but skips embeddings and
completed layers. It is not an unmodified CacheBlend forward or online admission.
"""
from dataclasses import dataclass, field


@dataclass
class DenseLoopContinuation:
    request_id: str
    generation: int
    completed_depth: int
    hidden_states: object
    residual: object
    active_positions: tuple
    attention_metadata: object
    working_kv: object
    inner_model: object
    position_tensor: object
    executed_layers: list = field(default_factory=list)
    consumed: bool = False

    @classmethod
    def detach(cls, session, *, request_id, generation, inner_model, boundary, positions):
        n = len(session.token_ids)
        if (not request_id or generation < 0 or not session._started or session._finished
                or session.exact_prefix_tokens or session.commits
                or session._pending_target_positions is not None
                or session.current_layer != boundary - 1
                or not 1 <= session.current_layer < len(inner_model.layers)
                or session.active_positions != tuple(range(n)) or positions.shape != (n,)):
            raise ValueError("handoff requires a live dense zero-Prefix state at boundary-1")
        h, r = session.hidden_states, session.residual
        if (h is None or r is None or h.shape != r.shape or len(h.shape) != 2
                or h.shape[0] != n or h.dtype != r.dtype or h.device != r.device):
            raise ValueError("handoff hidden/residual geometry or dtype mismatch")
        result = cls(request_id, generation, session.current_layer, h, r,
                     session.active_positions, session.attention_metadata,
                     session.working_kv, inner_model, positions)
        # A move, not a second resumable owner. All public session operations
        # reject _finished; retained layer audit/timing events remain readable.
        session._finished = True
        session.hidden_states = session.residual = None
        return result

    def run(self, *, request_id, generation, positions, attention_metadata, working_kv):
        if (self.consumed or request_id != self.request_id or generation != self.generation
                or attention_metadata is not self.attention_metadata
                or working_kv is not self.working_kv
                or positions is not self.position_tensor):
            raise RuntimeError("stale, replayed or mismatched continuation")
        model, metadata = self.inner_model, self.inner_model.cache_fuse_metadata
        if (metadata.get("probekv_resumable") or not metadata.get("check")
                or metadata.get("check_layers") != [self.completed_depth]
                or metadata.get("exact_prefix_tokens", 0)
                or not metadata.get("probekv_matched_boundary_source_kv")):
            raise RuntimeError("continuation requires matched normal-loop metadata")
        self.consumed = True  # exceptions must not make partially executed state replayable
        hidden, residual = self.hidden_states, self.residual
        metadata.update(org_seq_len=len(self.active_positions), org_pos=positions,
                        fake_q=None, attn_bias=None, imp_indices=None,
                        original_slot_mapping=None, our_slot_mapping=None)
        try:
            for index in range(self.completed_depth, len(model.layers)):
                status = 1 if index == self.completed_depth else 2
                if status == 1:
                    metadata["check_layer"] = index
                hidden, residual = model.layers[index](
                    positions, hidden, working_kv[index], attention_metadata, residual,
                    status=status, cache_fuse_metadata=metadata, old_kv=model.old_kvs[index])
                self.executed_layers.append(index + 1)
                if status == 1:
                    positions = positions[metadata["imp_indices"]]
            hidden, _ = model.norm(hidden, residual)
            return hidden
        finally:
            self.hidden_states = self.residual = None
