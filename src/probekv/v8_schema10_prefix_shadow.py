"""Bounded creation-verified CPU Prefix shadows. Block IDs are not identity."""
from collections import OrderedDict

from .v8_schema10_execution import digest_json
from .v8_schema10_storage import tensor_digest


class PrefixShadowStore:
    def __init__(self, *, model_signature, num_layers, kv_heads, head_dim, capacity_bytes):
        self.signature, self.num_layers = model_signature, num_layers
        self.geometry, self.capacity_bytes = (kv_heads, head_dim), capacity_bytes
        if not model_signature or min(num_layers, kv_heads, head_dim, capacity_bytes) <= 0:
            raise ValueError("invalid Prefix shadow capacity/geometry")
        self.entries = OrderedDict()
        self.retained = {}

    @property
    def resident_bytes(self):
        # Retained A/B snapshots own host tensors too; resetting the live cache
        # must not make those bytes disappear from the global host budget.
        rows = {id(row): row for row in self.entries.values()}
        for snapshot in self.retained.values():
            rows.update({id(row): row for row in snapshot.values()})
        return sum(row["bytes"] for row in rows.values())

    def retain(self, snapshot_id):
        if snapshot_id in self.retained:
            raise ValueError("Prefix snapshot is already retained")
        self.retained[snapshot_id] = self.entries.copy()

    def restore_retained(self, snapshot_id):
        self.entries = self.retained[snapshot_id].copy()

    def release_retained(self, snapshot_id):
        del self.retained[snapshot_id]

    def publish(self, token_ids, layers, *, origin):
        import torch
        tokens = tuple(token_ids)
        if origin != "exact_dense_full_prefill" or not tokens or len(layers) != self.num_layers:
            raise ValueError("Prefix shadow requires complete canonical full prefill")
        shape = (len(tokens),) + self.geometry
        if any(t.dtype != torch.bfloat16 or tuple(t.shape) != shape for pair in layers for t in pair):
            raise ValueError("Prefix shadow geometry/dtype differs")
        size = len(tokens) * self.geometry[0] * self.geometry[1] * self.num_layers * 4
        if size > self.capacity_bytes:
            return False
        key = digest_json([self.signature, tokens])
        self.entries.pop(key, None)
        while self.entries and self.resident_bytes + size > self.capacity_bytes:
            self.entries.popitem(last=False)
        if self.resident_bytes + size > self.capacity_bytes:
            return False  # retained snapshots cannot be evicted or undercounted
        cpu = tuple(tuple(t.detach().to(device="cpu", copy=True).contiguous() for t in pair) for pair in layers)
        self.entries[key] = {"tokens": tokens, "layers": cpu, "bytes": size,
                             "logical_digest": tensor_digest(t for p in cpu for t in p)}
        return True

    def lookup(self, token_ids, block_ids, *, native_request):
        tokens = tuple(token_ids)
        if (tuple(block_ids) != native_request.cached_block_ids
                or len(tokens) != native_request.cached_prefix_tokens
                or tuple(native_request.sequence.get_prompt_token_ids()[:len(tokens)]) != tokens
                or native_request.closed or not native_request.allocated):
            raise RuntimeError("stale native block lease for Prefix shadow")
        for key, row in tuple(self.entries.items()):
            if row["tokens"][:len(tokens)] == tokens:
                self.entries.move_to_end(key)
                # Context takes its own GPU working copy under HBM reservation.
                return tuple(tuple(t[:len(tokens)] for t in pair) for pair in row["layers"])
        return None

    def clear(self):
        self.entries.clear()

    def descriptor(self):
        return {"model_signature": self.signature, "capacity_bytes": self.capacity_bytes,
                "rows": [{"tokens": list(r["tokens"]), "logical_digest": r["logical_digest"]}
                         for r in self.entries.values()]}
