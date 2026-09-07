"""Transactional, single-writer tensor/file backing for schema10.

SelectionState is a separate object/file: comparing it never opens full KV.
Only this store owns mutable tensors. Callers receive copies for comparison or
leased read-only-by-contract KV. Full hashes run at creation/migration, not at
every online lookup. Quiescent snapshots retain their backing objects until
explicitly dropped and are diagnostic overhead, not online storage capacity.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import uuid

from .contracts import KVLocation
from .v7_contracts import CanonicalKVArtifact, ReplicaState
from .v8_schema10_execution import digest_json
from .v8_schema10_pool import Schema10SourcePool
from .v8_schema10_source_metadata import validate_publication_metadata


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(tensors) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        # Identical logical encoding to TorchLayerwiseSourceLoader._digest.
        # Placement/serialization bytes are deliberately a different digest.
        h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(value.view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


@dataclass
class BackingObject:
    source_id: str
    tier: KVLocation
    size_bytes: int
    layers: object
    selection: object
    metadata: dict
    kv_path: str | None
    selection_path: str | None
    kv_file_digest: str | None
    selection_file_digest: str | None
    selection_digest: str
    creation_epoch: int


class TensorFileSourceStore:
    def __init__(self, pool: Schema10SourcePool, root, *, cpu_bytes: int,
                 ssd_bytes: int, pin_cpu: bool = True):
        if min(cpu_bytes, ssd_bytes) < 0 or cpu_bytes + ssd_bytes <= 0:
            raise ValueError("positive explicit storage budget required")
        if pool.tier_capacity_bytes:
            raise ValueError("physical store must be the sole tier-capacity owner, not the historical value evictor")
        self.pool, self.root = pool, Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cpu_bytes, self.ssd_bytes, self.pin_cpu = cpu_bytes, ssd_bytes, pin_cpu
        self.objects: dict[str, BackingObject] = {}
        self.events: list[dict] = []
        self._snapshots: dict[str, tuple] = {}

    def _quiescent(self):
        if any(self.pool.logical_lease_counts.values()) or any(
                p.busy for v in self.pool._variants.values() for p in v.replicas.values()):
            raise RuntimeError("storage mutation/snapshot requires a quiescent single-request pool")

    def _clone_pool(self):
        clone = object.__new__(Schema10SourcePool)
        clone.__dict__.update(deepcopy({k: v for k, v in self.pool.__dict__.items()
                                       if k != "mutation_lock"}))
        clone.mutation_lock = self.pool.mutation_lock
        return clone

    def _install(self, pool, objects, events):
        self.pool.__dict__.update({k: v for k, v in pool.__dict__.items() if k != "mutation_lock"})
        self.objects = objects
        self.events.extend(events)

    def _delete_owned(self, path):
        value = Path(path).resolve()
        if value.parent != self.root or value.suffix != ".pt":
            raise RuntimeError("refusing to remove a non-store file")
        value.unlink(missing_ok=True)

    def _gc(self, paths):
        retained = {p for o in self.objects.values() for p in (o.kv_path, o.selection_path) if p}
        for _, objects, _ in self._snapshots.values():
            retained.update(p for o in objects.values() for p in (o.kv_path, o.selection_path) if p)
        for path in set(paths) - retained:
            self._delete_owned(path)

    def _save(self, value, created):
        import torch
        destination = self.root / (uuid.uuid4().hex + ".pt")
        temporary = destination.with_suffix(".tmp")
        try:
            with temporary.open("xb") as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            created.append(str(destination))
            # Deserialize our own restricted artifact and validate before publish.
            restored = torch.load(destination, weights_only=True, map_location="cpu")
            return str(destination), file_digest(destination), restored
        finally:
            temporary.unlink(missing_ok=True)

    def _read(self, obj, *, selection=False, verify_full=False):
        import torch
        if obj.tier is KVLocation.PINNED_CPU:
            return obj.selection if selection else obj.layers
        path = obj.selection_path if selection else obj.kv_path
        expected = obj.selection_file_digest if selection else obj.kv_file_digest
        # SelectionState is lightweight. Full-KV hashing is reserved for
        # creation/migration/qualification, never per-request immutable reads.
        if path is None or not Path(path).is_file() or ((selection or verify_full) and file_digest(Path(path)) != expected):
            raise RuntimeError("corrupt SSD SelectionState" if selection else "corrupt SSD Artifact")
        return torch.load(path, weights_only=True, map_location="cpu")

    def _cpu(self, tensor):
        result = tensor.detach().cpu().contiguous().clone()
        return result.pin_memory() if self.pin_cpu else result

    def _move_object(self, obj, tier, created):
        import torch
        layers, states = self._read(obj, verify_full=True), self._read(obj, selection=True)
        if tier is KVLocation.PINNED_CPU:
            cpu_size = sum(t.numel() * t.element_size() for pair in layers for t in pair)
            cpu_size += sum(t.numel() * t.element_size() for t in states.values()) + len(json.dumps(obj.metadata).encode())
            return BackingObject(obj.source_id, tier, cpu_size,
                                 tuple((self._cpu(k), self._cpu(v)) for k, v in layers),
                                 {d: self._cpu(k) for d, k in states.items()}, deepcopy(obj.metadata),
                                 None, None, None, None, obj.selection_digest, obj.creation_epoch)
        kp, kh, check = self._save(layers, created)
        if tensor_digest(t for pair in check for t in pair) != tensor_digest(t for pair in layers for t in pair):
            raise RuntimeError("SSD destination full KV verification failed")
        sp, sh, check = self._save(states, created)
        if tensor_digest(check[d] for d in sorted(check)) != obj.selection_digest:
            raise RuntimeError("SSD destination SelectionState verification failed")
        size = os.path.getsize(kp) + os.path.getsize(sp) + len(json.dumps(obj.metadata).encode())
        return BackingObject(obj.source_id, tier, size, None, None, deepcopy(obj.metadata),
                             kp, sp, kh, sh, obj.selection_digest, obj.creation_epoch)

    def _attach(self, pool, obj):
        row = next(v for v in pool._variants.values() if v.source_variant_id == obj.source_id)
        for replica in row.replicas.values():
            if replica.busy:
                raise RuntimeError("cannot change a busy backing")
            if replica.is_backing:
                replica.is_backing = False
                replica.state = ReplicaState.DELETED
        pool.attach_replica(row.identity.model_math_signature, row.identity.reuse_content_key,
                            obj.source_id, tier=obj.tier, locator_value=obj.kv_path or ("cpu:" + obj.source_id),
                            layout_signature="bf16-contiguous" if not obj.kv_path else "torch-weights-only-v1",
                            bytes_digest=obj.kv_file_digest or row.canonical_source_state_digest,
                            size_bytes=obj.size_bytes, is_backing=True)

    def _room(self, pool, objects, tier, size, protected, created, events):
        capacity = self.cpu_bytes if tier is KVLocation.PINNED_CPU else self.ssd_bytes
        if size > capacity:
            raise MemoryError("Artifact exceeds backing tier capacity")
        while sum(o.size_bytes for o in objects.values() if o.tier is tier) + size > capacity:
            candidates = [v for v in pool._variants.values()
                          if v.source_variant_id in objects and objects[v.source_variant_id].tier is tier
                          and v.source_variant_id not in protected and not pool._probation_protected(v)
                          and not pool.logical_lease_counts.get(v.source_variant_id)
                          and not any(p.busy for p in v.replicas.values())]
            if not candidates:
                raise MemoryError("all backing capacity victims are protected")
            victim = min(candidates, key=lambda v: (v.last_request_use_epoch, v.registered_order, v.source_variant_id))
            sid = victim.source_variant_id
            if tier is KVLocation.PINNED_CPU:
                moved = self._move_object(objects[sid], KVLocation.SSD, created)
                self._room(pool, objects, KVLocation.SSD, moved.size_bytes, protected | {sid}, created, events)
                objects[sid] = moved
                self._attach(pool, moved)
                events.append({"action": "demote_cpu_to_ssd", "source_id": sid})
            else:
                del objects[sid]
                pool._evict_variant(victim, "ssd_request_lru")
                events.append({"action": "delete_ssd_lru", "source_id": sid})

    def publish_exact_dense(self, identity, *, layers, selection_states, metadata,
                            request_epoch: int, whole_request_origin: str,
                            materialization_reason: str, replacement_transaction=None):
        import torch
        if whole_request_origin != "exact_dense_full_prefill":
            raise ValueError("only whole-request exact dense can create canonical KV")
        if request_epoch < 0 or not layers or not selection_states or not metadata:
            raise ValueError("incomplete canonical materialization")
        shape = tuple(layers[0][0].shape)
        if len(shape) != 3 or min(shape) <= 0:
            raise ValueError("KV must be [token, kv_head, head_dim]")
        if any(len(pair) != 2 for pair in layers):
            raise ValueError("every canonical layer must contain exactly K and V")
        validate_publication_metadata(metadata, token_count=shape[0], num_layers=len(layers))
        if any(t.dtype != torch.bfloat16 or tuple(t.shape) != shape for pair in layers for t in pair):
            raise ValueError("canonical KV must have consistent exact BF16 geometry")
        if any(not 1 <= d < len(layers) or k.dtype != torch.bfloat16 or tuple(k.shape) != shape
               for d, k in selection_states.items()):
            raise ValueError("invalid independent completed-depth SelectionState")
        # It must describe K entering the block after d completed blocks.
        if any(not torch.equal(k.cpu(), layers[d][0].cpu()) for d, k in selection_states.items()):
            raise ValueError("SelectionState differs from canonical observation-layer K")
        metadata = {**metadata, "selection_completed_depths": sorted(selection_states)}
        source_id = identity.source_variant_id
        with self.pool.mutation_lock:
            self._quiescent()
            if source_id in self.objects:
                raise ValueError("Source identity already published")
            plan = replacement_transaction or self.pool.replacement_transaction(
                identity.model_math_signature, identity.reuse_content_key)
            clone, objects, events, created = self._clone_pool(), dict(self.objects), [], []
            try:
                cpu_layers = tuple((self._cpu(k), self._cpu(v)) for k, v in layers)
                states = {d: self._cpu(k) for d, k in selection_states.items()}
                sd = tensor_digest(states[d] for d in sorted(states))
                logical = tensor_digest(t for pair in cpu_layers for t in pair)
                summary = digest_json({"selection_digest": sd, "metadata": metadata})
                row = clone.register_variant(identity, canonical_source_state_digest=logical,
                                             summary_digest=summary, replacement_transaction=plan,
                                             materialization_reason=materialization_reason)
                if plan.victim_source_variant_id:
                    objects.pop(plan.victim_source_variant_id)
                artifact = CanonicalKVArtifact(digest_json([source_id, logical]), source_id, 1,
                                              logical, logical, logical, len(layers), shape[1], shape[2])
                clone.register_artifact(identity.model_math_signature, identity.reuse_content_key, source_id, artifact)
                size = sum(t.numel() * t.element_size() for pair in cpu_layers for t in pair)
                size += sum(t.numel() * t.element_size() for t in states.values()) + len(json.dumps(metadata).encode())
                obj = BackingObject(source_id, KVLocation.PINNED_CPU, size, cpu_layers, states,
                                    deepcopy(metadata), None, None, None, None, sd, request_epoch)
                if size > self.cpu_bytes:
                    obj = self._move_object(obj, KVLocation.SSD, created)
                self._room(clone, objects, obj.tier, obj.size_bytes, {source_id}, created, events)
                self._attach(clone, obj)
                objects[source_id] = obj
                events.append({"action": "publish_exact_dense", "source_id": source_id,
                               "creation_epoch": request_epoch, "bytes": obj.size_bytes})
                old_paths = [p for o in self.objects.values() for p in (o.kv_path, o.selection_path) if p]
                self._install(clone, objects, events)
            except Exception:
                for path in created:
                    self._delete_owned(path)
                raise
            self._gc(old_paths + created)
            return self.pool._get(identity.model_math_signature, identity.reuse_content_key, source_id)

    def read_selection(self, source_id, completed_depth):
        with self.pool.mutation_lock:
            states = self._read(self.objects[source_id], selection=True)
            if completed_depth not in states:
                raise KeyError("SelectionState unavailable; full-KV fallback is prohibited")
            return states[completed_depth].clone()

    def promote_request_use(self, source_id):
        with self.pool.mutation_lock:
            self._quiescent()
            obj = self.objects[source_id]
            clone, objects, events, created = self._clone_pool(), dict(self.objects), [], []
            row = next(v for v in clone._variants.values() if v.source_variant_id == source_id)
            # Request selection/binding is a use; migration itself never ticks it.
            row.last_request_use_epoch = clone._tick()
            try:
                if obj.tier is KVLocation.SSD:
                    moved = self._move_object(obj, KVLocation.PINNED_CPU, created)
                    if moved.size_bytes <= self.cpu_bytes:
                        self._room(clone, objects, moved.tier, moved.size_bytes, {source_id}, created, events)
                        objects[source_id] = moved
                        self._attach(clone, moved)
                        events.append({"action": "promote_ssd_to_cpu", "source_id": source_id})
                old_paths = [p for o in self.objects.values() for p in (o.kv_path, o.selection_path) if p]
                self._install(clone, objects, events)
            except Exception:
                for path in created:
                    self._delete_owned(path)
                raise
            self._gc(old_paths + created)

    @contextmanager
    def leased_winner(self, model, content, source_id, *, expected_generation):
        with self.pool.freeze_source(model, content, source_id, expected_generation=expected_generation) as row:
            backing = row.healthy_backing_replicas[0]
            with self.pool.lease_replica(model, content, source_id, backing.replica_id):
                # Full KV may only be opened after the logical+physical lease.
                yield self._read(self.objects[source_id])

    def snapshot(self):
        with self.pool.mutation_lock:
            self._quiescent()
            descriptor = self.snapshot_descriptor()
            state, key = descriptor["state"], descriptor["snapshot_sha256"]
            self._snapshots[key] = (self._clone_pool(), dict(self.objects), deepcopy(self.events))
            return descriptor

    def snapshot_descriptor(self):
        state = self.describe()
        return {"snapshot_sha256": digest_json(state), "state": state,
                "scope": "quiescent_logical_and_backing", "ssd_page_cache_controlled": False}

    def release_snapshot(self, snapshot):
        key = snapshot["snapshot_sha256"]
        with self.pool.mutation_lock:
            _, objects, _ = self._snapshots.pop(key)
            self._gc([p for o in objects.values() for p in (o.kv_path, o.selection_path) if p])

    def close(self):
        """Discard only this replay's owned regenerable backing files."""
        with self.pool.mutation_lock:
            self._quiescent()
            objects = list(self.objects.values())
            for _, retained, _ in self._snapshots.values():
                objects.extend(retained.values())
            self.objects, self._snapshots = {}, {}
            self._gc([p for o in objects for p in (o.kv_path, o.selection_path) if p])

    def restore(self, snapshot):
        with self.pool.mutation_lock:
            self._quiescent()
            key = snapshot["snapshot_sha256"]
            if digest_json(snapshot["state"]) != key or key not in self._snapshots:
                raise ValueError("unknown or corrupt pool snapshot")
            pool, objects, events = self._snapshots[key]
            old_paths = [p for o in self.objects.values() for p in (o.kv_path, o.selection_path) if p]
            # Validate SSD files before changing any live registry state.
            for obj in objects.values():
                if obj.tier is KVLocation.SSD:
                    self._read(obj, verify_full=True)
                    self._read(obj, selection=True)
            self.pool.__dict__.update(deepcopy({k: v for k, v in pool.__dict__.items() if k != "mutation_lock"}))
            self.objects, self.events = dict(objects), deepcopy(events)
            if self.describe() != snapshot["state"]:
                raise RuntimeError("pool restoration mismatch")
            self._gc(old_paths)

    def describe(self):
        from dataclasses import asdict
        return {"clock": self.pool._clock, "placement_epoch": self.pool._placement_epoch,
                "replica_generation": self.pool._replica_generation,
                "capacity": self.pool.max_variants_per_content,
                "cpu_budget": self.cpu_bytes, "ssd_budget": self.ssd_bytes,
                "cpu_pinned": self.pin_cpu,
                "variants": [asdict(v) for v in sorted(self.pool._variants.values(), key=lambda r: r.source_variant_id)],
                "objects": [{"source_id": o.source_id, "tier": o.tier.value, "bytes": o.size_bytes,
                             "creation_epoch": o.creation_epoch, "metadata": deepcopy(o.metadata),
                             "kv_path": o.kv_path, "selection_path": o.selection_path,
                             "selection_digest": o.selection_digest}
                            for o in sorted(self.objects.values(), key=lambda x: x.source_id)]}
