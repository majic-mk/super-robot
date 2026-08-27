import unittest

from probekv.contracts import KVLocation
from probekv.v8_leases import (
    LeaseLifecycle,
    LeasePurpose,
    ReplicaLeaseRequest,
    ReplicaLifecycle,
    V8LeaseManager,
    V8ReplicaResource,
)
from probekv.v8_selection_state_store import (
    SelectionStateStore,
    SelectionStateUnavailable,
    build_selection_state,
)


class V8SelectionStateStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = SelectionStateStore()
        self.state = build_selection_state(
            source_variant_id="source", completed_depth=1, token_count=128,
            num_kv_heads=4, head_dim=128,
            parent_source_state_digest="parent", logical_digest="logical",
        )
        self.store.register(self.state)

    def test_backing_and_relocation_do_not_change_logical_identity(self):
        replica = self.store.attach_replica(
            self.state.selection_state_id, tier=KVLocation.PINNED_CPU,
            locator="cpu:1", layout_signature="contiguous", bytes_digest="bytes",
            size_bytes=1024,
        )
        moved = self.store.relocate_replica(
            self.state.selection_state_id, replica.state_replica_id, locator="cpu:2"
        )
        self.assertEqual(moved.state_replica_id, replica.state_replica_id)
        self.assertGreater(moved.placement_epoch, replica.placement_epoch)
        self.assertEqual(self.store.require_state(self.state.selection_state_id), self.state)

    def test_missing_or_corrupt_state_never_falls_back_to_full_kv(self):
        with self.assertRaises(SelectionStateUnavailable):
            self.store.require_state(self.state.selection_state_id)
        replica = self.store.attach_replica(
            self.state.selection_state_id, tier=KVLocation.SSD,
            locator="file:1", layout_signature="v1", bytes_digest="bytes",
            size_bytes=1024,
        )
        self.store.mark_corrupt(self.state.selection_state_id, replica.state_replica_id)
        with self.assertRaises(SelectionStateUnavailable):
            self.store.require_state(self.state.selection_state_id)

    def test_gpu_cannot_be_persistent_selection_state_backing(self):
        with self.assertRaises(ValueError):
            self.store.attach_replica(
                self.state.selection_state_id, tier=KVLocation.GPU,
                locator="gpu:1", layout_signature="scratch", bytes_digest="bytes",
                size_bytes=1024, is_backing=True,
            )


def setup_manager():
    manager = V8LeaseManager(ttl_floor_s=30, orphan_grace_s=5)
    manager.register_source("source-a", "artifact-a", "model-a")
    manager.register_source("source-b", "artifact-b", "model-a")
    for source, artifact, suffix in (
        ("source-a", "artifact-a", "a"), ("source-b", "artifact-b", "b")
    ):
        manager.register_replica(V8ReplicaResource(
            "cpu-" + suffix, source, artifact, KVLocation.PINNED_CPU,
            1, 1, 100, True,
        ))
        manager.register_replica(V8ReplicaResource(
            "gpu-" + suffix, source, artifact, KVLocation.GPU,
            1, 1, 100, False,
        ))
    return manager


class V8LeaseTests(unittest.TestCase):
    def test_freeze_is_atomic_and_cannot_switch_source(self):
        manager = setup_manager()
        first = manager.freeze_and_acquire_logical(
            request_id="r", request_generation=1, segment_id="c",
            source_variant_id="source-a", predicted_remaining_s=1, now_s=0,
        )
        with self.assertRaises(RuntimeError):
            manager.freeze_and_acquire_logical(
                request_id="r", request_generation=1, segment_id="c",
                source_variant_id="source-b", predicted_remaining_s=1, now_s=0,
            )
        self.assertEqual(first.purpose, LeasePurpose.LOGICAL_SOURCE)

    def test_multi_reader_refcount_and_batch_is_all_or_nothing(self):
        manager = setup_manager()
        for request in ("r1", "r2"):
            manager.freeze_and_acquire_logical(
                request_id=request, request_generation=1, segment_id="c",
                source_variant_id="source-a", predicted_remaining_s=1, now_s=0,
            )
        self.assertEqual(manager.sources["source-a"].logical_lease_refcount, 2)
        request = ReplicaLeaseRequest(
            "c", "source-a", "artifact-a", "gpu-a", 1, 1,
            LeasePurpose.EXECUTION,
        )
        leases = manager.compare_and_lease_batch(
            request_id="r1", request_generation=1, requests=[request],
            predicted_remaining_s=1, now_s=0,
        )
        self.assertEqual(manager.sources["source-a"].replicas["gpu-a"].lease_refcount, 1)
        stale = ReplicaLeaseRequest(
            "c", "source-a", "artifact-a", "gpu-a", 2, 1,
            LeasePurpose.EXECUTION,
        )
        with self.assertRaises(RuntimeError):
            manager.compare_and_lease_batch(
                request_id="r2", request_generation=1, requests=[stale],
                predicted_remaining_s=1, now_s=0,
            )
        self.assertEqual(manager.sources["source-a"].replicas["gpu-a"].lease_refcount, 1)
        manager.release(leases[0].lease_id)

    def test_unrelated_replica_does_not_stale_local_epoch(self):
        manager = setup_manager()
        manager.freeze_and_acquire_logical(
            request_id="r", request_generation=1, segment_id="c",
            source_variant_id="source-a", predicted_remaining_s=1, now_s=0,
        )
        request = ReplicaLeaseRequest(
            "c", "source-a", "artifact-a", "gpu-a", 1, 1,
            LeasePurpose.EXECUTION,
        )
        manager.sources["source-b"].replicas["gpu-b"].placement_epoch = 99
        leases = manager.compare_and_lease_batch(
            request_id="r", request_generation=1, requests=[request],
            predicted_remaining_s=1, now_s=0,
        )
        self.assertEqual(len(leases), 1)

    def test_ttl_marks_suspect_before_orphan_release(self):
        manager = setup_manager()
        lease = manager.freeze_and_acquire_logical(
            request_id="r", request_generation=1, segment_id="c",
            source_variant_id="source-a", predicted_remaining_s=1, now_s=0,
        )
        self.assertEqual(manager.recover_expired(live_owners=(), now_s=31), ())
        self.assertEqual(manager.leases[lease.lease_id].state, LeaseLifecycle.SUSPECT)
        released = manager.recover_expired(live_owners=(), now_s=37)
        self.assertEqual(released, (lease.lease_id,))
        self.assertEqual(manager.leases[lease.lease_id].state, LeaseLifecycle.RELEASED)

    def test_logical_lease_protects_backing_and_purge(self):
        manager = setup_manager()
        logical = manager.freeze_and_acquire_logical(
            request_id="r", request_generation=1, segment_id="c",
            source_variant_id="source-a", predicted_remaining_s=1, now_s=0,
        )
        with self.assertRaises(RuntimeError):
            manager.evict_replica("source-a", "cpu-a")
        with self.assertRaises(RuntimeError):
            manager.purge_namespace("model-a")
        manager.release(logical.lease_id)
        manager.evict_replica("source-a", "cpu-a")
        self.assertEqual(
            manager.sources["source-a"].replicas["cpu-a"].lifecycle,
            ReplicaLifecycle.DELETED,
        )


if __name__ == "__main__":
    unittest.main()
