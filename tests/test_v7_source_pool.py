import unittest

from probekv.contracts import KVLocation
from probekv.global_source_pool import ModelServingMode
from probekv.v7_contracts import (
    CanonicalKVArtifact,
    SourceVariantIdentity,
)
from probekv.v7_source_pool import V7SourcePool


def identity(model="model-a", content="content-a", occurrence="occ-a"):
    return SourceVariantIdentity(
        reuse_content_key=content,
        historical_prefix_digest="prefix-" + occurrence,
        position_ids_digest="positions-" + occurrence,
        occurrence_id=occurrence,
        model_math_signature=model,
    )


def artifact(source_id, name="artifact-a"):
    return CanonicalKVArtifact(
        artifact_id=name,
        source_variant_id=source_id,
        generation=1,
        parent_source_state_digest="state",
        artifact_logical_digest="logical",
        artifact_bytes_digest="bytes",
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
    )


class V7SourcePoolTests(unittest.TestCase):
    def setUp(self):
        self.pool = V7SourcePool(
            serving_mode=ModelServingMode.MULTI,
            tier_capacity_bytes={
                KVLocation.GPU: 1000,
                KVLocation.PINNED_CPU: 2000,
                KVLocation.SSD: 4000,
            },
        )
        self.pool.activate_namespace("model-a")
        item = self.pool.register_variant(
            identity(), canonical_source_state_digest="state", summary_digest="summary"
        )
        self.item = item
        self.pool.register_artifact(
            "model-a", "content-a", item.source_variant_id, artifact(item.source_variant_id)
        )

    def test_exactly_one_artifact_and_summary_is_separate(self):
        self.assertEqual(self.item.summary_digest, "summary")
        with self.assertRaises(ValueError):
            self.pool.register_artifact(
                "model-a",
                "content-a",
                self.item.source_variant_id,
                artifact(self.item.source_variant_id, "artifact-b"),
            )

    def test_multiple_tier_replicas_keep_artifact_identity(self):
        cpu = self.pool.attach_replica(
            "model-a",
            "content-a",
            self.item.source_variant_id,
            tier=KVLocation.PINNED_CPU,
            locator_value="cpu:0",
            layout_signature="contiguous-bf16",
            bytes_digest="cpu-bytes",
            size_bytes=500,
        )
        gpu = self.pool.attach_replica(
            "model-a",
            "content-a",
            self.item.source_variant_id,
            tier=KVLocation.GPU,
            locator_value="blocks:1-4",
            layout_signature="paged-bf16",
            bytes_digest="gpu-bytes",
            size_bytes=500,
            derived_from_replica_id=cpu.replica_id,
        )
        self.assertEqual(cpu.artifact_id, gpu.artifact_id)
        self.assertNotEqual(cpu.replica_id, gpu.replica_id)
        with self.assertRaises(ValueError):
            self.pool.attach_replica(
                "model-a",
                "content-a",
                self.item.source_variant_id,
                tier=KVLocation.GPU,
                locator_value="blocks:8-11",
                layout_signature="paged-bf16",
                bytes_digest="gpu-2",
                size_bytes=500,
            )

    def test_relocation_changes_epoch_not_replica_identity(self):
        replica = self.pool.attach_replica(
            "model-a",
            "content-a",
            self.item.source_variant_id,
            tier=KVLocation.GPU,
            locator_value="blocks:1",
            layout_signature="paged",
            bytes_digest="gpu",
            size_bytes=100,
        )
        old_id = replica.replica_id
        old_epoch = replica.locator.placement_epoch
        moved = self.pool.relocate_replica(
            "model-a",
            "content-a",
            self.item.source_variant_id,
            replica.replica_id,
            locator_value="blocks:9",
        )
        self.assertEqual(moved.replica_id, old_id)
        self.assertGreater(moved.locator.placement_epoch, old_epoch)

    def test_stale_binding_and_lease_protection(self):
        replica = self.pool.attach_replica(
            "model-a",
            "content-a",
            self.item.source_variant_id,
            tier=KVLocation.GPU,
            locator_value="blocks:1",
            layout_signature="paged",
            bytes_digest="gpu",
            size_bytes=900,
        )
        old_epoch = replica.locator.placement_epoch
        self.pool.relocate_replica(
            "model-a", "content-a", self.item.source_variant_id, replica.replica_id,
            locator_value="blocks:2",
        )
        with self.assertRaises(RuntimeError):
            self.pool.bind_replica(
                "model-a", "content-a", self.item.source_variant_id, replica.replica_id,
                artifact_generation=1,
                replica_generation=replica.generation,
                placement_epoch=old_epoch,
            )
        with self.pool.lease_replica(
            "model-a", "content-a", self.item.source_variant_id, replica.replica_id
        ):
            with self.assertRaises(MemoryError):
                self.pool.attach_replica(
                    "model-a", "content-a", self.item.source_variant_id,
                    tier=KVLocation.PINNED_CPU,
                    locator_value="cpu:0", layout_signature="packed",
                    bytes_digest="cpu", size_bytes=2500,
                )

    def test_copy_and_execution_inflight_are_eviction_protected(self):
        replica = self.pool.attach_replica(
            "model-a", "content-a", self.item.source_variant_id,
            tier=KVLocation.GPU, locator_value="gpu:busy",
            layout_signature="paged", bytes_digest="gpu", size_bytes=900,
        )
        for guard in (self.pool.copy_replica, self.pool.execute_replica):
            with guard(
                "model-a", "content-a", self.item.source_variant_id,
                replica.replica_id,
            ):
                with self.assertRaises(RuntimeError):
                    self.pool.purge_namespace("model-a")

    def test_model_namespace_purge_is_isolated(self):
        self.pool.activate_namespace("model-b")
        other = self.pool.register_variant(
            identity("model-b", "content-b", "occ-b"),
            canonical_source_state_digest="state-b",
            summary_digest="summary-b",
        )
        self.pool.purge_namespace("model-a")
        self.assertEqual(
            self.pool.variants_for_content("model-b", "content-b", include_unavailable=True),
            (other,),
        )

    def test_backing_replica_probation_precedes_value_eviction(self):
        pool = V7SourcePool(
            serving_mode=ModelServingMode.MULTI,
            tier_capacity_bytes={KVLocation.GPU: 100},
            probation_observations=2,
        )
        pool.activate_namespace("model-a")
        first = pool.register_variant(
            identity(occurrence="first"),
            canonical_source_state_digest="state",
            summary_digest="summary-first",
        )
        pool.register_artifact(
            "model-a", "content-a", first.source_variant_id,
            artifact(first.source_variant_id, "artifact-first"),
        )
        pool.attach_replica(
            "model-a", "content-a", first.source_variant_id,
            tier=KVLocation.GPU, locator_value="gpu:first",
            layout_signature="paged", bytes_digest="bytes-first", size_bytes=100,
        )
        second = pool.register_variant(
            identity(occurrence="second"),
            canonical_source_state_digest="state",
            summary_digest="summary-second",
        )
        pool.register_artifact(
            "model-a", "content-a", second.source_variant_id,
            artifact(second.source_variant_id, "artifact-second"),
        )
        with self.assertRaises(MemoryError):
            pool.attach_replica(
                "model-a", "content-a", second.source_variant_id,
                tier=KVLocation.GPU, locator_value="gpu:second",
                layout_signature="paged", bytes_digest="bytes-second", size_bytes=100,
            )
        for _ in range(2):
            pool.record_observation(
                "model-a", "content-a", first.source_variant_id,
                lookup_hit=True, compared=True,
            )
        pool.attach_replica(
            "model-a", "content-a", second.source_variant_id,
            tier=KVLocation.GPU, locator_value="gpu:second",
            layout_signature="paged", bytes_digest="bytes-second", size_bytes=100,
        )
        self.assertFalse(first.runtime_available)
        self.assertTrue(second.runtime_available)

    def test_single_model_switch_purges_only_old_namespace(self):
        pool = V7SourcePool(serving_mode=ModelServingMode.SINGLE)
        pool.activate_namespace("model-a")
        first = pool.register_variant(
            identity(), canonical_source_state_digest="state", summary_digest="summary"
        )
        pool.register_artifact(
            "model-a", "content-a", first.source_variant_id,
            artifact(first.source_variant_id),
        )
        pool.attach_replica(
            "model-a", "content-a", first.source_variant_id,
            tier=KVLocation.PINNED_CPU, locator_value="cpu:a",
            layout_signature="packed", bytes_digest="bytes", size_bytes=100,
        )
        pool.switch_single_model("model-b")
        self.assertEqual(
            pool.variants_for_content("model-a", "content-a", include_unavailable=True),
            (),
        )
        second = pool.register_variant(
            identity("model-b", "content-b", "occ-b"),
            canonical_source_state_digest="state-b", summary_digest="summary-b",
        )
        self.assertEqual(second.identity.model_math_signature, "model-b")


if __name__ == "__main__":
    unittest.main()
