import unittest
from dataclasses import replace

from probekv.contracts import CaseSpec, SourceOrigin
from probekv.contracts import KVLocation
from probekv.source_store import (
    ReplicaEvictionPolicy,
    SourceEvictionPolicy,
    SourceStore,
)

from tests.helpers import canonical_source


class SourceContractTests(unittest.TestCase):
    def test_canonical_full_prefill_is_accepted(self):
        store = SourceStore(online_kmax=4)
        source = canonical_source()
        store.register(source)
        self.assertEqual(store.get("hash-c", "s1"), source)

    def test_selective_repair_cannot_be_promoted(self):
        store = SourceStore()
        repaired = replace(
            canonical_source(), origin=SourceOrigin.SELECTIVE_REPAIR, exact=False
        )
        with self.assertRaisesRegex(ValueError, "exact|promoted"):
            store.register(repaired)

    def test_each_context_requires_an_independent_source(self):
        store = SourceStore()
        store.register(canonical_source("s1", context_id="A"))
        with self.assertRaisesRegex(ValueError, "collision"):
            store.register(canonical_source("s1", context_id="B"))

    def test_online_kmax_is_enforced(self):
        store = SourceStore(online_kmax=2)
        store.register(canonical_source("s1", context_id="A"))
        store.register(canonical_source("s2", context_id="B"))
        with self.assertRaisesRegex(ValueError, "Kmax"):
            store.register(canonical_source("s3", context_id="E"))

    def test_case_rejects_cross_segment_source(self):
        case = CaseSpec(
            case_id="c",
            current_prompt="prompt",
            content_hash="hash-c",
            model_signature="model@revision",
            sources=(canonical_source(content_hash="different"),),
            split="test",
            regime="low-prefix/different-order",
            segment_length=512,
        )
        with self.assertRaisesRegex(ValueError, "same segment"):
            case.validate()

    def test_fifo_version_eviction_is_explicit_and_audited(self):
        store = SourceStore(
            online_kmax=2,
            eviction_policy=SourceEvictionPolicy.FIFO,
        )
        store.register(canonical_source("s1", context_id="A"))
        store.register(canonical_source("s2", context_id="B"))
        store.register(canonical_source("s3", context_id="C"))
        self.assertEqual(
            [source.source_id for source in store.candidates("hash-c")],
            ["s2", "s3"],
        )
        self.assertEqual(store.eviction_events[-1].reason, "version_fifo")

    def test_leased_source_is_not_evicted(self):
        store = SourceStore(
            online_kmax=1,
            eviction_policy=SourceEvictionPolicy.FIFO,
        )
        store.register(canonical_source("s1", context_id="A"))
        store.lease("hash-c", "s1")
        with self.assertRaisesRegex(RuntimeError, "leased"):
            store.register(canonical_source("s2", context_id="B"))

    def test_byte_aware_replica_lru_does_not_remove_canonical_source(self):
        store = SourceStore(
            online_kmax=2,
            replica_eviction_policy=ReplicaEvictionPolicy.LRU,
            tier_capacity_bytes={KVLocation.GPU: 100},
        )
        store.register(canonical_source("s1", context_id="A"))
        store.register(canonical_source("s2", context_id="B"))
        store.attach_replica("hash-c", "s1", KVLocation.GPU, 80)
        events = store.attach_replica(
            "hash-c", "s2", KVLocation.GPU, 80
        )
        self.assertEqual(events[0].reason, "replica_lru")
        self.assertEqual(len(store.candidates("hash-c")), 2)
        self.assertNotIn(
            KVLocation.GPU,
            store.lifecycle("hash-c", "s1").replicas,
        )

    def test_model_signature_is_part_of_store_identity(self):
        from dataclasses import replace

        store = SourceStore()
        store.register(canonical_source())
        store.register(
            replace(
                canonical_source(),
                model_signature="another@revision",
            )
        )
        with self.assertRaisesRegex(ValueError, "model_signature"):
            store.get("hash-c", "s1")


if __name__ == "__main__":
    unittest.main()
