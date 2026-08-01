import unittest

from probekv.contracts import HistoricalSource, KVLocation, SourceOrigin
from probekv.global_source_pool import (
    GlobalEvictionPolicy,
    GlobalSourcePool,
    ModelNamespaceState,
    ModelServingMode,
)
from probekv.manifest import token_content_hash
from tests.test_v6_contracts_budget import structured_signature


def make_source(model, content, index):
    return HistoricalSource(
        source_id="%s-s%d" % (content[:8], index),
        content_hash=content,
        context_id="ctx-%s-%d" % (content[:8], index),
        model_signature=model,
        token_count=2,
        exact=True,
        origin=SourceOrigin.FULL_PREFILL,
        kv_location=KVLocation.PINNED_CPU,
    )


class GlobalSourcePoolTests(unittest.TestCase):
    def test_sixteen_variants_and_value_eviction(self):
        model = structured_signature("pool")
        content = token_content_hash((1, 2))
        pool = GlobalSourcePool(max_variants_per_content=16)
        pool.activate_model(model)
        for index in range(16):
            source = make_source(model, content, index)
            pool.register(source, 100)
            # End the two-observation probation before the cap is exercised.
            pool.record_comparison(model, content, source.source_id)
            pool.record_comparison(model, content, source.source_id)
        pool.record_selection(model, content, make_source(model, content, 15).source_id)
        pool.record_admission(
            model, content, make_source(model, content, 15).source_id, 100.0
        )

        pool.register(make_source(model, content, 16), 100)
        ids = {source.source_id for source in pool.candidates(model, content)}
        self.assertEqual(len(ids), 16)
        self.assertIn(make_source(model, content, 15).source_id, ids)
        self.assertIn(make_source(model, content, 16).source_id, ids)
        self.assertTrue(
            any(event.reason == "per_content_variant_cap" for event in pool.events)
        )

    def test_global_capacity_can_remove_entire_low_value_content(self):
        model = structured_signature("capacity")
        low = token_content_hash((10, 11))
        high = token_content_hash((20, 21))
        incoming = token_content_hash((30, 31))
        pool = GlobalSourcePool(canonical_capacity_bytes=200)
        pool.activate_model(model)
        for content in (low, high):
            source = make_source(model, content, 0)
            pool.register(source, 100)
            pool.record_comparison(model, content, source.source_id)
            pool.record_comparison(model, content, source.source_id)
        high_source = make_source(model, high, 0)
        pool.record_selection(model, high, high_source.source_id)
        pool.record_admission(model, high, high_source.source_id, 80.0)

        pool.register(make_source(model, incoming, 0), 100)
        self.assertEqual(pool.candidates(model, low), ())
        self.assertEqual(len(pool.candidates(model, high)), 1)
        self.assertEqual(len(pool.candidates(model, incoming)), 1)

    def test_busy_source_and_replica_are_protected(self):
        model = structured_signature("busy")
        content_a = token_content_hash((40, 41))
        content_b = token_content_hash((50, 51))
        source_a = make_source(model, content_a, 0)
        source_b = make_source(model, content_b, 0)
        pool = GlobalSourcePool(
            tier_capacity_bytes={KVLocation.GPU: 100},
            probation_observations=0,
        )
        pool.activate_model(model)
        pool.register(source_a, 10)
        pool.register(source_b, 10)
        pool.attach_replica(model, content_a, source_a.source_id, KVLocation.GPU, 100)
        pool.lease(model, content_a, source_a.source_id)
        with self.assertRaisesRegex(MemoryError, "unleased"):
            pool.attach_replica(
                model, content_b, source_b.source_id, KVLocation.GPU, 100
            )
        pool.release(model, content_a, source_a.source_id)
        pool.attach_replica(model, content_b, source_b.source_id, KVLocation.GPU, 100)
        self.assertNotIn(KVLocation.GPU, pool.lifecycle(model, content_a, source_a.source_id).replicas)

    def test_canonical_bytes_count_against_physical_tier_capacity(self):
        model = structured_signature("tier")
        first = token_content_hash((91, 92))
        second = token_content_hash((93, 94))
        pool = GlobalSourcePool(
            tier_capacity_bytes={KVLocation.PINNED_CPU: 100},
            probation_observations=0,
        )
        pool.activate_model(model)
        pool.register(make_source(model, first, 0), 100)
        pool.register(make_source(model, second, 0), 100)
        self.assertEqual(pool.candidates(model, first), ())
        self.assertEqual(len(pool.candidates(model, second)), 1)
        self.assertTrue(
            any(event.reason == "tier_content_eviction" for event in pool.events)
        )
        snapshot = pool.audit_snapshot()
        self.assertEqual(snapshot["tier_used_bytes"]["pinned_cpu"], 100)
        self.assertEqual(
            snapshot["model_namespaces"][model]["state"], "active"
        )

    def test_request_observation_updates_hit_and_miss_probabilities(self):
        model = structured_signature("hit")
        first = token_content_hash((95, 96))
        second = token_content_hash((97, 98))
        source_first = make_source(model, first, 0)
        source_second = make_source(model, second, 0)
        pool = GlobalSourcePool()
        pool.activate_model(model)
        pool.register(source_first, 10)
        pool.register(source_second, 10)
        pool.record_request_observation(model, (first,))
        first_stats = pool.lifecycle(model, first, source_first.source_id).stats
        second_stats = pool.lifecycle(model, second, source_second.source_id).stats
        self.assertEqual((first_stats.lookup_opportunities, first_stats.lookup_hits), (1, 1))
        self.assertEqual((second_stats.lookup_opportunities, second_stats.lookup_hits), (1, 0))

    def test_failed_registration_rolls_back_all_evictions(self):
        model = structured_signature("atomic")
        content = token_content_hash((101, 102))
        pool = GlobalSourcePool(
            max_variants_per_content=2,
            tier_capacity_bytes={KVLocation.PINNED_CPU: 200},
            probation_observations=0,
        )
        pool.activate_model(model)
        pool.register(make_source(model, content, 0), 100)
        pool.register(make_source(model, content, 1), 100)
        before_ids = tuple(source.source_id for source in pool.candidates(model, content))
        before_events = pool.events
        with self.assertRaises(MemoryError):
            pool.register(make_source(model, content, 2), 300)
        self.assertEqual(
            tuple(source.source_id for source in pool.candidates(model, content)),
            before_ids,
        )
        self.assertEqual(pool.events, before_events)

    def test_single_model_switch_drains_then_purges_only_old_model(self):
        old = structured_signature("old")
        new = structured_signature("new")
        content = token_content_hash((60, 61))
        source = make_source(old, content, 0)
        pool = GlobalSourcePool(serving_mode=ModelServingMode.SINGLE)
        pool.activate_model(old)
        pool.register(source, 10)
        pool.lease(old, content, source.source_id)

        self.assertFalse(pool.switch_single_model(new))
        self.assertEqual(pool.namespace_state(old), ModelNamespaceState.DRAINING)
        self.assertEqual(pool.candidates(old, content), ())
        pool.release(old, content, source.source_id)
        self.assertTrue(pool.switch_single_model(new))
        self.assertEqual(pool.namespace_state(old), ModelNamespaceState.DELETED)
        self.assertEqual(pool.namespace_state(new), ModelNamespaceState.ACTIVE)

    def test_multi_model_targeted_purge_preserves_other_namespace(self):
        first = structured_signature("first")
        second = structured_signature("second")
        content = token_content_hash((70, 71))
        pool = GlobalSourcePool(serving_mode=ModelServingMode.MULTI)
        pool.activate_model(first)
        pool.activate_model(second)
        pool.register(make_source(first, content, 0), 10)
        pool.register(make_source(second, content, 0), 10)

        pool.purge_model(first)
        self.assertEqual(pool.namespace_state(first), ModelNamespaceState.DELETED)
        self.assertEqual(pool.namespace_state(second), ModelNamespaceState.ACTIVE)
        self.assertEqual(len(pool.candidates(second, content)), 1)

    def test_model_soft_quota_prioritizes_eviction_under_global_pressure(self):
        first = structured_signature("quota-first")
        second = structured_signature("quota-second")
        content_first = token_content_hash((111, 112))
        content_second = token_content_hash((113, 114))
        incoming = token_content_hash((115, 116))
        pool = GlobalSourcePool(
            serving_mode=ModelServingMode.MULTI,
            tier_capacity_bytes={KVLocation.PINNED_CPU: 200},
            probation_observations=0,
        )
        pool.register_namespace(first, {KVLocation.PINNED_CPU: 50})
        pool.register_namespace(second, {KVLocation.PINNED_CPU: 200})
        pool.activate_model(first)
        pool.activate_model(second)
        source_first = make_source(first, content_first, 0)
        pool.register(source_first, 100)
        pool.record_selection(first, content_first, source_first.source_id)
        pool.record_admission(first, content_first, source_first.source_id, 100.0)
        pool.register(make_source(second, content_second, 0), 100)
        pool.register(make_source(second, incoming, 0), 100)
        self.assertEqual(pool.candidates(first, content_first), ())
        self.assertEqual(len(pool.candidates(second, content_second)), 1)
        self.assertEqual(len(pool.candidates(second, incoming)), 1)

    def test_cachecraft_fr_baseline_is_separate_policy(self):
        model = structured_signature("fr")
        content = token_content_hash((80, 81))
        pool = GlobalSourcePool(
            eviction_policy=GlobalEvictionPolicy.CACHE_CRAFT_FR,
            probation_observations=0,
        )
        pool.activate_model(model)
        source = make_source(model, content, 0)
        pool.register(source, 10)
        pool.record_cachecraft_access(model, content, source.source_id, 0.25)
        self.assertAlmostEqual(
            pool.lifecycle(model, content, source.source_id).stats.cache_craft_fr,
            4.0,
        )
        pool.record_cachecraft_access(model, content, source.source_id, 0.0)
        self.assertTrue(
            pool.lifecycle(model, content, source.source_id).stats.cache_craft_fr
            >= 1_000_004.0
        )


if __name__ == "__main__":
    unittest.main()
