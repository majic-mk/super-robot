import unittest
from probekv.selection_result_cache import SelectionCacheKey, SelectionCacheEntry, SelectionResultCache

class SelectionResultCacheTest(unittest.TestCase):
    def key(self, generation=1):
        return SelectionCacheKey('t','m','tok','p','d',generation,'s')
    def test_strict_key_and_pool_invalidation(self):
        c = SelectionResultCache(); k = self.key()
        e = SelectionCacheEntry(k, 'A', 1, .1, 'e')
        c.put(e); self.assertEqual(c.get(k), e)
        c.invalidate_pool(2); self.assertIsNone(c.get(k))
    def test_different_prefix_or_dispatch_does_not_hit(self):
        c = SelectionResultCache(); k = self.key()
        c.put(SelectionCacheEntry(k, 'A', 1, .1, 'e'))
        self.assertIsNone(c.get(SelectionCacheKey('t','m','tok','other','d',1,'s')))
        self.assertIsNone(c.get(SelectionCacheKey('t','m','tok','p','other',1,'s')))

    def test_source_digest_and_generation_are_part_of_identity(self):
        c = SelectionResultCache(); k = self.key()
        c.put(SelectionCacheEntry(k, 'A', 1, .1, 'e'))
        changed = SelectionCacheKey('t','m','tok','p','d',1,'changed')
        self.assertIsNone(c.get(changed))
        self.assertIsNone(c.get(self.key(2)))

    def test_entry_requires_complete_evidence_match(self):
        e = SelectionCacheEntry(self.key(), 'A', 1, .1, 'e', 'c', 'g', 'r', 'p')
        self.assertTrue(e.admissible_for(candidate_set_digest='c', gate1_evidence_digest='g', repair_support_digest='r', planner_snapshot_digest='p'))
        self.assertFalse(e.admissible_for(candidate_set_digest='changed', gate1_evidence_digest='g', repair_support_digest='r', planner_snapshot_digest='p'))

if __name__ == '__main__': unittest.main()
