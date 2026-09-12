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

if __name__ == '__main__': unittest.main()
