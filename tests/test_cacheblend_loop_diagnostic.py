import unittest
from probekv.cacheblend_loop_diagnostic import loop_metadata, install_loop_metadata


class CacheBlendLoopDiagnosticTests(unittest.TestCase):
    def metadata(self, **changes):
        values = dict(positions=range(288, 800), prompt_tokens=832,
                      suffix_tokens=32, boundary=2, ratio=.15, cached_prefix_tokens=0)
        values.update(changes)
        return loop_metadata(**values)

    def test_native_prefix_is_explicitly_unsupported_not_silently_disabled(self):
        with self.assertRaisesRegex(ValueError, "native Prefix"):
            self.metadata(cached_prefix_tokens=256)

    def test_original_forward_check_layer_with_segment_adaptation(self):
        m = self.metadata()
        self.assertFalse(m["probekv_resumable"])
        self.assertEqual(m["check_layers"], [1])
        self.assertEqual(m["prefix_len"], 0)  # current prefix/bridge stay dense
        self.assertEqual(m["repair_regions"], [dict(segment_id="C", start=288,
                                                 length=512, recomp_ratio=.15)])
        self.assertEqual(m["repair_rounding_policy"], "ceil")

    def test_r1_uses_the_same_loop_not_a_dense_substitution(self):
        m = self.metadata(ratio=1.0)
        self.assertTrue(m["check"])
        self.assertEqual(m["repair_regions"][0]["recomp_ratio"], 1.0)

    def test_original_loop_cannot_inherit_resumable_local_indices(self):
        for stale in (None, [1, 2, 3]):
            m = dict(local_imp_indices=stale)
            install_loop_metadata(m, positions=range(288, 800), prompt_tokens=832,
                suffix_tokens=32, boundary=2, ratio=1.0, cached_prefix_tokens=0)
            self.assertNotIn("local_imp_indices", m)
            m["imp_indices"] = [0, 1, 2]
            self.assertEqual(m.get("local_imp_indices", m["imp_indices"]), [0, 1, 2])

    def test_bad_regions_fail_closed(self):
        for changes in (dict(positions=[]), dict(positions=[288, 290]),
                        dict(suffix_tokens=0), dict(boundary=1), dict(ratio=.3),
                        dict(positions=range(288, 802))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.metadata(**changes)


if __name__ == "__main__":
    unittest.main()
