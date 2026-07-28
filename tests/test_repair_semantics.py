import unittest

from probekv.repair_semantics import (
    TokenRegions,
    assert_nested_selections,
    repaired_segment_token_count,
    select_repair_tokens,
)


class RepairCountTests(unittest.TestCase):
    def test_endpoints_and_floor_are_frozen(self):
        self.assertEqual(repaired_segment_token_count(7, 0.0), 0)
        self.assertEqual(repaired_segment_token_count(7, 0.16), 1)
        self.assertEqual(repaired_segment_token_count(7, 1.0), 7)

    def test_only_c_is_ranked_and_suffix_is_mandatory(self):
        regions = TokenRegions(prefix_tokens=2, segment_tokens=4, suffix_tokens=2)
        scores = [100, 99, 1, 4, 3, 2, 1000, 1000]
        selection = select_repair_tokens(scores, regions, 0.5)
        self.assertEqual(selection.selected_segment_indices, (3, 4))
        self.assertEqual(selection.mandatory_suffix_indices, (6, 7))
        self.assertEqual(selection.execution_indices, (3, 4, 6, 7))
        self.assertAlmostEqual(selection.effective_ratio, 0.5)

    def test_ratio_grid_is_nested_with_stable_ties(self):
        regions = TokenRegions(prefix_tokens=1, segment_tokens=5, suffix_tokens=1)
        scores = [999, 2, 2, 2, 1, 0, 999]
        selections = [
            select_repair_tokens(scores, regions, ratio)
            for ratio in (0.0, 0.2, 0.5, 0.75, 1.0)
        ]
        assert_nested_selections(selections)
        self.assertEqual(selections[1].selected_segment_indices, (1,))

    def test_regions_must_cover_prompt(self):
        with self.assertRaisesRegex(ValueError, "cover"):
            select_repair_tokens(
                [0.0] * 7,
                TokenRegions(prefix_tokens=2, segment_tokens=4, suffix_tokens=2),
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
