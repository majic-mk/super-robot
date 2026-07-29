import unittest

from probekv.causal_mask import (
    absolute_causal_rows,
    bottom_right_causal_rows,
)


class CausalMaskGeometryTests(unittest.TestCase):
    def test_arbitrary_repaired_rows_keep_absolute_causality(self):
        query_positions = (2, 5, 8, 9)
        actual = absolute_causal_rows(query_positions, 10)
        self.assertTrue(actual[0][2])
        self.assertFalse(actual[0][3])
        self.assertTrue(actual[1][5])
        self.assertFalse(actual[1][6])
        self.assertNotEqual(actual, bottom_right_causal_rows(4, 10))


if __name__ == "__main__":
    unittest.main()
