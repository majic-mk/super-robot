import unittest

from probekv.matrix import cartesian_rows, main_rag_matrix, profile_matrix


class MatrixTests(unittest.TestCase):
    def test_main_matrix_has_all_primary_cells(self):
        self.assertEqual(sum(1 for _ in main_rag_matrix()), 360)

    def test_profile_matrix_counts_optional_ssd(self):
        self.assertEqual(sum(1 for _ in profile_matrix(False)), 1620)
        self.assertEqual(sum(1 for _ in profile_matrix(True)), 2430)

    def test_empty_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            list(cartesian_rows({"model": []}))


if __name__ == "__main__":
    unittest.main()
