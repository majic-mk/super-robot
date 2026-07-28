import unittest

from probekv.metrics import best_answer_f1, token_f1_text, token_id_f1


class MetricTests(unittest.TestCase):
    def test_answer_normalization_matches_standard_qa_behavior(self):
        self.assertEqual(token_f1_text("The Eiffel Tower.", "Eiffel Tower"), 1.0)
        self.assertEqual(best_answer_f1("Paris", ("London", "Paris")), 1.0)

    def test_output_token_f1_is_multiset_based(self):
        self.assertAlmostEqual(token_id_f1((1, 2, 2), (1, 2, 3)), 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
