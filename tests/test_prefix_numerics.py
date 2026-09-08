import unittest
import torch
from probekv.v8_schema10_prefix_numerics import tensor_difference


class PrefixNumericControlTests(unittest.TestCase):
    def test_exact_and_perturbed_layers_are_reported_not_declared_passed(self):
        a = torch.ones(5, 2, 4, dtype=torch.bfloat16)
        self.assertTrue(tensor_difference(a, a.clone())['equal'])
        b = a.clone()
        b[3] *= 2
        result = tensor_difference(a, b)
        self.assertFalse(result['equal'])
        self.assertGreater(result['relative_l2'], 0)
        self.assertNotIn('passed', result)

    def test_bad_geometry_or_nan_rejected(self):
        with self.assertRaises(ValueError):
            tensor_difference(torch.ones(2, 3), torch.ones(3, 2))
        with self.assertRaises(ValueError):
            tensor_difference(torch.ones(2, 3), torch.full((2, 3), float('nan')))
