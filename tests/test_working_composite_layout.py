import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from probekv.cacheblend_v6_online_engine import (
    allocate_working_composite, source_row_index,
)


class WorkingCompositeLayoutTests(unittest.TestCase):
    def test_packed_allocation_one_zero_call_private_contiguous_layers(self):
        source = [(torch.randn(5, 2, 4), torch.randn(5, 2, 4)) for _ in range(4)]
        api = SimpleNamespace(zeros=Mock(wraps=torch.zeros))
        rows = allocate_working_composite(api, source, 13, "cpu", packed=True)
        self.assertEqual(api.zeros.call_count, 1)
        for layer in rows:
            for tensor in layer:
                self.assertEqual(tuple(tensor.shape), (13, 2, 4))
                self.assertTrue(tensor.is_contiguous())
                self.assertEqual(tensor.count_nonzero().item(), 0)
        rows[0][0].fill_(7)
        self.assertEqual(rows[0][1].count_nonzero().item(), 0)
        self.assertEqual(rows[1][0].count_nonzero().item(), 0)
        for old_pair in source:
            for old in old_pair:
                self.assertNotEqual(old.data_ptr(), rows[0][0].data_ptr())

    def test_legacy_and_packed_full_composite_equal_source_unchanged(self):
        generator = torch.Generator().manual_seed(73)
        source = [(torch.randn(5, 2, 4, generator=generator, dtype=torch.bfloat16),
                   torch.randn(5, 2, 4, generator=generator, dtype=torch.bfloat16))
                  for _ in range(3)]
        prefix = [(torch.ones(3, 2, 4, dtype=torch.bfloat16),
                   torch.ones(3, 2, 4, dtype=torch.bfloat16) * 2) for _ in source]
        before = [[t.clone() for t in p] for p in source]
        for positions in ((5, 6, 7, 8, 9), (9, 5, 7, 6, 8)):
            results = []
            for packed in (False, True):
                rows = allocate_working_composite(torch, source, 13, "cpu", packed=packed)
                index = source_row_index(positions, contiguous_copy=packed)
                for layer, pair in enumerate(rows):
                    for j, t in enumerate(pair):
                        t[:3] = prefix[layer][j]
                        t[index] = source[layer][j]
                results.append(rows)
            for a, b in zip(results[0], results[1]):
                for x, y in zip(a, b):
                    self.assertTrue(torch.equal(x, y))
                    self.assertEqual(x[10:].count_nonzero().item(), 0)
            for a, b in zip(source, before):
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

    def test_contiguous_index_only_when_exactly_contiguous(self):
        self.assertEqual(source_row_index((4, 5, 6), contiguous_copy=True), slice(4, 7))
        for positions in ((), (4, 6), (6, 5), (4, 4)):
            self.assertEqual(source_row_index(positions, contiguous_copy=True), list(positions))
        self.assertEqual(source_row_index((4, 5), contiguous_copy=False), [4, 5])

    def test_packed_mixed_geometry_and_dtype_rejected(self):
        for second in (torch.zeros(2, 4), torch.zeros(2, 3, dtype=torch.float64)):
            with self.assertRaisesRegex(ValueError, "homogeneous"):
                allocate_working_composite(torch, [(torch.zeros(2, 3), second)], 8, "cpu", packed=True)


if __name__ == "__main__":
    unittest.main()
