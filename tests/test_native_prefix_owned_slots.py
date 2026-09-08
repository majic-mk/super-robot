import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from probekv.cacheblend_patch import patch_files_for_mode
from probekv.v8_cfo import streaming_qk_attention_mass, eager_qk_attention_mass


ROOT = Path(__file__).resolve().parents[1]


class NativePrefixOwnedSlotTests(unittest.TestCase):
    def setUp(self):
        paths = patch_files_for_mode(ROOT / 'patches/cacheblend/manifest.json',
                                    'probekv_v8_variant_growth_counterfactual')
        self.patch_text = paths[-1].read_text(encoding='utf-8')
        # Execute the actual dependency-free helper shipped in the patch,
        # not a second local implementation of the ownership calculation.
        hunk = self.patch_text.split('@@')[2]
        code = '\n'.join(line[1:] for line in hunk.splitlines() if line.startswith('+'))
        namespace = {}
        exec(code, namespace)
        self.rows = namespace['_probekv_owned_cache_rows']

    def test_prefix_rows_are_excluded_but_all_owned_kv_is_written(self):
        key = torch.arange(448 * 2).reshape(448, 2)
        value = key + 1000
        meta = SimpleNamespace(slot_mapping=torch.arange(192), num_prefill_tokens=192, num_decode_tokens=0)
        k, v = self.rows(key, value, meta, dict(exact_prefix_tokens=256, org_seq_len=448))
        self.assertTrue(torch.equal(k, key[256:]))
        self.assertTrue(torch.equal(v, value[256:]))
        self.assertEqual(k.shape[0], meta.slot_mapping.numel())

    def test_bad_mapping_fails_before_cuda_write(self):
        key = torch.zeros(448, 2)
        meta = SimpleNamespace(slot_mapping=torch.arange(448), num_prefill_tokens=192, num_decode_tokens=0)
        with self.assertRaisesRegex(ValueError, 'slot count'):
            self.rows(key, key, meta, dict(exact_prefix_tokens=256, org_seq_len=448))

    def test_no_prefix_retains_full_composite(self):
        key = torch.zeros(448, 2)
        meta = SimpleNamespace(slot_mapping=torch.arange(448), num_prefill_tokens=448, num_decode_tokens=0)
        k, _ = self.rows(key, key, meta, dict(exact_prefix_tokens=0, org_seq_len=448))
        self.assertEqual(k.shape[0], 448)
        old = patch_files_for_mode(ROOT / 'patches/cacheblend/manifest.json',
                                  'probekv_v8_absolute_residual_variant_admission')
        self.assertEqual(len(old), 8)

    def test_resumable_queries_keep_full_keys_and_use_explicit_bias(self):
        self.assertIn('+            if not resumable_mode:', self.patch_text)
        self.assertIn('or (resumable_mode and attention_status in [1, 2])', self.patch_text)

    def test_cfo_block_transfer_preserves_gqa_reference_without_scalar_sync(self):
        generator = torch.Generator().manual_seed(37)
        q = torch.randn(19, 4, 8, generator=generator)
        k = torch.randn(19, 2, 8, generator=generator)
        ids = tuple(['a#0', 'b#0', 'a#1', 'a#0', 'c#0'][i % 5] for i in range(19))
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('per-pair scalar synchronization')):
            reference = eager_qk_attention_mass(q, k, ids, scale=8 ** -.5)
            for size in (1, 7, 64):
                observed = streaming_qk_attention_mass(q, k, ids, scale=8 ** -.5, block_size=size)
                for name in ('inter_mass_by_pair', 'intra_mass_by_occurrence'):
                    a, b = getattr(reference, name), getattr(observed, name)
                    self.assertEqual(set(a), set(b))
                    for identity in a:
                        self.assertAlmostEqual(a[identity], b[identity], places=5)


if __name__ == '__main__':
    unittest.main()
