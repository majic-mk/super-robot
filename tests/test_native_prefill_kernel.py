import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from probekv.cacheblend_patch import validate_unified_diff


class MatchedPrefillKernelTests(unittest.TestCase):
    def test_actual_patch_helper_preserves_legacy_and_flattens_gqa(self):
        path = Path(__file__).resolve().parents[1] / 'patches/cacheblend/0011-probekv-matched-prefill-kernel.patch'
        validate_unified_diff(path)
        first_hunk = path.read_text(encoding='utf-8').split('@@')[2]
        code = '\n'.join(line[1:] for line in first_hunk.splitlines() if line.startswith('+'))
        calls = []
        def forward(q, k, v, **kwargs):
            calls.append((q, k, v, kwargs))
            return q
        scope = dict(torch=torch, xops=SimpleNamespace(
            memory_efficient_attention_forward=forward,
            MemoryEfficientAttentionCutlassOp=('cutlass-fw', 'cutlass-bw')))
        exec(code, scope)
        for groups, heads_per_group in ((8, 4), (4, 7)):
            q = torch.randn(1, 3, groups, heads_per_group, 8)
            k = torch.randn(1, 5, groups, 1, 8).expand(1, 5, groups, heads_per_group, 8)
            v = k + 1
            bias = torch.zeros(1, groups, heads_per_group, 3, 5)
            scope['_PROBEKV_PREFILL_KERNEL'] = None
            scope['_probekv_attention_forward'](q, k, v, attn_bias=bias, p=0., scale=.125)
            self.assertIs(calls[-1][0], q)
            self.assertNotIn('op', calls[-1][-1])
            scope['_PROBEKV_PREFILL_KERNEL'] = 'cutlass_mha'
            scope['_probekv_attention_forward'](q, k, v, attn_bias=bias, p=0., scale=.125)
            actual = calls[-1]
            self.assertEqual(actual[0].shape, (1, 3, groups * heads_per_group, 8))
            self.assertTrue(torch.equal(actual[1], k.flatten(2, 3)))
            self.assertEqual(actual[-1]['attn_bias'].shape, (1, groups * heads_per_group, 3, 5))
            self.assertEqual(actual[-1]['op'], 'cutlass-fw')
        scope['_PROBEKV_PREFILL_KERNEL'] = 'unrecorded'
        with self.assertRaises(RuntimeError):
            scope['_probekv_attention_forward'](q, k, v, attn_bias=bias, p=0., scale=.125)
