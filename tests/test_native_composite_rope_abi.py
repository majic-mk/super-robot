import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from probekv.cacheblend_patch import validate_unified_diff


class CompositeRopeABITests(unittest.TestCase):
    def test_both_model_patches_flatten_every_kv_head_with_token_stride(self):
        path = Path(__file__).resolve().parents[1] / 'patches/cacheblend/0010-probekv-composite-rope-flat-heads.patch'
        validate_unified_diff(path)
        pieces = path.read_text(encoding='utf-8').split('diff --git ')[1:]
        self.assertEqual(len(pieces), 2)
        for piece in pieces:
            code = textwrap.dedent('\n'.join(line[1:] for line in piece.splitlines()
                                            if line.startswith('+') and not line.startswith('+++')))
            for heads in (4, 8):
                for structured in (False, True):
                    key = torch.arange(5 * heads * 128).reshape(5, heads, 128).float()
                    original = key.clone()
                    if not structured:
                        key = key.view(5, -1)
                    calls = []
                    def rotary(positions, query, flat):
                        # The actual pinned C++ entry uses precisely these
                        # two dimensions, not Tensor.numel()/token_count.
                        self.assertEqual(flat.size(-1) // 128, heads)
                        self.assertEqual(flat.stride(-2), heads * 128)
                        self.assertEqual(tuple(flat.shape), (5, heads * 128))
                        calls.append(True)
                        flat.add_(1)  # stand-in for the in-place CUDA kernel
                        return query, flat
                    state = dict(self=SimpleNamespace(rotary_emb=rotary), old_kv=[key, None],
                                 cache_fuse_metadata={'org_pos': torch.arange(5), 'fake_q': torch.zeros(5, 4096)})
                    exec(code, state)
                    self.assertEqual(calls, [True])
                    self.assertEqual(state['old_kv'][0].shape, key.shape)
                    self.assertTrue(torch.equal(state['old_kv'][0].reshape_as(original), original + 1))


if __name__ == '__main__':
    unittest.main()
