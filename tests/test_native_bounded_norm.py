import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from probekv.cacheblend_patch import validate_unified_diff
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_numerical_policy import validate_numerical_policy, apply_numerical_policy, assert_numerical_policy


class BoundedNormTests(unittest.TestCase):
    def test_actual_patch_preserves_inplace_semantics_and_legacy(self):
        path = Path(__file__).resolve().parents[1] / 'patches/cacheblend/0012-probekv-bounded-fused-norm.patch'
        validate_unified_diff(path)
        hunk = path.read_text(encoding='utf-8').split('@@')[2]
        code = '\n'.join(line[1:] for line in hunk.splitlines() if line.startswith('+'))
        calls = []
        def fused(x, residual, weight, eps):
            calls.append(len(x))
            residual.add_(x)
            x.copy_(residual * weight)
        scope = dict(ops=SimpleNamespace(fused_add_rms_norm=fused))
        exec(code, scope)
        for limit in (None, 128):
            scope['_PROBEKV_FUSED_NORM_MAX_ROWS'] = limit
            for n in (1, 127, 128, 192, 256, 448):
                calls.clear()
                x, residual, weight = torch.ones(n, 8), torch.full((n, 8), 2.), torch.full((8,), 3.)
                scope['_probekv_fused_add_rms_norm'](x, residual, weight, 1e-5)
                self.assertTrue(torch.equal(x, torch.full_like(x, 9.)))
                self.assertTrue(torch.equal(residual, torch.full_like(residual, 3.)))
                self.assertEqual(sum(calls), n)
                self.assertLessEqual(max(calls), n if limit is None else 128)
        with self.assertRaises(ValueError):
            scope['_probekv_fused_add_rms_norm'](x[:, ::2], residual[:, ::2], weight[::2], 1e-5)

    def test_norm_setting_is_bound_and_guarded(self):
        p = dict(allow_bf16_reduced_precision_reduction=False, fused_norm_max_rows=128)
        r = dict(numerical_execution_policy=p, cost_provenance={'numerical_execution_policy_sha256': digest_json(p)})
        self.assertEqual(validate_numerical_policy(r), p)
        norm = SimpleNamespace(_PROBEKV_FUSED_NORM_MAX_ROWS=None)
        t = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_bf16_reduced_precision_reduction=True))))
        with patch('probekv.v8_schema10_numerical_policy.import_module', return_value=norm):
            apply_numerical_policy(t, p)
            self.assertEqual(norm._PROBEKV_FUSED_NORM_MAX_ROWS, 128)
            norm._PROBEKV_FUSED_NORM_MAX_ROWS = 256
            with self.assertRaises(RuntimeError):
                assert_numerical_policy(t, p)
        p['fused_norm_max_rows'] = True
        with self.assertRaises(ValueError):
            validate_numerical_policy(r)
