import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from probekv.v8_schema10_measured_costs import MeasuredRequestCostProvider as Provider
from probekv.v8_schema10_cost_provider import EXECUTION_SHAPE_KEY, LEGACY_IDENTITY_KEY
from probekv.v8_schema10_execution import digest_json


class IdentityMemoTests(unittest.TestCase):
    def context(self):
        return NS(request={'token_ids': [1, 2, 3]}, cached_prefix_tokens=0,
                  prefix_cache_mode='native_exact_blocks', sampling_signature={'max_new_tokens': 32})

    def test_static_tokens_hashed_once_and_mutation_invalidates(self):
        c = self.context()
        with patch('probekv.v8_schema10_measured_costs.digest_json', wraps=digest_json) as h:
            first = Provider.identity(c)
            self.assertEqual(first, Provider.identity(c))
            self.assertEqual(h.call_count, 1)
            c.request['token_ids'][1] = 99
            second = Provider.identity(c)
            self.assertNotEqual(first, second)
            self.assertEqual(h.call_count, 2)
        self.assertEqual(second['prompt_token_ids_sha256'], digest_json(c.request['token_ids']))

    def test_prefix_sampling_not_memoized_and_returns_no_mutable_alias(self):
        c = self.context()
        first = Provider.identity(c)
        first['sampling']['max_new_tokens'] = 1
        self.assertEqual(c.sampling_signature['max_new_tokens'], 32)
        c.cached_prefix_tokens = 2
        c.sampling_signature['max_new_tokens'] = 64
        second = Provider.identity(c)
        self.assertEqual(second['cached_prefix_tokens'], 2)
        self.assertEqual(second['sampling']['max_new_tokens'], 64)

    def test_invalid_token_types_cannot_alias_integer_cache(self):
        c = self.context()
        Provider.identity(c)
        c.request['token_ids'][0] = True
        with self.assertRaises(ValueError):
            Provider.identity(c)

    def test_shape_key_does_not_build_unused_legacy_identity(self):
        p = object.__new__(Provider)
        p.key_contract = EXECUTION_SHAPE_KEY
        c = self.context()
        c.source_measurement_shape = lambda *args: dict(prompt_tokens=4, prefix_tokens=0,
            positions=[1, 2], completed_depth=1, num_layers=4, dtype='bf16', kv_heads=2,
            head_dim=4, tier='pinned_cpu', bytes=64, layout='contiguous')
        def forbidden():
            raise AssertionError('legacy query should not be materialized')
        result = p._source_query(c, 'C', 'S', 1, 'source_future', forbidden)
        self.assertIsInstance(result, dict)
        p.key_contract = LEGACY_IDENTITY_KEY
        self.assertEqual(p._source_query(c, 'C', 'S', 1, 'source_future', lambda: {'legacy': 1}), {'legacy': 1})


if __name__ == '__main__': unittest.main()
