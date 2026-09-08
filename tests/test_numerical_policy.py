import unittest
from types import SimpleNamespace as NS
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_numerical_policy import validate_numerical_policy, apply_numerical_policy, assert_numerical_policy


class NumericalPolicyTests(unittest.TestCase):
    def test_policy_bound_to_costs_and_checked_before_requests(self):
        policy = {'allow_bf16_reduced_precision_reduction': False}
        runtime = dict(numerical_execution_policy=policy,
                       cost_provenance={'numerical_execution_policy_sha256': digest_json(policy)})
        self.assertEqual(validate_numerical_policy(runtime), policy)
        module = NS(backends=NS(cuda=NS(matmul=NS(allow_bf16_reduced_precision_reduction=True))))
        apply_numerical_policy(module, policy)
        assert_numerical_policy(module, policy)
        module.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        with self.assertRaises(RuntimeError):
            assert_numerical_policy(module, policy)
        runtime['cost_provenance']['numerical_execution_policy_sha256'] = 'old-costs'
        with self.assertRaises(ValueError):
            validate_numerical_policy(runtime)

    def test_historical_manifests_not_silently_reinterpreted(self):
        self.assertIsNone(validate_numerical_policy({}))
        with self.assertRaises(ValueError):
            validate_numerical_policy({'numerical_execution_policy': {'unknown': 1}, 'cost_provenance': {}})
