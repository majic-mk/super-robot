"""Explicit process-local arithmetic provenance; BF16 KV format is unchanged."""
from .v8_schema10_execution import digest_json


def validate_numerical_policy(runtime):
    policy = runtime.get('numerical_execution_policy')
    if policy is None:
        return None  # historical manifests are read without a silent upgrade
    if (set(policy) != {'allow_bf16_reduced_precision_reduction'}
            or type(policy['allow_bf16_reduced_precision_reduction']) is not bool):
        raise ValueError('unsupported numerical execution policy')
    if runtime['cost_provenance'].get('numerical_execution_policy_sha256') != digest_json(policy):
        raise ValueError('numerical policy differs from measurement provenance')
    return dict(policy)


def apply_numerical_policy(torch_module, policy):
    if policy is not None:
        torch_module.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = policy['allow_bf16_reduced_precision_reduction']
        assert_numerical_policy(torch_module, policy)


def assert_numerical_policy(torch_module, policy):
    if policy is not None and torch_module.backends.cuda.matmul.allow_bf16_reduced_precision_reduction != policy['allow_bf16_reduced_precision_reduction']:
        raise RuntimeError('numerical execution policy changed since backend creation')
