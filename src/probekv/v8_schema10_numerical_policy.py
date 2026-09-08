"""Explicit process-local arithmetic provenance; BF16 KV format is unchanged."""
from importlib import import_module
from .v8_schema10_execution import digest_json


def validate_numerical_policy(runtime):
    policy = runtime.get('numerical_execution_policy')
    if policy is None:
        return None  # historical manifests are read without a silent upgrade
    if (set(policy) not in ({'allow_bf16_reduced_precision_reduction'},
                           {'allow_bf16_reduced_precision_reduction', 'prefill_attention_kernel'})
            or type(policy['allow_bf16_reduced_precision_reduction']) is not bool):
        raise ValueError('unsupported numerical execution policy')
    if 'prefill_attention_kernel' in policy and policy['prefill_attention_kernel'] != 'cutlass_mha':
        raise ValueError('unsupported prefill attention kernel')
    if runtime['cost_provenance'].get('numerical_execution_policy_sha256') != digest_json(policy):
        raise ValueError('numerical policy differs from measurement provenance')
    return dict(policy)


def apply_numerical_policy(torch_module, policy):
    if policy is not None:
        if 'prefill_attention_kernel' in policy:
            backend = import_module('vllm.attention.backends.xformers')
            if not hasattr(backend, '_PROBEKV_PREFILL_KERNEL'):
                raise RuntimeError('pinned backend lacks explicit prefill kernel policy')
            backend._PROBEKV_PREFILL_KERNEL = policy['prefill_attention_kernel']
        torch_module.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = policy['allow_bf16_reduced_precision_reduction']
        assert_numerical_policy(torch_module, policy)


def assert_numerical_policy(torch_module, policy):
    if policy is not None and torch_module.backends.cuda.matmul.allow_bf16_reduced_precision_reduction != policy['allow_bf16_reduced_precision_reduction']:
        raise RuntimeError('numerical execution policy changed since backend creation')
    if policy is not None and 'prefill_attention_kernel' in policy:
        backend = import_module('vllm.attention.backends.xformers')
        if getattr(backend, '_PROBEKV_PREFILL_KERNEL', None) != policy['prefill_attention_kernel']:
            raise RuntimeError('prefill attention kernel changed since backend creation')
