"""Fail-closed diagnostic identity checks, independent of ranking thresholds."""
from .v7_contracts import SourceVariantIdentity
from .v8_schema10_execution import digest_json


def audit_source_identity(source_request, target_request, source_id, metadata, model_signature):
    a, b = source_request['segments'][0], target_request['segments'][0]
    for request, segment in ((source_request, a), (target_request, b)):
        positions = segment['positions']
        if (not positions or positions != list(range(positions[0], positions[0]+len(positions))) or
                positions[-1] >= len(request['token_ids']) or positions[0] < 0 or
                [request['token_ids'][p] for p in positions] != segment['token_ids']):
            raise ValueError('absolute position/token identity mismatch')
    expected = SourceVariantIdentity(a['content_key'],
        digest_json(source_request['token_ids'][:a['positions'][0]]), digest_json(a['positions']),
        source_request['request_id']+':'+a['segment_id'], model_signature)
    if expected.source_variant_id != source_id:
        raise ValueError('Source identity does not bind historical prefix/positions/model')
    if (a['content_key'] != b['content_key'] or a['token_ids'] != b['token_ids'] or
            metadata['token_ids'] != a['token_ids']):
        raise ValueError('Source/target content rows differ')
    return dict(source_id=source_id, historical_positions=a['positions'],
        target_positions=b['positions'], token_ids_sha256=digest_json(a['token_ids']),
        historical_prefix_sha256=expected.historical_prefix_digest,
        mapping='content_ordinal_to_target_absolute_position',
        k_semantics='pre_rope', extra_rope_transform_for_selection=False,
        identity_and_position_contract_passed=True)


def audit_self_observation(capture, source_id, limit=1e-4):
    from .source_policy_replay import validate_observation
    observation = capture['observation']
    validate_observation(observation)
    rows = []
    for depth in observation['depth_observations']:
        state = depth['sources'][source_id]
        rows.append(dict(completed_depth=depth['completed_depth'],
            relative_l2=state['relative_l2'],
            exact_digest_equal=state['selection_k_digest'] == depth['current_k_digest']))
    return dict(source_id=source_id, depth_rows=rows, relative_l2_limit=limit,
        passed=all(0 <= r['relative_l2'] <= limit for r in rows),
        scope='same_original_tokens_prefix_positions_canonical_vs_live_dense')
