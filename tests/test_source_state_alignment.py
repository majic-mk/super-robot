import copy
import unittest
import torch
from probekv.source_state_alignment import audit_source_identity, audit_self_observation
from probekv.source_policy_replay import build_observation, observe_depth_k, replay_observation, validate_observation
from probekv.v7_contracts import SourceVariantIdentity
from probekv.v8_schema10_execution import digest_json
from test_source_policy_replay import observation, reseal


class AlignmentTests(unittest.TestCase):
    def test_legacy_depths_self_zero_and_no_policy_replay(self):
        old = observation(1)
        k = torch.ones(20, 2, 4, dtype=torch.bfloat16)
        value = build_observation(provenance=old['provenance'], absolute_positions=old['absolute_positions'],
            correctness_eligible_source_ids=['s0'], legacy_diagnostic=True,
            depth_observations=[observe_depth_k(k, {'s0': k.clone()}, completed_depth=d) for d in (1,2,4,5,8)])
        self.assertTrue(audit_self_observation({'observation': value}, 's0')['passed'])
        with self.assertRaisesRegex(ValueError, 'masquerade'):
            replay_observation(value)
        value['depth_observations'].pop()
        reseal(value)
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            validate_observation(value)

    def test_self_drift_is_not_accepted(self):
        self.assertFalse(audit_self_observation({'observation': observation(1)}, 's0')['passed'])

    def test_content_ordinal_alignment_not_equal_absolute_position(self):
        q = dict(request_id='old', token_ids=[8,9,1,2,3],
                 segments=[dict(segment_id='C', content_key='c', positions=[2,3], token_ids=[1,2])])
        target = dict(request_id='new', token_ids=[7,8,9,1,2,3],
                 segments=[dict(segment_id='C', content_key='c', positions=[3,4], token_ids=[1,2])])
        sid = SourceVariantIdentity('c', digest_json([8,9]), digest_json([2,3]), 'old:C', 'model').source_variant_id
        args = (q, target, sid, {'token_ids':[1,2]}, 'model')
        self.assertTrue(audit_source_identity(*args)['identity_and_position_contract_passed'])
        bad = copy.deepcopy(target)
        bad['segments'][0]['positions'] = [2,3]
        with self.assertRaisesRegex(ValueError, 'position/token'):
            audit_source_identity(q, bad, sid, args[3], 'model')
        with self.assertRaisesRegex(ValueError, 'historical prefix'):
            audit_source_identity(q, target, sid, args[3], 'other-model')
