import unittest
from probekv.v8_schema10_cost_provider import digest_joint_query, RequestExecutionShape
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema6_planner import JointTimelineContext


class JointQueryEncodingTests(unittest.TestCase):
    def test_exact_canonical_hash_and_mutation(self):
        rows = list(range(512))
        query = {'category': 'joint_future', 'key_contract': 'execution_shape_v1',
                 'geometry': {'layer_active_positions': {str(i): rows for i in range(1, 33)},
                              'sampling': {'text': '中文', 'temperature': 0.0}}}
        before = digest_joint_query(query)
        self.assertEqual(before, digest_json(query))
        rows[12] = 900
        self.assertEqual(digest_joint_query(query), digest_json(query))
        self.assertNotEqual(before, digest_joint_query(query))

    def test_numerically_equal_different_json_types_not_collapsed(self):
        query = {'geometry': {'layer_active_positions': {'1': [1], '2': [True], '3': [1.0]}}}
        self.assertEqual(digest_joint_query(query), digest_json(query))
        self.assertEqual(digest_joint_query({'legacy': [1, 2]}), digest_json({'legacy': [1, 2]}))

    def test_masks_reuse_only_equal_support_and_rebuild_on_mutation(self):
        repair = {'a': {2: (3, 4), 3: (3, 4), 4: (4,)}}
        shape = RequestExecutionShape(10, 2, 4, 1, {'a': (3, 4, 5)}, repair, {}, {}, {'sampling': {}})
        context = JointTimelineContext(('a',), ('a',), (), (), {'a': 2}, 'mask', 'snapshot')
        masks = shape.masks(context)
        self.assertIs(masks[2], masks[3])
        self.assertEqual(masks[2], (2, 3, 4, 6, 7, 8, 9))
        self.assertEqual(masks[4], (2, 4, 6, 7, 8, 9))
        repair['a'][3] = (5,)
        self.assertEqual(shape.masks(context)[3], (2, 5, 6, 7, 8, 9))


if __name__ == '__main__':
    unittest.main()
