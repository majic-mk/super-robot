import copy
import unittest
from probekv.source_quality_partition import freeze_group_roles


class PartitionTests(unittest.TestCase):
    def groups(self):
        return [dict(group_id=g, content_key=g, source_origin_ids=[g+str(i) for i in range(4)],
                     target_origin_ids=[g+str(i) for i in range(4, 9)]) for g in ('a', 'b')]

    def test_deterministic_content_isolation(self):
        rows = self.groups()
        roles = freeze_group_roles(rows)
        self.assertEqual(roles, freeze_group_roles(rows[::-1]))
        self.assertEqual(set(roles.values()), {'fit', 'validation'})

    def test_overlap_and_insufficient_history_rejected(self):
        for field in ('group_id', 'content_key', 'source_origin_ids', 'target_origin_ids'):
            rows = copy.deepcopy(self.groups())
            rows[1][field] = rows[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                freeze_group_roles(rows)
        rows = self.groups()
        rows[0]['source_origin_ids'].pop()
        with self.assertRaises(ValueError):
            freeze_group_roles(rows)
