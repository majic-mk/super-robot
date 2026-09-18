import unittest
from probekv.source_quality_partition import validation_only_representatives, validation_only_roles, freeze_group_roles


class ValidationOnlyTests(unittest.TestCase):
    def test_one_slice_selected_deterministically_before_outcomes(self):
        rows = [dict(group_id='g', case_id='b'), dict(group_id='g', case_id='a')]
        self.assertEqual(validation_only_representatives(rows), [rows[1]])
        self.assertEqual(validation_only_representatives(rows[::-1]), [rows[1]])

    def test_one_group_no_fit_does_not_weaken_default(self):
        row = dict(group_id='g', content_key='c', source_origin_ids=list('abcd'), target_origin_ids=list('efghi'))
        self.assertEqual(validation_only_roles([row]), {'g':'validation'})
        with self.assertRaisesRegex(ValueError, 'at least two'):
            freeze_group_roles([row])
        with self.assertRaises(ValueError):
            validation_only_roles([row, row])
        with self.assertRaises(ValueError):
            validation_only_roles([dict(row, target_origin_ids=list('abcde'))])
