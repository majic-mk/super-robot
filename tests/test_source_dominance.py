import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('dominance', Path(__file__).resolve().parents[1]/'scripts/server/audit_source_dominance.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DominanceTests(unittest.TestCase):
    def test_fixed_source_dominates_and_ties_do_not_prove_complementarity(self):
        result = module.summarize_group([dict(dense_f1=1., f1_by_source={'a':1.,'b':0.}),
                                         dict(dense_f1=1., f1_by_source={'a':1.,'b':1.})])
        self.assertEqual(result['fixed_sources_safe_on_every_target'], ['a'])
        self.assertEqual(result['fixed_sources_max_f1_on_every_target'], ['a'])

    def test_complementarity_and_cohort_mismatch(self):
        rows = [dict(dense_f1=1., f1_by_source={'a':1.,'b':0.}),
                dict(dense_f1=1., f1_by_source={'a':0.,'b':1.})]
        result = module.summarize_group(rows)
        self.assertEqual(result['fixed_sources_safe_on_every_target'], [])
        self.assertEqual(result['oracle_safe_coverage'], 1.)
        rows[1]['f1_by_source'] = {'c':1.}
        with self.assertRaises(ValueError):
            module.summarize_group(rows)
