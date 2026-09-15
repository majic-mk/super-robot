import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('recovery_plan', Path(__file__).resolve().parents[1] / 'scripts/server/generate_single_segment_recovery_plan.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecoveryPlanTests(unittest.TestCase):
    def test_plan_is_deterministic_and_dependencies_precede_consumers(self):
        args = ('a'*40, 'b'*64, '/env/python', '/cacheblend', '/audit.json', '/results')
        plan = module.build_plan(*args)
        self.assertEqual(plan, module.build_plan(*args))
        seen = set()
        for job in plan['jobs']:
            self.assertTrue(set(job['requires']) <= seen)
            self.assertNotIn(job['job_id'], seen)
            seen.add(job['job_id'])
        self.assertEqual(len(plan['jobs']), 36)
        self.assertIsNone(plan['cost_measurement_sha256'])
        self.assertFalse(plan['online_trace_execution_allowed'])
        self.assertEqual(sum(not p['warmup'] for p in plan['pair_schedule']), 20)

    def test_short_sha_rejected(self):
        with self.assertRaises(ValueError):
            module.build_plan('abc', '', '', '', '', '')
