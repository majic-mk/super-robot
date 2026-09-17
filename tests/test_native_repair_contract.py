from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from probekv.native_repair_contract import (
    PRIMARY_REPAIR_METRIC, historical_repair_metric, validate_repair_cost_evidence)
from probekv.v8_schema10_native_adapter import NativeRequestContext
from probekv.source_policy_development import development_experiment_spec


class NativeRepairContractTests(unittest.TestCase):
    def test_new_entrypoints_explicitly_select_joint_kv(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('run_locked_single_segment.py', 'run_schema10_native_correctness.py'):
            source = (root / 'scripts/server' / name).read_text(encoding='utf-8')
            self.assertIn('--repair-metric', source)
            self.assertIn('default=' + ('"' if name == 'run_schema10_native_correctness.py' else "'")
                          + PRIMARY_REPAIR_METRIC, source)
        self.assertEqual(development_experiment_spec()['repair_metric_contract'], PRIMARY_REPAIR_METRIC)

    def test_old_results_not_silently_reinterpreted(self):
        self.assertEqual(historical_repair_metric({}), 'normalized_v_legacy')
        self.assertEqual(validate_repair_cost_evidence({}, [{}]), {})
        runtime = {'repair_metric': PRIMARY_REPAIR_METRIC}
        for observation in ({}, {'winner_repair_metric': 'normalized_v_legacy'},
                            {'winner_repair_metric': 'external_fixed_mask'}):
            with self.assertRaisesRegex(ValueError, 'repair metric differs'):
                validate_repair_cost_evidence(runtime, [observation])
        self.assertEqual(validate_repair_cost_evidence(runtime,
            [{'winner_repair_metric': PRIMARY_REPAIR_METRIC}]), runtime)

    def test_actual_ready_path_uses_k_and_v_and_only_segment_rows(self):
        def tensor(values):
            return torch.tensor(values, dtype=torch.bfloat16).reshape(-1, 1, 1)
        # V-only prefers row 129; K deviation reverses the winner to row 128.
        current_k, current_v = tensor([1, 1, 1, 1]), tensor([1, 1, 1, 1])
        source_k, source_v = tensor([0, 1]), tensor([1, .5])
        c = NativeRequestContext.__new__(NativeRequestContext)
        c.selection_closed = True
        c.engine = SimpleNamespace(session=SimpleNamespace(
            current_layer=1,
            active_positions=(127, 128, 129, 130),
            observe_repair_check_pre_rope_kv=lambda depth: (current_k, current_v)))
        c.adapter = SimpleNamespace(native_repair_metric=PRIMARY_REPAIR_METRIC,
                                    spec=SimpleNamespace(num_layers=3))
        c.segments = {'C': {'positions': (128, 129)}}
        c.supports, c.repair_ratio = {}, .15
        c.actual_repair_check_sunk_ms, c.generation = 0., 1
        c.register_ready_hot_replicas = lambda: None
        ticket = SimpleNamespace(layer_events={2: SimpleNamespace(synchronize=lambda: None)},
                                 layer_tensors={2: (source_k, source_v)})
        before = (source_k.clone(), source_v.clone())
        ready, _ = c.ready_for_final_commit({'C': ticket})
        self.assertEqual(ready, {'C': 2})
        self.assertEqual(c.supports['C'], {2: (128,), 3: (128,)})
        c.adapter.native_repair_metric = 'normalized_v_legacy'
        c.ready_for_final_commit({'C': ticket})
        self.assertEqual(c.supports['C'][2], (129,))
        self.assertTrue(torch.equal(before[0], source_k))
        self.assertTrue(torch.equal(before[1], source_v))


if __name__ == '__main__':
    unittest.main()
