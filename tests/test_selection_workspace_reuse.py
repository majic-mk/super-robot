"""CPU arithmetic/lifetime tests, not GPU performance qualification."""
import gc
import math
import unittest
import weakref
from types import SimpleNamespace as NS
from unittest.mock import patch, Mock

import torch

from probekv.selection_comparison import prepare_current_k, residual_scores
from probekv.v8_schema10_native_adapter import _defer_layer_timing_capability
from probekv.v8_schema10_online_backend import Schema10OnlineExperimentBackend
from probekv.v8_schema10_execution import SelectionCostLedger, SelectionCostPolicy
from probekv.v8_schema6_hbm import UnifiedHBMReservationManager


class SelectionWorkspaceTests(unittest.TestCase):
    def test_original_arithmetic_and_stable_sort_preserved(self):
        gen = torch.Generator().manual_seed(20260726)
        for n in (1, 128, 512, 640):
            current = torch.randn(n, 2, 4, generator=gen).bfloat16()
            current[0] = 0
            sources = torch.randn(4, n, 2, 4, generator=gen).bfloat16()
            sources[1] = sources[0]  # deterministic ties
            for ratio in (.05, .15, .30, 1.):
                drift = (sources.float() - current.float()).square().sum((2, 3)).sqrt()
                drift /= current.float().square().sum((1, 2)).sqrt().clamp_min(1e-12)
                order = drift.argsort(dim=1, descending=True, stable=True)
                trim = min(n - 1, math.ceil(ratio * n))
                old = drift.gather(1, order)[:, trim:].mean(1)
                new = residual_scores(sources, *prepare_current_k(current), ratio)
                self.assertTrue(torch.equal(old, new), (n, ratio))

    def make_backend(self):
        current = torch.arange(32).reshape(4, 2, 4).bfloat16()
        unit = current.numel() * 32 + current.shape[0] * 32
        backend = object.__new__(Schema10OnlineExperimentBackend)
        backend.hbm = UnifiedHBMReservationManager(allocator_capacity_bytes=unit * 3, safety_bytes=0)
        backend.store = NS(read_selection=lambda sid, depth: current + int(sid) + depth,
                           pool=NS(record_observation=Mock()))
        backend.provenance = {'model_signature': 'model'}
        backend.costs = NS(gate1=lambda *a: None)
        context = NS(segments={'C': {'positions': (0, 1, 2, 3), 'content_key': 'content'}},
                     selector=NS(variant_profile=NS(source_residual_trim_ratio=.15)))
        rows = [NS(source_variant_id=str(i)) for i in range(5)]
        return backend, context, rows, current

    def test_normalization_once_per_checkpoint_and_cleanup(self):
        b, c, rows, current = self.make_backend()
        with patch('probekv.v8_schema10_online_backend.prepare_current_k', wraps=prepare_current_k) as prep:
            for depth in (1, 2):
                result = b._compare(c, 'C', depth, rows, current + depth,
                                    SelectionCostLedger(100, SelectionCostPolicy()), 'request')
                self.assertEqual(len(result[0]), 5)
                self.assertEqual(b.hbm.active_reserved_bytes, 0)
            self.assertEqual(prep.call_count, 2)  # not 2x3 microbatches
            self.assertFalse(torch.equal(prep.call_args_list[0].args[0], prep.call_args_list[1].args[0]))

    def test_exception_releases_shared_and_batch_reservations(self):
        b, c, rows, current = self.make_backend()
        with patch('probekv.v8_schema10_online_backend.residual_scores', side_effect=RuntimeError('kernel')):
            with self.assertRaisesRegex(RuntimeError, 'kernel'):
                b._compare(c, 'C', 1, rows, current, SelectionCostLedger(100, SelectionCostPolicy()), 'r')
        self.assertEqual(b.hbm.active_reserved_bytes, 0)

    def test_no_headroom_does_not_allocate_normalization(self):
        b, c, rows, current = self.make_backend()
        b.hbm.allocator_capacity_bytes = 0
        with patch('probekv.v8_schema10_online_backend.prepare_current_k') as prep:
            result = b._compare(c, 'C', 1, rows, current, SelectionCostLedger(100, SelectionCostPolicy()), 'r')
            prep.assert_not_called()
        self.assertEqual(result[0], [])


class CapabilityLifetimeTests(unittest.TestCase):
    def tearDown(self):
        _defer_layer_timing_capability.cache_clear()

    def test_cache_does_not_retain_model_owner(self):
        class Model:
            def advance(self): pass
        model = Model()
        ref = weakref.ref(model)
        with patch('inspect.getsource', return_value='probekv_defer_layer_timing') as probe:
            self.assertTrue(_defer_layer_timing_capability(model.advance))
            self.assertTrue(_defer_layer_timing_capability(Model().advance))
            self.assertEqual(probe.call_count, 1)
        del model
        gc.collect()
        self.assertIsNone(ref())

    def test_replaced_implementation_is_rechecked_and_missing_source_rejected(self):
        first, second = lambda: None, lambda: None
        with patch('inspect.getsource', side_effect=['probekv_defer_layer_timing', 'old']):
            self.assertTrue(_defer_layer_timing_capability(first))
            self.assertFalse(_defer_layer_timing_capability(second))
        with patch('inspect.getsource', side_effect=OSError('source unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'cannot be audited'):
                _defer_layer_timing_capability(lambda: None)


if __name__ == '__main__':
    unittest.main()
