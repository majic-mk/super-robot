import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from probekv.v8_schema10_native_correctness import release_diagnostic_hot_caches
from probekv.v8_schema6_hbm import UnifiedHBMReservationManager, HBMReservationKind


class HotCacheTeardownTests(unittest.TestCase):
    def setup_backend(self):
        hbm = UnifiedHBMReservationManager(allocator_capacity_bytes=1024, safety_bytes=0)
        def adapter():
            return SimpleNamespace(active=None, hot_layer_cache={}, hot_reservations={},
                torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock())))
        fast, legacy = adapter(), adapter()
        backend = SimpleNamespace(hbm=hbm, pending={}, adapters={'fast': fast, 'legacy': legacy})
        hot, unknown = hbm.reserve_batch(owner_request_id='q', rows=[
            ('hot', 64, HBMReservationKind.WINNER_PREFETCH),
            ('unknown', 32, HBMReservationKind.SELECTION_WORKSPACE)])
        legacy.hot_reservations['source'] = hot
        legacy.hot_layer_cache['source'] = object()
        return backend, fast, legacy, hot, unknown

    def test_releases_owner_not_selected_adapter_and_preserves_unknown(self):
        backend, fast, legacy, hot, unknown = self.setup_backend()
        self.assertEqual(release_diagnostic_hot_caches(backend), [hot.reservation_id])
        self.assertEqual(backend.hbm.active_reserved_bytes, 32)
        self.assertFalse(unknown.released)
        self.assertFalse(legacy.hot_layer_cache)
        self.assertFalse(legacy.hot_reservations)
        self.assertEqual(release_diagnostic_hot_caches(backend), [])

    def test_failed_fence_keeps_owned_resources(self):
        backend, fast, legacy, hot, _ = self.setup_backend()
        legacy.torch.cuda.synchronize.side_effect = RuntimeError('CUDA failure')
        with self.assertRaisesRegex(RuntimeError, 'CUDA failure'):
            release_diagnostic_hot_caches(backend)
        self.assertFalse(hot.released)
        self.assertIn('source', legacy.hot_layer_cache)

    def test_active_request_is_not_force_closed(self):
        backend, fast, legacy, hot, _ = self.setup_backend()
        fast.active = 'running'
        with self.assertRaisesRegex(RuntimeError, 'quiescent'):
            release_diagnostic_hot_caches(backend)
        self.assertFalse(hot.released)

    def test_pending_request_is_rejected(self):
        backend, _, _, hot, _ = self.setup_backend()
        backend.pending = {'q': object()}
        with self.assertRaisesRegex(RuntimeError, 'quiescent'):
            release_diagnostic_hot_caches(backend)
        self.assertFalse(hot.released)
