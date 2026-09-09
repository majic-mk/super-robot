import unittest
from pathlib import Path
from types import SimpleNamespace
from probekv.cacheblend_patch import validate_unified_diff
from probekv.resumable_prefill import LayerAdvanceResult, ProbeKVResumablePrefillSession


class DeferredLayerTimingTests(unittest.TestCase):
    def test_pending_measurement_is_null_not_zero_and_requires_events(self):
        with self.assertRaises(ValueError):
            LayerAdvanceResult(None, None, None, gpu_ms=None)
        row = LayerAdvanceResult(None, None, None, gpu_ms=None, timing_events=(object(), object()))
        self.assertIsNone(row.gpu_ms)

    def test_resolves_completed_events_without_host_wait(self):
        start = SimpleNamespace(elapsed_time=lambda _: 2.5)
        end = SimpleNamespace(query=lambda: True)
        session = object.__new__(ProbeKVResumablePrefillSession)
        session.layer_audit = [{"layer": 1, "active_after": [1, 2], "gpu_ms": None}]
        session.pending_timing_events = {1: (start, end)}
        session.resolve_completed_layer_timings()
        self.assertEqual(session.layer_audit[0]["gpu_ms"], 2.5)
        self.assertFalse(session.pending_timing_events)

    def test_unfinished_event_does_not_become_successful_timing(self):
        session = object.__new__(ProbeKVResumablePrefillSession)
        session.layer_audit = [{"layer": 1, "active_after": [1], "gpu_ms": None}]
        session.pending_timing_events = {1: (object(), SimpleNamespace(query=lambda: False))}
        with self.assertRaises(RuntimeError):
            session.resolve_completed_layer_timings()
        self.assertIsNone(session.layer_audit[0]["gpu_ms"])

    def test_independent_patch_is_well_formed_and_opt_in(self):
        path = Path(__file__).resolve().parents[1] / "patches/cacheblend/0013-probekv-deferred-layer-timing.patch"
        validate_unified_diff(path)
        self.assertEqual(path.read_text().count('metadata.get("probekv_defer_layer_timing", False)'), 2)
