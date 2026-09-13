import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from probekv.cacheblend_v6_online_engine import LayerwiseLoadTicket, TorchLayerwiseSourceLoader
from probekv.v8_schema10_native_adapter import NativeRequestContext
from probekv.request_wallclock import partition_request_wallclock


class FallbackAccountingTests(unittest.TestCase):
    def ticket(self):
        pair = (torch.ones(4, dtype=torch.bfloat16), torch.ones(4, dtype=torch.bfloat16))
        event = Mock()
        return LayerwiseLoadTicket(segment_id="C", source_id="S", started_host_ms=0,
            requested_bytes=32, layer_tensors={1: pair}, start_event=event,
            layer_events={1: event}, source_digest_before="", source_digest_after="",
            segment_positions=(1, 2, 3, 4), pending_layers={2: pair}, expected_layer_count=2)

    def test_cancel_preserves_inflight_and_prevents_new_submissions(self):
        t = self.ticket()
        event, tensors = t.layer_events[1], t.layer_tensors[1]
        row = t.cancel_pending()
        self.assertEqual(row["not_submitted_bytes"], 16)
        self.assertEqual(row["not_submitted_layers"], [2])
        self.assertIs(t.layer_events[1], event)
        self.assertIs(t.layer_tensors[1], tensors)
        event.synchronize.assert_not_called()
        self.assertFalse(t.fully_ready())
        loader = object.__new__(TorchLayerwiseSourceLoader)
        # No torch/device exists: this must return before entering any stream.
        loader.prefetch_pending(t, 2)
        self.assertEqual(set(t.pending_layers), {2})
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            t.wait_all(loader)

    def test_partial_rejection_keeps_committed_winner_and_leases(self):
        accepted, rejected = self.ticket(), self.ticket()
        context = SimpleNamespace(prepared={"A": accepted, "B": rejected},
                                  committed={"A": 2}, generation=3)
        result = NativeRequestContext.cancel_uncommitted_preparation(context)
        self.assertEqual([r["segment_id"] for r in result], ["B"])
        self.assertFalse(accepted.preparation_cancelled)
        self.assertTrue(rejected.preparation_cancelled)
        self.assertEqual(context.generation, 4)

    def test_exact_contiguous_ns_not_rounded_residual(self):
        report = partition_request_wallclock(100, 115223843,
            [("context", 4172772), ("ready", 44696387)])
        self.assertEqual(report["ttft_ns"], 115223743)
        self.assertEqual(sum(r["duration_ns"] for r in report["intervals"]), 115223743)
        self.assertEqual(report["unaccounted_ns"], 0)

    def test_backward_and_out_of_range_landmarks_rejected(self):
        for rows in ([("a", 12), ("b", 11)], [("a", 21)], [("a", 0)]):
            with self.assertRaises(ValueError):
                partition_request_wallclock(1, 20, rows)


if __name__ == "__main__":
    unittest.main()
