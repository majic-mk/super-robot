import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from probekv.cacheblend_v6_online_engine import LayerwiseLoadTicket, TorchLayerwiseSourceLoader
from probekv.v8_schema10_native_adapter import NativeRequestContext
from probekv.request_wallclock import partition_request_wallclock


class FallbackAccountingTests(unittest.TestCase):
    def dense_context(self):
        calls = []
        class Layer:
            def __init__(self, index):
                self.index = index
            def __call__(self, positions, hidden, kv, metadata, residual, status, fuse_metadata, old_kv):
                calls.append((self.index, status, old_kv, positions.tolist()))
                return hidden + self.index, residual + 1
        session = SimpleNamespace(current_layer=1, commits={}, active_positions=(2, 3),
            _pending_target_positions=None, _pending_reuse_commit=False,
            hidden_states=torch.ones(2, 1), residual=torch.ones(2, 1),
            working_kv=[None]*3, attention_metadata=object(), pending_timing_events={}, layer_audit=[])
        adapter = SimpleNamespace(inner=SimpleNamespace(layers=[Layer(i) for i in range(3)],
                cache_fuse_metadata={}), spec=SimpleNamespace(num_layers=3), check_deadline=lambda: None,
                torch=SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw: Mock())))
        adapter.kv = session.working_kv
        context = SimpleNamespace(adapter=adapter, engine=SimpleNamespace(session=session),
            cached_prefix_tokens=2, request={"token_ids": [1, 2, 3, 4]}, committed={},
            _prepared_inputs=(None, torch.tensor([2, 3])), generation=1,
            attention=session.attention_metadata)
        return context, calls

    def test_native_continuation_preserves_state_and_skips_completed_layer(self):
        context, calls = self.dense_context()
        NativeRequestContext._advance_native_dense_remaining(context)
        self.assertEqual(calls, [(1, 0, (None, None), [2, 3]), (2, 0, (None, None), [2, 3])])
        self.assertEqual(context.engine.session.current_layer, 3)
        self.assertTrue(torch.equal(context.engine.session.hidden_states, torch.full((2, 1), 4.)))
        self.assertEqual([r["layer"] for r in context.engine.session.layer_audit], [2, 3])

    def test_native_continuation_rejects_selective_or_shrunk_state(self):
        for case in ("commit", "shrunk", "pending"):
            context, calls = self.dense_context()
            if case == "commit":
                context.engine.session.commits = {"C": 2}
            elif case == "shrunk":
                context.engine.session.active_positions = (3,)
            else:
                context.engine.session._pending_target_positions = (3,)
            with self.assertRaises(RuntimeError):
                NativeRequestContext._advance_native_dense_remaining(context)
            self.assertEqual(calls, [])

    def test_native_continuation_rejects_composite_or_foreign_attention(self):
        for foreign in ('kv', 'attention'):
            context, calls = self.dense_context()
            if foreign == 'kv':
                context.engine.session.working_kv = [object()] * 3
            else:
                context.engine.session.attention_metadata = object()
            with self.assertRaisesRegex(RuntimeError, 'original paged KV'):
                NativeRequestContext._advance_native_dense_remaining(context)
            self.assertEqual(calls, [])
            self.assertEqual(context.engine.session.current_layer, 1)

    def test_depth_two_continuation_does_not_replay_layer_one_or_two(self):
        context, calls = self.dense_context()
        context.engine.session.current_layer = 2
        NativeRequestContext._advance_native_dense_remaining(context)
        self.assertEqual([row[0] for row in calls], [2])

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
