import unittest
from types import SimpleNamespace as NS
from contextlib import nullcontext
from probekv.cacheblend_v6_online_engine import CacheBlendV6OnlineEngine


class ResidentPreinstallSafetyTests(unittest.TestCase):
    def fixture(self):
        log = []
        class Destination:
            def __setitem__(self, index, value):
                log.append(('copy', value))
        source = NS(device=NS(type='cuda'))
        ticket = NS(supplied_resident_replica=True, pending_layers={},
                    preparation_cancelled=False, transfer_failed=False,
                    layer_tensors={1: (source, source), 2: (source, source)},
                    layer_events={1: 'event1', 2: 'event2'}, installed_layers=set())
        engine = object.__new__(CacheBlendV6OnlineEngine)
        engine.tickets = {'C': ticket}
        engine.model_spec = NS(num_layers=2)
        engine.session = NS(exact_prefix_tokens=0)
        engine._exact_prefix_layers = ()
        engine._source_row_indices = {'C': slice(3, 5)}
        engine._composite_old_kvs = [[Destination(), Destination()] for _ in range(2)]
        stream = NS(wait_event=lambda event: log.append(('wait', event)))
        engine.source_loader = NS(torch=NS(cuda=NS(current_stream=lambda: stream)))
        engine._component = lambda name: nullcontext()
        return engine, ticket, log

    def test_wait_precedes_copy_and_repeat_is_noop(self):
        engine, ticket, log = self.fixture()
        engine.preinstall_resident_diagnostic('C', segment_count=1)
        self.assertEqual([x[0] for x in log], ['wait', 'copy', 'copy'] * 2)
        self.assertEqual(ticket.installed_layers, {1, 2})
        engine.preinstall_resident_diagnostic('C', segment_count=1)
        self.assertEqual(len(log), 6)

    def test_cpu_ssd_completion_is_not_residency(self):
        engine, ticket, log = self.fixture()
        ticket.supplied_resident_replica = False
        with self.assertRaises(RuntimeError):
            engine.preinstall_resident_diagnostic('C', segment_count=1)
        self.assertEqual(log, [])

    def test_prefix_multisegment_missing_event_and_failed_copy_rejected(self):
        for condition in ('prefix', 'multi', 'event', 'pending', 'failed', 'cpu'):
            engine, ticket, log = self.fixture()
            count = 1
            if condition == 'prefix': engine.session.exact_prefix_tokens = 16
            if condition == 'multi': count = 2
            if condition == 'event': del ticket.layer_events[2]
            if condition == 'pending': ticket.pending_layers = {2: None}
            if condition == 'failed': ticket.transfer_failed = True
            if condition == 'cpu': ticket.layer_tensors[1][0].device.type = 'cpu'
            with self.subTest(condition=condition), self.assertRaises(RuntimeError):
                engine.preinstall_resident_diagnostic('C', segment_count=count)
            self.assertEqual(log, [])
