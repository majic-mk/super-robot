import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from probekv.request_wallclock import partition_request_wallclock, refine_request_wallclock

spec = importlib.util.spec_from_file_location('timing_analysis', Path(__file__).resolve().parents[1] / 'scripts/server/analyze_single_segment_timing.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class WallclockRefinementTests(unittest.TestCase):
    def setUp(self):
        self.partition = partition_request_wallclock(100, 200, [('probe', 150)])

    def test_nested_overlap_is_counted_once(self):
        events = [dict(event_id='compare', start_ns=120, end_ns=170),
                  dict(event_id='prepare', start_ns=140, end_ns=180)]
        result = refine_request_wallclock(self.partition, events)
        self.assertEqual(sum(r['duration_ns'] for r in result['intervals']), 100)
        both = [r for r in result['intervals'] if len(r['active_event_ids']) == 2]
        self.assertEqual(sum(r['duration_ns'] for r in both), 30)
        self.assertEqual(result['unaccounted_ns'], 0)

    def test_missing_instrumentation_remains_gap(self):
        result = refine_request_wallclock(self.partition, [])
        self.assertTrue(all(r['classification'] == 'uninstrumented_host_gap' for r in result['intervals']))

    def test_bad_accounting_bounds_and_duplicate_events_fail(self):
        for change in ('accounted_ns', 'ttft_ns', 'unaccounted_ns'):
            broken = copy.deepcopy(self.partition)
            broken[change] += 1
            with self.assertRaises(ValueError):
                refine_request_wallclock(broken, [])
        event = dict(event_id='x', start_ns=120, end_ns=130)
        for events in ([event, event], [dict(event, start_ns=99)], [dict(event, event_id='')]):
            with self.assertRaises(ValueError):
                refine_request_wallclock(self.partition, events)

    def test_zero_duration_does_not_invent_time(self):
        result = refine_request_wallclock(partition_request_wallclock(100, 100, []), [])
        self.assertEqual(result['accounted_ns'], 0)

    def test_analysis_rejects_shifted_endpoints_and_preserves_missing_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'outcome.json'
            row = dict(arrival_ns=100, first_token_ns=200, request_ttft_ms=0.0001,
                       request_wallclock=self.partition)
            path.write_text(json.dumps(row))
            result = analysis.analyze(path)
            self.assertIsNone(result['committed_sources'])
            self.assertFalse(result['performance_improvement_proven'])
            row.update(arrival_ns=101, first_token_ns=201)
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                analysis.analyze(path)
