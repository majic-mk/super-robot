import importlib.util
from pathlib import Path
import sys
import unittest
from probekv.v8_schema10_execution import digest_json

SERVER = Path(__file__).resolve().parents[1] / 'scripts/server'
sys.path.insert(0, str(SERVER))
spec = importlib.util.spec_from_file_location('qa_pilot_runner', SERVER / 'run_multitarget_qa_pilot.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
sys.path.remove(str(SERVER))


class PilotValidationTests(unittest.TestCase):
    def fixture(self):
        requests = [dict(request_id=str(i), request_epoch=i+1, token_ids=[1,2,3,4],
                    segments=[dict(segment_id='C', token_ids=[2,3], positions=[1,2], content_key='k')],
                    mandatory_suffix_positions=[3], partition_role='fit', content_group='g',
                    development_partition_digest='a'*64, locked_test_accessed=False)
                    for i in range(9)]
        return dict(kind='multitarget_qa_pilot_partition_v1', repair_metric='normalized_kv_deviation',
                    repair_ratio=.15, common_first_reuse_layer=9, locked_test_accessed=False,
                    input_sha256={'partition': 'a'*64},
                    groups=[dict(source_requests=requests[:4], target_requests=requests[4:])])

    def seal(self, p):
        p['partition_sha256'] = digest_json({k:v for k,v in p.items() if k != 'partition_sha256'})
        return p

    def test_valid_and_tampered_plan(self):
        p = self.seal(self.fixture())
        runner.validate_pilot(p)
        p['repair_ratio'] = .1
        with self.assertRaises(ValueError):
            runner.validate_pilot(p)

    def test_future_source_or_partition_leak_rejected(self):
        for field, value in [('request_epoch', 0), ('partition_role', 'validation')]:
            p = self.fixture()
            p['groups'][0]['target_requests'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.validate_pilot(self.seal(p))
