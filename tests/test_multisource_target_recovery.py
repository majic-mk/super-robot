import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

spec = importlib.util.spec_from_file_location('target_recovery', Path(__file__).resolve().parents[1] / 'scripts/server/recover_multisource_targets.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TargetRecoveryTests(unittest.TestCase):
    def test_only_later_distinct_allowed_content_candidates(self):
        sources = [{'origin_example_id': str(i), 'context_id': str(i) + ':doc', 'historical_context': 'old'} for i in range(4)]
        case = {'case_id': '4:corpus-repeat', 'group_id': 'g', 'target_document_id': 'doc',
                'sources': sources, 'current_context': 'current'}
        examples = [NS(example_id=str(i), documents=[NS(document_id=d)], question='q', answers=['a'])
                    for i, d in [(0, 'doc'), (4, 'doc'), (5, 'other'), (6, 'doc')]]
        with patch.object(module, 'normalize_example', side_effect=lambda dataset, raw: raw), \
                patch.object(module, 'rank_event', side_effect=lambda s: s.split(':')[0]), \
                patch.object(module, 'render_preceding_context', return_value='new'):
            result = module.recover_targets([case], examples, 'MuSiQue')[case['case_id']]
        self.assertEqual([r['origin_example_id'] for r in result], ['6'])
        self.assertFalse(result[0]['online_execution_allowed'])
        self.assertTrue(result[0]['requires_token_and_partition_audit'])
