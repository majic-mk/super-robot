import importlib.util
from pathlib import Path
import unittest
from probekv.rag_data import RAGDocument, RAGExample, render_preceding_context, segment_text

spec = importlib.util.spec_from_file_location('origin_audit', Path(__file__).resolve().parents[1] / 'scripts/server/audit_multisource_origins.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class OriginAuditTests(unittest.TestCase):
    def fixture(self):
        target = RAGDocument('doc', 'title', 'shared text', True, 1)
        examples = {}
        for i in range(5):
            prefix = RAGDocument(str(i), str(i), 'prefix ' + str(i), False, 0)
            examples[str(i)] = RAGExample('MuSiQue', str(i), 'question', ('answer',), (prefix, target))
        sources = [{'origin_example_id': str(i), 'historical_context': render_preceding_context(examples[str(i)].documents[:1])} for i in range(4)]
        row = {'case_id': '4:corpus-repeat-c000', 'sources': sources,
               'target_document_id': 'doc', 'segment_token_ids': list(map(ord, segment_text(target))),
               'current_context': render_preceding_context(examples['4'].documents[:1]),
               'question': 'question', 'answers': ['answer']}
        return row, examples

    def test_exact_origins_and_tokens(self):
        row, examples = self.fixture()
        report = audit.audit_case(row, examples, lambda x: map(ord, x))
        self.assertTrue(report['origin_token_verification_passed'])
        self.assertEqual(report['distinct_contexts'], 5)
        self.assertFalse(report['production_frequency_verified'])

    def test_prefix_or_token_corruption_rejected(self):
        for kind in ('prefix', 'tokens', 'target_leak', 'missing'):
            row, examples = self.fixture()
            if kind == 'prefix': row['sources'][0]['historical_context'] = 'wrong'
            if kind == 'tokens': row['segment_token_ids'][0] += 1
            if kind == 'target_leak': row['sources'][0]['origin_example_id'] = '4'
            if kind == 'missing': del examples['0']
            with self.subTest(kind=kind):
                self.assertFalse(audit.audit_case(row, examples, lambda x: map(ord, x))['origin_token_verification_passed'])
