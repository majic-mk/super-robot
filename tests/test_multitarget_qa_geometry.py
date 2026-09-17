import importlib.util
from pathlib import Path
import unittest

from probekv.rag_data import RAGDocument, RAGExample, segment_text

spec = importlib.util.spec_from_file_location('qa_geometry', Path(__file__).resolve().parents[1] / 'scripts/server/audit_multitarget_qa_geometry.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class GeometryTests(unittest.TestCase):
    def test_full_context_count_and_recapture(self):
        docs = tuple(RAGDocument(str(i), 'title', 'text' * 100, True, i) for i in range(9))
        example = RAGExample('MuSiQue', 'x', 'q', ('a',), docs)
        case = dict(target_document_id='6', segment_token_ids=list(map(ord, segment_text(docs[6]))))
        result = audit.audit_member(example, case, lambda s: list(map(ord, s)), old_prefix='truncated')
        self.assertTrue(result['source_recapture_required'])
        self.assertEqual(result['original_document_count'], 9)
        self.assertGreater(result['prompt_tokens'], 9 * 400)
        self.assertEqual(result['required_model_len'], result['prompt_tokens'] + 32)

    def test_no_token_mismatch_accepted(self):
        doc = RAGDocument('x', 't', 'text', True, 0)
        example = RAGExample('MuSiQue', 'x', 'q', ('a',), (doc,))
        with self.assertRaisesRegex(ValueError, 'reconstruction'):
            audit.audit_member(example, dict(target_document_id='x', segment_token_ids=[1]), list)
