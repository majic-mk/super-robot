import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from probekv.rag_data import normalize_example
from probekv.v8_schema10_execution import digest_json

spec = importlib.util.spec_from_file_location('relevance_audit', Path(__file__).resolve().parents[1]/'scripts/server/audit_source_relevance.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RelevanceAuditTests(unittest.TestCase):
    def test_label_does_not_use_answers_or_claim_token_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = dict(_id='q', question='where?', answer='a',
                context=[['shared', ['text']], ['other', ['text']]], supporting_facts=[['other',0]])
            example = normalize_example('HotPotQA', raw)
            (root/'raw.json').write_text(json.dumps([raw]))
            case = dict(case_id='c', target_document_id=example.documents[0].document_id)
            (root/'cases.jsonl').write_text(json.dumps(case)+'\n')
            pilot = dict(dataset_scope='HotPotQA', locked_test_accessed=False,
                input_sha256={k:module.file_sha(root/f) for k,f in [('raw','raw.json'),('cases','cases.jsonl')]},
                groups=[dict(case_id='c', group_id='g', target_requests=[dict(request_id='r',
                    origin_example_id='q', origin_example_digest=digest_json(example.to_row()))])])
            pilot['partition_sha256'] = digest_json(pilot)
            (root/'pilot.json').write_text(json.dumps(pilot))
            argv = ['audit', '--pilot', str(root/'pilot.json'), '--cases', str(root/'cases.jsonl'),
                    '--raw', str(root/'raw.json'), '--output', str(root/'result.json')]
            with patch('sys.argv', argv):
                module.main()
            result = json.loads((root/'result.json').read_text())
            self.assertEqual(result['supporting_targets'], 0)
            self.assertIsNone(result['rows'][0]['shared_segment_contains_supporting_sentence'])
            self.assertFalse(result['answers_or_model_outcomes_used'])
            argv[-1] = str(root/'bad.json')
            (root/'raw.json').write_text('[]')
            with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                module.main()
