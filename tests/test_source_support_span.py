import unittest
import importlib.util
import sys
from pathlib import Path
from probekv.rag_data import normalize_example, segment_text
from probekv.source_support_span import classify_support


class SupportSpanTests(unittest.TestCase):
    def test_extension_excludes_previous_groups_and_noncalibration(self):
        scripts = Path(__file__).resolve().parents[1]/'scripts'/'server'
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location('support_census', scripts/'census_source_support_extension.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        base = dict(dataset='HotPotQA', regime='corpus-repeat', split='calibration', group_id='new')
        rows = [base, dict(base,group_id='old'), dict(base,split='test'), dict(base,split='train')]
        partition = [dict(group_id='old', partition_role='development_profile_freeze', locked_test_accessed=False)]
        self.assertEqual(module.unused_calibration(rows,partition,'HotPotQA'),[base])
        with self.assertRaises(ValueError):
            module.unused_calibration(rows,[dict(partition[0],locked_test_accessed=True)],'HotPotQA')

    def test_sentence_inside_partial_outside_and_identity(self):
        raw = dict(_id='q', question='q', answer='a', context=[['T',['first.', '答案在这里。']]],
                   supporting_facts=[['T',1]])
        d = normalize_example('HotPotQA', raw).documents[0]
        text = segment_text(d)
        ids = list(range(len(text)))
        encoded = dict(input_ids=ids, offset_mapping=[(i,i+1) for i in ids])
        start, end = text.index('答案'), text.index('答案')+len('答案在这里。')
        def run(a,b, tokens=ids):
            return classify_support(raw,d,parent_token_ids=tokens,encoded=encoded,token_start=a,token_end=b)
        self.assertEqual(run(start,end), 'full_support_sentence')
        self.assertEqual(run(start+1,end), 'partial_support_sentence')
        self.assertEqual(run(0,start), 'support_elsewhere')
        with self.assertRaisesRegex(ValueError,'canonical tokens'):
            run(start,end,ids[::-1])

    def test_no_support_and_invalid_annotation(self):
        raw = dict(_id='q',question='q',answer='a',context=[['T',['one']]],supporting_facts=[])
        d = normalize_example('HotPotQA',raw).documents[0]
        text=segment_text(d); ids=list(range(len(text)))
        args=dict(parent_token_ids=ids,encoded=dict(input_ids=ids,offset_mapping=[(i,i+1) for i in ids]),token_start=0,token_end=len(ids))
        self.assertEqual(classify_support(raw,d,**args),'non_supporting')
        raw['supporting_facts']=[['T',9]]
        with self.assertRaises(ValueError):
            classify_support(raw,d,**args)
