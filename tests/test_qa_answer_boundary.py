import unittest
from types import SimpleNamespace
from probekv.v8_schema10_qa import answer_evidence, validate_answer_evidence, bounded_answer


class AnswerBoundaryTests(unittest.TestCase):
    def test_raw_preserved_and_scoring_recomputable(self):
        q = dict(token_ids=[1], answers=['Asia'], answer_boundary_contract='qa_next_question_boundary_v1')
        tokenizer = SimpleNamespace(decode=lambda *a, **k: 'Asia\n\nQuestion: next')
        row = answer_evidence([2, 3], tokenizer=tokenizer, request=q)['qa_evidence']
        self.assertEqual(row['answer_f1'], 1.)
        self.assertEqual(row['raw_generated_text'], 'Asia\n\nQuestion: next')
        self.assertEqual(validate_answer_evidence(row, request=q), 1.)
        with self.assertRaises(ValueError):
            validate_answer_evidence(row, request=dict(token_ids=[1], answers=['Asia']))

    def test_no_first_line_or_reference_based_truncation(self):
        q = dict(answer_boundary_contract='qa_next_question_boundary_v1')
        for text in ['wrong\nAsia', 'Asia\nQuest', 'Question: in an answer']:
            self.assertEqual(bounded_answer(text, q), (text, None))
        self.assertEqual(bounded_answer('wrong\nQuestion: x', q), ('wrong', '\nQuestion:'))
        self.assertEqual(bounded_answer('Asia\nQuestion: x', {}), ('Asia\nQuestion: x', None))

    def test_teacher_and_unknown_contract_rejected(self):
        for q in [dict(answer_boundary_contract='unknown'),
                  dict(answer_boundary_contract='qa_next_question_boundary_v1', teacher_token_ids=[])]:
            with self.assertRaises(ValueError):
                bounded_answer('Asia', q)
