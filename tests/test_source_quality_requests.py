import unittest

from probekv.rag_data import RAGDocument, RAGExample, segment_text
from probekv.source_quality_requests import build_quality_request


class QualityRequestTests(unittest.TestCase):
    def fixture(self):
        docs = tuple(RAGDocument(str(i), "title" + str(i), "evidence" + str(i),
                                 True, i) for i in range(9))
        example = RAGExample("MuSiQue", "target", "question", ("answer",), docs)
        parent = list(map(ord, segment_text(docs[6])))
        case = dict(target_document_id="6", canonical_parent_left_token_ids=parent[:5],
                    segment_token_ids=parent[5:20], canonical_parent_right_token_ids=parent[20:],
                    reuse_content_key="key", group_id="group")
        return example, case

    def build(self, example, case, **kwargs):
        options = dict(request_id="target", request_epoch=5, partition_role="validation",
                       partition_digest="audited-partition", max_model_len=4096)
        options.update(kwargs)
        return build_quality_request(example, case, lambda s: list(map(ord, s)), **options)

    def test_preserves_all_documents_answers_and_exact_slice(self):
        example, case = self.fixture()
        request = self.build(example, case)
        decoded = "".join(map(chr, request["token_ids"]))
        for i in range(9):
            self.assertIn("evidence" + str(i), decoded)
        self.assertEqual(request["answers"], ["answer"])
        segment = request["segments"][0]
        self.assertEqual([request["token_ids"][p] for p in segment["positions"]],
                         case["segment_token_ids"])
        self.assertFalse(set(segment["positions"]) & set(request["mandatory_suffix_positions"]))

    def test_rejects_truncation_and_unaudited_role(self):
        example, case = self.fixture()
        for options in ({"max_model_len": 20}, {"partition_role": "development"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.build(example, case, **options)

    def test_rejects_changed_shared_document(self):
        example, case = self.fixture()
        case["segment_token_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "reconstruction"):
            self.build(example, case)
