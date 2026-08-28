import json
import tempfile
import unittest
from pathlib import Path

from probekv.rag_data import (
    RAGDocument,
    RAGExample,
    build_controlled_cases,
    build_corpus_repeat_cases,
    build_streaming_pilot_cases,
    iter_raw_records,
    load_raw_records,
    normalize_2wiki,
    normalize_hotpotqa,
    normalize_musique,
)
from probekv.manifest import validate_manifest


def encode(text):
    return [ord(character) for character in text]


def documents(count=6, shared=None, prefix_tag=""):
    result = []
    for index in range(count):
        title = "Shared" if shared is not None and index == 2 else "Title %d" % index
        text = (
            shared
            if shared is not None and index == 2
            else "Document text %s %d" % (prefix_tag, index)
        )
        result.append(
            RAGDocument(
                "%s-%s" % (title, text), title, text, index == 2, index
            )
        )
    return tuple(result)


class AdapterTests(unittest.TestCase):
    def test_hotpot_parallel_context(self):
        example = normalize_hotpotqa(
            {
                "_id": "hp-1",
                "question": "Question?",
                "answer": "Answer",
                "supporting_facts": {"title": ["A"], "sent_id": [0]},
                "context": {
                    "title": ["A", "B"],
                    "sentences": [["a1", "a2"], ["b1"]],
                },
            }
        )
        self.assertEqual(example.dataset, "HotPotQA")
        self.assertTrue(example.documents[0].supporting)
        self.assertEqual(example.documents[0].text, "a1 a2")

    def test_single_string_huggingface_fields_are_not_split_into_characters(self):
        example = normalize_hotpotqa(
            {
                "id": "hp-single",
                "question": "Question?",
                "answers": {"text": "Whole answer"},
                "supporting_facts": {"title": "Only title", "sent_id": 0},
                "context": {"title": "Only title", "text": "Whole paragraph"},
            }
        )
        self.assertEqual(example.answers, ("Whole answer",))
        self.assertEqual(len(example.documents), 1)
        self.assertTrue(example.documents[0].supporting)

    def test_2wiki_pair_context(self):
        example = normalize_2wiki(
            {
                "id": "tw-1",
                "question": "Question?",
                "answer": "Answer",
                "supporting_facts": [["B", 0]],
                "context": [["A", ["a1"]], ["B", ["b1", "b2"]]],
            }
        )
        self.assertEqual(example.dataset, "2WikiMultiHopQA")
        self.assertTrue(example.documents[1].supporting)

    def test_musique_paragraphs(self):
        example = normalize_musique(
            {
                "id": "mq-1",
                "question": "Question?",
                "answer": "Answer",
                "paragraphs": [
                    {"idx": 0, "title": "A", "paragraph_text": "a", "is_supporting": True},
                    {"idx": 1, "title": "B", "paragraph_text": "b", "is_supporting": False},
                ],
            }
        )
        self.assertEqual(example.dataset, "MuSiQue")
        self.assertTrue(example.documents[0].supporting)

    def test_json_array_and_jsonl_loaders(self):
        with tempfile.TemporaryDirectory() as temporary:
            array_path = Path(temporary) / "array.json"
            lines_path = Path(temporary) / "lines.jsonl"
            array_path.write_text('[{"id": 1}]', encoding="utf-8")
            lines_path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
            self.assertEqual(len(load_raw_records(array_path)), 1)
            self.assertEqual(len(load_raw_records(lines_path)), 2)

    def test_streaming_json_array_and_jsonl_loaders(self):
        try:
            import ijson  # noqa: F401
        except ImportError:
            self.skipTest("ijson is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            array_path = Path(temporary) / "array.json"
            lines_path = Path(temporary) / "lines.jsonl"
            array_path.write_text('[{"id": 1}, {"id": 2}]', encoding="utf-8")
            lines_path.write_text(
                '{"id": 1}\n{"id": 2}\n', encoding="utf-8"
            )
            self.assertEqual(
                [row["id"] for row in iter_raw_records(array_path)],
                [1, 2],
            )
            self.assertEqual(
                [row["id"] for row in iter_raw_records(lines_path)],
                [1, 2],
            )


class ConstructionTests(unittest.TestCase):
    def test_controlled_case_has_four_distinct_regimes(self):
        example = RAGExample("fixture", "e1", "Question?", ("Answer",), documents())
        cases = build_controlled_cases([example], encode, "model@revision")
        self.assertEqual(len(cases), 1)
        self.assertEqual(len(cases[0].sources), 4)
        self.assertEqual(
            {source.regime for source in cases[0].sources},
            {
                "high-prefix/same-order",
                "low-prefix/same-order",
                "high-prefix/different-order",
                "low-prefix/different-order",
            },
        )
        self.assertEqual(len({source.historical_context for source in cases[0].sources}), 4)
        self.assertNotIn(cases[0].current_context, {source.historical_context for source in cases[0].sources})
        self.assertNotIn("Question?", cases[0].current_context)
        prefix_lengths = [
            len(encode(cases[0].current_context)),
            *[
                len(encode(source.historical_context))
                for source in cases[0].sources
            ],
        ]
        self.assertEqual(len(prefix_lengths), len(set(prefix_lengths)))

    def test_corpus_repeat_requires_five_distinct_occurrences(self):
        examples = [
            RAGExample(
                "fixture",
                "e%d" % index,
                "Distinct question %d?" % index,
                ("Answer",),
                documents(shared="Exactly repeated document", prefix_tag=str(index)),
            )
            for index in range(5)
        ]
        cases = build_corpus_repeat_cases(examples, encode, "model@revision")
        repeated = [
            case
            for case in cases
            if case.construction == "corpus_repeat_pseudotime"
            and case.target_document_id == "Shared-Exactly repeated document"
        ]
        self.assertEqual(len(repeated), 1)
        self.assertEqual(len(repeated[0].sources), 4)
        self.assertEqual(len({source.origin_example_id for source in repeated[0].sources}), 4)
        self.assertIn(
            "Document text",
            repeated[0].current_suffix_context,
        )
        self.assertNotIn(
            repeated[0].question,
            repeated[0].current_suffix_context,
        )

    def test_controlled_and_corpus_views_share_split_group(self):
        examples = [
            RAGExample(
                "fixture",
                "e%d" % index,
                "Distinct question %d?" % index,
                ("Answer",),
                documents(shared="Exactly repeated document", prefix_tag=str(index)),
            )
            for index in range(5)
        ]
        controlled = build_controlled_cases(examples, encode, "model@revision")
        corpus = build_corpus_repeat_cases(examples, encode, "model@revision")
        validate_manifest(controlled + corpus)
        shared_controlled = [
            case
            for case in controlled
            if case.target_document_id == "Shared-Exactly repeated document"
        ]
        shared_corpus = [
            case
            for case in corpus
            if case.target_document_id == "Shared-Exactly repeated document"
        ]
        self.assertEqual(shared_controlled[0].group_id, shared_corpus[0].group_id)

    def test_document_split_is_stable_across_model_tokenizers(self):
        example = RAGExample(
            "fixture", "e1", "Question?", ("Answer",), documents()
        )

        def other_model_encode(text):
            return [1000, *[ord(character) + 1 for character in text]]

        first = build_controlled_cases([example], encode, "model-a@revision")[0]
        second = build_controlled_cases(
            [example], other_model_encode, "model-b@revision"
        )[0]
        self.assertEqual(first.group_id, second.group_id)
        self.assertEqual(first.split, second.split)
        self.assertNotEqual(first.content_hash, second.content_hash)

    def test_corpus_repeat_tokenizes_each_unique_document_once(self):
        examples = [
            RAGExample(
                "fixture",
                "e%d" % index,
                "Distinct question %d?" % index,
                ("Answer",),
                documents(shared="Exactly repeated document", prefix_tag=str(index)),
            )
            for index in range(5)
        ]
        calls = []

        def counting_encode(text):
            calls.append(text)
            return encode(text)

        build_corpus_repeat_cases(examples, counting_encode, "model@revision")
        # Five examples contribute five unique non-shared documents each and
        # all five share one exact repeated document: 5 * 5 + 1.
        self.assertEqual(len(calls), 26)

    def test_controlled_construction_stops_without_scanning_case_outputs(self):
        examples = [
            RAGExample(
                "fixture",
                "e%d" % index,
                "Question %d?" % index,
                ("Answer",),
                documents(prefix_tag=str(index)),
            )
            for index in range(50)
        ]
        cases = build_controlled_cases(
            examples, encode, "model@revision", max_cases=3
        )
        self.assertEqual(len(cases), 3)

    def test_streaming_builder_scans_factory_twice_with_bounded_outputs(self):
        examples = [
            RAGExample(
                "fixture",
                "e%d" % index,
                "Distinct question %d?" % index,
                ("Answer",),
                documents(
                    shared="Exactly repeated document",
                    prefix_tag=str(index),
                ),
            )
            for index in range(8)
        ]
        factory_calls = []

        def factory():
            factory_calls.append(True)
            return iter(examples)

        cases, sampled, audit = build_streaming_pilot_cases(
            factory,
            encode,
            "model@revision",
            max_controlled_cases=3,
            max_corpus_repeat_cases=1,
        )
        self.assertEqual(len(factory_calls), 2)
        self.assertEqual(
            sum(
                case.construction == "controlled_document_order"
                for case in cases
            ),
            3,
        )
        self.assertEqual(
            sum(
                case.construction == "corpus_repeat_pseudotime"
                for case in cases
            ),
            1,
        )
        self.assertLessEqual(len(sampled), 64)
        self.assertEqual(audit["normalized_examples_scanned"], 8)


if __name__ == "__main__":
    unittest.main()
