import unittest
from dataclasses import replace

from probekv.manifest import ManifestCase, ManifestSource, token_content_hash
from probekv.v8_profile_manifest import canonicalize_v8_profile_cases
from probekv.semantic_segment import SemanticWindowConfig


def _case(*, split="calibration", token_count=700):
    tokens = tuple(range(1, token_count + 1))
    return ManifestCase(
        case_id="case",
        dataset="dataset",
        document_id="document",
        group_id="group",
        split=split,
        regime="natural",
        model_signature="model@revision",
        segment_text="source text",
        segment_token_ids=tokens,
        content_hash=token_content_hash(tokens),
        current_context="current",
        sources=(ManifestSource("s0", "historical", "context"),),
    )


class V8ProfileManifestTests(unittest.TestCase):
    def test_semantic_manifest_uses_exact_offsets_and_preserves_parent_tokens(self):
        case = _case()
        text = 'a' * 500 + '。' + 'b' * 199
        case = replace(case, segment_text=text)
        audit = []
        rows = canonicalize_v8_profile_cases((case,), tokenizer_signature='tok', document_revision='doc',
            decode=lambda ids: ''.join(text[i - 1] for i in ids), chunker_config=SemanticWindowConfig(),
            encode_with_offsets=lambda value: {'input_ids': case.segment_token_ids,
                'offset_mapping': [(i, i + 1) for i in range(len(value))]}, segmentation_audits=audit)
        self.assertEqual(len(rows[0].segment_token_ids), 501)
        self.assertEqual(tuple(t for row in rows for t in row.segment_token_ids), case.segment_token_ids)
        self.assertEqual(rows[1].canonical_parent_left_token_ids, case.segment_token_ids[:501])
        self.assertEqual(audit[0]['cuts'][0]['reason'], 'sentence')

    def test_semantic_manifest_refuses_tokenization_change(self):
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            canonicalize_v8_profile_cases((_case(),), tokenizer_signature='tok', document_revision='doc',
                decode=lambda ids: '', chunker_config=SemanticWindowConfig(),
                encode_with_offsets=lambda text: {'input_ids': [999], 'offset_mapping': [(0, 1)]})

    def test_upgrade_is_deterministic_exact_and_canonical(self):
        kwargs = {
            "tokenizer_signature": "tokenizer-sha",
            "document_revision": "dataset-revision",
            "decode": lambda values: " ".join(str(value) for value in values),
        }
        first = canonicalize_v8_profile_cases((_case(),), **kwargs)
        second = canonicalize_v8_profile_cases((_case(),), **kwargs)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        self.assertEqual(
            tuple(token for case in first for token in case.segment_token_ids),
            _case().segment_token_ids,
        )
        self.assertTrue(all(case.protocol_version == 8 for case in first))
        self.assertTrue(all(case.reuse_content_key for case in first))
        self.assertTrue(all(case.canonicalizer_signature for case in first))

    def test_profile_upgrade_rejects_non_development_rows(self):
        with self.assertRaisesRegex(ValueError, "cannot access pilot/test"):
            canonicalize_v8_profile_cases(
                (_case(split="test"),),
                tokenizer_signature="tokenizer-sha",
                document_revision="dataset-revision",
                decode=lambda values: "text",
            )


if __name__ == "__main__":
    unittest.main()
