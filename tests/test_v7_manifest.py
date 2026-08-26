import unittest

from probekv.manifest import (
    ManifestCase,
    ManifestSource,
    manifest_case_from_row,
    token_content_hash,
    validate_manifest,
)
from probekv.v6_h1_runtime import compose_manifest_prompt_regions


def _sources(count):
    return tuple(
        ManifestSource(
            source_id="s%d" % index,
            historical_context="history-%d" % index,
            context_id="context-%d" % index,
        )
        for index in range(count)
    )


class V7ManifestTests(unittest.TestCase):
    def make_case(self, sources=16):
        left = (1, 2)
        segment = (3, 4, 5)
        right = (6, 7)
        return ManifestCase(
            case_id="v7-case",
            dataset="dataset",
            document_id="document",
            group_id="group",
            split="pilot",
            regime="controlled",
            model_signature="model@revision",
            segment_text="segment",
            segment_token_ids=segment,
            content_hash=token_content_hash(segment),
            current_context="current",
            sources=_sources(sources),
            protocol_version=7,
            canonicalizer_signature="chunker",
            segment_provenance_id="provenance",
            reuse_content_key="reuse-key",
            canonical_parent_content_hash=token_content_hash(left + segment + right),
            canonical_parent_left_token_ids=left,
            canonical_parent_right_token_ids=right,
        )

    def test_v7_supports_sixteen_variants_and_round_trips_parent_tokens(self):
        case = self.make_case()
        validate_manifest((case,))
        restored = manifest_case_from_row(case.to_row())
        self.assertEqual(restored, case)
        self.assertEqual(
            restored.canonical_parent_left_token_ids
            + restored.segment_token_ids
            + restored.canonical_parent_right_token_ids,
            (1, 2, 3, 4, 5, 6, 7),
        )

    def test_v7_rejects_incomplete_parent_reconstruction(self):
        row = self.make_case(4).to_row()
        row["canonical_parent_right_token_ids"] = [99]
        with self.assertRaisesRegex(ValueError, "parent token sequence"):
            manifest_case_from_row(row)

    def test_h1_runtime_restores_left_and_right_parent_tokens(self):
        case = self.make_case(4)
        prefix, segment, suffix = compose_manifest_prompt_regions(
            case, (100,), case.segment_token_ids, (200,)
        )
        self.assertEqual(prefix, (100, 1, 2))
        self.assertEqual(segment, (3, 4, 5))
        self.assertEqual(suffix, (6, 7, 200))

    def test_legacy_default_still_rejects_more_than_four_sources(self):
        v7 = self.make_case(5)
        row = v7.to_row()
        row.update(
            protocol_version=0,
            canonicalizer_signature="",
            segment_provenance_id="",
            reuse_content_key="",
            canonical_parent_content_hash="",
            canonical_parent_left_token_ids=[],
            canonical_parent_right_token_ids=[],
        )
        with self.assertRaisesRegex(ValueError, "online Kmax"):
            manifest_case_from_row(row)


if __name__ == "__main__":
    unittest.main()
