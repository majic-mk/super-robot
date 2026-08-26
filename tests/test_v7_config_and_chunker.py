import json
import tempfile
import unittest
from pathlib import Path

from probekv.canonical_segment import (
    CanonicalizerConfig,
    SemanticBoundary,
    canonicalize_token_ids,
    normalize_retrieved_text,
)
from probekv.config import load_config
from probekv.v7_simulation import run_v7_local_simulation


ROOT = Path(__file__).resolve().parents[1]


class V7ConfigTests(unittest.TestCase):
    def test_frozen_configs_load_and_old_v6_still_loads(self):
        causal = load_config(str(ROOT / "configs/local_system_v7_causal_wait.json"))
        immediate = load_config(
            str(ROOT / "configs/local_system_v7_immediate_staggered.json")
        )
        legacy = load_config(str(ROOT / "configs/local_system_v6.json"))
        self.assertEqual(causal.protocol_version, 7)
        self.assertEqual(immediate.protocol_version, 7)
        self.assertEqual(legacy.protocol_version, 6)
        self.assertEqual(causal.max_artifacts_per_source_variant, 1)

    def test_v7_rejects_multi_artifact_and_legacy_kmax(self):
        raw = json.loads(
            (ROOT / "configs/local_system_v7_causal_wait.json").read_text(
                encoding="utf-8"
            )
        )
        for key, value in (
            ("max_artifacts_per_source_variant", 2),
            ("lossy_full_kv_artifacts_enabled", True),
            ("online_kmax", 4),
        ):
            changed = dict(raw)
            changed[key] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(str(path))

    def test_local_matrix_covers_37_segments_and_16_variants(self):
        config = load_config(str(ROOT / "configs/local_system_v7_causal_wait.json"))
        result = run_v7_local_simulation(config)
        self.assertTrue(all(gate["passed"] for gate in result["gates"]))
        target = [
            row for row in result["rows"]
            if row["detected_segment_count"] == 37
            and row["stored_variants_per_segment"] == 16
        ]
        self.assertTrue(target)
        self.assertTrue(
            all(row["planned_segment_count"] == 37 for row in target)
        )


class CanonicalChunkerTests(unittest.TestCase):
    def test_deterministic_no_padding_and_exact_reconstruction(self):
        tokens = tuple(range(1025))
        kwargs = dict(
            tokenizer_signature="tokenizer-a",
            document_revision="doc-r1",
            semantic_boundaries={
                512: SemanticBoundary.PARAGRAPH,
                896: SemanticBoundary.SENTENCE,
            },
        )
        first = canonicalize_token_ids(tokens, **kwargs)
        second = canonicalize_token_ids(tokens, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(
            tuple(token for segment in first for token in segment.token_ids), tokens
        )
        self.assertNotIn(1, [segment.token_count for segment in first])
        self.assertEqual(sum(segment.token_count for segment in first), 1025)

    def test_semantic_boundary_can_beat_alignment(self):
        tokens = tuple(range(900))
        segments = canonicalize_token_ids(
            tokens,
            tokenizer_signature="tok",
            document_revision="doc",
            semantic_boundaries={501: SemanticBoundary.PARAGRAPH},
        )
        self.assertEqual(segments[0].token_end, 501)
        self.assertNotEqual(segments[0].token_count % 16, 0)

    def test_reuse_key_ignores_chunker_provenance_but_is_model_scoped(self):
        tokens = tuple(range(300))
        one = canonicalize_token_ids(
            tokens,
            tokenizer_signature="tok",
            document_revision="r1",
        )[0]
        two = canonicalize_token_ids(
            tokens,
            tokenizer_signature="tok",
            document_revision="r2",
            config=CanonicalizerConfig(search_window_tokens=32),
        )[0]
        self.assertNotEqual(one.canonicalizer_signature, two.canonicalizer_signature)
        self.assertEqual(
            one.reuse_content_key("model-a", "tok"),
            two.reuse_content_key("model-a", "tok"),
        )
        self.assertNotEqual(
            one.reuse_content_key("model-a", "tok"),
            one.reuse_content_key("model-b", "tok"),
        )

    def test_normalization_is_stable(self):
        self.assertEqual(normalize_retrieved_text("A  \r\nB\r\n"), "A\nB")


if __name__ == "__main__":
    unittest.main()
