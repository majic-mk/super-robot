import unittest
from dataclasses import replace

from probekv.contracts import CaseSpec, SourceOrigin
from probekv.source_store import SourceStore

from tests.helpers import canonical_source


class SourceContractTests(unittest.TestCase):
    def test_canonical_full_prefill_is_accepted(self):
        store = SourceStore(online_kmax=4)
        source = canonical_source()
        store.register(source)
        self.assertEqual(store.get("hash-c", "s1"), source)

    def test_selective_repair_cannot_be_promoted(self):
        store = SourceStore()
        repaired = replace(
            canonical_source(), origin=SourceOrigin.SELECTIVE_REPAIR, exact=False
        )
        with self.assertRaisesRegex(ValueError, "exact|promoted"):
            store.register(repaired)

    def test_each_context_requires_an_independent_source(self):
        store = SourceStore()
        store.register(canonical_source("s1", context_id="A"))
        with self.assertRaisesRegex(ValueError, "collision"):
            store.register(canonical_source("s1", context_id="B"))

    def test_online_kmax_is_enforced(self):
        store = SourceStore(online_kmax=2)
        store.register(canonical_source("s1", context_id="A"))
        store.register(canonical_source("s2", context_id="B"))
        with self.assertRaisesRegex(ValueError, "Kmax"):
            store.register(canonical_source("s3", context_id="E"))

    def test_case_rejects_cross_segment_source(self):
        case = CaseSpec(
            case_id="c",
            current_prompt="prompt",
            content_hash="hash-c",
            model_signature="model@revision",
            sources=(canonical_source(content_hash="different"),),
            split="test",
            regime="low-prefix/different-order",
            segment_length=512,
        )
        with self.assertRaisesRegex(ValueError, "same segment"):
            case.validate()


if __name__ == "__main__":
    unittest.main()
