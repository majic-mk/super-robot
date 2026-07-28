import unittest

from probekv.cacheblend_backend import CacheBlendBackend, RuntimeRepairMeasurement
from probekv.contracts import KVLocation

from tests.helpers import canonical_source


class FakeRuntime:
    def __init__(self, mutate=False):
        self.mutate = mutate

    def stage_canonical_source(self, source, target):
        return 3.5

    def selective_repair(self, source, start_layer, ratio):
        selected = int(source.token_count * ratio)
        return RuntimeRepairMeasurement(
            0.99,
            0.96,
            8.0,
            "before",
            "after" if self.mutate else "before",
            requested_ratio=ratio,
            eligible_segment_tokens=source.token_count,
            selected_segment_tokens=selected,
            effective_ratio=selected / float(source.token_count),
            mandatory_suffix_tokens=12,
            reuse_start_layer=start_layer,
            repair_gpu_ms=7.5,
            repair_host_ms=8.0,
            output_token_ids=(1, 2),
            output_hash="hash",
        )

    def dense_remaining_ms(self, token_count, start_layer):
        return 40.0

    def provenance(self):
        return {
            "cacheblend_commit": "b72d7945",
            "cacheblend_patch_sha256": "patch",
            "cacheblend_tree": "tree",
            "vllm": "0.4.1",
            "torch": "2.2.1",
            "cuda": "12.1",
        }


class CacheBlendAdapterTests(unittest.TestCase):
    def test_valid_runtime_is_adapted(self):
        backend = CacheBlendBackend(FakeRuntime(), total_layers=32)
        self.assertEqual(
            backend.prepare_source(canonical_source(), KVLocation.GPU), 3.5
        )
        self.assertEqual(backend.repair(canonical_source(), 5, 0.2).token_f1, 0.96)
        self.assertEqual(backend.full_remaining(512, 5), 40.0)
        self.assertEqual(backend.provenance()["vllm"], "0.4.1")

    def test_source_mutation_is_rejected(self):
        backend = CacheBlendBackend(FakeRuntime(mutate=True), total_layers=32)
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            backend.repair(canonical_source(), 5, 0.2)


if __name__ == "__main__":
    unittest.main()
