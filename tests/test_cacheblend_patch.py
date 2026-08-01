import unittest
from pathlib import Path

from probekv.cacheblend_patch import (
    combined_patch_sha256,
    load_patch_manifest,
    patch_files_for_mode,
    validate_unified_diff,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "patches" / "cacheblend" / "manifest.json"


class CacheBlendPatchTests(unittest.TestCase):
    def test_modes_are_tracked_and_have_distinct_hashes(self):
        manifest = load_patch_manifest(MANIFEST)
        self.assertFalse(manifest["innovation_claim"])
        cb0 = patch_files_for_mode(MANIFEST, "cb0")
        probekv = patch_files_for_mode(MANIFEST, "probekv")
        closed_loop = patch_files_for_mode(
            MANIFEST, "probekv_closed_loop"
        )
        multiregion = patch_files_for_mode(
            MANIFEST, "probekv_v6_multiregion"
        )
        self.assertEqual(len(cb0), 1)
        self.assertEqual(len(probekv), 2)
        self.assertEqual(closed_loop, probekv)
        self.assertEqual(len(multiregion), 3)
        self.assertEqual(multiregion[:2], probekv)
        self.assertNotEqual(
            combined_patch_sha256(cb0),
            combined_patch_sha256(probekv),
        )
        self.assertNotEqual(
            combined_patch_sha256(probekv),
            combined_patch_sha256(multiregion),
        )
        self.assertTrue(
            manifest["runtime_modes"]["closed_loop_v5"][
                "layer_resumable_prefill"
            ]
        )

    def test_runtime_patch_freezes_segment_only_denominator(self):
        patch = patch_files_for_mode(MANIFEST, "probekv")[1].read_text(
            encoding="utf-8"
        )
        additions = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        self.assertIn("segment_len * ratio", additions)
        self.assertIn("mandatory_suffix_tokens", additions)
        self.assertNotIn(
            "(total_len-last_len)*cache_fuse_metadata", additions
        )

    def test_all_tracked_patches_have_valid_hunk_counts(self):
        for path in patch_files_for_mode(
            MANIFEST, "probekv_v6_multiregion"
        ):
            validate_unified_diff(path)

    def test_partial_repair_uses_absolute_query_causal_rows(self):
        patch = patch_files_for_mode(MANIFEST, "probekv")[1].read_text(
            encoding="utf-8"
        )
        additions = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        self.assertIn("attn_bias = _make_partial_bias_gqa", additions)
        self.assertNotIn(
            "attn_bias = LowerTriangularFromBottomRightMask", additions
        )

    def test_v6_patch_builds_union_mask_without_changing_v5_mode(self):
        paths = patch_files_for_mode(MANIFEST, "probekv_v6_multiregion")
        patch = paths[-1].read_text(encoding="utf-8")
        additions = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        self.assertIn('cache_fuse_metadata.get("repair_regions")', additions)
        self.assertIn("dense_mask[:prefix_len] = False", additions)
        self.assertIn("torch.unique(top_indices)", additions)
        self.assertEqual(
            patch_files_for_mode(MANIFEST, "probekv_closed_loop"),
            paths[:2],
        )


if __name__ == "__main__":
    unittest.main()
