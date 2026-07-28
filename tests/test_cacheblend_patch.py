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
        self.assertEqual(len(cb0), 1)
        self.assertEqual(len(probekv), 2)
        self.assertNotEqual(
            combined_patch_sha256(cb0),
            combined_patch_sha256(probekv),
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
        for path in patch_files_for_mode(MANIFEST, "probekv"):
            validate_unified_diff(path)


if __name__ == "__main__":
    unittest.main()
