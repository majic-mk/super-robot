import unittest
from pathlib import Path

from probekv.cacheblend_patch import (
    DEFERRED_TIMING_PATCH, combined_patch_sha256, load_patch_manifest,
    native_patch_files, validate_native_patch_audit,
)


class NativePatchProvenanceTests(unittest.TestCase):
    manifest = Path(__file__).resolve().parents[1] / "patches/cacheblend/manifest.json"
    mode = "probekv_v8_variant_growth_counterfactual"

    def audit(self, extras=()):
        paths = native_patch_files(self.manifest, self.mode, extras)
        return {"patch_mode": self.mode, "patches": [p.name for p in paths],
                "extra_patches": list(extras),
                "cacheblend_commit": load_patch_manifest(self.manifest)["base_commit"],
                "cacheblend_patch_sha256": combined_patch_sha256(paths),
                "cacheblend_tree": "a" * 40, "expected_cacheblend_tree": "a" * 40,
                "verification_method": "independent_clean_clone_ordered_patchset"}

    def test_optional_timing_is_bound_without_changing_historical_patchset(self):
        original = self.manifest.read_bytes()
        base, deferred = self.audit(), self.audit((DEFERRED_TIMING_PATCH,))
        validate_native_patch_audit(base, self.manifest)
        validate_native_patch_audit(deferred, self.manifest, deferred_timing=True)
        self.assertNotEqual(base["cacheblend_patch_sha256"], deferred["cacheblend_patch_sha256"])
        self.assertEqual(self.manifest.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "independently audited patch"):
            validate_native_patch_audit(base, self.manifest, deferred_timing=True)

    def test_old_digest_cannot_attest_to_new_timing_patch(self):
        deferred = self.audit((DEFERRED_TIMING_PATCH,))
        deferred["cacheblend_patch_sha256"] = self.audit()["cacheblend_patch_sha256"]
        with self.assertRaisesRegex(ValueError, "digest differs"):
            validate_native_patch_audit(deferred, self.manifest, deferred_timing=True)

    def test_tree_edit_and_omitted_patch_are_rejected(self):
        for field, value in (("expected_cacheblend_tree", "b" * 40),
                             ("verification_method", "manually_copied_tree"),
                             ("patches", self.audit()["patches"][:-1])):
            audit = self.audit()
            audit[field] = value
            with self.assertRaises(ValueError):
                validate_native_patch_audit(audit, self.manifest)

    def test_extras_cannot_inject_paths_or_duplicate_patches(self):
        for extras in (("../unknown.patch",), (DEFERRED_TIMING_PATCH,) * 2):
            with self.assertRaises(ValueError):
                native_patch_files(self.manifest, self.mode, extras)


if __name__ == "__main__":
    unittest.main()
