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
        staggered = patch_files_for_mode(
            MANIFEST, "probekv_v6_staggered_runtime"
        )
        self.assertEqual(len(cb0), 1)
        self.assertEqual(len(probekv), 2)
        self.assertEqual(closed_loop, probekv)
        self.assertEqual(len(multiregion), 3)
        self.assertEqual(multiregion[:2], probekv)
        self.assertEqual(len(staggered), 4)
        self.assertEqual(staggered[:3], multiregion)
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
        self.assertEqual(
            manifest["runtime_modes"]["staggered_runtime_v6"]["status"],
            "concrete_engine_hook_complete_requires_a800_qualification",
        )
        self.assertTrue(
            manifest["runtime_modes"]["closed_loop_v6"][
                "absolute_union_mask"
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
            MANIFEST, "probekv_v6_staggered_runtime"
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

    def test_staggered_runtime_patch_ports_qwen_and_preserves_old_mode(self):
        paths = patch_files_for_mode(MANIFEST, "probekv_v6_staggered_runtime")
        patch = paths[-1].read_text(encoding="utf-8")
        for marker in (
            "class Qwen2Model", "probekv_begin_prefill",
            "probekv_advance_prefill", "target_active_positions",
            "local_imp_indices", "BlockDiagonalCausalMask.from_seqlens",
        ):
            self.assertIn(marker, patch)
        self.assertIn(
            " attn_metadata: AttentionMetadata,\n"
            "         residual: Optional[torch.Tensor],\n"
            "+        status: int,",
            patch,
        )
        self.assertNotIn(
            "return hidden_states\n \n"
            "+        if self.model.cache_fuse_metadata.get(\"capture_logits\"",
            patch,
        )
        self.assertIn(
            "+        if self.model.cache_fuse_metadata.get(\"capture_logits\"",
            patch,
        )
        self.assertIn("self.num_queries_per_kv not in (1, 2, 4, 8)", patch)
        self.assertIn("partial_bias = partial_bias.view(", patch)
        self.assertIn("already-expanded GQA tensors", patch)
        self.assertEqual(
            patch_files_for_mode(MANIFEST, "probekv_v6_multiregion"),
            paths[:3],
        )


if __name__ == "__main__":
    unittest.main()
