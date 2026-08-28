import math
import unittest

from probekv.features import cache_craft_cfo_operational, cache_craft_cfo_raw
from probekv.v8_cfo import (
    CFOFullPrefillCollector,
    CanonicalChunkOccurrence,
    build_source_cfo_metadata,
    compute_cachecraft_cfo,
    eager_qk_attention_mass,
    streaming_qk_attention_mass,
)
from probekv.v8_schema6_profile import (
    bootstrap_p95_ucb,
    build_schema6_runtime_cost_profile,
    make_measurement_cell,
    validate_schema6_runtime_cost_profile,
)


class Schema6CFOTests(unittest.TestCase):
    def test_raw_equation_is_distinct_from_operational_clip(self):
        self.assertEqual(cache_craft_cfo_raw(0, 0, 1, alpha=2), 2)
        self.assertEqual(cache_craft_cfo_operational(0, 0, 1, alpha=2), 1)

    def test_occurrence_overlap_order_and_degenerate_rules(self):
        old = (
            CanonicalChunkOccurrence("a", 0, 0, 10),
            CanonicalChunkOccurrence("b", 0, 10, 10),
        )
        metadata = build_source_cfo_metadata(
            historical_prefix_chunk_occurrences=old,
            inter_mass_by_layer_and_occurrence=(
                {"a#0": 2.0, "b#0": 1.0},
                {"a#0": 2.0, "b#0": 1.0},
            ),
            intra_mass_by_layer=(3.0, 3.0),
            target_token_count=10,
        )
        new = (
            CanonicalChunkOccurrence("b", 0, 0, 10),
            CanonicalChunkOccurrence("a", 0, 10, 10),
        )
        result = compute_cachecraft_cfo(metadata, new)
        self.assertEqual(result.prefix_overlap, 1)
        self.assertEqual(result.order_penalty, 1)
        self.assertEqual(result.adjusted_prefix_overlap, 0)
        self.assertAlmostEqual(result.cfo_raw, metadata.cci)

        zero = build_source_cfo_metadata(
            historical_prefix_chunk_occurrences=(old[0],),
            inter_mass_by_layer_and_occurrence=({"a#0": 0.0},),
            intra_mass_by_layer=(0.0,), target_token_count=10,
        )
        self.assertEqual(zero.cci, 0.5)
        no_overlap = compute_cachecraft_cfo(zero, ())
        self.assertEqual(no_overlap.prefix_overlap, 0)
        self.assertEqual(no_overlap.order_penalty, 0)

    def test_streaming_logsumexp_matches_eager_gqa_attention(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        torch.manual_seed(7)
        q = torch.randn(4, 4, 3)
        k = torch.randn(4, 2, 3)
        occurrences = ("a", "a", "target", "target")
        scale = 1 / math.sqrt(3)
        observed = streaming_qk_attention_mass(
            q, k, occurrences, scale=scale, block_size=2
        )
        head_map = torch.arange(4) // 2
        expanded = k.index_select(1, head_map)
        logits = torch.einsum("qhd,khd->qhk", q.float(), expanded.float()) * scale
        mask = torch.arange(4).view(1, 1, -1) <= torch.arange(4).view(-1, 1, 1)
        probs = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1).mean(dim=1)
        expected_inter = float(probs[2:, :2].sum().item())
        expected_intra = float(
            probs[2, 2].item() + probs[3, 2:4].sum().item()
        )
        self.assertAlmostEqual(observed.inter_mass_by_pair[("a", "target")], expected_inter, places=6)
        self.assertAlmostEqual(observed.intra_mass_by_occurrence["target"], expected_intra, places=6)
        eager = eager_qk_attention_mass(q, k, occurrences, scale=scale)
        self.assertAlmostEqual(
            eager.intra_mass_by_occurrence["target"], expected_intra, places=6
        )

    def test_full_prefill_collector_reshapes_hook_payload_and_discards_qk(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        torch.manual_seed(11)
        occurrences = ("p#0", "p#0", "target#0", "target#0")
        collector = CFOFullPrefillCollector(
            occurrences, expected_layers=2, block_size=2,
            eager_reference=True, eager_tolerance=1e-5,
        )
        for _ in range(2):
            collector(
                q=torch.randn(4, 12), k=torch.randn(4, 6),
                positions=torch.arange(4), scale=1 / math.sqrt(3),
                q_heads=4, kv_heads=2, head_dim=3,
            )
        metadata, audit = collector.finalize(
            prefix_occurrences=(CanonicalChunkOccurrence("p", 0, 0, 2),),
            target_occurrence=CanonicalChunkOccurrence("target", 0, 2, 2),
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["captured_layers"], 2)
        self.assertEqual(len(collector.layer_masses), 2)
        self.assertTrue(metadata.metadata_digest)


class Schema6RuntimeProfileTests(unittest.TestCase):
    def test_bootstrap_contract_is_deterministic(self):
        first = bootstrap_p95_ucb([1, 2, 3, 4, 5], resamples=100, seed=9)
        second = bootstrap_p95_ucb([1, 2, 3, 4, 5], resamples=100, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_prng"], "PCG64")
        self.assertEqual(first["quantile_method"], "linear")

    def test_sparse_profile_is_valid_but_not_frozen(self):
        cell = make_measurement_cell(
            "comparison_batch",
            axes={"k": 1, "token_count": 512, "completed_depth": 1, "backing_tier": "pinned_cpu"},
            measurements_ms=[1, 1.1, 0.9, 1.2, 1.0], warmups=2, resamples=50,
        )
        profile = build_schema6_runtime_cost_profile(
            model_key="mistral", policy="causal_commit_wait", code_commit="sha",
            cacheblend_patch_sha256="patch", gpu_uuid="gpu",
            hardware_compatibility_signature="hardware", measurement_cells=(cell,),
            measurement_sha256="measurement", frozen=False,
        )
        validate_schema6_runtime_cost_profile(profile, require_frozen=False)
        with self.assertRaises(ValueError):
            validate_schema6_runtime_cost_profile(profile, require_frozen=True)


if __name__ == "__main__":
    unittest.main()
