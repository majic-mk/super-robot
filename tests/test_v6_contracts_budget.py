import unittest
from dataclasses import replace

from probekv.candidate_budget import (
    VariantComparisonCandidate,
    allocate_variant_comparisons,
)
from probekv.contracts import (
    CandidateBounds,
    HistoricalSource,
    KVLocation,
    SelectionReason,
    SourceOrigin,
)
from probekv.manifest import token_content_hash
from probekv.manifest import ManifestSource
from probekv.model_signature import ModelSignature
from probekv.multisegment_selector import MultiSegmentProbeSelector
from probekv.selector import DynamicProbeSelector, ProbePolicy, SelectorPolicy
from probekv.v6_contracts import (
    RegionKind,
    RegionSpec,
    RequestSpec,
    SegmentSpec,
)
from probekv.v6_manifest import (
    RequestManifestCase,
    SegmentManifest,
    request_manifest_digest,
    request_case_from_mapping,
    validate_request_manifest,
)


def structured_signature(label="model"):
    return ModelSignature(
        weights_revision=label + "-weights",
        tokenizer_revision=label + "-tokenizer",
        rope_signature="rope-default",
        dtype="bf16",
        runtime_signature="cacheblend-b72d7945",
    ).encode()


def request_with_segments(count=5, variants=1):
    signature = structured_signature()
    token_ids = [10, 11]
    regions = [RegionSpec("prefix", RegionKind.PREFIX_EXACT, 0, 2)]
    segments = []
    cursor = 2
    for index in range(count):
        segment_tokens = (100 + index * 2, 101 + index * 2)
        start = cursor
        token_ids.extend(segment_tokens)
        cursor += 2
        content_hash = token_content_hash(segment_tokens)
        sources = tuple(
            HistoricalSource(
                source_id="c%d-s%d" % (index, source_index),
                content_hash=content_hash,
                context_id="c%d-context-%d" % (index, source_index),
                model_signature=signature,
                token_count=2,
                exact=True,
                origin=SourceOrigin.FULL_PREFILL,
                kv_location=KVLocation.PINNED_CPU,
            )
            for source_index in range(variants)
        )
        segment_id = "c%d" % index
        regions.append(
            RegionSpec(
                "region-" + segment_id,
                RegionKind.REUSE_CANDIDATE,
                start,
                cursor,
                segment_id,
            )
        )
        segments.append(
            SegmentSpec(
                segment_id,
                index,
                start,
                cursor,
                content_hash,
                segment_tokens,
                sources,
            )
        )
        if index < count - 1:
            token_ids.append(500 + index)
            regions.append(
                RegionSpec(
                    "dense-%d" % index,
                    RegionKind.DENSE,
                    cursor,
                    cursor + 1,
                )
            )
            cursor += 1
    token_ids.extend((900, 901))
    regions.append(
        RegionSpec(
            "suffix", RegionKind.MANDATORY_SUFFIX, cursor, cursor + 2
        )
    )
    request = RequestSpec(
        request_id="request",
        model_signature=signature,
        token_ids=tuple(token_ids),
        regions=tuple(regions),
        segments=tuple(segments),
        exact_prefix_tokens=2,
        mandatory_suffix_tokens=2,
    )
    request.validate()
    return request


class V6RequestContractTests(unittest.TestCase):
    def test_five_segments_are_all_represented_without_hard_cap(self):
        request = request_with_segments(5, 16)
        self.assertEqual(len(request.segments), 5)
        self.assertEqual(sum(len(item.sources) for item in request.segments), 80)

    def test_unstructured_model_signature_is_rejected(self):
        request = request_with_segments(1, 1)
        invalid = RequestSpec(
            request.request_id,
            "model@revision",
            request.token_ids,
            request.regions,
            request.segments,
            request.exact_prefix_tokens,
            request.mandatory_suffix_tokens,
        )
        with self.assertRaisesRegex(ValueError, "structured"):
            invalid.validate()

    def test_current_context_cannot_enter_historical_candidates(self):
        request = request_with_segments(1, 1)
        invalid = replace(
            request,
            current_context_id=request.segments[0].sources[0].context_id,
        )
        with self.assertRaisesRegex(ValueError, "current request context"):
            invalid.validate()

    def test_ten_segment_request_manifest_round_trip_is_deterministic(self):
        request = request_with_segments(10, 16)
        segments = tuple(
            SegmentManifest(
                segment.segment_id,
                segment.order,
                "document-%d" % segment.order,
                segment.token_start,
                segment.token_end,
                "segment-%d" % segment.order,
                segment.token_ids,
                segment.content_hash,
                tuple(
                    ManifestSource(
                        source.source_id,
                        "historical-%s" % source.context_id,
                        source.context_id,
                    )
                    for source in segment.sources
                ),
            )
            for segment in request.segments
        )
        case = RequestManifestCase(
            "case",
            "dataset",
            "group",
            "train",
            request.model_signature,
            request.token_ids,
            request.regions,
            segments,
            request.exact_prefix_tokens,
            request.mandatory_suffix_tokens,
        )
        validate_request_manifest((case,))
        row = case.to_row()
        self.assertEqual(len(row["segments"]), 10)
        self.assertEqual(len(row["segments"][-1]["sources"]), 16)
        self.assertEqual(request_manifest_digest((case,)), request_manifest_digest((case,)))
        self.assertEqual(request_case_from_mapping(row), case)

    def test_any_segment_content_crossing_splits_is_rejected(self):
        request = request_with_segments(1, 1)
        segment = request.segments[0]
        manifest_segment = SegmentManifest(
            segment.segment_id,
            0,
            "shared-document",
            segment.token_start,
            segment.token_end,
            "segment",
            segment.token_ids,
            segment.content_hash,
            (ManifestSource("s", "history", "ctx"),),
        )
        common = dict(
            dataset="dataset",
            model_signature=request.model_signature,
            token_ids=request.token_ids,
            regions=request.regions,
            segments=(manifest_segment,),
            exact_prefix_tokens=request.exact_prefix_tokens,
            mandatory_suffix_tokens=request.mandatory_suffix_tokens,
        )
        first = RequestManifestCase(
            case_id="first", group_id="g0", split="train", **common
        )
        second = RequestManifestCase(
            case_id="second", group_id="g1", split="test", **common
        )
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_request_manifest((first, second))


class CandidateBudgetTests(unittest.TestCase):
    def test_all_sixteen_variants_are_compared_when_budget_allows(self):
        candidates = [
            VariantComparisonCandidate(
                "c0", "s%d" % index, index / 16.0, 20.0, 0.01
            )
            for index in range(16)
        ]
        allocation = allocate_variant_comparisons(
            candidates,
            full_reference_ms=100.0,
            probe_ms=1.0,
            metadata_ms=0.1,
        )
        self.assertEqual(allocation.audits[0].compared_k, 16)
        self.assertEqual(allocation.audits[0].dropped_source_ids, ())

    def test_request_budget_reduces_candidates_deterministically(self):
        candidates = [
            VariantComparisonCandidate(
                "c%d" % segment,
                "c%d-s%d" % (segment, source),
                float(source),
                10.0 - segment,
                0.08,
            )
            for segment in range(5)
            for source in range(4)
        ]
        allocation = allocate_variant_comparisons(
            candidates,
            full_reference_ms=10.0,
            probe_ms=0.2,
            metadata_ms=0.2,
        )
        self.assertLess(sum(audit.compared_k for audit in allocation.audits), 20)
        self.assertLessEqual(
            allocation.budget_used_ms, allocation.budget_available_ms
        )

    def test_comparison_count_is_sum_of_k_not_cartesian_product(self):
        candidates = tuple(
            VariantComparisonCandidate(
                "c%d" % segment,
                "c%d-s%d" % (segment, source),
                float(source),
                10.0,
                0.01,
            )
            for segment in range(3)
            for source in range(2)
        )
        allocation = allocate_variant_comparisons(
            candidates,
            full_reference_ms=100.0,
            probe_ms=0.1,
            metadata_ms=0.1,
            segment_ids=("c0", "c1", "c2"),
        )
        self.assertEqual(sum(item.compared_k for item in allocation.audits), 6)

    def test_best_sixteenth_source_can_win_at_first_probe_layer(self):
        candidates = [
            VariantComparisonCandidate("c0", "s%d" % index, index, 20.0, 0.001)
            for index in range(16)
        ]
        allocation = allocate_variant_comparisons(
            candidates,
            full_reference_ms=100.0,
            probe_ms=0.1,
            metadata_ms=0.1,
        )
        bounds = {
            "c0": {
                1: tuple(
                    CandidateBounds(
                        "s%d" % index,
                        0.1,
                        9.0 if index == 15 else 29.0 + index,
                        10.0 if index == 15 else 30.0 + index,
                    )
                    for index in range(16)
                )
            }
        }
        selector = MultiSegmentProbeSelector(
            DynamicProbeSelector(
                ProbePolicy(
                    (1,),
                    1,
                    SelectorPolicy.STRICT_INTERVAL,
                    preliminary_economic_filter=True,
                )
            )
        )
        plan = selector.select("request", allocation, bounds)
        decision = plan.segment_decisions[0].source_decision
        self.assertEqual(decision.selected_source_id, "s15")
        self.assertEqual(decision.probe_layer, 1)
        self.assertEqual(decision.selection_reason, SelectionReason.EARLY_CONFIDENT)

    def test_no_comparison_budget_forces_abstention(self):
        candidate = VariantComparisonCandidate("c0", "s0", 0.0, 10.0, 1.0)
        allocation = allocate_variant_comparisons(
            [candidate],
            full_reference_ms=10.0,
            probe_ms=0.5,
            metadata_ms=0.1,
        )
        selector = MultiSegmentProbeSelector(
            DynamicProbeSelector(ProbePolicy((1,), 1))
        )
        plan = selector.select("request", allocation, {"c0": {}})
        decision = plan.segment_decisions[0].source_decision
        self.assertTrue(decision.abstained)
        self.assertEqual(
            decision.selection_reason,
            SelectionReason.COMPARISON_BUDGET_EXHAUSTED,
        )

    def test_all_miss_segment_remains_in_plan_and_goes_dense(self):
        allocation = allocate_variant_comparisons(
            [],
            full_reference_ms=10.0,
            probe_ms=0.1,
            metadata_ms=0.1,
            segment_ids=("c0",),
            stored_count_by_segment={"c0": 0},
        )
        selector = MultiSegmentProbeSelector(
            DynamicProbeSelector(ProbePolicy((1,), 1))
        )
        plan = selector.select("request", allocation, {})
        self.assertEqual(len(plan.segment_decisions), 1)
        decision = plan.segment_decisions[0]
        self.assertEqual(decision.comparison.stored_k, 0)
        self.assertTrue(decision.source_decision.abstained)
        self.assertEqual(
            decision.source_decision.selection_reason,
            SelectionReason.NO_QUALITY_SAFE_SOURCE,
        )

    def test_ineligible_variant_is_counted_as_stored_and_dropped(self):
        allocation = allocate_variant_comparisons(
            (
                VariantComparisonCandidate("c0", "safe", 0.1, 10.0, 0.01),
                VariantComparisonCandidate(
                    "c0", "outside-support", 0.0, 20.0, 0.01, eligible=False
                ),
            ),
            full_reference_ms=100.0,
            probe_ms=1.0,
            metadata_ms=0.1,
        )
        audit = allocation.audits[0]
        self.assertEqual((audit.stored_k, audit.eligible_k, audit.compared_k), (2, 1, 1))
        self.assertEqual(audit.compared_source_ids, ("safe",))
        self.assertEqual(audit.dropped_source_ids, ("outside-support",))

    def test_one_segment_abstention_does_not_block_another_segment(self):
        allocation = allocate_variant_comparisons(
            (VariantComparisonCandidate("c0", "c0-s", 0.0, 20.0, 0.01),),
            full_reference_ms=100.0,
            probe_ms=0.1,
            metadata_ms=0.1,
            segment_ids=("c0", "c1"),
            stored_count_by_segment={"c0": 1, "c1": 0},
        )
        selector = MultiSegmentProbeSelector(
            DynamicProbeSelector(ProbePolicy((1,), 1))
        )
        plan = selector.select(
            "request",
            allocation,
            {"c0": {1: (CandidateBounds("c0-s", 0.1, 10.0, 20.0),)}},
        )
        self.assertEqual(plan.segment_decisions[0].source_decision.selected_source_id, "c0-s")
        self.assertTrue(plan.segment_decisions[1].source_decision.abstained)


if __name__ == "__main__":
    unittest.main()
