import unittest
from dataclasses import replace

from probekv.cacheblend_backend import (
    MultiSegmentCacheBlendBackend,
    MultiSegmentRepairMeasurement,
    MultiSourceStageMeasurement,
    SegmentRuntimeRepairMeasurement,
    StaggeredMultiSegmentRepairMeasurement,
)
from probekv.causal_mask import absolute_causal_rows
from probekv.contracts import KVLocation
from probekv.repair_semantics import (
    assert_nested_multisegment_selections,
    select_multisegment_repair_tokens,
    select_staggered_multisegment_repair_tokens,
)
from tests.test_v6_contracts_budget import request_with_segments


class FakeMultiRegionRuntime:
    def __init__(
        self,
        mutate_on_stage=False,
        mutate_on_repair=False,
        dense_mismatch=False,
    ):
        self.mutate_on_stage = mutate_on_stage
        self.mutate_on_repair = mutate_on_repair
        self.dense_mismatch = dense_mismatch

    def stage_canonical_sources(self, request, sources_by_segment, target):
        ids = {segment: source.source_id for segment, source in sources_by_segment.items()}
        before = {segment: "digest-" + source_id for segment, source_id in ids.items()}
        after = dict(before)
        if self.mutate_on_stage and after:
            after[next(iter(after))] += "-mutated"
        return MultiSourceStageMeasurement(
            selected_source_ids=ids,
            load_start_ms_by_segment={segment: 0.0 for segment in ids},
            ready_ms_by_segment={segment: 2.0 for segment in ids},
            layer_ready_ms_by_segment={
                segment: {1: 1.0, 5: 2.0} for segment in ids
            },
            transferred_bytes_by_segment={segment: 100 for segment in ids},
            digest_before_by_segment=before,
            digest_after_by_segment=after,
        )

    def execute_multisegment_prefill(
        self, request, sources_by_segment, start_layer, repair_selection
    ):
        segment_map = {segment.segment_id: segment for segment in request.segments}
        rows = []
        for segment_id, source in sources_by_segment.items():
            selected = repair_selection.selected_indices_by_segment[segment_id]
            before = "digest-" + source.source_id
            after = before + ("-mutated" if self.mutate_on_repair else "")
            rows.append(
                SegmentRuntimeRepairMeasurement(
                    segment_id,
                    source.source_id,
                    repair_selection.requested_ratios[segment_id],
                    segment_map[segment_id].token_count,
                    selected,
                    len(selected) / float(segment_map[segment_id].token_count),
                    before,
                    after,
                )
            )
        return MultiSegmentRepairMeasurement(
            request_id=request.request_id,
            reuse_start_layer=start_layer,
            segments=tuple(rows),
            union_execution_indices=repair_selection.execution_indices,
            union_mask_digest=repair_selection.union_mask_digest,
            repair_gpu_ms=1.0,
            repair_host_ms=1.2,
            output_token_ids=(7, 8),
            output_hash="output-hash",
            output_text="answer",
            teacher_forced_logit_relative_l2=1e-5,
            teacher_forced_logit_positions=32,
            dense_reference_token_ids=(7, 9) if self.dense_mismatch else (7, 8),
            causal_mask_mode="absolute_query_positions",
            rope_alignment_mode="pre_rope_derotate_rerotate",
        )

    def dense_remaining_profile(self, request, start_layer):
        return float(33 - start_layer)

    def execute_staggered_multisegment_prefill(
        self, request, sources_by_segment, repair_plan
    ):
        before = {
            segment_id: "digest-" + source.source_id
            for segment_id, source in sources_by_segment.items()
        }
        after = dict(before)
        if self.mutate_on_repair and after:
            after[next(iter(after))] += "-mutated"
        return StaggeredMultiSegmentRepairMeasurement(
            request_id=request.request_id,
            boundary_by_segment=dict(repair_plan.boundary_by_segment),
            selected_indices_by_segment_layer={
                segment_id: dict(by_layer)
                for segment_id, by_layer in (
                    repair_plan.selected_indices_by_segment_layer.items()
                )
            },
            execution_indices_by_layer=dict(
                repair_plan.execution_indices_by_layer
            ),
            union_mask_digest_by_layer=dict(
                repair_plan.union_mask_digest_by_layer
            ),
            digest_before_by_segment=before,
            digest_after_by_segment=after,
            repair_gpu_ms=2.0,
            repair_host_ms=2.2,
            output_token_ids=(7, 8),
            dense_reference_token_ids=(7, 8),
            teacher_forced_logit_relative_l2=1e-5,
            teacher_forced_logit_positions=32,
            causal_mask_mode="absolute_query_positions_per_layer",
            rope_alignment_mode="pre_rope_derotate_rerotate",
        )

    def provenance(self):
        return {
            "cacheblend_commit": "b72d7945",
            "cacheblend_patch_sha256": "patch",
            "cacheblend_tree": "tree",
            "patch_mode": "probekv_v6_multiregion",
            "vllm": "0.4.1",
            "torch": "2.2.1",
            "cuda": "12.1",
        }


class MultiSegmentRepairTests(unittest.TestCase):
    def test_staggered_masks_change_by_layer_and_keep_unaccepted_segments_dense(self):
        request = request_with_segments(3, 1)
        scores = {
            layer: [float((index + layer) % 7) for index in range(len(request.token_ids))]
            for layer in range(1, 7)
        }
        plan = select_staggered_multisegment_repair_tokens(
            scores,
            request,
            {"c0": 0.5, "c2": 0.5},
            {"c0": 2, "c2": 5},
            6,
        )
        c0 = request.segments[0]
        c1 = request.segments[1]
        c2 = request.segments[2]
        self.assertTrue(
            set(range(c0.token_start, c0.token_end)).issubset(
                plan.execution_indices_by_layer[1]
            )
        )
        self.assertEqual(
            len(
                set(range(c0.token_start, c0.token_end))
                & set(plan.execution_indices_by_layer[2])
            ),
            1,
        )
        self.assertTrue(
            set(range(c2.token_start, c2.token_end)).issubset(
                plan.execution_indices_by_layer[4]
            )
        )
        self.assertEqual(
            len(
                set(range(c2.token_start, c2.token_end))
                & set(plan.execution_indices_by_layer[5])
            ),
            1,
        )
        for layer in range(1, 7):
            self.assertTrue(
                set(range(c1.token_start, c1.token_end)).issubset(
                    plan.execution_indices_by_layer[layer]
                )
            )

    def test_staggered_mask_builder_accepts_thirty_seven_segments(self):
        request = request_with_segments(37, 1)
        scores = {
            layer: [float(index % 11) for index in range(len(request.token_ids))]
            for layer in range(1, 3)
        }
        ratios = {segment.segment_id: 0.5 for segment in request.segments}
        boundaries = {
            segment.segment_id: 1 + (segment.order % 2)
            for segment in request.segments
        }
        plan = select_staggered_multisegment_repair_tokens(
            scores, request, ratios, boundaries, 2
        )
        self.assertEqual(len(plan.boundary_by_segment), 37)

    def test_union_mask_uses_absolute_positions_and_keeps_other_regions_dense(self):
        request = request_with_segments(3, 1)
        scores = [float(index) for index in range(len(request.token_ids))]
        selection = select_multisegment_repair_tokens(
            scores, request, {"c0": 0.5, "c2": 0.5}
        )
        self.assertFalse(
            set(range(request.exact_prefix_tokens))
            & set(selection.execution_indices)
        )
        c1 = next(segment for segment in request.segments if segment.segment_id == "c1")
        self.assertTrue(
            set(range(c1.token_start, c1.token_end)).issubset(
                selection.dense_indices
            )
        )
        rows = absolute_causal_rows(selection.execution_indices, len(request.token_ids))
        for query_position, row in zip(selection.execution_indices, rows):
            self.assertTrue(all(row[: query_position + 1]))
            self.assertFalse(any(row[query_position + 1 :]))

    def test_all_candidate_ratios_one_equals_dense_nonprefix_execution(self):
        request = request_with_segments(5, 1)
        selection = select_multisegment_repair_tokens(
            [0.0] * len(request.token_ids),
            request,
            {segment.segment_id: 1.0 for segment in request.segments},
        )
        self.assertEqual(
            selection.execution_indices,
            tuple(range(request.exact_prefix_tokens, len(request.token_ids))),
        )

    def test_per_segment_ratio_masks_are_nested(self):
        request = request_with_segments(2, 1)
        scores = [float(index % 5) for index in range(len(request.token_ids))]
        selections = [
            select_multisegment_repair_tokens(scores, request, {"c0": ratio})
            for ratio in (0.0, 0.5, 1.0)
        ]
        assert_nested_multisegment_selections(selections, "c0")

    def test_external_mask_cannot_omit_suffix_or_dense_region(self):
        request = request_with_segments(2, 1)
        selection = select_multisegment_repair_tokens(
            [0.0] * len(request.token_ids), request, {"c0": 0.5}
        )
        invalid = replace(
            selection, dense_indices=selection.dense_indices[:-1]
        )
        with self.assertRaisesRegex(ValueError, "dense mask"):
            invalid.validate(request)


class MultiSegmentCacheBlendAdapterTests(unittest.TestCase):
    def test_staggered_backend_validates_per_layer_masks_and_boundaries(self):
        request = request_with_segments(3, 1)
        sources = {
            segment.segment_id: segment.sources[0]
            for segment in request.segments
        }
        plan = select_staggered_multisegment_repair_tokens(
            {
                layer: [float(index % 5) for index in range(len(request.token_ids))]
                for layer in range(1, 33)
            },
            request,
            {segment.segment_id: 1.0 for segment in request.segments},
            {"c0": 2, "c1": 5, "c2": 3},
            32,
        )
        backend = MultiSegmentCacheBlendBackend(FakeMultiRegionRuntime(), 32)
        measured = backend.repair_request_staggered(request, sources, plan)
        self.assertEqual(
            measured.boundary_by_segment, {"c0": 2, "c1": 5, "c2": 3}
        )
        self.assertEqual(len(measured.execution_indices_by_layer), 32)

    def test_staging_and_repair_preserve_canonical_sources(self):
        request = request_with_segments(3, 1)
        sources = {
            "c0": request.segments[0].sources[0],
            "c2": request.segments[2].sources[0],
        }
        selection = select_multisegment_repair_tokens(
            [float(index) for index in range(len(request.token_ids))],
            request,
            {"c0": 0.5, "c2": 1.0},
        )
        backend = MultiSegmentCacheBlendBackend(FakeMultiRegionRuntime(), 32)
        staged = backend.prepare_sources(request, sources, KVLocation.GPU)
        measured = backend.repair_request(request, sources, 5, selection)
        self.assertEqual(set(staged.selected_source_ids), {"c0", "c2"})
        self.assertEqual(measured.union_mask_digest, selection.union_mask_digest)
        self.assertEqual(backend.dense_remaining_profile(request, 5), 28.0)
        self.assertEqual(backend.provenance()["vllm"], "0.4.1")

    def test_stage_digest_mutation_is_rejected(self):
        request = request_with_segments(1, 1)
        sources = {"c0": request.segments[0].sources[0]}
        backend = MultiSegmentCacheBlendBackend(
            FakeMultiRegionRuntime(mutate_on_stage=True), 32
        )
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            backend.prepare_sources(request, sources)

    def test_repair_digest_mutation_is_rejected(self):
        request = request_with_segments(1, 1)
        sources = {"c0": request.segments[0].sources[0]}
        selection = select_multisegment_repair_tokens(
            [0.0] * len(request.token_ids), request, {"c0": 1.0}
        )
        backend = MultiSegmentCacheBlendBackend(
            FakeMultiRegionRuntime(mutate_on_repair=True), 32
        )
        with self.assertRaisesRegex(RuntimeError, "mutated"):
            backend.repair_request(request, sources, 5, selection)

    def test_all_r_one_must_match_dense_reference_exactly(self):
        request = request_with_segments(2, 1)
        sources = {
            segment.segment_id: segment.sources[0]
            for segment in request.segments
        }
        selection = select_multisegment_repair_tokens(
            [0.0] * len(request.token_ids),
            request,
            {segment.segment_id: 1.0 for segment in request.segments},
        )
        backend = MultiSegmentCacheBlendBackend(
            FakeMultiRegionRuntime(dense_mismatch=True), 32
        )
        with self.assertRaisesRegex(RuntimeError, "r=1 output"):
            backend.repair_request(request, sources, 5, selection)


if __name__ == "__main__":
    unittest.main()
