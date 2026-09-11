import math
import unittest

import torch

from probekv.source_policy_development import (
    residual_tail_curve, plan_depth2_shortlist, audit_depth2_pruning,
    cacheblend_pinned_value_scores,
    cacheblend_kv_deviation_scores, rank_winner_kv_positions,
    development_experiment_spec,
)
from probekv.semantic_segment import (
    SemanticWindowConfig, segment_document_tokens, punctuation_token_boundaries,
)
from probekv.v8_schema7_repair import source_score_from_k_drifts


class ResidualCurveTests(unittest.TestCase):
    def test_one_curve_matches_existing_reference_scores_at_all_points(self):
        values = tuple((j % 7) / 9 for j in range(131))
        positions = tuple(range(32, 163))
        curve = residual_tail_curve(values, positions)
        for point in curve:
            expected, _ = source_score_from_k_drifts(values, trim_ratio=point.nominal_ratio,
                                                    absolute_positions=positions)
            self.assertAlmostEqual(point.residual_mean, expected)
            self.assertEqual(point.trim_count, math.ceil(point.nominal_ratio * len(values)))
        self.assertFalse(hasattr(curve[0], 'repair_positions'))

    def test_five_and_fifteen_can_reverse_source_order(self):
        a, b = [100., 100.] + [0.] * 18, [1.] * 20
        ca, cb = residual_tail_curve(a, range(20)), residual_tail_curve(b, range(20))
        by_ratio_a = {point.nominal_ratio: point for point in ca}
        by_ratio_b = {point.nominal_ratio: point for point in cb}
        self.assertGreater(by_ratio_a[0.05].residual_mean, by_ratio_b[0.05].residual_mean)
        self.assertLess(by_ratio_a[0.15].residual_mean, by_ratio_b[0.15].residual_mean)

    def test_empty_tail_and_invalid_inputs_rejected(self):
        for values, positions, ratios in (([1.], [0], (.15,)),
                ([1., 2.], [0, 1], (1.,)), ([float('nan'), 0.], [0, 1], (.15,)),
                ([1., 2.], [0, 0], (.15,)), ([1., 2.], [True, 2], (.15,))):
            with self.assertRaises(ValueError):
                residual_tail_curve(values, positions, ratios=ratios)

    def test_pinned_v_ranking_is_not_normalized_v(self):
        current = torch.tensor([100., 2.], dtype=torch.bfloat16).reshape(2, 1, 1)
        source = torch.tensor([90., 1.], dtype=torch.bfloat16).reshape(2, 1, 1)
        scores = cacheblend_pinned_value_scores(current, source)
        self.assertTrue(torch.equal(scores, torch.sum((current - source) ** 2, dim=(1, 2))))
        self.assertEqual(scores.argmax().item(), 0)
        self.assertEqual(((current.float() - source.float()).abs() / current.float()).argmax().item(), 1)
        with self.assertRaises(ValueError):
            cacheblend_pinned_value_scores(current, source.float())
        with self.assertRaises(ValueError):
            cacheblend_pinned_value_scores(current * float('nan'), source)

    def test_runtime_winner_ranking_keeps_metric_and_absolute_positions_separate(self):
        from probekv.source_policy_development import rank_winner_v_positions
        current = torch.tensor([100.,2.],dtype=torch.bfloat16).reshape(2,1,1)
        source = torch.tensor([90.,1.],dtype=torch.bfloat16).reshape(2,1,1)
        self.assertEqual(rank_winner_v_positions(current,source,(128,129),metric="normalized_v_legacy"),(129,128))
        self.assertEqual(rank_winner_v_positions(current,source,(128,129),metric="value_squared_l2_pinned_dtype"),(128,129))
        self.assertEqual(rank_winner_v_positions(current,current,(128,129),metric="value_squared_l2_pinned_dtype"),(128,129))
        with self.assertRaises(ValueError):
            rank_winner_v_positions(current,source,(128,128),metric="value_squared_l2_pinned_dtype")

    def test_kv_deviation_is_winner_only_and_has_distinct_ranking(self):
        current_k = torch.tensor([1., 2.], dtype=torch.float32).reshape(2, 1, 1)
        source_k = torch.tensor([1., 1.], dtype=torch.float32).reshape(2, 1, 1)
        current_v = torch.tensor([10., 1.], dtype=torch.float32).reshape(2, 1, 1)
        source_v = torch.tensor([9., 0.5], dtype=torch.float32).reshape(2, 1, 1)
        scores = cacheblend_kv_deviation_scores(current_k, source_k, current_v, source_v)
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(rank_winner_kv_positions(
            current_k, source_k, current_v, source_v, (128, 129)), (129, 128))
        with self.assertRaises(ValueError):
            rank_winner_kv_positions(
                current_k, source_k, current_v, source_v, (128, 129), metric="normalized_v_legacy")

    def test_kv_deviation_rejects_empty_or_mixed_kv_geometry(self):
        empty = torch.empty((0, 1, 1), dtype=torch.float32)
        one = torch.ones((1, 1, 1), dtype=torch.float32)
        with self.assertRaises(ValueError):
            cacheblend_kv_deviation_scores(empty, empty, empty, empty)
        with self.assertRaises(ValueError):
            cacheblend_kv_deviation_scores(one, one, one.to(torch.bfloat16), one.to(torch.bfloat16))


class CascadeTests(unittest.TestCase):
    def make(self, scores, **kw):
        args = dict(request_binding='q:segment:g1', source_inventory_digest='inventory',
                    correctness_eligible_k=len(scores))
        args.update(kw)
        return plan_depth2_shortlist(scores, **args)

    def test_half_saves_second_depth_comparisons_not_claimed_complete(self):
        plan = self.make({str(i): float(i) for i in range(16)})
        self.assertEqual(len(plan.retained_source_ids), 8)
        self.assertFalse(plan.scope_complete_at_depth2)
        self.assertEqual(plan.correctness_eligible_k, 16)
        rank = plan.validate_depth2({s: 1. for s in plan.retained_source_ids},
            request_binding=plan.request_binding, source_inventory_digest='inventory', reference_trim_ratio=.15)
        self.assertEqual(len(rank), 8)

    def test_flat_or_boundary_ties_not_arbitrarily_halved(self):
        self.assertEqual(len(self.make({str(i): 0. for i in range(16)}).retained_source_ids), 16)
        self.assertEqual(len(self.make({'a': 0., 'b': 1., 'c': 1.0000001, 'd': 2.}).retained_source_ids), 3)

    def test_minimum_two_and_two_different_k_one_cases(self):
        self.assertEqual(len(self.make({'a': 0., 'b': 1., 'c': 2.}).retained_source_ids), 2)
        one = self.make({'a': 0.})
        self.assertTrue(one.scope_complete_at_depth2)
        self.assertFalse(one.insufficient_ranking_coverage)
        partial = self.make({'a': 0.}, correctness_eligible_k=16)
        self.assertTrue(partial.insufficient_ranking_coverage)
        self.assertFalse(partial.scope_complete_at_depth2)

    def test_stale_request_or_rho_and_missing_retained_source_rejected(self):
        plan = self.make({'a': 0., 'b': 1., 'c': 2., 'd': 3.})
        args = dict(request_binding=plan.request_binding, source_inventory_digest='inventory', reference_trim_ratio=.15)
        for k, v in (('request_binding', 'another'), ('source_inventory_digest', 'changed'),
                     ('reference_trim_ratio', .05)):
            with self.assertRaises(ValueError):
                plan.validate_depth2({'a': 1., 'b': 2.}, **{**args, k: v})
        with self.assertRaises(ValueError):
            plan.validate_depth2({'a': 1.}, **args)

    def test_shadow_detects_lost_depth2_winner_without_readmitting_it(self):
        plan = self.make({'a': 0., 'b': 1., 'c': 2., 'd': 3.})
        result = audit_depth2_pruning(plan, {'a': 1., 'b': 2., 'c': 0., 'd': 3.})
        self.assertFalse(result['oracle_winner_retained'])
        self.assertEqual(result['absolute_residual_regret'], 1.)
        self.assertNotIn('c', plan.retained_source_ids)
        self.assertFalse(result['production_admission_allowed'])
        with self.assertRaises(ValueError):
            audit_depth2_pruning(plan, {'a': 1., 'b': 2.})

    def test_d0_never_prunes_or_locks(self):
        with self.assertRaises(ValueError):
            self.make({'a': 0., 'b': 1.}, completed_depth=0)

    def test_factorized_spec_is_not_runtime_qualification(self):
        value = development_experiment_spec()
        self.assertEqual(value, development_experiment_spec())
        self.assertEqual(len(value['selection_cells']), 4)
        self.assertEqual(len(value['chunking_cells']), 9)
        self.assertEqual(value['default_execution_objective'], 'efficiency_first')
        self.assertIsNone(value['quality_profile'])
        self.assertFalse(value['gpu_execution_allowed'])
        self.assertEqual(value['candidate_comparison_policy_default'], 'full_compare')
        self.assertTrue(value['cfo_and_anchor_are_mutually_exclusive'])
        self.assertEqual(tuple(value['repair_ratio_candidates']),
                         (.05, .075, .10, .125, .15, .175, .20, .25, .30))


class SemanticWindowTests(unittest.TestCase):
    def split(self, text, **cfg):
        config = SemanticWindowConfig(target_tokens=20, search_window_tokens=5,
                                      min_tail_tokens=5, **cfg)
        return segment_document_tokens(range(len(text)), text=text,
            offsets=[(i, i + 1) for i in range(len(text))], tokenizer_signature='tok',
            document_revision='doc1', config=config)

    def test_paragraph_then_sentence_then_clause_priority(self):
        # Paragraph at exclusive char/token 24 wins over a sentence at 20.
        text = 'a' * 19 + '。' + 'bb\n\n' + 'c' * 40
        segments, audit = self.split(text)
        self.assertEqual(segments[0].token_end, 24)
        self.assertEqual(audit['cuts'][0]['reason'], 'paragraph')
        text = 'a' * 19 + '。' + 'c' * 40
        self.assertEqual(self.split(text)[1]['cuts'][0]['reason'], 'sentence')
        text = 'a' * 19 + '，' + 'c' * 40
        self.assertEqual(self.split(text)[1]['cuts'][0]['reason'], 'clause')

    def test_fixed_and_forced_fallback_are_explicit_not_fake_sentence_safety(self):
        text = 'a' * 61
        semantic, audit = self.split(text)
        self.assertTrue(audit['cuts'][0]['forced_token_cut'])
        self.assertFalse(audit['cuts'][0]['sentence_integrity_guaranteed'])
        self.assertEqual(semantic[0].token_count, 20)
        fixed, fa = self.split(text, policy='fixed_tokens_v1')
        self.assertEqual([s.token_count for s in fixed], [20, 20, 20, 1])
        self.assertNotEqual(audit['canonicalizer_signature'], fa['canonicalizer_signature'])

    def test_deterministic_exact_tokens_and_identity(self):
        text = 'a' * 20 + '。' + 'b' * 41
        first, a = self.split(text)
        second, b = self.split(text)
        self.assertEqual(first, second)
        self.assertEqual(a, b)
        self.assertEqual(tuple(t for s in first for t in s.token_ids), tuple(range(len(text))))
        self.assertFalse(a['runtime_timing_switch_allowed'])
        self.assertFalse(a['gpu_runtime_qualified'])

    def test_windows_are_token_not_character_lengths(self):
        text = 'a' * 40 + '。 ' + 'b' * 60
        offsets = [(i, min(i + 2, len(text))) for i in range(0, len(text), 2)]
        segments, _ = segment_document_tokens(range(len(offsets)), text=text, offsets=offsets,
            tokenizer_signature='tok', document_revision='doc',
            config=SemanticWindowConfig(20, 5, 5))
        self.assertEqual(segments[0].token_end, 21)

    def test_offset_overlap_and_token_containing_next_word_not_cut(self):
        self.assertIn(2, punctuation_token_boundaries('。x', [(0, 1), (0, 1), (1, 2)]))
        self.assertEqual(punctuation_token_boundaries('。x', [(0, 2)]), {})
        with self.assertRaises(ValueError):
            punctuation_token_boundaries('ab', [(1, 2), (0, 1)])

    def test_english_abbreviation_and_decimal_are_not_sentence_ends(self):
        text = 'Dr. Jones paid 3.14 dollars. Next.'
        boundaries = punctuation_token_boundaries(text, [(i, i + 1) for i in range(len(text))])
        self.assertNotIn(3, boundaries)
        self.assertNotIn(text.index('3.14') + 2, boundaries)
        self.assertIn(text.index('dollars.') + len('dollars.'), boundaries)

    def test_semantic_mode_requires_offsets_instead_of_silent_fallback(self):
        with self.assertRaises(ValueError):
            segment_document_tokens([1, 2], text='ab', offsets=None,
                tokenizer_signature='tok', document_revision='doc')
        with self.assertRaises(ValueError):
            SemanticWindowConfig(search_window_tokens=-1)


if __name__ == '__main__':
    unittest.main()
