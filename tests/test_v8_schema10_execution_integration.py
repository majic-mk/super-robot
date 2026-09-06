import copy
import json
import time
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from probekv.config import ExperimentConfig
from probekv.v8_contracts import CandidateCounts, ResidualCandidate
from probekv.v8_schema8_planner import Gate1LocalPlan, Gate1MarginalLowerBound
from probekv.v8_schema10_contracts import AbsoluteResidualThreshold, Gate1Mode
from probekv.v8_schema10_profile import VariantAdmissionProfileV10, PreparationPolicyProfile, AbsoluteResidualThresholdPointV10
from probekv.v8_schema10_selector import Schema10CheckpointSelector
from probekv.v8_schema10_execution import (
    SelectionCostPolicy, SelectionCostLedger, ProductionSelectionSession, replay_selection_events,
)
from probekv.v8_schema10_experiments import (
    SourceOutcome, measured_quality_cost_oracle, run_source_oracle,
    run_gate1_pairs, run_causal_capacity_traces, run_serving_trace,
)
from probekv.v8_schema10_execution import digest_json


def online_row(request, dispatch, arrival_ns):
    first = time.perf_counter_ns()
    return dict(request_id=request['request_id'], execution_kind='online_policy',
                forced_source=False, final_commit_executed=True,
                selection_events=[{'decision': {'selected_source_variant_id': 's'}}],
                runtime_events=[{'kind': 'final_commit'}], dispatch_config=dict(dispatch),
                arrival_ns=arrival_ns, first_token_ns=first, completion_ns=time.perf_counter_ns(),
                request_ttft_ms=(first-arrival_ns)/1e6, quality_passed=True,
                code_commit='code', model_signature='model')


def make_selector(depths=(1, 2)):
    profile = VariantAdmissionProfileV10(
        'code', 'patch', 'model', 'rev', 'tok', .15,
        tuple(AbsoluteResidualThreshold(d, .5) for d in (1, 2)),
        threshold_table=tuple(AbsoluteResidualThresholdPointV10(d, .15, .5)
                              for d in sorted(set(depths) | {1, 2})),
    )
    return Schema10CheckpointSelector(
        variant_profile=profile,
        preparation_profile=PreparationPolicyProfile('code', 'model', 'policy', Gate1Mode.EXPLICIT_BARRIER),
        strong_margin=.6, stable_margin=.3, residual_band_relative_tolerance=.05,
        checkpoint_depths=depths,
    )


def plan(source, depth, cost=1):
    return Gate1LocalPlan(source, depth, depth, depth+1, 0, Gate1MarginalLowerBound(0, 0, cost), 10)


class ExecutionIntegrationTests(unittest.TestCase):
    def test_end_to_end_does_not_reject_twelve_percent_comparison(self):
        self.assertTrue(SelectionCostLedger(100, SelectionCostPolicy()).may_compare(12))
        self.assertFalse(SelectionCostLedger(100, SelectionCostPolicy('legacy_fixed_fraction')).may_compare(12))

    def test_new_budget_config_is_explicit_and_cannot_enter_legacy_schema(self):
        root = Path(__file__).resolve().parents[1]
        old = json.loads((root / 'configs/local_system_v8_schema10_gate1_barrier.json').read_text())
        new = json.loads((root / 'configs/local_system_v8_schema10_gate1_barrier_end_to_end.json').read_text())
        self.assertEqual(ExperimentConfig.from_mapping(old).selection_budget_policy, 'legacy_fixed_fraction')
        self.assertEqual(ExperimentConfig.from_mapping(new).selection_budget_policy, 'end_to_end_aware')
        with self.assertRaisesRegex(ValueError, 'schema10'):
            ExperimentConfig.from_mapping(dict(new, v8_schema_version=9))

    def test_unresolved_plan_is_not_required_to_already_pass_gamma(self):
        ledger = SelectionCostLedger(100, SelectionCostPolicy())
        self.assertFalse(ledger.continuation_proven_infeasible(30, joint_future_lower_ms=60,
                                                             complete_scope_lower_bound=False))
        self.assertTrue(ledger.continuation_proven_infeasible(30, joint_future_lower_ms=60,
                                                            complete_scope_lower_bound=True))
        self.assertTrue(ledger.may_compare(12))
        self.assertFalse(ledger.may_compare(12, resource_available=False))

    def test_ledger_unions_overlap_and_preserves_actual_cost(self):
        ledger = SelectionCostLedger(100, SelectionCostPolicy())
        ledger.reserve('a', 8); ledger.reserve('b', 8)
        ledger.settle('a', 0, 10_000_000); ledger.settle('b', 5_000_000, 15_000_000)
        self.assertEqual(ledger.actual_active_ms, 15)
        with self.assertRaises(ValueError): ledger.reserve('a', 1)
        with self.assertRaises(ValueError): ledger.may_compare(float('nan'))

    def test_d1_d2_and_legacy_use_same_selector_and_replay(self):
        for depths in ((1,), (1, 2), (1, 2, 4, 5, 8)):
            session = ProductionSelectionSession('q', ['c'], make_selector(depths), SelectionCostLedger(100, SelectionCostPolicy()))
            for depth in depths:
                # Neither strong nor stable: winner alternates until the final checkpoint.
                winner = 'a' if depths.index(depth) % 2 == 0 else 'b'
                candidates = [ResidualCandidate(s, .1 if s == winner else .11, 1, i) for i, s in enumerate(('a', 'b'))]
                session.step('c', completed_depth=depth, counts=CandidateCounts(2, 2, 2, 2, 2),
                             candidates=candidates, gate1_plan_by_source={s: plan(s, depth) for s in ('a', 'b')})
            self.assertTrue(session.closed)
            replay = ProductionSelectionSession('q', ['c'], make_selector(depths), SelectionCostLedger(100, SelectionCostPolicy()))
            self.assertEqual(replay_selection_events(replay, session.events), tuple(session.events))
            with self.assertRaises(RuntimeError):
                session.step('c', completed_depth=depths[-1], counts=CandidateCounts(0, 0, 0, 0, 0), candidates=[], gate1_plan_by_source={})

    def test_oracle_is_actual_quality_cost_not_minimum_residual(self):
        base = dict(request_id='q', repair_ratio=.15, first_reuse_layer=3,
                    timing_scope='request_ttft', provenance_sha256='a'*64, source_digest_unchanged=True)
        rows = [SourceOutcome(**base, source_id=None, ttft_ms=100, answer_f1=1),
                SourceOutcome(**base, source_id='fast_bad', ttft_ms=40, answer_f1=.5),
                SourceOutcome(**base, source_id='good', ttft_ms=65, answer_f1=.99)]
        result = measured_quality_cost_oracle(rows, chosen_source_id='fast_bad', max_answer_f1_drop=.02)
        self.assertEqual(result['oracle_source_id'], 'good')
        self.assertFalse(result['chosen_quality_passed'])
        self.assertIsNone(result['normalized_ttft_regret'])
        with self.assertRaises(ValueError):
            SourceOutcome(**base, source_id='s', ttft_ms=1, answer_f1=1, measurement_kind='predicted')

    def test_causal_coverage_builds_separate_capacity_pools(self):
        class Backend:
            capabilities = {}
            def __init__(self): self.resets = []
            def reset(self, *, capacity, global_byte_budget):
                self.capacity, self.global_byte_budget = capacity, global_byte_budget
                self.pool = {}; self.resets.append(capacity)
            def execute(self, request, dispatch, *, arrival_ns):
                ids = list(self.pool)
                event = dict(request_id=request['request_id'], execution_kind='online',
                            request_epoch=request['request_epoch'], capacity=self.capacity,
                            global_byte_budget=self.global_byte_budget,
                            visible_variant_creation_epochs=dict(self.pool), compatible_variant_ids=ids,
                            selected_variant_ids=ids, committed_variant_ids=ids,
                            quality_passed=True, matched_dense_ttft_ms=100, actual_ttft_ms=70)
                return dict(online_row(request, dispatch, arrival_ns), coverage_event=event)
            def finalize_request(self, request, outcome):
                key = request['request_id']; self.pool[key] = request['request_epoch']
                if len(self.pool) > self.capacity: del self.pool[next(iter(self.pool))]
        requests = [dict(request_id=f'q{i}', request_epoch=i) for i in range(1, 4)]
        backend = Backend()
        result = run_causal_capacity_traces(backend, requests, {}, capacities=(1, 2), global_byte_budget=1000)
        self.assertEqual(result['curves'][0]['commit_coverage'], 2/3)
        self.assertEqual(backend.resets, [1, 2])
        self.assertEqual(result['traces'][2][0]['visible_variant_creation_epochs'], {})

    def test_real_arrival_harness_retains_failures(self):
        class Backend:
            capabilities = {'max_integrated_concurrency': 2}
            def execute(self, request, dispatch, *, arrival_ns):
                if request['request_id'] == 'bad': raise RuntimeError('failed')
                return online_row(request, dispatch, arrival_ns)
            def finalize_request(self, request, outcome): pass
        result = run_serving_trace(Backend(), [dict(request_id='ok', arrival_offset_ms=0), dict(request_id='bad', arrival_offset_ms=0)],
                                  {}, concurrency=2, slo_ttft_ms=100)
        self.assertEqual(result['failed'], 1)
        self.assertEqual(len(result['rows']), 2)
        self.assertEqual(result['completed'], 1)
        self.assertTrue(result['integrated_concurrent_requests'])

    def test_paired_gate1_executes_both_arms_on_identical_pool(self):
        class Backend:
            capabilities = {'quiescent_snapshot_restore': True}
            def __init__(self): self.state = {'lru': 1}; self.calls = []
            def snapshot(self): return copy.deepcopy(self.state)
            def restore(self, snapshot): self.state = copy.deepcopy(snapshot)
            def execute(self, request, dispatch, *, arrival_ns):
                row = online_row(request, dispatch, arrival_ns)
                row['initial_pool_snapshot_sha256'] = digest_json(self.snapshot())
                self.calls.append((dispatch['gate1_mode'], self.snapshot()))
                self.state['lru'] += 1
                return row
        backend = Backend()
        result = run_gate1_pairs(backend, [{'request_id': 'q'}], {'repair': .15})
        self.assertEqual({mode for mode, _ in backend.calls}, {'explicit_barrier', 'fused_advisory'})
        self.assertEqual([state for _, state in backend.calls], [{'lru': 1}, {'lru': 1}])
        self.assertEqual(backend.snapshot(), {'lru': 1})
        self.assertEqual(len(result), 1)

    def test_copy_interference_cannot_claim_request_concurrency(self):
        backend = SimpleNamespace(capabilities={'max_integrated_concurrency': 1})
        with self.assertRaisesRegex(RuntimeError, 'true integrated'):
            run_serving_trace(backend, [{'request_id': 'q', 'arrival_offset_ms': 0}], {}, concurrency=2, slo_ttft_ms=100)

    def test_final_commit_preserves_full_inventory_and_rejects_only_costly_segment(self):
        from probekv.v8_schema10_live import LiveSelectionBridge
        from probekv.v8_schema6_contracts import PlannerSnapshot
        from probekv.v8_schema6_planner import JointTimelineEstimate
        from probekv.v8_schema7_planner import FinalCommitPlanner
        contexts = []
        class Estimator:
            def estimate(self, context):
                contexts.append(context)
                # Complete joint future, not one TTFT per Segment. c1 adds
                # interference; leaving it dense makes c0 reuse admissible.
                cost = 60 if set(context.reuse_segment_ids) == {'c0'} else 95
                return JointTimelineEstimate(cost, {'request_critical_path': cost})
        session = ProductionSelectionSession('q', ['c0', 'c1', 'c2'], make_selector(), SelectionCostLedger(100, SelectionCostPolicy()))
        for segment in ('c0', 'c1'):
            session.step(segment, completed_depth=1, counts=CandidateCounts(1,1,1,1,1),
                         candidates=[ResidualCandidate(segment, .1, 1, 0)],
                         gate1_plan_by_source={segment: plan(segment, 1)})
        for depth in (1,2):
            session.step('c2', completed_depth=depth, counts=CandidateCounts(0,0,0,0,0), candidates=[], gate1_plan_by_source={})
        snapshot = PlannerSnapshot(1,1,'scheduler',1,'a'*64)
        bridge = LiveSelectionBridge(session, gate1_provider=None, final_planner=FinalCommitPlanner(Estimator()),
                                     snapshot_provider=lambda: snapshot, hbm_manager=None,
                                     metadata_order_by_segment={}, batch_time_predictor=None,
                                     arrival_ns=1_000_000_000)
        with patch('probekv.v8_schema10_live.time.perf_counter_ns', return_value=1_015_000_000):
            result = bridge.final_commit(boundary_by_segment={'c0':2,'c1':2}, union_mask_digest='m')
        self.assertEqual(result.accepted_ready_segment_ids, ('c0',))
        self.assertEqual(result.rejected_ready_segment_ids, ('c1',))
        self.assertEqual(result.untouched_segment_ids, ('c2',))
        self.assertEqual(result.request_total_ms, 75)
        self.assertTrue(all('c2' in c.dense_fallback_segment_ids for c in contexts))
        self.assertEqual(session.decisions['c1'].selected_source_variant_id, 'c1')
        with self.assertRaisesRegex(RuntimeError, 'irreversible'):
            bridge.final_commit(boundary_by_segment={'c0':2}, union_mask_digest='m')

    def test_final_commit_stale_snapshot_and_nonfinite_cost_are_rejected(self):
        from probekv.v8_schema6_contracts import PlannerSnapshot
        from probekv.v8_schema6_planner import JointTimelineEstimate
        from probekv.v8_schema7_planner import FinalCommitPlanner
        snapshot = PlannerSnapshot(1,1,'scheduler',1,'a'*64)
        args = dict(inventory_segment_ids=('c',), eligible_ready_segment_ids=('c',), committed_segment_ids=(),
                    actual_boundary_by_segment={'c':2}, actual_sunk_ms=0, dense_reference_total_ms=100,
                    snapshot=snapshot, current_snapshot=snapshot, union_mask_digest='m')
        planner = FinalCommitPlanner(SimpleNamespace(estimate=lambda _: JointTimelineEstimate(10, {'future':10})))
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            planner.plan_ready_subset(**dict(args, current_snapshot=PlannerSnapshot(1,1,'scheduler',2,'a'*64)))
        with self.assertRaises(ValueError):
            planner.plan_ready_subset(**dict(args, actual_sunk_ms=float('nan')))

    def test_replay_cannot_change_budget_policy(self):
        session = ProductionSelectionSession('q', ['c'], make_selector(), SelectionCostLedger(100, SelectionCostPolicy()))
        session.step('c', completed_depth=1, counts=CandidateCounts(1,1,1,1,1), candidates=[ResidualCandidate('s', .1, 1, 0)],
                     gate1_plan_by_source={'s':plan('s',1)})
        replay = ProductionSelectionSession('q', ['c'], make_selector(), SelectionCostLedger(100, SelectionCostPolicy('legacy_fixed_fraction')))
        with self.assertRaisesRegex(ValueError, 'cost policy'):
            replay_selection_events(replay, session.events)

    def test_full_source_failure_cannot_masquerade_as_dense_oracle(self):
        def measure(source, layer, ratio):
            return SourceOutcome('q', None, layer, ratio, 5, 1, True, 'request_ttft', 'a'*64)
        with self.assertRaisesRegex(ValueError, 'another Source'):
            run_source_oracle('q', ['s'], first_reuse_layer=2, repair_ratio=.15,
                              chosen_source_id=None, max_answer_f1_drop=.02, measure=measure)

    def test_live_bridge_reads_current_engine_not_dense_fixture_labels(self):
        import torch
        from probekv.v8_schema10_live import LiveSelectionBridge
        from probekv.v8_schema6_hbm import UnifiedHBMReservationManager
        current = torch.ones(16, 2, 4, dtype=torch.bfloat16)
        states = tuple(tuple(current * (1 if v == 15 else 2) for _ in range(3)) for v in range(16))
        class Engine:
            current_layer = 0
            active_positions = tuple(range(16))
            def __init__(self): self.session = self
            def advance_to_layer(self, d): self.current_layer = d
            def observe_pre_rope_k(self, d): return current
        fixture = SimpleNamespace(segment_positions=(tuple(range(16)),), selection_variants=(states,),
                                  canonical_variants=(tuple(None for _ in range(16)),),
                                  canonical_variant_digests=(tuple(str(i).zfill(64) for i in range(16)),),
                                  exact_prefix_tokens=0)
        session = ProductionSelectionSession('q', ['c0'], make_selector(), SelectionCostLedger(100, SelectionCostPolicy()))
        hbm = UnifiedHBMReservationManager(allocator_capacity_bytes=1024*1024, safety_bytes=0)
        bridge = LiveSelectionBridge(session=session, hbm_manager=hbm,
            metadata_order_by_segment={'c0': tuple(range(16))},
            gate1_provider=lambda seg, source, d: plan(source, d),
            batch_time_predictor=lambda d, k, n: 12,
            final_planner=None, snapshot_provider=lambda: None)
        selected = bridge.select(SimpleNamespace(torch=torch), Engine(), fixture)
        self.assertEqual(selected, {0: 15})
        self.assertEqual(session.events[0]['counts']['compared_k'], 16)
        self.assertEqual(hbm.active_reserved_bytes, 0)


if __name__ == '__main__': unittest.main()
