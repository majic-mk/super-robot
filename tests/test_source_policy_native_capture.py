from contextlib import contextmanager
import copy
import time
import unittest
from unittest.mock import patch

import test_schema10_online_integration as fixtures
from test_v8_schema10_execution_integration import make_selector, plan
from probekv.source_policy_native_capture import capture_native_source_observation
from probekv.source_policy_replay import replay_observation
from probekv.v8_schema10_contracts import CostUnsupportedSourceObservation
from probekv.v8_contracts import CandidateCounts, ResidualCandidate
from probekv.v8_schema10_execution import (
    digest_json, ProductionSelectionSession, SelectionCostLedger,
    SelectionCostPolicy, replay_selection_events,
)


class NativeCaptureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnlineIntegration()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.b = self.fixture.backend
        self.b.provenance["code_commit"] = "a" * 40
        self.fixture.execute(1)
        self.sid = next(iter(self.b.store.objects))
        self.q = fixtures.request(2)
        self.q.update(partition_role="development", locked_test_accessed=False,
                      development_partition_digest=digest_json("development"))
        self.binding = {k: self.b.provenance[k] for k in ("code_commit", "model_signature", "tokenizer_hash")}
        self.binding.update(patch_sha256="b" * 64, config_sha256="c" * 64)

    def run_capture(self, **overrides):
        args = dict(request=self.q, selection_path="d1_d2_rescue", source_ids=[self.sid],
                    binding=self.binding, host_budget_bytes=2**20, allow_cpu_test=True)
        args.update(overrides)
        return capture_native_source_observation(self.b, **args)

    def test_real_context_reads_selection_only_and_preserves_pool(self):
        before = self.b.store.snapshot_descriptor()
        original = self.b.store._read
        def read(obj, *, selection=False, verify_full=False):
            self.assertTrue(selection, "capture must never read full KV")
            return original(obj, selection=selection, verify_full=verify_full)
        with patch.object(self.b.store, "_read", side_effect=read), \
             patch.object(self.b.store, "leased_winner", side_effect=AssertionError("no winner")), \
             patch.object(self.b.store, "publish_exact_dense", side_effect=AssertionError("no materialization")):
            captured = self.run_capture()
        self.assertEqual(before, self.b.store.snapshot_descriptor())
        self.assertTrue(captured["pool_unchanged"])
        self.assertEqual(captured["comparison_execution_device"], "cpu")
        self.assertEqual(len(replay_observation(captured["observation"])["cells"]), 4)
        self.assertGreaterEqual(captured["diagnostic_total_including_restore_wall_ms"], captured["diagnostic_first_token_wall_ms"])
        self.assertFalse(captured["gpu_runtime_qualified"])
        self.assertFalse(self.b.hbm.active_reserved_bytes)

    def test_no_cpu_context_can_claim_native_evidence(self):
        with self.assertRaisesRegex(ValueError, "native factory"):
            self.run_capture(allow_cpu_test=False)

    def test_host_budget_and_missing_selection_fail_without_pool_mutation(self):
        before = self.b.store.snapshot_descriptor()
        with self.assertRaises(MemoryError):
            self.run_capture(host_budget_bytes=1)
        self.assertEqual(before, self.b.store.snapshot_descriptor())
        with patch.object(self.b.store, "read_selection", side_effect=KeyError("missing")):
            with self.assertRaises(KeyError):
                self.run_capture()
        self.assertFalse(self.b.poisoned_error)
        self.assertFalse(self.b.hbm.active_reserved_bytes)

    def test_future_source_and_locked_partition_rejected(self):
        q = dict(self.q, request_epoch=1)
        with self.assertRaisesRegex(ValueError, "future"):
            self.run_capture(request=q)
        with self.assertRaises(ValueError):
            self.run_capture(request=dict(self.q, partition_role="locked"))

    def test_prefix_covered_and_partial_tail_cannot_enter_comparison(self):
        adapter = self.b.adapters["d1_d2_rescue"]
        for prefix in (1, 4):
            @contextmanager
            def opened(request, *, arrival_ns):
                ctx = fixtures.TinyLiveContext(request)
                ctx.cached_prefix_tokens = prefix
                yield ctx
            with patch.object(adapter, "open_request", opened):
                with self.assertRaisesRegex(RuntimeError, "Prefix-covered"):
                    self.run_capture()

    def test_active_reservation_is_not_freed_by_diagnostic(self):
        from probekv.v8_schema6_hbm import HBMReservationKind
        reservation = self.b.hbm.reserve_batch(owner_request_id="other",
            rows=(("s", 1024, HBMReservationKind.SELECTION_WORKSPACE),))[0]
        with self.assertRaisesRegex(RuntimeError, "quiescent"):
            self.run_capture()
        self.assertFalse(reservation.released)
        self.b.hbm.release(reservation.reservation_id)

    def test_one_unpriced_source_does_not_remove_other_real_comparisons(self):
        ctx = fixtures.TinyLiveContext(fixtures.request(2))
        capture = ctx.export_exact_dense()["s0"]
        metadata = ctx.materialization_metadata()["s0"]
        self.b.store.publish_exact_dense(metadata["identity"], layers=capture["layers"],
            selection_states=capture["selection_states"], metadata=metadata["source_metadata"],
            request_epoch=2, whole_request_origin="exact_dense_full_prefill", materialization_reason="content_miss")
        ids = list(self.b.store.objects)
        self.assertEqual(len(ids), 2)
        original = self.fixture.costs.candidate_future_ms
        with patch.object(self.fixture.costs, "candidate_future_ms",
                          side_effect=lambda ctx, seg, source, depth: None if source == ids[0] else original(ctx, seg, source, depth)):
            row = self.fixture.execute(3)
        self.assertEqual(row["selected_source_variant_ids"], [ids[1]])
        self.assertEqual(row["selection_events"][-1]["counts"]["compared_k"], 2)
        self.assertEqual(len(row["coverage_event"]["compatible_variant_ids"]), 2)
        self.assertIsNone(next(c for c in row["selection_events"][-1]["candidates"] if c["source_variant_id"] == ids[0])["predicted_future_upper_ms"])

    def test_live_cascade_compares_only_retained_sources_without_inventing_oracle(self):
        self.b.reset(capacity=16,global_byte_budget=2000000)
        ids = []
        for i in range(4):
            ctx = fixtures.TinyLiveContext(fixtures.request(i+1))
            capture = ctx.export_exact_dense()["s0"]
            metadata = ctx.materialization_metadata()["s0"]
            states = {1:capture["selection_states"][1]+.25*(i+1),
                      2:capture["selection_states"][2]+.25*(4-i)}
            layers = tuple((states.get(layer,k),v) for layer,(k,v) in enumerate(capture["layers"]))
            row = self.b.store.publish_exact_dense(metadata["identity"],layers=layers,
                selection_states=states,metadata=metadata["source_metadata"],request_epoch=i+1,
                whole_request_origin="exact_dense_full_prefill",materialization_reason="content_miss")
            ids.append(row.source_variant_id)
        original = self.b.selector_factory
        def selector(dispatch, capacity):
            value = original(dispatch,capacity)
            value.strong_margin = value.stable_margin = 1.
            value.depth2_keep_fraction = .5
            return value
        with patch.object(self.b,"selector_factory",side_effect=selector), \
             patch.object(self.b.store,"read_selection",wraps=self.b.store.read_selection) as reads:
            row = self.fixture.execute(5)
        self.assertEqual(len([c for c in reads.call_args_list if c.args[1] == 1]),4)
        self.assertEqual(len([c for c in reads.call_args_list if c.args[1] == 2]),2)
        last = row["selection_events"][-1]
        self.assertEqual(last["counts"]["correctness_eligible_k"],4)
        self.assertEqual(last["counts"]["compared_k"],2)
        self.assertFalse(last["decision"]["selection_scope_complete"])
        self.assertEqual(row["selected_source_variant_ids"],[ids[1]])
        event = next(e for e in row["runtime_events"] if e["kind"] == "depth2_shortlist")
        self.assertFalse(event["full_depth2_oracle_observed"])


class QualifiedCostSelectionTests(unittest.TestCase):
    def test_future_cost_not_gate1_lower_bound_and_outside_legacy_band(self):
        candidates = [ResidualCandidate("a", .05, 80, 0), ResidualCandidate("b", .2, 10, 1)]
        plans = {"a": plan("a", 2, cost=.1), "b": plan("b", 2, cost=1)}
        selector = make_selector()
        old = selector.decide(completed_depth=2, counts=CandidateCounts(2,2,2,2,2), candidates=candidates, gate1_plan_by_source=plans)
        self.assertEqual(old.selected_source_variant_id, "a")
        selector.source_cost_selection_policy = "absolute_qualified_future_cost"
        new = selector.decide(completed_depth=2, counts=CandidateCounts(2,2,2,2,2), candidates=candidates, gate1_plan_by_source=plans)
        self.assertEqual(new.selected_source_variant_id, "b")

    def test_early_residual_winner_cost_disagreement_continues(self):
        selector = make_selector()
        selector.source_cost_selection_policy = "absolute_qualified_future_cost"
        candidates = [ResidualCandidate("a", .01, 80, 0), ResidualCandidate("b", .2, 10, 1)]
        result = selector.decide(completed_depth=1, counts=CandidateCounts(2,2,2,2,2), candidates=candidates,
                                 gate1_plan_by_source={s: plan(s, 1) for s in ("a","b")})
        self.assertEqual(result.reason, "early_residual_and_cost_winners_disagree")
        self.assertEqual(result.state, "continue_probe")

    def test_unknown_cost_event_replays_and_policy_cannot_change(self):
        selector = make_selector((2,))
        selector.source_cost_selection_policy = "absolute_qualified_future_cost"
        def session():
            return ProductionSelectionSession("q", ("s",), selector, SelectionCostLedger(100, SelectionCostPolicy()))
        original = session()
        original.step("s", completed_depth=2, counts=CandidateCounts(2,2,2,2,2),
            candidates=[CostUnsupportedSourceObservation("a", .01, None, 0, "missing"), ResidualCandidate("b", .1, 4, 1)],
            gate1_plan_by_source={"b":plan("b", 2)})
        self.assertEqual(original.events[-1]["decision"]["selected_source_variant_id"], "b")
        self.assertEqual(replay_selection_events(session(), original.events), tuple(original.events))
        selector.source_cost_selection_policy = "legacy_residual_band"
        with self.assertRaisesRegex(ValueError, "Source cost policy"):
            replay_selection_events(session(), original.events)


if __name__ == "__main__":
    unittest.main()
