import unittest

from probekv.resumable_prefill import LayerAdvanceResult, ProbeKVResumablePrefillSession
from probekv.v8_schema6_contracts import PlannerSnapshot
from probekv.v8_schema6_planner import DeterministicJointTimelineEstimator
from probekv.v8_schema7_planner import FinalCommitPlanner
from probekv.v8_schema7_planner import PreparationAdmissionPlanner
from probekv.v8_schema6_planner import FrozenSegmentCandidate


class CpuAdapter:
    adapter_name = "schema7-cpu"
    total_layers = 4

    def begin_prefill(self, *, token_ids, **kwargs):
        return tuple(float(value) for value in token_ids), None

    def advance_layer(
        self, *, layer, hidden_states, active_positions,
        target_active_positions, working_kv, **kwargs
    ):
        values = dict(zip(active_positions, hidden_states))
        return LayerAdvanceResult(
            tuple(values[position] + layer for position in target_active_positions),
            None, working_kv,
        )

    def finish_prefill(self, *, hidden_states, **kwargs):
        return hidden_states

    def observe_pre_rope_kv(self, *, hidden_states, **kwargs):
        rows = tuple((value,) for value in hidden_states)
        return rows, rows


class Schema7RuntimeTests(unittest.TestCase):
    def test_dense_repair_check_observes_kv_for_next_layer(self):
        session = ProbeKVResumablePrefillSession(
            adapter=CpuAdapter(), model_signature="m", token_ids=(1, 2),
            absolute_positions=(1, 2), attention_metadata={}, working_kv=[],
            exact_prefix_tokens=1,
        )
        session.begin_prefill()
        with self.assertRaises(ValueError):
            session.observe_repair_check_pre_rope_kv(0)
        session.advance_to_layer(1)
        key, value = session.observe_repair_check_pre_rope_kv(1)
        self.assertEqual(key, value)
        self.assertEqual(session.layer_audit[-1]["first_selective_reuse_layer"], 2)

    def test_preparation_admission_prices_unresolved_as_joint_dense_fallback(self):
        estimator = DeterministicJointTimelineEstimator(
            base_future_ms=0,
            dense_cost_ms_by_segment={"c1": 40, "c2": 40},
            reuse_cost_ms_by_segment={"c1": 10},
        )
        snapshot = PlannerSnapshot(1, 1, "scheduler", 1, "profile")
        decision = PreparationAdmissionPlanner(estimator).plan_incremental(
            inventory_segment_ids=("c1", "c2"),
            frozen_candidates=(FrozenSegmentCandidate("c1", "s1", 2, 1),),
            existing_provisional_segment_ids=(), existing_deferred_segment_ids=(),
            predicted_dense_segment_ids=(), committed_segment_ids=(),
            actual_sunk_ms=1, dense_reference_total_ms=100, selection_closed=False,
            snapshot=snapshot, current_snapshot=snapshot, union_mask_digest="mask",
        )
        self.assertEqual(decision.predicted_request_total_ms, 51)
        self.assertEqual(decision.disposition_by_segment["c1"], "provisional_reuse")

    def test_gradual_support_can_only_shrink_after_commit(self):
        session = ProbeKVResumablePrefillSession(
            adapter=CpuAdapter(), model_signature="m", token_ids=(1, 2, 3, 4),
            absolute_positions=(1, 2, 3, 4), attention_metadata={}, working_kv=[],
            exact_prefix_tokens=1,
        )
        session.begin_prefill()
        session.register_source_handle("c", "s", object())
        session.commit_segment_reuse(
            segment_id="c", source_id="s", boundary=1,
            segment_positions=(1, 2, 3), repair_positions=(1, 2),
        )
        session.advance_to_layer(1)
        session.shrink_segment_repair_support(
            segment_id="c", consumer_layer=2, repair_positions=(2,),
        )
        session.advance_to_layer(2)
        self.assertEqual(session.active_positions, (2, 4))
        with self.assertRaises(RuntimeError):
            session.shrink_segment_repair_support(
                segment_id="c", consumer_layer=3, repair_positions=(1, 2),
            )

    def test_final_commit_returns_partial_subset_without_gate3_api(self):
        estimator = DeterministicJointTimelineEstimator(
            base_future_ms=10,
            dense_cost_ms_by_segment={"c1": 50, "c2": 50},
            reuse_cost_ms_by_segment={"c1": 10, "c2": 70},
        )
        snapshot = PlannerSnapshot(1, 1, "scheduler", 1, "runtime-profile")
        decision = FinalCommitPlanner(estimator).plan_ready_subset(
            inventory_segment_ids=("c1", "c2"),
            eligible_ready_segment_ids=("c1", "c2"),
            committed_segment_ids=(),
            actual_boundary_by_segment={"c1": 2, "c2": 4},
            actual_sunk_ms=0,
            dense_reference_total_ms=100,
            snapshot=snapshot,
            current_snapshot=snapshot,
            union_mask_digest="mask",
        )
        self.assertEqual(decision.accepted_ready_segment_ids, ("c1",))
        self.assertEqual(decision.rejected_ready_segment_ids, ("c2",))
        self.assertEqual(decision.planner_snapshot, snapshot)

    def test_stale_final_commit_snapshot_fails(self):
        planner = FinalCommitPlanner(
            DeterministicJointTimelineEstimator(
                base_future_ms=0,
                dense_cost_ms_by_segment={"c": 10},
                reuse_cost_ms_by_segment={"c": 1},
            )
        )
        with self.assertRaises(RuntimeError):
            planner.plan_ready_subset(
                inventory_segment_ids=("c",), eligible_ready_segment_ids=("c",),
                committed_segment_ids=(), actual_boundary_by_segment={"c": 2},
                actual_sunk_ms=0, dense_reference_total_ms=100,
                snapshot=PlannerSnapshot(1, 1, "a", 1, "p"),
                current_snapshot=PlannerSnapshot(1, 1, "b", 1, "p"),
                union_mask_digest="mask",
            )


if __name__ == "__main__":
    unittest.main()
