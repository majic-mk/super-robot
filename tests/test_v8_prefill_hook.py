import unittest

from probekv.resumable_prefill import (
    LayerAdvanceResult,
    ProbeKVResumablePrefillSession,
)


class FakeCompletedDepthAdapter:
    adapter_name = "fake-v8"
    total_layers = 3

    def begin_prefill(self, **kwargs):
        return ("h0", "r0")

    def observe_pre_rope_k(self, **kwargs):
        return {
            "depth": kwargs["completed_depth"],
            "hidden": kwargs["hidden_states"],
        }

    def advance_layer(self, **kwargs):
        layer = kwargs["layer"]
        return LayerAdvanceResult(
            hidden_states="h%d" % layer,
            residual="r%d" % layer,
            working_kv=kwargs["working_kv"],
            union_mask_digest="mask-%d" % layer,
        )

    def finish_prefill(self, **kwargs):
        return kwargs["hidden_states"]


class V8CompletedDepthHookTests(unittest.TestCase):
    def test_d0_observes_layer1_input_and_d1_observes_layer2_input(self):
        session = ProbeKVResumablePrefillSession(
            adapter=FakeCompletedDepthAdapter(), model_signature="model",
            token_ids=(1, 2, 3), attention_metadata={}, working_kv={},
        )
        session.begin_prefill()
        self.assertEqual(session.observe_pre_rope_k(0), {"depth": 0, "hidden": "h0"})
        session.advance_to_layer(1)
        self.assertEqual(session.observe_pre_rope_k(1), {"depth": 1, "hidden": "h1"})
        observations = [row for row in session.layer_audit if row.get("event") == "selection_k_observation"]
        self.assertEqual(
            [(row["completed_depth"], row["k_observation_layer_1based"]) for row in observations],
            [(0, 1), (1, 2)],
        )

    def test_hook_rejects_future_or_completed_model_depth(self):
        session = ProbeKVResumablePrefillSession(
            adapter=FakeCompletedDepthAdapter(), model_signature="model",
            token_ids=(1, 2), attention_metadata={}, working_kv={},
        )
        session.begin_prefill()
        with self.assertRaises(ValueError):
            session.observe_pre_rope_k(1)
        session.advance_to_layer(3)
        with self.assertRaises(ValueError):
            session.observe_pre_rope_k(3)


if __name__ == "__main__":
    unittest.main()
