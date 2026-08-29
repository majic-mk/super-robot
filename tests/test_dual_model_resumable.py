import json
import tempfile
import unittest
from pathlib import Path

from probekv.model_adapters import (
    MISTRAL_SPEC,
    QWEN_SPEC,
    PinnedCacheBlendResumableAdapter,
    ResumableModelSpec,
    runtime_model_signature,
)
from probekv.resumable_prefill import (
    LayerAdvanceResult,
    ProbeKVResumablePrefillSession,
)
from probekv.runtime_source_audit import audit_runtime_sources
from probekv.server_storage import evaluate_storage_plan
from probekv.v6_a800_jobs import build_v6_a800_jobs
from probekv.v6_qualification_worker import dry_dispatch
from probekv.v6_qualification_worker import (
    QualificationJobResult,
    dispatch_qualification,
)


class CpuLayerAdapter:
    adapter_name = "cpu-reference"

    def __init__(self, total_layers=4):
        self.total_layers = total_layers
        self.reuse_transitions = []

    def begin_prefill(self, *, token_ids, **kwargs):
        return tuple(float(value) for value in token_ids), None

    def advance_layer(
        self,
        *,
        layer,
        hidden_states,
        active_positions,
        target_active_positions,
        working_kv,
        reuse_commit,
        **kwargs,
    ):
        if reuse_commit:
            self.reuse_transitions.append(layer)
        by_position = dict(zip(active_positions, hidden_states))
        hidden = tuple(by_position[position] + layer for position in target_active_positions)
        return LayerAdvanceResult(hidden, None, working_kv)

    def finish_prefill(self, *, hidden_states, **kwargs):
        return hidden_states


class ResumableSessionTests(unittest.TestCase):
    def session(self):
        return ProbeKVResumablePrefillSession(
            adapter=CpuLayerAdapter(),
            model_signature="signature",
            token_ids=(1, 2, 3, 4, 5, 6),
            absolute_positions=(1, 2, 3, 4, 5, 6),
            attention_metadata={},
            working_kv=[],
            exact_prefix_tokens=1,
        )

    def test_r1_resumable_equals_monolithic_cpu_reference(self):
        baseline = self.session()
        baseline.begin_prefill()
        expected = baseline.finish_prefill()
        observed = self.session()
        observed.begin_prefill()
        observed.register_source_handle("c", "s", object())
        observed.advance_to_layer(1)
        observed.commit_segment_reuse(
            segment_id="c", source_id="s", boundary=2,
            segment_positions=(1, 2, 3), repair_positions=(1, 2, 3),
        )
        self.assertEqual(observed.finish_prefill(), expected)
        self.assertTrue(all(
            row["active_before"] == row["active_after"]
            for row in observed.layer_audit
        ))
        self.assertEqual(observed.adapter.reuse_transitions, [2])

    def test_active_set_only_shrinks_and_two_segments_can_stagger(self):
        session = self.session()
        session.begin_prefill()
        session.register_source_handle("c1", "s1", object())
        session.register_source_handle("c2", "s2", object())
        session.commit_segment_reuse(
            segment_id="c1", source_id="s1", boundary=1,
            segment_positions=(1, 2), repair_positions=(2,),
        )
        session.advance_to_layer(1)
        session.commit_segment_reuse(
            segment_id="c2", source_id="s2", boundary=2,
            segment_positions=(3, 4), repair_positions=(4,),
        )
        session.finish_prefill()
        sizes = [len(row["active_after"]) for row in session.layer_audit]
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        self.assertEqual(session.active_positions, (2, 4, 5, 6))

    def test_reintroduced_token_and_unlocked_source_are_rejected(self):
        session = self.session()
        session.begin_prefill()
        with self.assertRaises(RuntimeError):
            session.commit_segment_reuse(
                segment_id="c", source_id="default", boundary=1,
                segment_positions=(1,), repair_positions=(),
            )

    def test_native_prefix_rows_cannot_be_recomputed(self):
        with self.assertRaises(ValueError):
            ProbeKVResumablePrefillSession(
                adapter=CpuLayerAdapter(),
                model_signature="signature",
                token_ids=(1, 2),
                absolute_positions=(0, 1),
                attention_metadata={},
                working_kv=[],
                exact_prefix_tokens=1,
            )


class DualModelContractTests(unittest.TestCase):
    def test_pre_rope_k_observation_cannot_mutate_live_prefill_state(self):
        import torch

        class MutatingNorm:
            def __call__(self, hidden, residual):
                hidden.add_(100)
                residual.add_(200)
                return hidden, residual

        class Projection:
            def __call__(self, normalized):
                return torch.cat((normalized, normalized, normalized), dim=-1)

        class Attention:
            head_dim = 2
            q_size = 2
            kv_size = 2
            qkv_proj = Projection()

        class Layer:
            input_layernorm = MutatingNorm()
            self_attn = Attention()

        class InnerModel:
            layers = [Layer()]

            def probekv_begin_prefill(self, *args, **kwargs):
                raise NotImplementedError

            def probekv_advance_prefill(self, *args, **kwargs):
                raise NotImplementedError

            def probekv_finish_prefill(self, *args, **kwargs):
                raise NotImplementedError

        spec = ResumableModelSpec(
            adapter_name="mutation-test",
            model_id="mutation-test",
            revision="r",
            architecture="TestModel",
            num_layers=1,
            num_attention_heads=1,
            num_kv_heads=1,
            rope_theta=1.0,
            rope_scaling=None,
            sliding_window=None,
            use_sliding_window=False,
            qkv_bias=False,
            checkpoints=(0,),
            max_context_tokens=8,
        )
        adapter = PinnedCacheBlendResumableAdapter(InnerModel(), spec)
        hidden = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 4.0]])
        hidden_before = hidden.clone()
        residual_before = residual.clone()

        observed = adapter.observe_pre_rope_k(
            completed_depth=0,
            hidden_states=hidden,
            residual=residual,
            active_positions=(0,),
        )

        self.assertTrue(torch.equal(hidden, hidden_before))
        self.assertTrue(torch.equal(residual, residual_before))
        self.assertEqual(tuple(observed.shape), (1, 1, 2))

    def test_frozen_adapter_geometry(self):
        self.assertEqual(MISTRAL_SPEC.checkpoints, (1, 2, 4, 6, 8))
        self.assertEqual(MISTRAL_SPEC.num_layers, 32)
        self.assertFalse(MISTRAL_SPEC.qkv_bias)
        self.assertEqual(QWEN_SPEC.checkpoints, (1, 2, 4, 5, 7))
        self.assertEqual(QWEN_SPEC.num_layers, 28)
        self.assertEqual(QWEN_SPEC.num_attention_heads, 28)
        self.assertEqual(QWEN_SPEC.num_kv_heads, 4)
        self.assertEqual(QWEN_SPEC.rope_theta, 1000000.0)
        self.assertFalse(QWEN_SPEC.use_sliding_window)
        self.assertTrue(QWEN_SPEC.qkv_bias)

    def test_qwen_adapter_rejects_yarn_or_sliding_window(self):
        base = {
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "rope_theta": 1000000.0,
            "rope_scaling": None,
            "use_sliding_window": False,
        }
        QWEN_SPEC.validate_config(base)
        with self.assertRaises(ValueError):
            QWEN_SPEC.validate_config({**base, "rope_scaling": {"type": "yarn"}})

    def test_model_signatures_isolate_namespaces(self):
        mistral = runtime_model_signature(
            MISTRAL_SPEC, tokenizer_hash="t", dtype="bfloat16",
            runtime_patch_sha="p")
        qwen = runtime_model_signature(
            QWEN_SPEC, tokenizer_hash="t", dtype="bfloat16",
            runtime_patch_sha="p")
        self.assertNotEqual(mistral, qwen)

    def test_unpatched_model_cannot_masquerade_as_runtime_adapter(self):
        with self.assertRaises(RuntimeError):
            PinnedCacheBlendResumableAdapter(object(), QWEN_SPEC)

    def test_both_qualification_matrices_have_140_jobs(self):
        raw = json.loads(Path("configs/v6_a800_microbench.json").read_text())
        for spec in (MISTRAL_SPEC, QWEN_SPEC):
            jobs = build_v6_a800_jobs(raw)
            self.assertEqual(len(jobs), 140, spec.adapter_name)
            self.assertEqual(dry_dispatch(jobs, spec.adapter_name)["planned"], 140)

    def test_static_runtime_source_audit_is_complete(self):
        result = audit_runtime_sources(Path(".").resolve())
        self.assertTrue(result["runtime_source_ready"], result["failures"])

    def test_fake_timing_and_failed_fidelity_cannot_qualify(self):
        raw = json.loads(Path("configs/v6_a800_microbench.json").read_text())
        jobs = build_v6_a800_jobs(raw)

        class Executor:
            concrete_engine_hook = True
            adapter_name = QWEN_SPEC.adapter_name

            @staticmethod
            def capabilities():
                return {
                    "async_multisource_loading": True,
                    "layer_resumable_prefill": True,
                    "layer_indexed_union_repair_masks": True,
                    "per_segment_staggered_boundaries": True,
                    "causal_commit_wait_execution": True,
                    "immediate_staggered_closed_loop_execution": True,
                    "policy_conditioned_probe_state": True,
                    "cuda_event_timing": True,
                }

            def __init__(self, **overrides):
                self.overrides = overrides

            def execute(self, job):
                values = {
                    "job_id": job.job_id,
                    "passed": True,
                    "cuda_event_timing": True,
                    "gpu_ms": 1.0,
                    "host_ms": 1.1,
                }
                values.update(self.overrides)
                return QualificationJobResult(**values)

        with self.assertRaisesRegex(RuntimeError, "host/fake timing"):
            dispatch_qualification(jobs, Executor(cuda_event_timing=False))
        with self.assertRaisesRegex(RuntimeError, "dense reference"):
            dispatch_qualification(
                jobs, Executor(r1_dense_token_ids_equal=False)
            )
        with self.assertRaisesRegex(RuntimeError, "relative-L2"):
            dispatch_qualification(
                jobs, Executor(teacher_forced_logit_relative_l2=2e-4)
            )


class StoragePlanTests(unittest.TestCase):
    def test_storage_decision_thresholds(self):
        base = {"largest_writable_free_gib": 60, "system_free_gib": 20}
        self.assertEqual(
            evaluate_storage_plan({**base, "combined_free_gib": 95})["storage_mode"],
            "dual_model_resident",
        )
        self.assertEqual(
            evaluate_storage_plan({**base, "combined_free_gib": 75})["storage_mode"],
            "sequential_mistral_then_qwen",
        )
        self.assertFalse(
            evaluate_storage_plan({**base, "combined_free_gib": 69})["storage_ready"]
        )


if __name__ == "__main__":
    unittest.main()
