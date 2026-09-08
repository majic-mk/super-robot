"""Read-only layer controls for Prefix arithmetic; never qualification success.

Projection hooks use private RMSNorm inputs and original positions. These
expensive observations are diagnostic costs, not online TTFT measurements.
"""
import time

from .model_adapters import PinnedCacheBlendResumableAdapter
from .v8_schema10_native_preflight import isolated_native_preflight
from .v8_schema10_storage import tensor_digest


def tensor_difference(reference, observed):
    import torch
    if reference.shape != observed.shape or not torch.isfinite(reference).all() or not torch.isfinite(observed).all():
        raise ValueError("layer control has incompatible/nonfinite tensors")
    a, b = reference.float(), observed.float()
    return {"relative_l2": float((a - b).norm() / a.norm().clamp_min(1e-12)),
            "max_absolute_error": float((a - b).abs().max()),
            "equal": torch.equal(reference, observed),
            "reference_digest": tensor_digest((reference,)),
            "observed_digest": tensor_digest((observed,))}


def run_prefix_layer_controls(adapter, *, request, warm_request):
    import torch
    bridge = PinnedCacheBlendResumableAdapter(adapter.inner, adapter.spec)
    captures = {}
    stage_captures = {}
    controls = {}
    with isolated_native_preflight(adapter):
        for mode in ("dense", "native_prefix", "resumable_prefix"):
            adapter.reset()
            if mode != "dense":
                warm = {**warm_request, "capture_original_full_prefill": True, "max_new_tokens": 1}
                with adapter.open_request(warm, arrival_ns=time.perf_counter_ns()) as context:
                    context.finish(lambda: None)
            rows, handles, stages = {}, [], {}
            def hook(depth):
                def capture(module, args):
                    positions, hidden = args[:2]
                    if positions.numel() <= 1:
                        return
                    if depth in rows:
                        raise RuntimeError("layer control captured prefill twice")
                    key, value = bridge.observe_pre_rope_kv(completed_depth=depth,
                        hidden_states=hidden, residual=args[4], active_positions=tuple(positions.tolist()))
                    rows[depth] = (positions.detach().cpu(), key.detach().cpu(), value.detach().cpu())
                return capture
            try:
                for depth, layer in enumerate(adapter.inner.layers):
                    handles.append(layer.register_forward_pre_hook(hook(depth)))
                    if depth < 10:
                        for name, module in (
                            ("attention", layer.self_attn.attn),
                            ("output_projection", layer.self_attn.o_proj),
                            ("gate_up_projection", layer.mlp.gate_up_proj),
                            ("activation", layer.mlp.act_fn),
                            ("down_projection", layer.mlp.down_proj),
                            ("block", layer),
                        ):
                            def stage_hook(module, args, output, depth=depth, name=name):
                                if depth not in rows or (depth, name) in stages:
                                    return
                                values = output if isinstance(output, tuple) else (output,)
                                stages[depth, name] = tuple(
                                    value.detach().cpu().clone() for value in values
                                    if torch.is_tensor(value))
                            handles.append(module.register_forward_hook(stage_hook))
                q = {**request, "max_new_tokens": 1}
                with adapter.open_request(q, arrival_ns=time.perf_counter_ns()) as context:
                    prefix = context.cached_prefix_tokens
                    if mode == "dense" and prefix or mode != "dense" and prefix < 128:
                        raise RuntimeError("layer control has incorrect Prefix ownership")
                    if mode != "dense":
                        shadow = context.native.prefix_shadow
                        controls[mode + "_shadow"] = [
                            {"layer": d + 1, "k": tensor_difference(captures['dense'][d][1][:prefix], pair[0]),
                             "v": tensor_difference(captures['dense'][d][2][:prefix], pair[1])}
                            for d, pair in enumerate(shadow)]
                    if mode == "resumable_prefix":
                        context.advance_to_depth(1)
                    result = context.finish(lambda: None)
                    if result['whole_request_origin'] == 'selective_reuse':
                        raise RuntimeError("Prefix numeric controls cannot use a Source")
            finally:
                torch.cuda.synchronize()
                for handle in handles:
                    handle.remove()
            if len(rows) != adapter.spec.num_layers:
                raise RuntimeError("layer controls omit Transformer blocks")
            captures[mode] = rows
            stage_captures[mode] = stages
        for mode in ('native_prefix', 'resumable_prefix'):
            controls[mode] = [{"layer": d + 1, "absolute_positions": row[0].tolist(),
                "k": tensor_difference(captures['dense'][d][1][row[0]], row[1]),
                "v": tensor_difference(captures['dense'][d][2][row[0]], row[2])}
                for d, row in captures[mode].items()]
        controls['resumable_vs_native'] = [{"layer": d + 1,
            "k": tensor_difference(captures['native_prefix'][d][1], row[1]),
            "v": tensor_difference(captures['native_prefix'][d][2], row[2])}
            for d, row in captures['resumable_prefix'].items()]
        stage_differences = {}
        for mode in ('native_prefix', 'resumable_prefix'):
            stage_differences[mode] = []
            for (depth, name), tensors in stage_captures[mode].items():
                positions = captures[mode][depth][0]
                ref = stage_captures['dense'][depth, name]
                stage_differences[mode].append({"layer": depth + 1, "stage": name,
                    "outputs": [tensor_difference(left[positions], right)
                                for left, right in zip(ref, tensors)]})
    return {"controls": controls, "origin": "real_cuda_execution", "fake_timing": False,
        "stage_controls": stage_differences,
        "diagnostic_only": True, "qualification_passed": False, "paper_evidence": False,
        "retained_cpu_capture_bytes": sum(t.numel()*t.element_size() for rows in captures.values()
                                          for triple in rows.values() for t in triple)}
