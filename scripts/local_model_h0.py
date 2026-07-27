"""Offline CPU H0 checks against a locally cached Hugging Face causal LM.

This verifies model-level invariants that do not require CacheBlend kernels:
independent context-conditioned sources, exact save/load, deterministic greedy
generation, and source read-only behavior. It is correctness evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import tempfile
import time
from pathlib import Path


def tensor_digest(tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    # Some older Windows research environments have a broken Linux-only
    # bitsandbytes package installed. Hide that optional package while
    # importing Transformers; this script does not request quantization.
    original_find_spec = importlib.util.find_spec

    def find_spec_without_bitsandbytes(name, *find_args, **find_kwargs):
        if name == "bitsandbytes" or name.startswith("bitsandbytes."):
            return None
        return original_find_spec(name, *find_args, **find_kwargs)

    importlib.util.find_spec = find_spec_without_bitsandbytes
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(max(1, args.threads))
        torch.manual_seed(20260726)
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
    finally:
        importlib.util.find_spec = original_find_spec
    model.eval()
    load_seconds = time.perf_counter() - started

    prefix_a = tokenizer.encode(
        "Historical context A: alpine climate records and snowfall. ",
        add_special_tokens=True,
    )
    prefix_b = tokenizer.encode(
        "Historical context B: tropical ocean currents and coral reefs. ",
        add_special_tokens=True,
    )
    # Equalize length so C uses the same absolute positions; observed KV
    # differences then come from preceding content rather than a RoPE offset.
    common_prefix_length = min(len(prefix_a), len(prefix_b))
    prefix_a = prefix_a[:common_prefix_length]
    prefix_b = prefix_b[:common_prefix_length]
    segment_c = tokenizer.encode(
        "The shared segment states that the instrument was calibrated twice.",
        add_special_tokens=False,
    )
    if not segment_c:
        raise RuntimeError("tokenized shared segment is empty")
    ids_a = torch.tensor([prefix_a + segment_c], dtype=torch.long)
    ids_b = torch.tensor([prefix_b + segment_c], dtype=torch.long)

    inference_started = time.perf_counter()
    with torch.inference_mode():
        output_a = model(ids_a, use_cache=True, return_dict=True)
        output_b = model(ids_b, use_cache=True, return_dict=True)
    inference_seconds = time.perf_counter() - inference_started
    cache_a = legacy_cache(output_a.past_key_values)
    cache_b = legacy_cache(output_b.past_key_values)
    layer_count = len(cache_a)
    c_length = len(segment_c)

    source_differences = []
    source_tensors = []
    raw_drift_by_source = {}
    for layer_index in {0, layer_count - 1}:
        key_a, value_a = cache_a[layer_index][0], cache_a[layer_index][1]
        key_b, value_b = cache_b[layer_index][0], cache_b[layer_index][1]
        c_key_a = key_a[:, :, -c_length:, :].clone()
        c_value_a = value_a[:, :, -c_length:, :].clone()
        c_key_b = key_b[:, :, -c_length:, :].clone()
        c_value_b = value_b[:, :, -c_length:, :].clone()
        source_tensors.extend([c_key_a, c_value_a])
        source_differences.append(
            {
                "layer": layer_index,
                "key_mean_abs_difference": float(
                    (c_key_a - c_key_b).abs().mean().item()
                ),
                "value_mean_abs_difference": float(
                    (c_value_a - c_value_b).abs().mean().item()
                ),
            }
        )
        if layer_index == layer_count - 1:
            # Treat the fresh A|C pass as the current request. The matching
            # canonical source must have lower raw K/V drift than B|C.
            raw_drift_by_source = {
                "source_A": 0.0,
                "source_B": float(
                    (c_key_a - c_key_b).abs().mean().item()
                    + (c_value_a - c_value_b).abs().mean().item()
                ),
            }

    before_hashes = [tensor_digest(tensor) for tensor in source_tensors]
    full_cache_a = [
        (layer[0].clone(), layer[1].clone()) for layer in cache_a
    ]
    with tempfile.TemporaryDirectory() as temporary:
        cache_path = Path(temporary) / "canonical_source.pt"
        torch.save(
            {"segment_tensors": source_tensors, "full_cache": full_cache_a},
            str(cache_path),
        )
        try:
            reloaded_payload = torch.load(
                str(cache_path), map_location="cpu", weights_only=True
            )
        except TypeError:
            reloaded_payload = torch.load(str(cache_path), map_location="cpu")
    reloaded = reloaded_payload["segment_tensors"]
    reloaded_full_cache = tuple(reloaded_payload["full_cache"])
    exact_save_load = all(
        torch.equal(original, restored)
        for original, restored in zip(source_tensors, reloaded)
    )

    # Read-only comparison: summaries are computed from detached tensors.
    _summaries = [float(tensor.float().mean().item()) for tensor in source_tensors]
    after_hashes = [tensor_digest(tensor) for tensor in source_tensors]
    source_read_only = before_hashes == after_hashes

    next_token = output_a.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    with torch.inference_mode():
        original_cache_next = model(
            next_token,
            past_key_values=tuple(full_cache_a),
            use_cache=True,
            return_dict=True,
        ).logits
        loaded_cache_next = model(
            next_token,
            past_key_values=reloaded_full_cache,
            use_cache=True,
            return_dict=True,
        ).logits
    loaded_cache_logits_exact = torch.equal(
        original_cache_next, loaded_cache_next
    )

    with torch.inference_mode():
        first_generation = model.generate(
            ids_a, max_new_tokens=4, do_sample=False, use_cache=True
        )
        second_generation = model.generate(
            ids_a, max_new_tokens=4, do_sample=False, use_cache=True
        )
    greedy_exact = torch.equal(first_generation, second_generation)
    contexts_condition_source = any(
        row["key_mean_abs_difference"] > 0
        or row["value_mean_abs_difference"] > 0
        for row in source_differences
    )
    raw_drift_selects_matching = (
        min(raw_drift_by_source, key=raw_drift_by_source.get) == "source_A"
        and raw_drift_by_source["source_B"] > 0.0
    )

    result = {
        "check": "local_hf_h0",
        "evidence_class": "local_correctness",
        "paper_performance_evidence": False,
        "model_path": str(Path(args.model).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": "cpu",
        "load_seconds": load_seconds,
        "two_prefills_seconds": inference_seconds,
        "layers": layer_count,
        "shared_segment_tokens": c_length,
        "equal_prefix_tokens": common_prefix_length,
        "independent_contexts_change_segment_kv": contexts_condition_source,
        "source_differences": source_differences,
        "raw_drift_by_source": raw_drift_by_source,
        "current_state_raw_drift_selects_matching_source": raw_drift_selects_matching,
        "canonical_save_load_exact": exact_save_load,
        "loaded_cache_next_logits_exact": loaded_cache_logits_exact,
        "source_read_only": source_read_only,
        "greedy_generation_exact": greedy_exact,
        "passed": all(
            [
                contexts_condition_source,
                raw_drift_selects_matching,
                exact_save_load,
                loaded_cache_logits_exact,
                source_read_only,
                greedy_exact,
            ]
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
