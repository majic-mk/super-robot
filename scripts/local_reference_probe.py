"""Real-model CPU smoke test for per-layer reference-state extraction."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from probekv.reference_hf import HuggingFaceReferenceStateBackend, _model_layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(20260726)
    started = time.perf_counter()
    backend = HuggingFaceReferenceStateBackend.from_pretrained(
        args.model, device="cpu", dtype="float32", local_files_only=True
    )
    load_seconds = time.perf_counter() - started
    tokenizer = backend.tokenizer
    prefix_a = tokenizer.encode(
        "Historical context A: alpine climate records and snowfall. ",
        add_special_tokens=True,
    )
    prefix_b = tokenizer.encode(
        "Historical context B: tropical ocean currents and coral reefs. ",
        add_special_tokens=True,
    )
    equal_length = min(len(prefix_a), len(prefix_b))
    prefix_a = prefix_a[:equal_length]
    prefix_b = prefix_b[:equal_length]
    segment = tokenizer.encode(
        "The shared segment states that the instrument was calibrated twice.",
        add_special_tokens=False,
    )
    total_layers = len(_model_layers(backend.model))
    capture_layers = tuple(range(max(1, int(total_layers * 0.25))))
    inference_started = time.perf_counter()
    current = backend.prefill_ids(prefix_a, segment, capture_layers)
    other = backend.prefill_ids(prefix_b, segment, capture_layers)
    inference_seconds = time.perf_counter() - inference_started
    matching = [
        backend.compare_layer(current.layer_states[layer], current.layer_states[layer])
        for layer in capture_layers
    ]
    different = [
        backend.compare_layer(current.layer_states[layer], other.layer_states[layer])
        for layer in capture_layers
    ]
    matching_zero = all(
        all(value is None or value == 0.0 for value in row.values())
        for row in matching
    )
    different_nonzero = any(
        any(value is not None and value > 0.0 for value in row.values())
        for row in different
    )
    query_and_pre_rope_k_captured = all(
        current.layer_states[layer].query is not None
        and current.layer_states[layer].key_pre_rope is not None
        for layer in capture_layers
    )
    result = {
        "check": "local_reference_probe",
        "model_path": str(Path(args.model).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": "cpu",
        "total_layers": total_layers,
        "captured_experiment_layers_1_based": [layer + 1 for layer in capture_layers],
        "segment_tokens": len(segment),
        "equal_prefix_tokens": equal_length,
        "load_seconds": load_seconds,
        "two_prefills_seconds": inference_seconds,
        "matching_source_all_drifts_zero": matching_zero,
        "different_context_has_nonzero_drift": different_nonzero,
        "query_and_pre_rope_k_captured": query_and_pre_rope_k_captured,
        "different_source_drift_by_layer": [
            dict(layer=layer + 1, **row)
            for layer, row in zip(capture_layers, different)
        ],
        "evidence_class": "local_correctness",
        "paper_performance_evidence": False,
        "passed": matching_zero and different_nonzero and query_and_pre_rope_k_captured,
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
