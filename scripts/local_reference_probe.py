"""Real-model smoke test for per-layer reference-state extraction."""

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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(20260726)
    started = time.perf_counter()
    backend = HuggingFaceReferenceStateBackend.from_pretrained(
        args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=True,
    )
    load_seconds = time.perf_counter() - started
    tokenizer = backend.tokenizer
    prefix_current = tokenizer.encode(
        "Current request context P: laboratory maintenance records, sensor logs, and the latest calibration schedule. ",
        add_special_tokens=True,
    )
    prefix_a = tokenizer.encode(
        "Historical context A: alpine climate records and snowfall. ",
        add_special_tokens=True,
    )
    prefix_b = tokenizer.encode(
        "Historical context B: coral reefs. ",
        add_special_tokens=True,
    )
    prefix_token_counts = {
        "current_P": len(prefix_current),
        "source_A": len(prefix_a),
        "source_B": len(prefix_b),
    }
    if len(set(prefix_token_counts.values())) != 3:
        raise RuntimeError(
            "P, A and B must have different natural token lengths: %r"
            % prefix_token_counts
        )
    segment = tokenizer.encode(
        "The shared segment states that the instrument was calibrated twice.",
        add_special_tokens=False,
    )
    total_layers = len(_model_layers(backend.model))
    capture_layers = tuple(range(max(1, int(total_layers * 0.25))))
    inference_started = time.perf_counter()
    current = backend.prefill_ids(prefix_current, segment, capture_layers)
    source_a = backend.prefill_ids(prefix_a, segment, capture_layers)
    source_b = backend.prefill_ids(prefix_b, segment, capture_layers)
    inference_seconds = time.perf_counter() - inference_started
    self_comparison = [
        backend.compare_layer(current.layer_states[layer], current.layer_states[layer])
        for layer in capture_layers
    ]
    source_drifts = {
        "source_A": [
            backend.compare_layer(
                current.layer_states[layer], source_a.layer_states[layer]
            )
            for layer in capture_layers
        ],
        "source_B": [
            backend.compare_layer(
                current.layer_states[layer], source_b.layer_states[layer]
            )
            for layer in capture_layers
        ],
    }
    self_comparison_zero = all(
        all(value is None or value == 0.0 for value in row.values())
        for row in self_comparison
    )
    historical_sources_nonzero = all(
        any(
            any(value is not None and value > 0.0 for value in row.values())
            for row in rows
        )
        for rows in source_drifts.values()
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
        "device": args.device,
        "dtype": args.dtype,
        "total_layers": total_layers,
        "captured_experiment_layers_1_based": [layer + 1 for layer in capture_layers],
        "segment_tokens": len(segment),
        "prefix_token_counts": prefix_token_counts,
        "load_seconds": load_seconds,
        "three_prefills_seconds": inference_seconds,
        "self_comparison_all_drifts_zero": self_comparison_zero,
        "all_historical_sources_differ_from_current": historical_sources_nonzero,
        "source_ranking_is_descriptive_only": True,
        "query_and_pre_rope_k_captured": query_and_pre_rope_k_captured,
        "historical_source_drift_by_layer": {
            source_id: [
                dict(layer=layer + 1, **row)
                for layer, row in zip(capture_layers, rows)
            ]
            for source_id, rows in source_drifts.items()
        },
        "evidence_class": (
            "server_correctness" if args.device == "cuda" else "local_correctness"
        ),
        "paper_performance_evidence": False,
        "passed": (
            self_comparison_zero
            and historical_sources_nonzero
            and query_and_pre_rope_k_captured
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
