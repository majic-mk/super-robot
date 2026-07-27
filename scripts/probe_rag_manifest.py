"""Extract real-model per-layer observations for an audited RAG manifest."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file, write_jsonl
from probekv.manifest import manifest_case_from_row, manifest_digest, validate_manifest
from probekv.reference_hf import HuggingFaceReferenceStateBackend, _model_layers


def read_cases(path: Path):
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                cases.append(manifest_case_from_row(json.loads(line)))
            except Exception as error:
                raise ValueError("invalid manifest row %d" % line_number) from error
    validate_manifest(cases)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(20260726)
    manifest_path = Path(args.manifest).resolve()
    cases = read_cases(manifest_path)
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    backend = HuggingFaceReferenceStateBackend.from_pretrained(
        args.model, device="cpu", dtype="float32", local_files_only=True
    )
    load_seconds = time.perf_counter() - load_started
    tokenizer = backend.tokenizer
    total_layers = len(_model_layers(backend.model))
    capture_layers = tuple(range(max(1, int(total_layers * 0.25))))
    observations = []
    case_rows = []
    inference_seconds = 0.0
    for case in cases:
        encoded_segment = tuple(
            int(token)
            for token in tokenizer.encode(case.segment_text, add_special_tokens=False)
        )
        if encoded_segment != case.segment_token_ids:
            raise ValueError(
                "manifest segment IDs do not match loaded tokenizer for case %s"
                % case.case_id
            )
        current_prefix = tokenizer.encode(
            case.current_context, add_special_tokens=True
        )
        started = time.perf_counter()
        current = backend.prefill_ids(
            current_prefix, case.segment_token_ids, capture_layers
        )
        inference_seconds += time.perf_counter() - started
        for source in case.sources:
            source_prefix = tokenizer.encode(
                source.historical_context, add_special_tokens=True
            )
            started = time.perf_counter()
            historical = backend.prefill_ids(
                source_prefix, case.segment_token_ids, capture_layers
            )
            inference_seconds += time.perf_counter() - started
            for layer in capture_layers:
                compare_started = time.perf_counter()
                drifts = backend.compare_layer(
                    current.layer_states[layer], historical.layer_states[layer]
                )
                comparison_latency_ms = (
                    time.perf_counter() - compare_started
                ) * 1000.0
                observations.append(
                    {
                        "case_id": case.case_id,
                        "dataset": case.dataset,
                        "split": case.split,
                        "construction": case.construction,
                        "source_id": source.source_id,
                        "source_regime": source.regime,
                        "layer": layer + 1,
                        "k_drift_pre_rope": drifts["k_drift_pre_rope"],
                        "v_drift": drifts["v_drift"],
                        "hidden_drift": drifts["hidden_drift"],
                        "query_drift": drifts["query_drift"],
                        "comparison_latency_ms_cpu": comparison_latency_ms,
                        "current_prefix_tokens": len(current_prefix),
                        "source_prefix_tokens": len(source_prefix),
                        "segment_tokens": len(case.segment_token_ids),
                        "evidence_class": "local_correctness",
                        "paper_performance_evidence": False,
                    }
                )
        case_rows.append(
            {
                "case_id": case.case_id,
                "sources": len(case.sources),
                "layers": len(capture_layers),
                "observations": len(case.sources) * len(capture_layers),
            }
        )
    write_jsonl(output / "probe_observations.jsonl", observations)
    summary = {
        "check": "rag_manifest_reference_probe",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selected_manifest_digest": manifest_digest(cases),
        "model_path": str(Path(args.model).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": "cpu",
        "total_layers": total_layers,
        "captured_experiment_layers_1_based": [layer + 1 for layer in capture_layers],
        "cases": len(cases),
        "sources": sum(len(case.sources) for case in cases),
        "observations": len(observations),
        "load_seconds": load_seconds,
        "prefill_seconds": inference_seconds,
        "case_rows": case_rows,
        "all_drifts_finite": all(
            value is None or math.isfinite(float(value))
            for row in observations
            for value in (
                row["k_drift_pre_rope"],
                row["v_drift"],
                row["hidden_drift"],
                row["query_drift"],
            )
        ),
        "evidence_class": "local_correctness",
        "paper_performance_evidence": False,
    }
    summary["passed"] = bool(observations) and summary["all_drifts_finite"]
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
