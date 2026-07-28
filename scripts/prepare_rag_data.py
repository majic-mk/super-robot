"""Normalize a RAG dataset and construct auditable four-source manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file, write_jsonl
from probekv.manifest import manifest_digest, validate_manifest
from probekv.rag_data import (
    build_controlled_cases,
    build_corpus_repeat_cases,
    construction_audit,
    load_raw_records,
    normalize_example,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("hotpotqa", "2wiki", "musique"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model-signature", required=True)
    parser.add_argument("--source-url", default="unspecified-local-fixture")
    parser.add_argument("--source-revision", default="unspecified-local-fixture")
    parser.add_argument("--license", default="unspecified-local-fixture")
    parser.add_argument(
        "--construction",
        choices=("controlled", "corpus-repeat", "both"),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for model-specific token hashes") from error

    input_path = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_records = load_raw_records(input_path)
    if args.limit_records:
        raw_records = raw_records[: args.limit_records]
    normalized = [normalize_example(args.dataset, row) for row in raw_records]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=not args.allow_download
    )

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    cases = []
    if args.construction in {"controlled", "both"}:
        cases.extend(
            build_controlled_cases(
                normalized,
                encode,
                args.model_signature,
                seed=args.seed,
                max_cases=args.max_cases,
            )
        )
    if args.construction in {"corpus-repeat", "both"}:
        remaining = max(0, args.max_cases - len(cases)) if args.max_cases else 0
        if not args.max_cases or remaining:
            cases.extend(
                build_corpus_repeat_cases(
                    normalized,
                    encode,
                    args.model_signature,
                    seed=args.seed,
                    max_cases=remaining,
                )
            )
    if not cases and not args.allow_empty:
        raise RuntimeError(
            "construction produced zero cases; inspect document counts/repetition or use --allow-empty"
        )
    if cases:
        validate_manifest(cases)
    write_jsonl(output / "normalized_examples.jsonl", [example.to_row() for example in normalized])
    write_jsonl(output / "cases.jsonl", [case.to_row() for case in cases])
    audit = construction_audit(normalized, cases)
    audit.update(
        {
            "dataset_argument": args.dataset,
            "construction_argument": args.construction,
            "model_signature": args.model_signature,
            "tokenizer": str(args.tokenizer),
            "seed": args.seed,
            "raw_input": str(input_path),
            "raw_input_sha256": sha256_file(input_path),
            "official_source_url": args.source_url,
            "official_source_revision": args.source_revision,
            "dataset_license": args.license,
            "source_split": "train",
            "manifest_digest": manifest_digest(cases) if cases else None,
            "corpus_repeat_is_production_trace": False,
            "evidence_class": "data_preparation",
            "paper_evidence": False,
        }
    )
    atomic_write_json(output / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
