"""Normalize a RAG dataset and construct auditable four-source manifests."""

from __future__ import annotations

import argparse
import json
import hashlib
from dataclasses import replace
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file, write_jsonl
from probekv.canonical_segment import canonicalize_token_ids
from probekv.manifest import token_content_hash
from probekv.manifest import manifest_digest, validate_manifest
from probekv.rag_data import (
    build_controlled_cases,
    build_corpus_repeat_cases,
    build_streaming_pilot_cases,
    construction_audit,
    iter_raw_records,
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
    parser.add_argument("--protocol-version", type=int, choices=(6, 7), default=6)
    parser.add_argument("--tokenizer-signature", default="")
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--max-controlled-cases",
        type=int,
        default=0,
        help="cap controlled cases without limiting the records scanned",
    )
    parser.add_argument(
        "--max-corpus-repeat-cases",
        type=int,
        default=0,
        help="cap corpus-repeat cases after scanning the complete input",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument(
        "--streaming-pilot",
        action="store_true",
        help="scan the full train split with bounded memory",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for model-specific token hashes") from error

    input_path = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=not args.allow_download
    )

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    streaming_audit = {}
    if args.streaming_pilot:
        if args.limit_records:
            raise ValueError(
                "--streaming-pilot must scan the complete input"
            )
        if args.construction != "both":
            raise ValueError(
                "--streaming-pilot requires --construction both"
            )

        def example_factory():
            return (
                normalize_example(args.dataset, row)
                for row in iter_raw_records(input_path)
            )

        cases, normalized, streaming_audit = build_streaming_pilot_cases(
            example_factory,
            encode,
            args.model_signature,
            seed=args.seed,
            max_controlled_cases=args.max_controlled_cases,
            max_corpus_repeat_cases=args.max_corpus_repeat_cases,
        )
    else:
        raw_records = load_raw_records(input_path)
        if args.limit_records:
            raw_records = raw_records[: args.limit_records]
        normalized = [
            normalize_example(args.dataset, row)
            for row in raw_records
        ]
        cases = []
        if args.construction in {"controlled", "both"}:
            controlled_limit = args.max_controlled_cases
            if args.max_cases and not controlled_limit:
                controlled_limit = args.max_cases
            cases.extend(
                build_controlled_cases(
                    normalized,
                    encode,
                    args.model_signature,
                    seed=args.seed,
                    max_cases=controlled_limit,
                )
            )
        if args.construction in {"corpus-repeat", "both"}:
            if args.max_cases:
                corpus_limit = max(0, args.max_cases - len(cases))
            else:
                corpus_limit = args.max_corpus_repeat_cases
            if not args.max_cases or corpus_limit:
                cases.extend(
                    build_corpus_repeat_cases(
                        normalized,
                        encode,
                        args.model_signature,
                        seed=args.seed,
                        max_cases=corpus_limit,
                    )
                )
    if args.protocol_version == 7:
        tokenizer_signature = args.tokenizer_signature or str(args.tokenizer)
        canonical_cases = []
        for case in cases:
            segments = canonicalize_token_ids(
                case.segment_token_ids,
                tokenizer_signature=tokenizer_signature,
                document_revision=args.source_revision,
            )
            for segment in segments:
                text = tokenizer.decode(
                    list(segment.token_ids), skip_special_tokens=False
                )
                roundtrip = tuple(
                    int(value)
                    for value in tokenizer.encode(text, add_special_tokens=False)
                )
                if roundtrip != segment.token_ids:
                    raise ValueError(
                        "canonical Segment text is not tokenizer round-trip exact"
                    )
                provenance = hashlib.sha256(
                    (
                        "%s|%s|%d|%d|%d"
                        % (
                            case.document_id,
                            args.source_revision,
                            segment.ordinal,
                            segment.token_start,
                            segment.token_end,
                        )
                    ).encode("utf-8")
                ).hexdigest()
                canonical_cases.append(
                    replace(
                        case,
                        case_id="%s-c%03d" % (case.case_id, segment.ordinal),
                        segment_text=text,
                        segment_token_ids=segment.token_ids,
                        content_hash=token_content_hash(segment.token_ids),
                        protocol_version=7,
                        canonicalizer_signature=segment.canonicalizer_signature,
                        segment_provenance_id=provenance,
                        reuse_content_key=segment.reuse_content_key(
                            args.model_signature, tokenizer_signature
                        ),
                        canonical_parent_content_hash=case.content_hash,
                        canonical_parent_left_token_ids=tuple(
                            int(value)
                            for value in case.segment_token_ids[:segment.token_start]
                        ),
                        canonical_parent_right_token_ids=tuple(
                            int(value)
                            for value in case.segment_token_ids[segment.token_end:]
                        ),
                    )
                )
        cases = canonical_cases
    if not cases and not args.allow_empty:
        raise RuntimeError(
            "construction produced zero cases; inspect document counts/repetition or use --allow-empty"
        )
    if cases:
        validate_manifest(cases)
    write_jsonl(output / "normalized_examples.jsonl", [example.to_row() for example in normalized])
    write_jsonl(output / "cases.jsonl", [case.to_row() for case in cases])
    audit = construction_audit(normalized, cases)
    audit.update(streaming_audit)
    audit.update(
        {
            "dataset_argument": args.dataset,
            "construction_argument": args.construction,
            "model_signature": args.model_signature,
            "tokenizer": str(args.tokenizer),
            "seed": args.seed,
            "max_cases": args.max_cases,
            "max_controlled_cases": args.max_controlled_cases,
            "max_corpus_repeat_cases": args.max_corpus_repeat_cases,
            "streaming_pilot": args.streaming_pilot,
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
            "protocol_version": args.protocol_version,
            "canonical_segmentation": args.protocol_version == 7,
        }
    )
    atomic_write_json(output / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
