#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file, write_jsonl
from probekv.manifest import manifest_case_from_row, manifest_digest
from probekv.v8_profile_manifest import canonicalize_v8_profile_cases
from probekv.semantic_segment import SemanticWindowConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-signature", required=True)
    parser.add_argument("--document-revision", required=True)
    parser.add_argument("--chunker-policy", choices=("legacy", "paragraph_sentence_clause_v2", "fixed_tokens_v1"), default="legacy")
    parser.add_argument("--target-tokens", type=int, choices=(512, 1024, 2048))
    parser.add_argument("--search-window-tokens", type=int, choices=(50, 100))
    args = parser.parse_args()
    if args.chunker_policy == "legacy" and (args.target_tokens is not None or args.search_window_tokens is not None):
        parser.error("target/window overrides require an explicit new chunker policy")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for manifest text decoding") from error

    source = Path(args.input).resolve()
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Filtering before parsing prevents this profile-only preparation step
    # from opening pilot/test rows at all.
    development_rows = [
        row for row in rows if row.get("split") in {"calibration", "development"}
    ]
    cases = tuple(manifest_case_from_row(row) for row in development_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    config = None if args.chunker_policy == "legacy" else SemanticWindowConfig(
        target_tokens=args.target_tokens or 512, search_window_tokens=args.search_window_tokens or 100,
        policy=args.chunker_policy)
    segmentation_audits = []
    canonical = canonicalize_v8_profile_cases(
        cases,
        tokenizer_signature=args.tokenizer_signature,
        document_revision=args.document_revision,
        decode=lambda values: tokenizer.decode(
            list(int(value) for value in values), skip_special_tokens=False
        ),
        chunker_config=config,
        encode_with_offsets=lambda text: tokenizer(text, add_special_tokens=False, return_offsets_mapping=True),
        segmentation_audits=segmentation_audits,
    )
    output = Path(args.output).resolve()
    write_jsonl(output, [case.to_row() for case in canonical])
    audit = {
        "stage": "v8_profile_case_canonicalization",
        "protocol_version": 8,
        "input_sha256": sha256_file(source),
        "output_sha256": sha256_file(output),
        "input_development_cases": len(cases),
        "output_canonical_segments": len(canonical),
        "manifest_digest": manifest_digest(canonical),
        "tokenizer_signature": args.tokenizer_signature,
        "document_revision": args.document_revision,
        "chunker_policy": args.chunker_policy,
        "segmentation_audits": segmentation_audits,
        "paper_evidence": False,
        "locked_test_accessed": False,
    }
    atomic_write_json(Path(args.audit_output).resolve(), audit)
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
