#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.v8_schema10_jobs import build_schema10_no_gpu_handoff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--model-key", choices=("mistral", "qwen"), required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-hash", required=True)
    parser.add_argument("--cacheblend-patch-sha256", required=True)
    parser.add_argument("--cacheblend-tree", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs-output")
    args = parser.parse_args()
    handoff = build_schema10_no_gpu_handoff(
        code_commit=args.code_commit,
        model_key=args.model_key,
        model_revision=args.model_revision,
        tokenizer_hash=args.tokenizer_hash,
        cacheblend_patch_sha256=args.cacheblend_patch_sha256,
        cacheblend_tree=args.cacheblend_tree,
        config_sha256=args.config_sha256,
        contract_sha256=args.contract_sha256,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.jobs_output:
        jobs_output = Path(args.jobs_output).resolve()
        jobs_output.parent.mkdir(parents=True, exist_ok=True)
        jobs_output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in handoff["profile_jobs"]),
            encoding="utf-8",
        )
    print(json.dumps({"output": str(output), "jobs_sha256": handoff["jobs_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
