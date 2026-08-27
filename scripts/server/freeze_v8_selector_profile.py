from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.v8_profile import freeze_selector_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-key", choices=("mistral", "qwen"), required=True)
    parser.add_argument("--policy", choices=("causal_commit_wait", "immediate_staggered_closed_loop"), required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-hash", required=True)
    parser.add_argument("--cacheblend-patch-sha256", required=True)
    parser.add_argument("--microbenchmark-sha256", required=True)
    parser.add_argument("--runtime-cost-profile", required=True)
    parser.add_argument("--profile-freeze-contract", required=True)
    parser.add_argument("--development-partition-sha256", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.rows).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runtime_cost_profile = json.loads(
        Path(args.runtime_cost_profile).read_text(encoding="utf-8")
    )
    profile_freeze_contract = json.loads(
        Path(args.profile_freeze_contract).read_text(encoding="utf-8")
    )
    profile = freeze_selector_profile(
        model_key=args.model_key,
        policy=args.policy,
        rows=rows,
        code_commit=args.code_commit,
        model_revision=args.model_revision,
        tokenizer_hash=args.tokenizer_hash,
        cacheblend_patch_sha256=args.cacheblend_patch_sha256,
        microbenchmark_sha256=args.microbenchmark_sha256,
        runtime_cost_profile=runtime_cost_profile,
        profile_freeze_contract=profile_freeze_contract,
        development_partition_sha256=args.development_partition_sha256,
    )
    atomic_write_json(Path(args.output).resolve(), profile)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
