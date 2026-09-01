#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.v8_schema10_jobs import build_schema10_qualification_jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=("mistral", "qwen"), required=True)
    parser.add_argument("--selection-depth-profile-sha256", required=True)
    parser.add_argument("--variant-admission-profile-sha256", required=True)
    parser.add_argument("--preparation-policy-profile-sha256", required=True)
    parser.add_argument("--repair-policy-profile-sha256", required=True)
    parser.add_argument("--runtime-cost-profile-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    jobs = build_schema10_qualification_jobs(
        model_key=args.model_key,
        selection_depth_profile_sha256=args.selection_depth_profile_sha256,
        variant_admission_profile_sha256=args.variant_admission_profile_sha256,
        preparation_policy_profile_sha256=args.preparation_policy_profile_sha256,
        repair_policy_profile_sha256=args.repair_policy_profile_sha256,
        runtime_cost_profile_sha256=args.runtime_cost_profile_sha256,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in jobs), encoding="utf-8")
    print(json.dumps({"output": str(output), "jobs": len(jobs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
