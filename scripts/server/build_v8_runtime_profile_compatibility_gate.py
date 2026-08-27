from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.v8_profile import evaluate_runtime_profile_compatibility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-cost-profile", required=True)
    parser.add_argument("--actual-gpu-uuid", required=True)
    parser.add_argument("--hardware-compatibility-signature", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--cacheblend-patch-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    profile = json.loads(
        Path(args.runtime_cost_profile).read_text(encoding="utf-8")
    )
    result = evaluate_runtime_profile_compatibility(
        profile,
        actual_gpu_uuid=args.actual_gpu_uuid,
        actual_hardware_compatibility_signature=args.hardware_compatibility_signature,
        code_commit=args.code_commit,
        cacheblend_patch_sha256=args.cacheblend_patch_sha256,
    )
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
