from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file
from probekv.v8_profile import build_runtime_cost_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--model-key", choices=("mistral", "qwen"), required=True)
    parser.add_argument(
        "--policy",
        choices=("causal_commit_wait", "immediate_staggered_closed_loop"),
        required=True,
    )
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--cacheblend-patch-sha256", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--hardware-compatibility-signature", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.measurements).resolve()
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    curve = {}
    for count in (1, 2, 4, 8, 16):
        values = sorted(
            float(row["selection_batch_gpu_ms"])
            for row in rows
            if int(row["compared_k"]) == count
            and row.get("cuda_event_timing") is True
            and row.get("fake_timing") is False
        )
        if not values:
            raise ValueError("missing real RuntimeCostProfile samples for K=%d" % count)
        curve[count] = values[-1]
    profile = build_runtime_cost_profile(
        model_key=args.model_key,
        policy=args.policy,
        code_commit=args.code_commit,
        cacheblend_patch_sha256=args.cacheblend_patch_sha256,
        gpu_uuid=args.gpu_uuid,
        hardware_compatibility_signature=args.hardware_compatibility_signature,
        comparison_batch_upper_ms=curve,
        measurement_sha256=sha256_file(path),
    )
    atomic_write_json(Path(args.output).resolve(), profile)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
