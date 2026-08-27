from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.v8_runtime_qualification import evaluate_v8_runtime_qualification


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--runtime-cost-profile", required=True)
    parser.add_argument("--profile-freeze-contract", required=True)
    parser.add_argument("--runtime-audit", required=True)
    parser.add_argument("--prefix-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gate = evaluate_v8_runtime_qualification(
        load(args.lock), load(args.manifest), load(args.profile),
        load(args.runtime_cost_profile), load(args.profile_freeze_contract),
        load(args.runtime_audit), load(args.prefix_audit),
    )
    atomic_write_json(Path(args.output).resolve(), gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if gate["gpu_runtime_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
