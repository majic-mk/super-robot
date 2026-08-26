from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.v7_runtime_qualification import build_v7_joint_gate


def _json(value: str) -> dict:
    return json.loads(Path(value).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--mistral-gate", required=True)
    parser.add_argument("--qwen-gate", required=True)
    parser.add_argument("--mistral-h1-sentinel", required=True)
    parser.add_argument("--qwen-h1-sentinel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
    ).strip()
    result = build_v7_joint_gate(
        code_commit=commit,
        mistral_gate=_json(args.mistral_gate),
        qwen_gate=_json(args.qwen_gate),
        mistral_h1_sentinel=_json(args.mistral_h1_sentinel),
        qwen_h1_sentinel=_json(args.qwen_h1_sentinel),
    )
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready_for_full_h1_pilot"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
