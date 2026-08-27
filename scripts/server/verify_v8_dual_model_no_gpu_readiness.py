from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.v8_server_readiness import evaluate_v8_no_gpu_readiness


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--mistral-causal-manifest", required=True)
    parser.add_argument("--mistral-immediate-manifest", required=True)
    parser.add_argument("--qwen-causal-manifest", required=True)
    parser.add_argument("--qwen-immediate-manifest", required=True)
    parser.add_argument("--mistral-model-audit", required=True)
    parser.add_argument("--qwen-model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--storage-ready", action="store_true")
    parser.add_argument("--runtime-source-ready", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
    ).strip()
    clean = not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(repo), text=True
    ).strip()
    result = evaluate_v8_no_gpu_readiness(
        load(args.lock),
        {
            "mistral_causal_wait": load(args.mistral_causal_manifest),
            "mistral_immediate_staggered": load(args.mistral_immediate_manifest),
            "qwen_causal_wait": load(args.qwen_causal_manifest),
            "qwen_immediate_staggered": load(args.qwen_immediate_manifest),
        },
        {
            "mistral": load(args.mistral_model_audit),
            "qwen": load(args.qwen_model_audit),
        },
        load(args.patch_audit),
        expected_code_commit=args.expected_code_commit,
        actual_code_commit=actual,
        git_clean=clean,
        storage_ready=args.storage_ready,
        runtime_source_ready=args.runtime_source_ready,
    )
    atomic_write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gpu_rental_ready_for_profile_freeze"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
