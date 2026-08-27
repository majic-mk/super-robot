from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file
from probekv.v8_profile import build_profile_freeze_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=str(repo), text=True
    ).strip():
        raise ValueError("Profile-freeze contract requires a clean checkout")
    code_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=str(repo), text=True
    ).strip()
    contract = build_profile_freeze_contract(
        code_commit=code_commit,
        development_partition_sha256=sha256_file(
            Path(args.development_manifest).resolve()
        ),
    )
    atomic_write_json(Path(args.output).resolve(), contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
