"""Offline re-verification; requires raw logits, never a GPU or a success flag."""
import argparse
from pathlib import Path

from probekv.v8_schema10_event_log import atomic_json
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_native_evidence import verify_native_primitive_directory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("verification report requires a fresh output path")
    row = verify_native_primitive_directory(args.input)
    row["verification_sha256"] = digest_json(row)
    atomic_json(output, row)
    print(row["verification_sha256"])


if __name__ == "__main__":
    main()
