from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .io import (
    environment_manifest,
    sha256_file,
    try_write_parquet,
    write_json,
    write_jsonl,
)
from .simulation import run_local_simulation


def _simulate(config_path: str, output_override: str = None) -> int:
    config = load_config(config_path)
    if config.evidence_class != "local_simulation":
        raise ValueError("simulate command only accepts local_simulation configs")
    result = run_local_simulation(config)
    output = Path(output_override or config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    workspace = Path.cwd()
    config_file = Path(config_path).resolve()
    manifest = environment_manifest(
        workspace,
        config.seed,
        config.evidence_class,
        model_signature="deterministic-simulation-v1",
        data_manifest_hash="synthetic-seed-%d" % config.seed,
        config_hash=sha256_file(config_file),
    )
    for row in result["rows"]:
        row.update(
            {
                "git_commit": manifest["git_commit"],
                "environment_hash": manifest["environment_hash"],
                "model_signature": manifest["model_signature"],
                "data_manifest_hash": manifest["data_manifest_hash"],
                "config_hash": manifest["config_hash"],
                "seed": config.seed,
                "timestamp_utc": manifest["timestamp_utc"],
            }
        )
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", result["summary"])
    write_json(output / "gates.json", result["gates"])
    write_jsonl(output / "cases.jsonl", result["rows"])
    parquet_written = try_write_parquet(output / "cases.parquet", result["rows"])
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": config.cases,
                "parquet_written": parquet_written,
                "paper_evidence": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _validate(config_path: str) -> int:
    config = load_config(config_path)
    print(
        json.dumps(
            {
                "name": config.name,
                "valid": True,
                "evidence_class": config.evidence_class,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probekv")
    subparsers = parser.add_subparsers(dest="command")
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--output")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        return _validate(args.config)
    if args.command == "simulate":
        return _simulate(args.config, args.output)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
