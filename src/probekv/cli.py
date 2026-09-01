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
from .local_e1e2 import run_local_e1e2
from .manifest import case_from_mapping, manifest_digest, validate_manifest
from .simulation import run_local_simulation
from .v6_simulation import run_v6_local_simulation
from .v7_simulation import run_v7_local_simulation
from .v8_simulation import run_v8_local_simulation
from .v8_schema7_simulation import run_v8_schema7_local_simulation
from .v8_schema8_simulation import run_v8_schema8_local_simulation
from .v8_schema9_simulation import run_v8_schema9_local_simulation
from .v6_manifest import (
    request_case_from_mapping,
    request_manifest_digest,
    validate_request_manifest,
)


def _simulate(config_path: str, output_override: str = None) -> int:
    config = load_config(config_path)
    if config.evidence_class != "local_simulation":
        raise ValueError("simulate command only accepts local_simulation configs")
    if config.protocol_version == 8:
        if config.v8_schema_version == 9:
            result = run_v8_schema9_local_simulation(config)
        elif config.v8_schema_version == 8:
            result = run_v8_schema8_local_simulation(config)
        elif config.v8_schema_version == 7:
            result = run_v8_schema7_local_simulation(config)
        else:
            result = run_v8_local_simulation(config)
    elif config.protocol_version == 7:
        result = run_v7_local_simulation(config)
    elif config.protocol_version == 6:
        result = run_v6_local_simulation(config)
    else:
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


def _local_e1e2(
    config_path: str, output_override: str = None, resume: bool = False
) -> int:
    config = load_config(config_path)
    output = Path(output_override or config.output_dir).resolve()
    summary = run_local_e1e2(config, output, resume=resume)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _build_manifest(
    input_path: str,
    output_path: str,
    model_signature: str,
    seed: int,
) -> int:
    rows = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError("invalid JSON on line %d" % line_number) from error
    cases = [case_from_mapping(row, model_signature, seed) for row in rows]
    validate_manifest(cases)
    output = Path(output_path).resolve()
    write_jsonl(output, [case.to_row() for case in cases])
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": len(cases),
                "manifest_digest": manifest_digest(cases),
                "paper_evidence": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _validate_request_manifest(input_path: str) -> int:
    cases = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                cases.append(request_case_from_mapping(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "invalid v6 request manifest row %d" % line_number
                ) from error
    validate_request_manifest(cases)
    print(
        json.dumps(
            {
                "input": str(Path(input_path).resolve()),
                "cases": len(cases),
                "manifest_digest": request_manifest_digest(cases),
                "protocol_version": 6,
                "valid": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probekv")
    parser.add_argument(
        "--config",
        help="direct local-simulation shorthand; equivalent to simulate --config",
    )
    parser.add_argument("--output", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command")
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--output")
    e1e2 = subparsers.add_parser("local-e1e2")
    e1e2.add_argument("--config", required=True)
    e1e2.add_argument("--output")
    e1e2.add_argument("--resume", action="store_true")
    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--input", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--model-signature", required=True)
    manifest.add_argument("--seed", type=int, default=20260726)
    request_manifest = subparsers.add_parser("validate-request-manifest")
    request_manifest.add_argument("--input", required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        return _validate(args.config)
    if args.command == "simulate":
        return _simulate(args.config, args.output)
    if args.command == "local-e1e2":
        return _local_e1e2(args.config, args.output, args.resume)
    if args.command == "build-manifest":
        return _build_manifest(
            args.input, args.output, args.model_signature, args.seed
        )
    if args.command == "validate-request-manifest":
        return _validate_request_manifest(args.input)
    if args.config:
        return _simulate(args.config, args.output)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
