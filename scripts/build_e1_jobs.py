"""Build the deterministic E1 repair-grid job manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.config import load_config
from probekv.experiment_jobs import generate_e1_jobs
from probekv.io import atomic_write_json, sha256_file, write_jsonl
from probekv.manifest import manifest_case_from_row, validate_manifest


def read_cases(path: Path):
    cases = [
        manifest_case_from_row(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_manifest(cases)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", default="pilot,train")
    parser.add_argument("--anchor-fraction", type=float, default=0.20)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    config = load_config(args.config)
    cases = read_cases(manifest_path)
    splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    if config.evidence_class == "server_pilot" and "test" in splits:
        raise ValueError("server_pilot cannot read the locked test split")
    jobs = generate_e1_jobs(
        cases,
        config.total_layers,
        config.repair_ratios,
        seed=config.seed,
        include_splits=splits,
        anchor_fraction=args.anchor_fraction,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "jobs.jsonl", [job.to_row() for job in jobs])
    selected_cases = {job.case_id for job in jobs}
    layer_counts = {}
    for job in jobs:
        layer_counts[str(job.reuse_layer)] = layer_counts.get(str(job.reuse_layer), 0) + 1
    audit = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(Path(args.config).resolve()),
        "include_splits": list(splits),
        "anchor_fraction": args.anchor_fraction,
        "cases": len(selected_cases),
        "sources": len({(job.case_id, job.source_id) for job in jobs}),
        "jobs": len(jobs),
        "layer_job_counts": layer_counts,
        "repair_ratios": list(config.repair_ratios),
        "seed": config.seed,
        "evidence_class": config.evidence_class,
        "paper_evidence": False,
    }
    atomic_write_json(output / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
