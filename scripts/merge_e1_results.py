"""Merge E1 shards, preserve failures and generate safe-ratio/H1 diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.e1_analysis import analyze_e1
from probekv.experiment_jobs import E1Job, E1Result, merge_e1_results
from probekv.io import atomic_write_json, write_jsonl


def read_rows(path: Path, constructor):
    return [
        constructor(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--result", action="append", default=[])
    parser.add_argument("--result-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-layers", type=int, default=32)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    jobs = read_rows(Path(args.jobs), E1Job.from_row)
    paths = [Path(value) for value in args.result]
    if args.result_dir:
        result_dir = Path(args.result_dir)
        paths.extend(sorted(result_dir.glob("shard-*.jsonl")))
        paths.extend(sorted(result_dir.glob("results-*.jsonl")))
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)
    results = []
    for path in unique_paths:
        results.extend(read_rows(path, E1Result.from_row))
    latest, audit = merge_e1_results(jobs, results)
    analysis = analyze_e1(
        jobs,
        latest,
        total_layers=args.total_layers,
        allow_test=args.allow_test,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "merged_results.jsonl", [result.to_row() for result in latest])
    write_jsonl(output / "safe_budget_labels.jsonl", analysis.pop("labels"))
    write_jsonl(output / "case_sensitivity.jsonl", analysis.pop("case_rows"))
    atomic_write_json(output / "merge_audit.json", audit)
    atomic_write_json(output / "analysis.json", analysis)
    print(json.dumps({"audit": audit, "analysis": analysis}, ensure_ascii=False))
    return 0 if (audit["all_completed"] or not args.require_complete) else 1


if __name__ == "__main__":
    raise SystemExit(main())
