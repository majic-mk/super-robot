"""Run one non-paper deterministic shard through the exact E1 result contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.experiment_jobs import (
    E1Job,
    E1Result,
    resumable_e1_jobs,
    select_job_shard,
    simulate_e1_results,
)
from probekv.io import atomic_write_json, write_jsonl


def read_jobs(path: Path):
    return [
        E1Job.from_row(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_results(path: Path):
    if not path.exists():
        return []
    return [
        E1Result.from_row(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--total-layers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    jobs = select_job_shard(
        read_jobs(Path(args.jobs)), args.shard_index, args.shard_count
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / ("shard-%05d.jsonl" % args.shard_index)
    existing = read_results(result_path) if args.resume else []
    pending, attempt_by_job = resumable_e1_jobs(jobs, existing)
    new_results = simulate_e1_results(
        pending,
        total_layers=args.total_layers,
        attempt_by_job=attempt_by_job,
    )
    combined = existing + new_results
    write_jsonl(result_path, [result.to_row() for result in combined])
    summary = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_jobs": len(jobs),
        "existing_results": len(existing),
        "new_results": len(new_results),
        "result_rows": len(combined),
        "paper_evidence": False,
    }
    atomic_write_json(output / ("shard-%05d-summary.json" % args.shard_index), summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
