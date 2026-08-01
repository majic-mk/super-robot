"""Freeze the non-paper v6 A800 correctness/microbenchmark job matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import write_json, write_jsonl
from probekv.v6_a800_jobs import build_v6_a800_jobs, v6_a800_job_digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v6_a800_microbench.json"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if raw.get("protocol_version") != 6 or raw.get("paper_evidence"):
        raise ValueError("v6 A800 bring-up matrix must remain non-paper")
    jobs = build_v6_a800_jobs(raw)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "jobs.jsonl", [job.to_row() for job in jobs])
    summary = {
        "protocol_version": 6,
        "paper_evidence": False,
        "jobs": len(jobs),
        "job_digest": v6_a800_job_digest(jobs),
    }
    write_json(output / "manifest.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
