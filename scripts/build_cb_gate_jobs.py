"""Build the frozen three-dataset CB1-CB3 correctness job set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.experiment_jobs import generate_e1_jobs
from probekv.io import atomic_write_json, write_jsonl
from probekv.manifest import manifest_case_from_row, validate_manifest


RATIOS = (0.0, 0.05, 0.10, 0.16, 0.20, 0.30, 0.50, 0.75, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    cases = [
        manifest_case_from_row(json.loads(line))
        for line in Path(args.manifest)
        .resolve()
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    validate_manifest(cases)
    by_dataset = {}
    for case in sorted(cases, key=lambda item: item.case_id):
        by_dataset.setdefault(case.dataset, case)
    expected = {"MuSiQue", "2WikiMultiHopQA", "HotPotQA"}
    if set(by_dataset) != expected:
        raise ValueError("CB gates require all three datasets")
    selected = [by_dataset[name] for name in sorted(by_dataset)]
    primary = generate_e1_jobs(
        selected,
        total_layers=32,
        repair_ratios=RATIOS,
        seed=args.seed,
        include_splits=("pilot",),
        anchor_fraction=0.0,
    )
    anchors = generate_e1_jobs(
        selected[:1],
        total_layers=32,
        repair_ratios=RATIOS,
        seed=args.seed,
        include_splits=("pilot",),
        anchor_fraction=1.0,
    )
    jobs = primary + [job for job in anchors if job.reuse_layer != 5]
    identifiers = [job.job_id for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("CB gate jobs contain duplicate identifiers")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "cb_gate_jobs.jsonl", [job.to_row() for job in jobs])
    audit = {
        "cases": [case.case_id for case in selected],
        "datasets": sorted(by_dataset),
        "jobs": len(jobs),
        "primary_jobs": len(primary),
        "anchor_jobs": len(jobs) - len(primary),
        "repair_ratios": list(RATIOS),
        "paper_evidence": False,
        "evidence_class": "server_pilot",
    }
    atomic_write_json(output / "cb_gate_jobs_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
