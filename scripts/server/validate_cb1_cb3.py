"""Validate CB1-CB3 from immutable jobs and CacheBlend server results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.experiment_jobs import E1Job, E1Result, ResultStatus
from probekv.io import atomic_write_json
from probekv.repair_semantics import assert_nested_selections, RepairSelection


def _read(path: Path, loader):
    return [
        loader(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    jobs = _read(Path(args.jobs).resolve(), E1Job.from_row)
    results = []
    for path in args.results:
        results.extend(_read(Path(path).resolve(), E1Result.from_row))
    job_by_id = {job.job_id: job for job in jobs}
    latest = {}
    for result in results:
        if result.job_id not in job_by_id:
            continue
        previous = latest.get(result.job_id)
        if previous is None or result.attempt > previous.attempt:
            latest[result.job_id] = result
    completed = {
        job_id: result
        for job_id, result in latest.items()
        if result.status is ResultStatus.COMPLETED
    }
    missing = sorted(set(job_by_id) - set(completed))
    cb1 = (
        not missing
        and all(
            result.source_digest_before == result.source_digest_after
            for result in completed.values()
        )
        and all(
            result.source_k_representation == "pre_rope"
            and result.rope_alignment_mode
            == "cacheblend_current_org_pos"
            and result.causal_mask_mode == "absolute_query_positions"
            for result in completed.values()
        )
        and len({job.dataset for job in jobs}) == 3
    )
    endpoint_failures = []
    groups = {}
    for job in jobs:
        result = completed.get(job.job_id)
        if result is None:
            continue
        key = (job.case_id, job.source_id, job.reuse_layer)
        groups.setdefault(key, []).append((job, result))
        if job.repair_ratio == 0.0 and (
            result.selected_segment_tokens != 0
            or result.mandatory_suffix_tokens <= 0
        ):
            endpoint_failures.append("%s:r0" % job.job_id)
        if job.repair_ratio == 1.0 and (
            result.selected_segment_tokens != result.eligible_segment_tokens
            or not result.output_ids_exact_full
            or tuple(result.output_token_ids)
            != tuple(result.full_output_token_ids)
            or result.logit_relative_l2 is None
            or float(result.logit_relative_l2) > 1e-4
            or result.logit_trace_mode != "matched_greedy_prefix"
            or result.logit_positions_compared <= 0
        ):
            endpoint_failures.append("%s:r1" % job.job_id)
    cb2 = not missing and not endpoint_failures
    grid_failures = []
    expected_ratios = {0.0, 0.05, 0.10, 0.16, 0.20, 0.30, 0.50, 0.75, 1.0}
    for key, pairs in sorted(groups.items()):
        if {job.repair_ratio for job, _ in pairs} != expected_ratios:
            grid_failures.append("%s:ratios" % (key,))
            continue
        selections = [
            RepairSelection(
                requested_ratio=job.repair_ratio,
                eligible_segment_tokens=int(result.eligible_segment_tokens),
                selected_segment_indices=tuple(result.selected_segment_indices),
                mandatory_suffix_indices=tuple(
                    range(
                        int(result.prefix_tokens)
                        + int(result.eligible_segment_tokens),
                        int(result.prefix_tokens)
                        + int(result.eligible_segment_tokens)
                        + int(result.mandatory_suffix_tokens),
                    )
                ),
            )
            for job, result in pairs
        ]
        try:
            assert_nested_selections(selections)
        except ValueError:
            grid_failures.append("%s:nesting" % (key,))
        if any(
            float(result.repair_gpu_ms) < 0
            or float(result.repair_host_ms) < 0
            or result.timing_warmup_runs < 2
            or result.timing_measurement_runs < 5
            for _, result in pairs
        ):
            grid_failures.append("%s:timing" % (key,))
    cb3 = not missing and not grid_failures
    payload = {
        "CB1": {"passed": cb1, "missing_jobs": len(missing)},
        "CB2": {
            "passed": cb2,
            "endpoint_failures": endpoint_failures,
        },
        "CB3": {"passed": cb3, "grid_failures": grid_failures},
        "stage_gate_passed": cb1 and cb2 and cb3,
        "paper_evidence": False,
        "evidence_class": "server_pilot",
    }
    atomic_write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["stage_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
