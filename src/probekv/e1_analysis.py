from __future__ import annotations

import statistics
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .experiment_jobs import E1Job, E1Result, ResultStatus
from .gates import gate_h1
from .labeling import RatioMeasurement, safe_repair_ratio
from .statistics import grouped_paired_bootstrap


def analyze_e1(
    jobs: Sequence[E1Job],
    results: Sequence[E1Result],
    total_layers: int,
    allow_test: bool = False,
    bootstrap_iterations: int = 2000,
    seed: int = 20260726,
) -> Dict[str, Any]:
    job_by_id = {job.job_id: job for job in jobs}
    if len(job_by_id) != len(jobs):
        raise ValueError("duplicate jobs are not analyzable")
    if not allow_test and any(job.split == "test" for job in jobs):
        raise ValueError("locked test jobs require explicit allow_test=True")
    completed = {
        result.job_id: result
        for result in results
        if result.status is ResultStatus.COMPLETED and result.job_id in job_by_id
    }
    expected_ratios: Dict[Tuple[str, str, int], set] = {}
    result_groups: Dict[Tuple[str, str, int], List[Tuple[E1Job, E1Result]]] = {}
    for job in jobs:
        key = (job.case_id, job.source_id, job.reuse_layer)
        expected_ratios.setdefault(key, set()).add(job.repair_ratio)
        if job.job_id in completed:
            result_groups.setdefault(key, []).append((job, completed[job.job_id]))

    labels = []
    incomplete_groups = []
    no_safe_groups = []
    for key, ratios in sorted(expected_ratios.items()):
        pairs = result_groups.get(key, [])
        observed = {job.repair_ratio for job, _ in pairs}
        if observed != ratios:
            incomplete_groups.append(
                {
                    "case_id": key[0],
                    "source_id": key[1],
                    "reuse_layer": key[2],
                    "missing_ratios": sorted(ratios - observed),
                }
            )
            continue
        measurements = [
            RatioMeasurement(
                job.repair_ratio,
                float(result.task_score_drop),
                float(result.token_f1),
            )
            for job, result in pairs
        ]
        safe = safe_repair_ratio(measurements)
        if safe is None:
            no_safe_groups.append(
                {
                    "case_id": key[0],
                    "source_id": key[1],
                    "reuse_layer": key[2],
                }
            )
            labels.append(
                {
                    "case_id": key[0],
                    "source_id": key[1],
                    "reuse_layer": key[2],
                    "safe_repair_ratio": None,
                    "repair_latency_ms": None,
                    "label_status": "no_safe_ratio",
                }
            )
            continue
        safe_result = next(
            result for job, result in pairs if job.repair_ratio == safe
        )
        labels.append(
            {
                "case_id": key[0],
                "source_id": key[1],
                "reuse_layer": key[2],
                "safe_repair_ratio": safe,
                "repair_latency_ms": safe_result.repair_latency_ms,
                "quality_score": safe_result.quality_score,
                "token_f1": safe_result.token_f1,
                "label_status": "safe",
            }
        )

    primary_layer = max(1, min(total_layers - 1, round(total_layers * 0.15)))
    primary_by_case: Dict[str, List[Mapping[str, Any]]] = {}
    source_count_by_case: Dict[str, set] = {}
    for job in jobs:
        source_count_by_case.setdefault(job.case_id, set()).add(job.source_id)
    for row in labels:
        if row["reuse_layer"] == primary_layer and row["label_status"] == "safe":
            primary_by_case.setdefault(row["case_id"], []).append(row)

    case_rows = []
    spreads = []
    s0_improvements = []
    last_source_improvements = []
    improvement_groups = {}
    for case_id, rows in sorted(primary_by_case.items()):
        expected_sources = len(source_count_by_case[case_id])
        if len(rows) != expected_sources:
            continue
        ordered = sorted(rows, key=lambda row: _source_order(row["source_id"]))
        ratios = [float(row["safe_repair_ratio"]) for row in ordered]
        costs = [float(row["repair_latency_ms"]) for row in ordered]
        oracle_index = min(range(len(costs)), key=lambda index: costs[index])
        oracle_cost = costs[oracle_index]
        s0_cost = costs[0]
        last_source_cost = costs[-1]
        s0_improvement = (
            (s0_cost - oracle_cost) / s0_cost if s0_cost > 0 else 0.0
        )
        last_source_improvement = (
            (last_source_cost - oracle_cost) / last_source_cost
            if last_source_cost > 0
            else 0.0
        )
        spread = max(ratios) - min(ratios)
        spreads.append(spread)
        s0_improvements.append(s0_improvement)
        last_source_improvements.append(last_source_improvement)
        improvement_groups[case_id] = [s0_improvement]
        case_rows.append(
            {
                "case_id": case_id,
                "reuse_layer": primary_layer,
                "source_spread": spread,
                "oracle_source": ordered[oracle_index]["source_id"],
                "oracle_cost_ms": oracle_cost,
                "single_source_s0_improvement": s0_improvement,
                "last_source_oracle_improvement": last_source_improvement,
            }
        )

    if improvement_groups:
        interval = grouped_paired_bootstrap(
            improvement_groups,
            iterations=bootstrap_iterations,
            seed=seed,
        )
        gate = gate_h1(spreads, s0_improvements, interval)
        interval_row = asdict(interval)
        gate_row = asdict(gate)
    else:
        interval_row = None
        gate_row = {
            "gate": "H1",
            "passed": False,
            "summary": "no complete primary-layer source groups",
        }
    paper_claimable = (
        bool(results)
        and len(results) == len(jobs)
        and len(completed) == len(jobs)
        and not incomplete_groups
        and not no_safe_groups
        and all(result.paper_evidence for result in results)
    )
    return {
        "primary_reuse_layer": primary_layer,
        "jobs": len(jobs),
        "completed_results": len(completed),
        "safe_labels": sum(row["label_status"] == "safe" for row in labels),
        "incomplete_groups": incomplete_groups,
        "no_safe_groups": no_safe_groups,
        "analyzable_cases": len(case_rows),
        "spread_ge_10pp_fraction": (
            sum(value >= 0.10 for value in spreads) / float(len(spreads))
            if spreads
            else None
        ),
        "mean_s0_oracle_improvement": (
            statistics.mean(s0_improvements) if s0_improvements else None
        ),
        "mean_last_source_oracle_improvement": (
            statistics.mean(last_source_improvements)
            if last_source_improvements
            else None
        ),
        "improvement_ci": interval_row,
        "gate_diagnostic": gate_row,
        "paper_gate_claimable": paper_claimable,
        "paper_evidence": paper_claimable,
        "labels": labels,
        "case_rows": case_rows,
    }


def _source_order(source_id: str) -> Tuple[int, str]:
    suffix = source_id[1:] if source_id.startswith("s") else ""
    return (int(suffix), source_id) if suffix.isdigit() else (10 ** 9, source_id)
