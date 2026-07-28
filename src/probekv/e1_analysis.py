from __future__ import annotations

import statistics
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .experiment_jobs import E1Job, E1Result, ResultStatus
from .labeling import RatioMeasurement, safe_repair_ratio
from .statistics import grouped_paired_bootstrap


def analyze_e1(
    jobs: Sequence[E1Job],
    results: Sequence[E1Result],
    total_layers: int,
    allow_test: bool = False,
    bootstrap_iterations: int = 2000,
    seed: int = 20260726,
    result_set_audit: Optional[Mapping[str, Any]] = None,
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
        safe_job = next(job for job, _ in pairs if job.repair_ratio == safe)
        remaining_layers = total_layers - key[2] + 1
        safe_cost = int(safe_result.selected_segment_tokens) * remaining_layers
        labels.append(
            {
                "case_id": key[0],
                "source_id": key[1],
                "reuse_layer": key[2],
                "safe_repair_ratio": safe,
                "repair_latency_ms": safe_result.repair_latency_ms,
                "repair_gpu_ms": safe_result.repair_gpu_ms,
                "repair_host_ms": safe_result.repair_host_ms,
                "selected_segment_tokens": safe_result.selected_segment_tokens,
                "eligible_segment_tokens": safe_result.eligible_segment_tokens,
                "safe_token_layer_cost": safe_cost,
                "dataset": safe_job.dataset,
                "construction": safe_job.construction,
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
    s0_improvement_groups = {}
    latest_improvement_groups = {}
    case_metadata = {}
    for job in jobs:
        case_metadata.setdefault(
            job.case_id,
            {"dataset": job.dataset, "construction": job.construction},
        )
    for case_id, rows in sorted(primary_by_case.items()):
        expected_sources = len(source_count_by_case[case_id])
        if len(rows) != expected_sources:
            continue
        ordered = sorted(rows, key=lambda row: _source_order(row["source_id"]))
        ratios = [float(row["safe_repair_ratio"]) for row in ordered]
        costs = [float(row["safe_token_layer_cost"]) for row in ordered]
        oracle_index = min(
            range(len(costs)),
            key=lambda index: (
                costs[index],
                float(ordered[index]["repair_gpu_ms"]),
                _source_order(ordered[index]["source_id"]),
            ),
        )
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
        s0_improvement_groups[case_id] = [s0_improvement]
        latest_improvement_groups[case_id] = [last_source_improvement]
        case_rows.append(
            {
                "case_id": case_id,
                "dataset": case_metadata[case_id]["dataset"],
                "construction": case_metadata[case_id]["construction"],
                "reuse_layer": primary_layer,
                "source_spread": spread,
                "oracle_source": ordered[oracle_index]["source_id"],
                "oracle_safe_token_layer_cost": oracle_cost,
                "single_source_s0_improvement": s0_improvement,
                "latest_source_oracle_improvement": last_source_improvement,
            }
        )

    if s0_improvement_groups:
        s0_interval = grouped_paired_bootstrap(
            s0_improvement_groups,
            iterations=bootstrap_iterations,
            seed=seed,
        )
        latest_interval = grouped_paired_bootstrap(
            latest_improvement_groups,
            iterations=bootstrap_iterations,
            seed=seed + 1,
        )
        interval_row = {
            "s0": asdict(s0_interval),
            "latest": asdict(latest_interval),
        }
    else:
        interval_row = None
    spread_fraction = (
        sum(value >= 0.10 for value in spreads) / float(len(spreads))
        if spreads
        else 0.0
    )
    mean_s0 = statistics.mean(s0_improvements) if s0_improvements else 0.0
    mean_latest = (
        statistics.mean(last_source_improvements)
        if last_source_improvements
        else 0.0
    )
    completed_fraction = len(completed) / float(len(jobs)) if jobs else 0.0
    runtime_endpoint_failures = []
    for key, pairs in sorted(result_groups.items()):
        full_ratio = next(
            (
                result
                for job, result in pairs
                if abs(job.repair_ratio - 1.0) <= 1e-12
            ),
            None,
        )
        if full_ratio is not None and (
            float(full_ratio.task_score_drop) > 0.10
            or float(full_ratio.token_f1) < 0.90
        ):
            runtime_endpoint_failures.append(
                {
                    "case_id": key[0],
                    "source_id": key[1],
                    "reuse_layer": key[2],
                }
            )
    if not case_rows:
        decision = "runtime_failure" if runtime_endpoint_failures else "incomplete"
    elif runtime_endpoint_failures:
        decision = "runtime_failure"
    elif (
        spread_fraction >= 0.25
        and mean_s0 >= 0.10
        and mean_latest >= 0.10
        and completed_fraction >= 0.90
    ):
        decision = "pass"
    elif spread_fraction < 0.10 and max(mean_s0, mean_latest) < 0.05:
        decision = "stop_multi_source"
    else:
        decision = "conditional"
    gate_row = {
        "gate": "H1-pilot",
        "passed": decision == "pass",
        "decision": decision,
        "spread_ge_10pp_fraction": spread_fraction,
        "mean_s0_oracle_improvement": mean_s0,
        "mean_latest_oracle_improvement": mean_latest,
        "completed_fraction": completed_fraction,
        "runtime_endpoint_failures": len(runtime_endpoint_failures),
    }
    if result_set_audit is None:
        observed_ids = {result.job_id for result in results}
        expected_ids = set(job_by_id)
        result_set_complete = (
            bool(jobs)
            and observed_ids == expected_ids
            and len(results) == len(jobs)
            and len(completed) == len(jobs)
        )
        run_environment_valid = bool(results) and all(
            result.code_commit
            and result.environment_hash
            and result.finished_at_utc
            for result in results
        )
        publication_ready = (
            result_set_complete
            and run_environment_valid
            and all(result.paper_evidence for result in results)
        )
    else:
        result_set_complete = bool(
            result_set_audit.get("result_set_complete", False)
        )
        run_environment_valid = bool(
            result_set_audit.get("run_environment_valid", False)
        )
        publication_ready = bool(
            result_set_audit.get("publication_ready", False)
        )
    return {
        "primary_reuse_layer": primary_layer,
        "jobs": len(jobs),
        "completed_results": len(completed),
        "safe_labels": sum(row["label_status"] == "safe" for row in labels),
        "incomplete_groups": incomplete_groups,
        "no_safe_groups": no_safe_groups,
        "analyzable_cases": len(case_rows),
        "spread_ge_10pp_fraction": spread_fraction if spreads else None,
        "mean_s0_oracle_improvement": (
            statistics.mean(s0_improvements) if s0_improvements else None
        ),
        "mean_latest_source_oracle_improvement": (
            statistics.mean(last_source_improvements)
            if last_source_improvements
            else None
        ),
        "completed_fraction": completed_fraction,
        "runtime_endpoint_failures": runtime_endpoint_failures,
        "improvement_ci": interval_row,
        "gate_diagnostic": gate_row,
        "run_environment_valid": run_environment_valid,
        "result_set_complete": result_set_complete,
        "publication_ready": publication_ready,
        "paper_gate_claimable": publication_ready,
        "paper_evidence": publication_ready,
        "labels": labels,
        "case_rows": case_rows,
    }


def _source_order(source_id: str) -> Tuple[int, str]:
    suffix = source_id[1:] if source_id.startswith("s") else ""
    return (int(suffix), source_id) if suffix.isdigit() else (10 ** 9, source_id)
