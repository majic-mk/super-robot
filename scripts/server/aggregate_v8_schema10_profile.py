#!/usr/bin/env python3
"""Aggregate diagnostics without manufacturing production evidence.

Historical causal_replay_of_preexisting_historical_variants is read-only.
Residual thresholds are not FinalCommit; generation latency is not TTFT.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from probekv.io import atomic_write_json, sha256_file
from probekv.v8_schema10_evidence import EVIDENCE_CONTRACT_VERSION, summarize_repair_evidence
from probekv.v8_schema10_profile import SCHEMA10_MODEL_CHECKPOINTS
from probekv.v8_schema10_profile_analysis import build_threshold_table, build_selection_candidates


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate_diagnostics(
    rows: Sequence[Mapping[str, Any]], audit: Mapping[str, Any],
) -> dict[str, Any]:
    if audit.get("real_gpu_measurements") is not True or audit.get("fake_timing") is not False:
        raise ValueError("schema10 aggregation requires real GPU evidence")
    if audit.get("failed") != 0 or audit.get("completed") != audit.get("planned") or len(rows) != audit.get("completed"):
        raise ValueError("schema10 Profile measurements are incomplete")
    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[str(row["kind"])].append(row)
    observations = [o for r in by_kind["selection_admission_sweep"] for o in r["measurement"]["observations"]]
    table, candidates = [], []
    if observations:
        checkpoints = SCHEMA10_MODEL_CHECKPOINTS[str(audit["model_key"])]
        table, thresholds = build_threshold_table(observations, checkpoints)
        candidates = build_selection_candidates(observations, checkpoints, thresholds, None)
    repair_rows = [o for r in by_kind["repair_policy_development_sweep"] for o in r["measurement"]["observations"]]
    repair = summarize_repair_evidence(repair_rows)
    failures = [
        "production_selector_validation_required",
        "disjoint_fit_validation_groups_required",
        "request_ttft_and_selection_event_accounting_required",
        "causal_pool_growth_and_actual_commit_events_required",
        "preregistered_task_quality_validation_required",
        "same_dispatch_gate1_paired_execution_required",
        "actual_final_commit_and_runtime_consistency_required",
    ]
    if repair["integrity_violations"] or repair["r1_failures"]:
        failures.append("repair_correctness_failed")
    if not repair["r1_coverage_complete"]:
        failures.append("actual_reuse_r1_coverage_incomplete")
    return {
        **{name: audit[name] for name in (
            "code_commit", "cacheblend_patch_sha256", "cacheblend_tree", "model_id",
            "model_revision", "tokenizer_hash", "gpu_uuid", "runtime_environment_hash",
            "server_lock_sha256", "config_sha256", "contract_sha256", "handoff_sha256",
            "development_partition_sha256", "development_case_manifest_sha256",
        ) if name in audit},
        "protocol_version": 8, "schema_version": 10,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
        "stage": "schema10_profile_diagnostics_aggregated",
        "evidence_scope": "diagnostic_measurements",
        "real_gpu_measurements": True, "fake_timing": False,
        "profile_freeze_events": [],
        "selector_analysis": {
            "kind": "in_sample_deep_residual_score_diagnostic",
            "not_production_selector_validation": True,
            "threshold_table": table, "candidates": candidates,
        },
        "repair_evidence": repair,
        "correctness_sentinel": [r["measurement"] for r in by_kind["correctness_sentinel"]],
        "runtime_microbenchmarks": {
            kind: [dict(r) for r in by_kind[kind]] for kind in (
                "reference_runtime", "factorized_selection", "factorized_transfer",
                "factorized_repair", "factorized_scheduler", "joint_anchor",
            )
        },
        "gate1_diagnostic_measurements": {
            "kind": "dense_vs_forced_reuse_not_gate1_ab",
            "policy_ab_verified": False,
            "rows": [r["measurement"] for r in by_kind["gate1_paired_ab"]],
        },
        "coverage_curves": None,
        "coverage_trace_kind": "unavailable_actual_causal_events_required",
        "dynamic_materialization_growth_certified": False,
        "final_consistency": {
            "passed": False, "selection_retuned": None,
            "operational_coverage_causal": None,
            "final_commit_gamma_violations": None, "runtime_cost_consistent": None,
        },
        "profile_freeze_allowed": False,
        "ready_for_schema10_runtime_qualification": False,
        "quality_tail_rate_1pct_certified": False,
        "gpu_runtime_qualified": False, "h1_h2_execution_allowed": False,
        "paper_evidence": False, "locked_test_accessed": False,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--runtime-audit", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = Path(args.results).resolve()
    audit = json.loads(Path(args.runtime_audit).resolve().read_text(encoding="utf-8"))
    development_path = Path(args.development_manifest).resolve()
    development = _jsonl(development_path)
    if len(development) != 90 or len({r["case_id"] for r in development}) != 90:
        raise ValueError("schema10 aggregation requires 90 unique development cases")
    claimed = audit.get("development_case_manifest_sha256")
    if claimed and claimed != sha256_file(development_path):
        raise ValueError("development manifest digest differs from runtime audit")
    payload = aggregate_diagnostics(_jsonl(results), audit)
    payload["results_sha256"] = sha256_file(results)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("diagnostic output must be a new file; preserve old evidence")
    atomic_write_json(output, payload)
    print(json.dumps({"output": str(output), "profile_freeze_allowed": False, "failures": payload["failures"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
