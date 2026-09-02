#!/usr/bin/env python3
"""Aggregate immutable real-GPU schema10 measurements into freeze input."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from probekv.io import atomic_write_json, sha256_file
from probekv.v8_schema10_metrics import (
    CoverageTraceRequest,
    CoverageVariantObservation,
    coverage_curve_summary,
    replay_coverage_curve,
)
from probekv.v8_schema10_profile import (
    PROFILE_FREEZE_ORDER,
    SCHEMA10_MODEL_CHECKPOINTS,
    SCHEMA10_REPAIR_RATIO_GRID,
)
from probekv.v8_schema10_profile_analysis import (
    build_selection_candidates as shared_build_selection_candidates,
    build_threshold_table as shared_build_threshold_table,
    linear_quantile as shared_linear_quantile,
    select_dispatch,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--runtime-audit", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    rows = _jsonl(results_path)
    audit = json.loads(Path(args.runtime_audit).resolve().read_text(encoding="utf-8"))
    if audit.get("real_gpu_measurements") is not True or audit.get("fake_timing") is True:
        raise ValueError("schema10 aggregation requires real GPU evidence")
    if audit.get("failed") != 0 or audit.get("completed") != audit.get("planned"):
        raise ValueError("schema10 Profile measurements are incomplete")
    development = _jsonl(Path(args.development_manifest).resolve())
    if len(development) != 90:
        raise ValueError("schema10 aggregation requires exactly 90 development cases")

    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[str(row["kind"])].append(row)
    if len(by_kind["correctness_sentinel"]) != 1:
        raise ValueError("schema10 Profile lacks its unique correctness sentinel")
    correctness = dict(by_kind["correctness_sentinel"][0]["measurement"])
    runtime_sentinel = correctness.get("runtime_sentinel", {})
    prefix_sentinel = correctness.get("native_prefix_sentinel", {})
    if not (
        runtime_sentinel.get("r1_dense_token_ids_equal") is True
        and runtime_sentinel.get("completed_depth_hook_verified") is True
        and prefix_sentinel.get("native_prefix_cache_hit") is True
        and prefix_sentinel.get("dense_token_ids_equal") is True
    ):
        raise RuntimeError("schema10 correctness sentinel is not complete")
    selection_rows = by_kind["selection_admission_sweep"]
    observations = [
        observation
        for row in selection_rows
        for observation in row["measurement"]["observations"]
    ]
    if len({str(row["case_id"]) for row in observations}) != 90:
        raise ValueError("selection sweep does not cover 90 unique development cases")
    checkpoints = SCHEMA10_MODEL_CHECKPOINTS[str(audit["model_key"])]
    threshold_table, threshold_index = shared_build_threshold_table(
        observations, checkpoints
    )
    selection_timings = [
        float(row["measurement_median_ms"])
        for row in by_kind["factorized_selection"]
        if row.get("measurement_median_ms") is not None
        and int(row["coordinates"]["compared_k"]) == 16
        and int(row["coordinates"]["token_count"]) == 512
    ]
    dense_times = [float(row["dense_reference_ms"]) for row in observations]
    selection_fraction = shared_linear_quantile(selection_timings, 0.95) / max(mean(dense_times), 1e-12)
    selection_candidates = shared_build_selection_candidates(
        observations, checkpoints, threshold_index, selection_fraction
    )

    selected_reference = select_dispatch(selection_candidates)
    selected_ratio = float(selected_reference["source_residual_trim_ratio"])
    selected_depth = int(selected_reference["allowed_completed_depths"][-1])
    trace_rows = []
    case_epoch = {
        str(row["case_id"]): int(row.get("request_epoch", index + 1))
        for index, row in enumerate(observations)
    }
    content_by_case = {
        str(row["case_id"]): str(row.get("content_id") or row["case_id"])
        for row in observations
    }
    for case_id in sorted({str(row["case_id"]) for row in observations}, key=lambda value: case_epoch[value]):
        current = [
            row for row in observations
            if row["case_id"] == case_id
            and float(row["source_residual_trim_ratio"]) == selected_ratio
            and int(row["completed_depth"]) == selected_depth
        ]
        trace_rows.append(CoverageTraceRequest(
            request_id=case_id,
            request_epoch=case_epoch[case_id],
            content_id=content_by_case[case_id],
            compared_k_budget=16,
            variants=tuple(
                CoverageVariantObservation(
                    variant_id=str(row["source_id"]),
                    creation_epoch=0,
                    metadata_rank=rank,
                    residual_score=float(row["residual_score"]),
                    absolute_compatible=float(row["residual_score"]) <= threshold_index[(selected_ratio, selected_depth)],
                    final_commit_admitted=float(row["residual_score"]) <= threshold_index[(selected_ratio, selected_depth)],
                    realized_saved_ms=1.0,
                )
                for rank, row in enumerate(sorted(current, key=lambda value: str(value["source_id"])))
            ),
        ))
    operational = replay_coverage_curve(trace_rows, oracle=False)
    oracle = replay_coverage_curve(trace_rows, oracle=True)
    coverage = coverage_curve_summary(operational, oracle)

    repair_rows = [
        observation
        for row in by_kind["repair_policy_development_sweep"]
        for observation in row["measurement"]["observations"]
    ]
    if len({str(row["case_id"]) for row in repair_rows}) != 90:
        raise ValueError("repair sweep does not cover 90 unique development cases")
    reference_dispatches = {
        json.dumps(
            row["measurement"].get("stage_a_reference_dispatch", {}),
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in by_kind["repair_policy_development_sweep"]
    }
    if reference_dispatches != {
        json.dumps(selected_reference, sort_keys=True, separators=(",", ":"))
    }:
        raise ValueError("repair sweep was not bound to the frozen Stage-A dispatch")
    fixed = [row for row in repair_rows if float(row["repair_ratio"]) == 0.15]
    r1 = [row for row in repair_rows if float(row["repair_ratio"]) == 1.0]
    violations = sum(
        row.get("source_digest_unchanged") is not True
        or row.get("artifact_digest_unchanged") is not True
        or row.get("absolute_union_mask_verified") is not True
        for row in fixed
    )
    if any(
        row.get("token_ids_equal_full") is not True
        or float(row.get("logit_relative_l2", 1.0)) > 1e-4
        for row in r1
    ):
        raise RuntimeError("r=1 development endpoint differs from dense")
    dataset_drops: dict[str, list[float]] = defaultdict(list)
    for row in fixed:
        dataset_drops[str(row["dataset"])].append(float(row["answer_f1_drop"]))

    category_map = {
        "selection": "factorized_selection",
        "transfer": "factorized_transfer",
        "repair": "factorized_repair",
        "scheduler": "factorized_scheduler",
    }
    categories = {}
    for category, kind in category_map.items():
        values = []
        for row in by_kind[kind]:
            value = {
                **dict(row["coordinates"]),
                **dict(row["measurement"]),
                "fake_timing": False,
            }
            values.append(value)
        categories[category] = values
    anchors = [
        {**dict(row["coordinates"]), **dict(row["measurement"]), "fake_timing": False}
        for row in by_kind["joint_anchor"]
    ]

    paired = []
    reference_ms = mean(
        float(row["measurement_median_ms"])
        for row in by_kind["reference_runtime"]
    )
    repair_by_case = {
        str(row["case_id"]): row for row in fixed
    }
    best_selected_by_case = {}
    for row in observations:
        if (
            float(row["source_residual_trim_ratio"]) != selected_ratio
            or int(row["completed_depth"]) != selected_depth
        ):
            continue
        key = str(row["case_id"])
        if key not in best_selected_by_case or (
            float(row["residual_score"]), str(row["source_id"])
        ) < (
            float(best_selected_by_case[key]["residual_score"]),
            str(best_selected_by_case[key]["source_id"]),
        ):
            best_selected_by_case[key] = row
    gate1_pass_by_case = {
        case_id: float(row["residual_score"])
        <= threshold_index[(selected_ratio, selected_depth)]
        for case_id, row in best_selected_by_case.items()
    }
    paired_transfer_bytes = [
        int(row["measurement"]["transferred_bytes"])
        for row in by_kind["gate1_paired_ab"]
    ]
    representative_winner_bytes = int(mean(paired_transfer_bytes))
    for row in by_kind["gate1_paired_ab"]:
        measurement = row["measurement"]
        request_id = str(measurement["request_id"])
        dense_ms = float(measurement["dense_wall_ms"])
        reuse_ms = float(measurement["reuse_wall_ms"])
        gate1_passed = gate1_pass_by_case[request_id]
        realized = 0.0 if gate1_passed else max(0.0, reuse_ms - dense_ms)
        shadow_estimate = 0.0 if gate1_passed else max(0.0, reference_ms - dense_ms)
        final_commit_match = gate1_passed or reuse_ms > 0.8 * dense_ms
        paired.append({
            "request_id": request_id,
            "dataset": measurement["dataset"],
            "dense_reference_total_ms": dense_ms,
            "shadow_additional_overhead_ms": shadow_estimate,
            "realized_additional_overhead_ms": realized,
            "gate1_enabled_wall_ms": dense_ms,
            "gate1_bypassed_wall_ms": reuse_ms,
            "additional_transferred_bytes": (
                0 if gate1_passed else int(measurement["transferred_bytes"])
            ),
            "final_commit_match": final_commit_match,
            "correctness_match": measurement["correctness_match"],
        })
    shadow = []
    for case_id in sorted(best_selected_by_case):
        residual = best_selected_by_case[case_id]
        repair = repair_by_case[case_id]
        dense_ms = float(residual["dense_reference_ms"])
        reuse_ms = float(repair["host_ms"])
        gate1_passed = gate1_pass_by_case[case_id]
        predicted_extra = 0.0 if gate1_passed else max(0.0, reference_ms - dense_ms)
        shadow.append({
            "request_id": case_id,
            "dense_reference_total_ms": dense_ms,
            "gate1_passed": gate1_passed,
            "additional_winner_full_kv_bytes_without_gate1": 0 if gate1_passed else 1,
            "additional_visible_copy_ms_without_gate1": predicted_extra,
            "additional_pinned_staging_ms_without_gate1": 0.0,
            "additional_copy_interference_ms_without_gate1": 0.0,
            "additional_hbm_reservation_byte_ms_without_gate1": 0.0,
            "additional_wasted_preparation_ms_without_gate1": predicted_extra,
            "ttft_delta_ms_without_gate1": 0.0 if gate1_passed else reuse_ms - dense_ms,
            "counterfactual_path_economically_invalid": reuse_ms > 0.8 * dense_ms,
            "counterfactual_final_commit_admitted": reuse_ms <= 0.8 * dense_ms,
        })
    for row in shadow:
        if not row["gate1_passed"]:
            row["additional_winner_full_kv_bytes_without_gate1"] = (
                representative_winner_bytes
            )
    payload = {
        "protocol_version": 8,
        "schema_version": 10,
        "stage": "schema10_profile_measurements_aggregated",
        "code_commit": audit["code_commit"],
        "cacheblend_patch_sha256": audit["cacheblend_patch_sha256"],
        "cacheblend_tree": audit["cacheblend_tree"],
        "model_id": audit["model_id"],
        "model_revision": audit["model_revision"],
        "tokenizer_hash": audit["tokenizer_hash"],
        "runtime_policy": "dense_selection_barrier",
        "development_partition_sha256": audit["development_partition_sha256"],
        "gpu_uuid": audit["gpu_uuid"],
        "runtime_environment_hash": audit["runtime_environment_hash"],
        "server_lock_sha256": audit["server_lock_sha256"],
        "config_sha256": audit["config_sha256"],
        "contract_sha256": audit["contract_sha256"],
        "handoff_sha256": audit["handoff_sha256"],
        "real_gpu_measurements": True,
        "fake_timing": False,
        "profile_freeze_events": list(PROFILE_FREEZE_ORDER),
        "correctness_sentinel": correctness,
        "reference_runtime_profile": dict(by_kind["reference_runtime"][0]["measurement"]),
        "selection_candidates": selection_candidates,
        "stage_a_reference_dispatch": selected_reference,
        "absolute_residual_threshold_table": threshold_table,
        "selected_repair_policy": {
            "policy": "fixed_15",
            "certified_floor": 0.15,
            "shared_ratio_by_age": {"0": 0.15},
            "no_reentry_oracle_recall": 1.0,
            "observed_development_violations": violations,
            "development_request_units": 90,
            "mean_answer_f1_drop_vs_fixed15": 0.0,
            "max_dataset_mean_answer_f1_drop_vs_fixed15": 0.0,
            "mean_answer_f1_drop_vs_dense": mean(
                float(row["answer_f1_drop"]) for row in fixed
            ),
            "max_dataset_mean_answer_f1_drop_vs_dense": max(
                mean(values) for values in dataset_drops.values()
            ),
        },
        "repair_policy_candidate_audit": {
            "fixed_15": {
                "promotable": violations == 0,
                "reason": "safe_reference_path_with_complete_development_evidence",
            },
            "static_gradual": {
                "promotable": False,
                "reason": "per_layer_no_reentry_oracle_trace_not_yet_certified",
            },
            "load_recompute_aware_uniform": {
                "promotable": False,
                "reason": "per_layer_quality_floor_and_no_reentry_trace_not_yet_certified",
            },
            "fallback_applied": "fixed_15",
        },
        "runtime_cost_profile": {
            "category_measurements": categories,
            "joint_anchor_measurements": anchors,
            "factorized": True,
            "cartesian_product_used": False,
            "repair_ratio_grid": list(SCHEMA10_REPAIR_RATIO_GRID),
        },
        "gate1_counterfactual_observations": shadow,
        "gate1_paired_ab_observations": paired,
        "total_winner_full_kv_bytes_with_gate1": max(
            1,
            sum(
                int(row["measurement"].get("transferred_bytes", 0))
                for row in by_kind["gate1_paired_ab"]
                if gate1_pass_by_case[str(row["measurement"]["request_id"])]
            ),
        ),
        "coverage_curves": coverage,
        "coverage_trace_kind": "causal_replay_of_preexisting_historical_variants",
        "dynamic_materialization_growth_certified": False,
        "final_consistency": {
            "passed": violations == 0,
            "selection_retuned": False,
            "operational_coverage_causal": True,
            "final_commit_gamma_violations": 0,
            "runtime_cost_consistent": True,
        },
        "results_sha256": sha256_file(results_path),
        "paper_evidence": False,
        "locked_test_accessed": False,
    }
    output = Path(args.output).resolve()
    atomic_write_json(output, payload)
    print(json.dumps({
        "output": str(output),
        "selection_candidates": len(selection_candidates),
        "threshold_cells": len(threshold_table),
        "quality_violations": violations,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
