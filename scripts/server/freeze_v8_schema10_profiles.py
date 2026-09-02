#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from probekv.io import atomic_write_json
from probekv.v8_schema10_contracts import AbsoluteResidualThreshold, Gate1Mode
from probekv.v8_schema10_metrics import one_sided_clopper_pearson_upper
from probekv.v8_schema10_preparation import (
    Gate1PairedABObservation,
    PreparationCostObservation,
    evaluate_gate1_counterfactual,
)
from probekv.v8_schema10_profile import (
    AbsoluteResidualThresholdPointV10,
    PreparationPolicyProfile,
    SCHEMA10_MODEL_CHECKPOINTS,
    SCHEMA10_REPAIR_RATIO_GRID,
    SCHEMA10_TRIM_GRID,
    build_preparation_policy_profile,
    build_repair_policy_profile_v10,
    build_runtime_cost_profile_v10,
    build_selection_depth_profile_v10,
    build_variant_admission_profile_v10,
    validate_schema10_profile_freeze_order,
)
from probekv.v8_schema10_profile_analysis import select_dispatch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--cacheblend-patch-sha256", required=True)
    parser.add_argument("--model-key", required=True, choices=("mistral", "qwen"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-hash", required=True)
    parser.add_argument("--runtime-policy", default="dense_selection_barrier")
    parser.add_argument("--development-partition-sha256", required=True)
    parser.add_argument("--development-case-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    measurement_path = Path(args.measurements).resolve()
    payload = json.loads(measurement_path.read_text(encoding="utf-8"))
    if (payload.get("protocol_version"), payload.get("schema_version")) != (8, 10):
        raise ValueError("schema10 Profile freeze requires schema10 measurements")
    if payload.get("real_gpu_measurements") is not True or payload.get("fake_timing") is True:
        raise ValueError("schema10 Profiles require real GPU development evidence")
    expected_provenance = {
        "code_commit": args.code_commit,
        "cacheblend_patch_sha256": args.cacheblend_patch_sha256,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "tokenizer_hash": args.tokenizer_hash,
        "runtime_policy": args.runtime_policy,
        "development_partition_sha256": args.development_partition_sha256,
        "development_case_manifest_sha256": args.development_case_manifest_sha256,
    }
    mismatched = [
        name for name, expected in expected_provenance.items()
        if payload.get(name) != expected
    ]
    if mismatched:
        raise ValueError("schema10 measurement provenance differs: " + ",".join(mismatched))
    validate_schema10_profile_freeze_order(payload.get("profile_freeze_events", ()))
    consistency = payload.get("final_consistency", {})
    if consistency.get("passed") is not True or consistency.get("selection_retuned") is True:
        raise ValueError("final Profile consistency failed or retuned selection")

    measurement_sha = _sha256(measurement_path)
    gpu_uuid = str(payload.get("gpu_uuid", ""))
    if not gpu_uuid:
        raise ValueError("schema10 Profile measurements lack GPU UUID")
    selected = select_dispatch(list(payload["selection_candidates"]))
    if dict(payload.get("stage_a_reference_dispatch", {})) != dict(selected):
        raise ValueError("Stage-B evidence differs from the frozen Stage-A dispatch")
    selected_ratio = float(selected["source_residual_trim_ratio"])
    if selected_ratio not in SCHEMA10_TRIM_GRID:
        raise ValueError("selected trim ratio is outside the frozen grid")
    allowed_depths = tuple(int(value) for value in selected["allowed_completed_depths"])
    expected_depths = SCHEMA10_MODEL_CHECKPOINTS[args.model_key]
    if selected["dispatch"] == "legacy_multicheckpoint" and allowed_depths != expected_depths:
        raise ValueError("legacy dispatch lacks its complete checkpoint set")

    table = tuple(
        AbsoluteResidualThresholdPointV10(
            int(row["completed_depth"]),
            float(row["source_residual_trim_ratio"]),
            float(row["upper_residual"]),
        )
        for row in payload["absolute_residual_threshold_table"]
    )
    required_cells = {
        (ratio, depth) for ratio in SCHEMA10_TRIM_GRID for depth in expected_depths
    }
    if {(row.source_residual_trim_ratio, row.completed_depth) for row in table} != required_cells:
        raise ValueError("absolute threshold table is not complete")
    # The online fast selector keeps its historical d1/d2 value object. The
    # complete legacy-depth table is carried separately and queried through
    # VariantAdmissionProfile.threshold_for_depth().
    thresholds = tuple(
        AbsoluteResidualThreshold(
            depth,
            next(
                row.upper_residual for row in table
                if row.completed_depth == depth
                and row.source_residual_trim_ratio == selected_ratio
            ),
        )
        for depth in (1, 2)
    )

    selection = build_selection_depth_profile_v10(
        code_commit=args.code_commit,
        cacheblend_patch_sha256=args.cacheblend_patch_sha256,
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_hash=args.tokenizer_hash,
        selected_dispatch=str(selected["dispatch"]),
        allowed_completed_depths=allowed_depths,
        source_residual_trim_ratio=selected_ratio,
        metrics=dict(selected["metrics"]),
        development_partition_sha256=args.development_partition_sha256,
        development_case_manifest_sha256=args.development_case_manifest_sha256,
        measurement_sha256=measurement_sha,
        gpu_uuid=gpu_uuid,
        frozen=True,
    )
    variant = build_variant_admission_profile_v10(
        code_commit=args.code_commit,
        cacheblend_patch_sha256=args.cacheblend_patch_sha256,
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_hash=args.tokenizer_hash,
        source_residual_trim_ratio=selected_ratio,
        thresholds=thresholds,
        threshold_table=table,
        materialization_budget_fraction=float(payload.get("materialization_budget_fraction", 0.02)),
        replacement_policy="per_content_variant_lru_full_scope_only",
        replacement_budget_fraction=float(payload.get("replacement_budget_fraction", 0.01)),
        exploration_quota_per_content=2,
        probation_comparison_observations=2,
        probation_lookup_opportunities=2,
        max_protected_probation_per_content=2,
        development_partition_sha256=args.development_partition_sha256,
        development_case_manifest_sha256=args.development_case_manifest_sha256,
        frozen=True,
    )

    repair_row = dict(payload["selected_repair_policy"])
    repair_audit = dict(payload.get("repair_policy_candidate_audit", {}))
    if (
        repair_row.get("policy") == "fixed_15"
        and repair_audit.get("fallback_applied") != "fixed_15"
    ):
        raise ValueError("fixed15 fallback lacks an explicit candidate audit")
    units = int(repair_row["development_request_units"])
    violations = int(repair_row["observed_development_violations"])
    upper = one_sided_clopper_pearson_upper(violations, units)
    if units != 90 or violations != 0:
        raise ValueError("Profile repair freeze requires 0/90 development violations")
    if float(repair_row.get("mean_answer_f1_drop_vs_fixed15", 1.0)) > 0.01:
        raise ValueError("repair policy mean answer-F1 drop exceeds 0.01")
    if float(repair_row.get("max_dataset_mean_answer_f1_drop_vs_fixed15", 1.0)) > 0.02:
        raise ValueError("repair policy dataset answer-F1 drop exceeds 0.02")
    repair = build_repair_policy_profile_v10(
        code_commit=args.code_commit,
        cacheblend_patch_sha256=args.cacheblend_patch_sha256,
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_hash=args.tokenizer_hash,
        policy=str(repair_row["policy"]),
        certified_floor=float(repair_row["certified_floor"]),
        shared_ratio_by_age={
            int(key): float(value)
            for key, value in repair_row["shared_ratio_by_age"].items()
        },
        no_reentry_oracle_recall=float(repair_row["no_reentry_oracle_recall"]),
        observed_development_violations=violations,
        development_request_units=units,
        one_sided_95_upper_bound=upper,
        quality_tail_rate_1pct_certified=False,
        ratio_grid=SCHEMA10_REPAIR_RATIO_GRID,
        development_partition_sha256=args.development_partition_sha256,
        development_case_manifest_sha256=args.development_case_manifest_sha256,
        measurement_sha256=measurement_sha,
        gpu_uuid=gpu_uuid,
        frozen=True,
    )

    runtime_row = payload["runtime_cost_profile"]
    runtime = build_runtime_cost_profile_v10(
        code_commit=args.code_commit,
        cacheblend_patch_sha256=args.cacheblend_patch_sha256,
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_hash=args.tokenizer_hash,
        category_measurements={
            key: tuple(rows)
            for key, rows in runtime_row["category_measurements"].items()
        },
        joint_anchor_measurements=tuple(runtime_row["joint_anchor_measurements"]),
        factorized=True,
        cartesian_product_used=False,
        development_case_manifest_sha256=args.development_case_manifest_sha256,
        measurement_sha256=measurement_sha,
        gpu_uuid=gpu_uuid,
        frozen=True,
    )

    provisional = PreparationPolicyProfile(
        code_commit=args.code_commit,
        model_id=args.model_id,
        runtime_policy=args.runtime_policy,
        gate1_mode=Gate1Mode.EXPLICIT_BARRIER,
    )
    shadow_rows = tuple(
        PreparationCostObservation(**row)
        for row in payload["gate1_counterfactual_observations"]
    )
    paired_rows = tuple(
        Gate1PairedABObservation(**row)
        for row in payload["gate1_paired_ab_observations"]
    )
    summary = evaluate_gate1_counterfactual(
        shadow_rows,
        total_winner_full_kv_bytes_with_gate1=int(
            payload["total_winner_full_kv_bytes_with_gate1"]
        ),
        profile=provisional,
        paired_observations=paired_rows,
    )
    preparation = build_preparation_policy_profile(
        code_commit=args.code_commit,
        model_id=args.model_id,
        runtime_policy=args.runtime_policy,
        gate1_mode=summary.recommended_gate1_mode,
        paired_observations=summary.paired_observations,
        paired_mean_error_fraction=summary.paired_mean_absolute_error_fraction,
        paired_p95_error_fraction=summary.paired_p95_absolute_error_fraction,
        development_partition_sha256=args.development_partition_sha256,
        development_case_manifest_sha256=args.development_case_manifest_sha256,
        frozen=True,
    )

    profiles = {
        "selection_depth_profile.json": selection,
        "variant_admission_profile.json": variant,
        "repair_policy_profile.json": repair,
        "runtime_cost_profile.json": runtime,
        "preparation_policy_profile.json": preparation,
    }
    coverage = dict(payload.get("coverage_curves", {}))
    if not coverage or payload.get("coverage_trace_kind") != (
        "causal_replay_of_preexisting_historical_variants"
    ):
        raise ValueError("schema10 coverage replay provenance is incomplete")
    auxiliary = {
        "correctness_sentinel.json": payload["correctness_sentinel"],
        "reference_runtime_profile.json": payload["reference_runtime_profile"],
        "coverage_curves_operational.json": coverage["operational"],
        "coverage_curves_oracle.json": coverage["oracle"],
        "gate1_paired_ab.json": payload["gate1_paired_ab_observations"],
        "gate1_shadow_summary.json": {
            "observations": payload["gate1_counterfactual_observations"],
            "summary": {
                **summary.__dict__,
                "recommended_gate1_mode": summary.recommended_gate1_mode.value,
            },
        },
        "final_consistency_report.json": consistency,
        "repair_policy_candidate_audit.json": repair_audit,
    }
    summary_payload = {
        **summary.__dict__,
        "recommended_gate1_mode": summary.recommended_gate1_mode.value,
    }
    provenance_names = (
        "cacheblend_tree",
        "runtime_environment_hash",
        "server_lock_sha256",
        "config_sha256",
        "contract_sha256",
        "handoff_sha256",
        "development_case_manifest_sha256",
    )
    missing_provenance = [name for name in provenance_names if not payload.get(name)]
    if missing_provenance:
        raise ValueError(
            "schema10 Profile bundle provenance is incomplete: "
            + ",".join(missing_provenance)
        )
    bundle = {
        "protocol_version": 8,
        "schema_version": 10,
        "stage": "schema10_profile_bundle_frozen",
        "code_commit": args.code_commit,
        "model_id": args.model_id,
        "gpu_uuid": gpu_uuid,
        "model_revision": args.model_revision,
        "tokenizer_hash": args.tokenizer_hash,
        "cacheblend_patch_sha256": args.cacheblend_patch_sha256,
        **{name: payload[name] for name in provenance_names},
        "measurement_sha256": measurement_sha,
        "profile_freeze_order_verified": True,
        "operational_coverage_causal": (
            consistency.get("operational_coverage_causal") is True
        ),
        "real_cuda_timing": True,
        "fake_timing": False,
        "profiles": {
            "selection_depth_profile_sha256": selection.profile_sha256,
            "variant_admission_profile_sha256": variant.profile_sha256,
            "repair_policy_profile_sha256": repair.profile_sha256,
            "runtime_cost_profile_sha256": runtime.profile_sha256,
            "preparation_policy_profile_sha256": preparation.profile_sha256,
        },
        "final_consistency": dict(consistency),
        "ready_for_schema10_runtime_qualification": True,
        "runtime_qualification_jobs_per_model": 140,
        "quality_tail_rate_1pct_certified": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "failures": [],
    }
    bundle["profile_bundle_sha256"] = hashlib.sha256(
        json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    # Nothing is written until every Profile, coverage artifact, consistency
    # rule and provenance binding has validated.  Individual files use atomic
    # replacement, so a failed freeze cannot masquerade as a partial bundle.
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("schema10 frozen Profile output must be a new directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, profile in profiles.items():
        atomic_write_json(output / name, profile.payload())
    for name, value in auxiliary.items():
        atomic_write_json(output / name, value)
    atomic_write_json(output / "gate1_counterfactual_summary.json", summary_payload)
    atomic_write_json(output / "profile_bundle_manifest.json", bundle)
    print(json.dumps(bundle, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
