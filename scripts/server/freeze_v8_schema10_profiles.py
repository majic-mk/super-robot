#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.v8_schema10_contracts import AbsoluteResidualThreshold, Gate1Mode
from probekv.v8_schema10_preparation import (
    PreparationCostObservation,
    evaluate_gate1_counterfactual,
)
from probekv.v8_schema10_profile import (
    PreparationPolicyProfile,
    build_preparation_policy_profile,
    build_variant_admission_profile_v10,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--cacheblend-patch-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-hash", required=True)
    parser.add_argument("--runtime-policy", required=True)
    parser.add_argument("--development-partition-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.measurements).read_text(encoding="utf-8"))
    if payload.get("protocol_version") != 8 or payload.get("schema_version") != 10:
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
    }
    mismatched = [
        name
        for name, expected in expected_provenance.items()
        if payload.get(name) != expected
    ]
    if mismatched:
        raise ValueError(
            "schema10 measurement provenance differs: " + ",".join(mismatched)
        )
    trim_ratio = float(payload["selected_source_residual_trim_ratio"])
    thresholds = tuple(
        AbsoluteResidualThreshold(int(row["completed_depth"]), float(row["upper_residual"]))
        for row in payload["absolute_residual_thresholds"]
    )
    variant = build_variant_admission_profile_v10(
        code_commit=args.code_commit,
        cacheblend_patch_sha256=args.cacheblend_patch_sha256,
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_hash=args.tokenizer_hash,
        source_residual_trim_ratio=trim_ratio,
        thresholds=thresholds,
        materialization_budget_fraction=float(payload.get("materialization_budget_fraction", 0.02)),
        replacement_policy=str(
            payload.get("replacement_policy", "value_density_v1_full_scope_only")
        ),
        replacement_budget_fraction=float(
            payload.get("replacement_budget_fraction", 0.01)
        ),
        exploration_quota_per_content=int(payload.get("exploration_quota_per_content", 2)),
        probation_comparison_observations=2,
        probation_lookup_opportunities=2,
        max_protected_probation_per_content=2,
        development_partition_sha256=args.development_partition_sha256,
        frozen=True,
    )
    provisional_preparation = PreparationPolicyProfile(
        code_commit=args.code_commit,
        model_id=args.model_id,
        runtime_policy=args.runtime_policy,
        gate1_mode=Gate1Mode.EXPLICIT_BARRIER,
    )
    observations = tuple(
        PreparationCostObservation(**row)
        for row in payload["gate1_counterfactual_observations"]
    )
    summary = evaluate_gate1_counterfactual(
        observations,
        total_winner_full_kv_bytes_with_gate1=int(payload["total_winner_full_kv_bytes_with_gate1"]),
        profile=provisional_preparation,
    )
    preparation = build_preparation_policy_profile(
        code_commit=args.code_commit,
        model_id=args.model_id,
        runtime_policy=args.runtime_policy,
        gate1_mode=summary.recommended_gate1_mode,
        development_partition_sha256=args.development_partition_sha256,
        frozen=True,
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "variant_admission_profile.json").write_text(
        json.dumps(variant.payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "preparation_policy_profile.json").write_text(
        json.dumps(preparation.payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "gate1_counterfactual_summary.json").write_text(
        json.dumps({**summary.__dict__, "recommended_gate1_mode": summary.recommended_gate1_mode.value}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"variant_admission_profile_sha256": variant.profile_sha256, "preparation_policy_profile_sha256": preparation.profile_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
