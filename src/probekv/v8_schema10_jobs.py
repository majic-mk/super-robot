from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .runtime_source_audit import audit_v8_schema10_runtime_sources
from .v8_schema10_contracts import schema10_no_gpu_gate
from .v8_schema10_profile import (
    SCHEMA10_MODEL_CHECKPOINTS,
    SCHEMA10_REPAIR_RATIO_GRID,
    SCHEMA10_TRIM_GRID,
)


def _job(
    kind: str,
    coordinates: Mapping[str, Any],
    *,
    phase: str = "profile",
    warmups: int = 0,
    repeats: int = 1,
) -> Dict[str, Any]:
    row = {
        "kind": kind,
        "coordinates": dict(coordinates),
        "paper_evidence": False,
        "locked_test_accessed": False,
        "requires_real_gpu": True,
        "phase": phase,
        "warmups": int(warmups),
        "repeats": int(repeats),
    }
    row["job_id"] = "schema10-%s-%s" % (
        kind,
        hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16],
    )
    return row


def build_schema10_profile_jobs(model_key: str) -> Tuple[Dict[str, Any], ...]:
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("schema10 supports only Mistral/Qwen")
    jobs = [
        _job(
            "correctness_sentinel",
            {
                "model": model_key,
                "native_prefix_cache": True,
                "completed_depth_hook": True,
                "r1_dense_equivalence": True,
                "source_and_artifact_digest": True,
                "repair_mask_integrity": True,
            },
            phase="correctness",
        ),
        _job(
            "reference_runtime",
            {
                "model": model_key,
                "repair_policy": "fixed_15",
                "gate1_mode": "explicit_barrier",
                "final_commit_gamma": 0.8,
            },
            phase="reference",
            warmups=2,
            repeats=5,
        )
    ]
    checkpoints = SCHEMA10_MODEL_CHECKPOINTS[model_key]
    # Six-case immutable shards keep a four-hour GPU session resumable without
    # treating a partially executed 90-case sweep as a successful job.
    for case_start in range(0, 90, 6):
        jobs.append(
            _job(
                "selection_admission_sweep",
                {
                    "model": model_key,
                    "source_residual_trim_ratio_grid": list(SCHEMA10_TRIM_GRID),
                    "completed_depths": list(checkpoints),
                    "candidate_coverage": "full_correctness_set",
                    "case_start": case_start,
                    "case_count": 6,
                    "reference_repair_policy": "fixed_15",
                    "reference_gate1_mode": "explicit_barrier",
                },
                phase="selection_admission",
            )
        )
    # Selection cost is part of the Stage-A hard Gate, so these cells must be
    # complete before any Stage-B repair sweep starts.
    for compared_k in (1, 2, 4, 8, 16):
        for token_count in (128, 512, 640):
            for completed_depth in checkpoints:
                jobs.append(
                    _job(
                        "factorized_selection",
                        {
                            "model": model_key,
                            "compared_k": compared_k,
                            "token_count": token_count,
                            "completed_depth": completed_depth,
                            "backing_tier": "pinned_cpu",
                        },
                        phase="selection_admission",
                        warmups=2,
                        repeats=5,
                    )
                )
    for policy in (
        "single_variant_no_growth",
        "complete_mismatch_growth_only",
        "schema10_controlled_exploration",
        "deep_oracle_growth",
    ):
        jobs.append(
            _job(
                "variant_growth_trace",
                {"model": model_key, "policy": policy, "initial_k": 1, "max_k": 16},
                phase="selection_admission",
            )
        )
    jobs.append(
        _job(
            "probation_lifecycle",
            {
                "model": model_key,
                "verification_comparisons": 2,
                "grace_lookup_opportunities": 2,
                "grace_capacity_per_content": 2,
            },
            phase="selection_admission",
        )
    )
    jobs.append(
        _job(
            "gate1_counterfactual_shadow",
            {"model": model_key, "cases": 90, "side_effect_free": True},
            phase="runtime_repair_preparation",
        )
    )
    for case_start in range(0, 90, 6):
        jobs.append(
            _job(
                "repair_policy_development_sweep",
                {
                    "model": model_key,
                    "case_start": case_start,
                    "case_count": 6,
                    "policies": [
                        "fixed_15",
                        "static_gradual",
                        "load_recompute_aware_uniform",
                    ],
                    "repair_ratio_grid": list(SCHEMA10_REPAIR_RATIO_GRID),
                },
                phase="runtime_repair_preparation",
            )
        )
    for dataset in ("musique", "2wikimultihopqa", "hotpotqa"):
        for anchor_index in range(6):
            jobs.append(
                _job(
                    "gate1_paired_ab",
                    {
                        "model": model_key,
                        "dataset": dataset,
                        "anchor_index": anchor_index,
                        "modes": ["explicit_barrier", "bypassed_for_measurement"],
                    },
                    phase="runtime_repair_preparation",
                )
            )

    # Remaining factorized cost cells: no Cartesian product across categories.
    for bytes_count in (4 << 20, 16 << 20, 64 << 20):
        for path in ("pinned_cpu_to_gpu", "ssd_staged_to_gpu"):
            for concurrency in (1, 2, 4):
                jobs.append(
                    _job(
                        "factorized_transfer",
                        {
                            "model": model_key,
                            "bytes": bytes_count,
                            "path": path,
                            "concurrency": concurrency,
                        },
                        phase="runtime_repair_preparation",
                        warmups=2,
                        repeats=5,
                    )
                )
    for active_rows in (128, 512, 2560, 18944):
        for repair_ratio in SCHEMA10_REPAIR_RATIO_GRID:
            jobs.append(
                _job(
                    "factorized_repair",
                    {
                        "model": model_key,
                        "active_rows": active_rows,
                        "repair_ratio": repair_ratio,
                    },
                    phase="runtime_repair_preparation",
                    warmups=2,
                    repeats=5,
                )
            )
    for segment_count in (1, 2, 5, 10, 37):
        for concurrency in (1, 2, 4):
            jobs.append(
                _job(
                    "factorized_scheduler",
                    {
                        "model": model_key,
                        "segment_count": segment_count,
                        "concurrency": concurrency,
                        "ready_state": "layerwise",
                    },
                    phase="runtime_repair_preparation",
                    warmups=2,
                    repeats=5,
                )
            )
    for segment_count in (1, 5, 37):
        for path in ("gpu_resident", "pinned_cpu_to_gpu", "ssd_staged_to_gpu"):
            for repair_ratio in (0.15, 0.30):
                anchor = segment_count == 5 and path == "pinned_cpu_to_gpu" and repair_ratio == 0.15
                jobs.append(
                    _job(
                        "joint_anchor",
                        {
                            "model": model_key,
                            "segment_count": segment_count,
                            "path": path,
                            "repair_ratio": repair_ratio,
                            "concurrency": 1,
                        },
                        phase="runtime_repair_preparation",
                        warmups=20 if anchor else 2,
                        repeats=100 if anchor else 5,
                    )
                )
    for segment_count in (5, 37):
        for path in ("pinned_cpu_to_gpu", "ssd_staged_to_gpu"):
            for concurrency in (2, 4):
                jobs.append(
                    _job(
                        "joint_anchor",
                        {
                            "model": model_key,
                            "segment_count": segment_count,
                            "path": path,
                            "repair_ratio": 0.15,
                            "concurrency": concurrency,
                        },
                        phase="runtime_repair_preparation",
                        warmups=2,
                        repeats=5,
                    )
                )
    jobs.append(
        _job(
            "final_profile_consistency",
            {
                "model": model_key,
                "retune_on_failure": False,
                "required_profile_count": 5,
            },
            phase="final_consistency",
        )
    )
    if len({row["job_id"] for row in jobs}) != len(jobs):
        raise RuntimeError("schema10 Profile jobs are not unique")
    return tuple(jobs)


def build_schema10_qualification_jobs(
    *,
    model_key: str,
    selection_depth_profile_sha256: str,
    variant_admission_profile_sha256: str,
    preparation_policy_profile_sha256: str,
    repair_policy_profile_sha256: str,
    runtime_cost_profile_sha256: str,
) -> Tuple[Dict[str, Any], ...]:
    profile_shas = (
        selection_depth_profile_sha256,
        variant_admission_profile_sha256,
        preparation_policy_profile_sha256,
        repair_policy_profile_sha256,
        runtime_cost_profile_sha256,
    )
    if model_key not in {"mistral", "qwen"} or any(
        len(value) != 64 for value in profile_shas
    ):
        raise ValueError("schema10 qualification provenance is incomplete")
    candidates = []
    for segment_count in (1, 2, 5, 10, 37):
        rows = []
        for compared_k, eligible_k in ((1, 1), (1, 16), (4, 16), (16, 16)):
            for outcome in (
                "absolute_compatible_reuse",
                "complete_mismatch_materialize",
                "truncated_exploration_materialize",
                "truncated_exploration_quota_reject",
            ):
                for gate1_mode in ("explicit_barrier", "fused_advisory"):
                    for tier in ("gpu", "pinned_cpu", "ssd_staged"):
                        rows.append(
                            {
                                "model": model_key,
                                "segment_count": segment_count,
                                "compared_k": compared_k,
                                "correctness_eligible_k": eligible_k,
                                "outcome": outcome,
                                "gate1_mode": gate1_mode,
                                "source_tier": tier,
                                "selection_depth_profile_sha256": profile_shas[0],
                                "variant_admission_profile_sha256": profile_shas[1],
                                "preparation_policy_profile_sha256": profile_shas[2],
                                "repair_policy_profile_sha256": profile_shas[3],
                                "runtime_cost_profile_sha256": profile_shas[4],
                            }
                        )
        rows.sort(
            key=lambda row: hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        candidates.extend(rows[:28])
    jobs = tuple(
        _job("schema10_runtime_qualification", {"qualification_index": i, **row})
        for i, row in enumerate(candidates)
    )
    if len(jobs) != 140 or len({row["job_id"] for row in jobs}) != 140:
        raise RuntimeError("schema10 qualification matrix must contain 140 unique jobs")
    return jobs


def build_schema10_h1_h5_manifests(model_key: str) -> Mapping[str, object]:
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("schema10 supports only Mistral/Qwen")
    common = {"model": model_key, "paper_evidence": False, "locked_test_accessed": False}
    return {
        "H1": {
            **common,
            "purpose": "variant_necessity_and_growth",
            "growth_policies": [
                "single_variant_no_growth",
                "complete_mismatch_growth_only",
                "schema10_controlled_exploration",
                "deep_oracle_growth",
            ],
            "metrics": [
                "stored_pool_mismatch_precision",
                "exploration_yield_at_32",
                "useful_materialization_precision_at_32",
                "mean_variant_count",
                "saturation_probability_k16",
                "write_amplification_bytes",
                "mean_first_selection_delay_requests",
                "miss_to_reuse_conversion_rate",
                "mean_marginal_reuse_admission_improvement_ms",
                "replacement_frequency",
                "warmup_mean_ttft_ms",
                "steady_state_mean_ttft_ms",
                "compatible_coverage_k",
                "selected_coverage_k",
                "commit_coverage_k",
                "marginal_coverage_gain_k",
                "operational_near_saturation_k",
                "oracle_near_saturation_k",
            ],
        },
        "H2": {
            **common,
            "purpose": "source_selection_profile_freeze",
            "source_residual_trim_ratio_grid": list(SCHEMA10_TRIM_GRID),
            "policies": ["d1_only", "d1_d2_rescue", "legacy_multicheckpoint", "deep_full_candidate_oracle"],
            "metrics": ["selection_coverage", "wrong_early_lock", "normalized_regret", "cfo_shortlist_recall", "budget_truncation_regret"],
            "outputs": ["SelectionDepthProfile", "VariantAdmissionProfile"],
        },
        "H3": {
            **common,
            "purpose": "repair_and_io_balance",
            "policies": ["fixed_15", "static_gradual", "load_recompute_aware_uniform"],
            "repair_ratio_grid": list(SCHEMA10_REPAIR_RATIO_GRID),
            "correctness_endpoint": "r1_dense_equivalence",
        },
        "H4": {
            **common,
            "purpose": "runtime_storage_concurrency_and_gate1_counterfactual",
            "metrics": [
                "cpu_ssd_lru",
                "per_content_variant_lru",
                "probation_verified_expired_counts",
                "pool_saturation",
                "replacement_churn",
                "gate1_additional_wasted_preparation_ms",
                "gate1_additional_transferred_bytes",
                "ttft",
                "throughput",
            ],
        },
        "H5": {
            **common,
            "purpose": "locked_end_to_end_after_h1_h4",
            "baselines": ["full", "native_prefix_cache", "single_variant_reuse", "cacheblend", "cachecraft_cfo", "probekv_schema10", "oracle"],
        },
    }


def build_schema10_no_gpu_handoff(
    *,
    code_commit: str,
    model_key: str,
    model_revision: str,
    tokenizer_hash: str,
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
    config_sha256: str,
    contract_sha256: str,
    server_lock_sha256: str,
    development_partition_sha256: str,
    repo_root: str = "",
) -> Mapping[str, object]:
    if not all((code_commit, model_revision, tokenizer_hash, cacheblend_patch_sha256, cacheblend_tree, config_sha256, contract_sha256, server_lock_sha256, development_partition_sha256)):
        raise ValueError("schema10 handoff provenance is incomplete")
    profile_jobs = build_schema10_profile_jobs(model_key)
    hypotheses = build_schema10_h1_h5_manifests(model_key)
    audit = audit_v8_schema10_runtime_sources(
        Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
    )
    payload = {
        "protocol_version": 8,
        "schema_version": 10,
        "runtime_patch_mode": "probekv_v8_variant_growth_counterfactual",
        "code_commit": code_commit,
        "model_key": model_key,
        "model_revision": model_revision,
        "tokenizer_hash": tokenizer_hash,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "cacheblend_tree": cacheblend_tree,
        "config_sha256": config_sha256,
        "contract_sha256": contract_sha256,
        "server_lock_sha256": server_lock_sha256,
        "development_partition_sha256": development_partition_sha256,
        "profile_jobs": list(profile_jobs),
        "h1_h5_manifests": hypotheses,
        "profiles_frozen": False,
        "gpu_jobs_started": False,
        "runtime_source_audit": audit,
        **schema10_no_gpu_gate(artifact_preparation_ready=audit.get("runtime_source_ready") is True),
    }
    payload["jobs_sha256"] = hashlib.sha256(
        json.dumps({"profile_jobs": profile_jobs, "h1_h5": hypotheses}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload
