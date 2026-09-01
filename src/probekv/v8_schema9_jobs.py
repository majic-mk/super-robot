from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Tuple

from .v8_schema9_contracts import schema9_no_gpu_gate
from .runtime_source_audit import audit_v8_schema9_runtime_sources


def _job(kind: str, coordinates: Mapping[str, Any]) -> Dict[str, Any]:
    row = {
        "kind": kind,
        "coordinates": dict(coordinates),
        "paper_evidence": False,
        "locked_test_accessed": False,
        "requires_real_gpu": True,
    }
    row["job_id"] = "schema9-%s-%s" % (
        kind,
        hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16],
    )
    return row


def build_schema9_profile_jobs(model_key: str) -> Tuple[Dict[str, Any], ...]:
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("schema9 supports only Mistral/Qwen")
    jobs = []
    for trim_ratio in (0.0, 0.10, 0.15, 0.30, 0.40):
        for completed_depth in (1, 2):
            jobs.append(
                _job(
                    "absolute_residual_threshold_freeze",
                    {
                        "model": model_key,
                        "source_score_trim_ratio": trim_ratio,
                        "completed_depth": completed_depth,
                        "candidate_coverage": "full_correctness_set",
                    },
                )
            )
    jobs.extend(
        (
            _job(
                "variant_growth_warmup",
                {"model": model_key, "max_variants": 16},
            ),
            _job(
                "materialization_budget",
                {"model": model_key, "budget_fraction": 0.02},
            ),
            _job(
                "variant_replacement",
                {
                    "model": model_key,
                    "policy": "value_density_then_lru",
                    "busy_protected": True,
                },
            ),
        )
    )
    return tuple(jobs)


def build_schema9_qualification_jobs(
    *,
    model_key: str,
    selection_depth_profile_sha256: str,
    variant_admission_profile_sha256: str,
    repair_policy_profile_sha256: str,
    runtime_cost_profile_sha256: str,
) -> Tuple[Dict[str, Any], ...]:
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("schema9 supports only Mistral/Qwen")
    profile_shas = (
        selection_depth_profile_sha256,
        variant_admission_profile_sha256,
        repair_policy_profile_sha256,
        runtime_cost_profile_sha256,
    )
    if any(len(value) != 64 for value in profile_shas):
        raise ValueError("schema9 qualification requires four Profile SHAs")
    patterns = []
    for segment_count in (1, 2, 5, 10, 37):
        segment_rows = []
        for compared_k in (1, 4, 16):
            for existing_variants in (0, 1, 4, 16):
                for outcome in (
                    "absolute_compatible_reuse",
                    "complete_mismatch_materialize",
                    "truncated_mismatch_dense_no_materialize",
                ):
                    for tier in ("gpu", "pinned_cpu", "ssd_staged"):
                        segment_rows.append(
                            {
                                "model": model_key,
                                "segment_count": segment_count,
                                "compared_k": compared_k,
                                "existing_variants": existing_variants,
                                "outcome": outcome,
                                "source_tier": tier,
                                "selection_depth_profile_sha256": profile_shas[0],
                                "variant_admission_profile_sha256": profile_shas[1],
                                "repair_policy_profile_sha256": profile_shas[2],
                                "runtime_cost_profile_sha256": profile_shas[3],
                            }
                        )
        segment_rows.sort(
            key=lambda row: hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        patterns.extend(segment_rows[:28])
    jobs = tuple(
        _job(
            "schema9_runtime_qualification",
            {"qualification_index": index, **coordinates},
        )
        for index, coordinates in enumerate(patterns)
    )
    if len(jobs) != 140 or len({row["job_id"] for row in jobs}) != 140:
        raise RuntimeError("schema9 qualification matrix must contain 140 unique jobs")
    return jobs


def build_schema9_h1_h5_manifests(model_key: str) -> Mapping[str, object]:
    if model_key not in {"mistral", "qwen"}:
        raise ValueError("schema9 supports only Mistral/Qwen")
    common = {
        "model": model_key,
        "paper_evidence": False,
        "locked_test_accessed": False,
    }
    return {
        "H1": {
            **common,
            "purpose": "source_opportunity_and_online_variant_growth",
            "metrics": [
                "source_residual_spread",
                "variant_count_by_request_epoch",
                "content_miss_to_reuse_conversion",
                "warmup_cost_ms",
                "r1_dense_equivalence",
            ],
        },
        "H2": {
            **common,
            "purpose": "trim_threshold_depth_and_dispatch_freeze",
            "trim_ratio_grid": [0.0, 0.10, 0.15, 0.30, 0.40],
            "policies": [
                "d1_only",
                "d1_d2_rescue",
                "legacy_multicheckpoint",
                "deep_full_candidate_oracle",
            ],
            "outputs": ["SelectionDepthProfile", "VariantAdmissionProfile"],
        },
        "H3": {
            **common,
            "purpose": "repair_quality_at_certified_absolute_admission",
            "policies": [
                "fixed_15",
                "static_gradual",
                "load_recompute_aware_uniform",
            ],
        },
        "H4": {
            **common,
            "purpose": "variant_churn_storage_and_concurrency",
            "metrics": [
                "write_amplification_bytes",
                "variant_evictions",
                "busy_replacement_rejections",
                "cpu_ssd_migrations",
                "warmup_to_steady_state_ttft",
            ],
        },
        "H5": {
            **common,
            "purpose": "locked_end_to_end_after_all_prior_gates",
            "locked_test_accessed": False,
            "baselines": [
                "full",
                "prefix_cache",
                "cacheblend",
                "cachecraft_variant_baseline",
                "single_variant_probekv",
                "multi_variant_without_absolute_threshold",
                "probekv_absolute_admission",
                "oracle_admission",
            ],
        },
    }


def build_schema9_no_gpu_handoff(
    *,
    code_commit: str,
    model_key: str,
    model_revision: str,
    tokenizer_hash: str,
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
    config_sha256: str,
    contract_sha256: str,
    repo_root: str = "",
) -> Mapping[str, object]:
    if not all(
        (
            code_commit,
            model_revision,
            tokenizer_hash,
            cacheblend_patch_sha256,
            cacheblend_tree,
            config_sha256,
            contract_sha256,
        )
    ):
        raise ValueError("schema9 handoff provenance is incomplete")
    profile_jobs = build_schema9_profile_jobs(model_key)
    hypotheses = build_schema9_h1_h5_manifests(model_key)
    from pathlib import Path

    audit = audit_v8_schema9_runtime_sources(
        Path(repo_root).resolve()
        if repo_root
        else Path(__file__).resolve().parents[2]
    )
    payload = {
        "protocol_version": 8,
        "schema_version": 9,
        "runtime_patch_mode": "probekv_v8_absolute_residual_variant_admission",
        "code_commit": code_commit,
        "model_key": model_key,
        "model_revision": model_revision,
        "tokenizer_hash": tokenizer_hash,
        "cacheblend_patch_sha256": cacheblend_patch_sha256,
        "cacheblend_tree": cacheblend_tree,
        "config_sha256": config_sha256,
        "contract_sha256": contract_sha256,
        "profile_jobs": list(profile_jobs),
        "h1_h5_manifests": hypotheses,
        "profiles_frozen": False,
        "gpu_jobs_started": False,
        "runtime_source_audit": audit,
        **schema9_no_gpu_gate(
            artifact_preparation_ready=audit.get("runtime_source_ready") is True
        ),
    }
    payload["jobs_sha256"] = hashlib.sha256(
        json.dumps(
            {"profile_jobs": profile_jobs, "h1_h5": hypotheses},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload
