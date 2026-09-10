"""Read-only full-cohort d1/d2 observations and offline policy replay.

Not a runtime dispatch or a quality certificate. In particular, a full-cohort
shadow cannot measure the latency of a pruned, policy-conditioned execution.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math

from .source_policy_development import (
    REFERENCE_TRIM_CANDIDATES, audit_depth2_pruning, plan_depth2_shortlist,
    residual_tail_curve,
)
from .v8_schema10_execution import digest_json


KIND = "source_policy_full_cohort_observation_v1"
PROVENANCE_FIELDS = (
    "request_id", "segment_id", "model_signature", "tokenizer_signature",
    "request_token_ids_sha256", "source_inventory_digest", "code_commit",
    "patch_sha256", "config_sha256", "development_partition_digest",
)


def _seal(value, field):
    result = dict(value)
    result[field] = digest_json(result)
    return result


def _hash_tensor(tensor):
    import torch
    # Only the comparison K state, never a full KV Artifact.
    value = tensor.detach().contiguous().cpu()
    header = (str(value.dtype), list(value.shape))
    return hashlib.sha256(digest_json(header).encode("ascii") +
                          value.view(torch.uint8).numpy().tobytes()).hexdigest()


def observe_depth_k(current_k, source_k_by_id, *, completed_depth):
    """Diagnostic extraction from independent pre-RoPE SelectionState tensors.

    The caller owns the live K hook and full-cohort capture. This function does
    not read any Artifact, mutate the pool, freeze a Source or issue a KV copy.
    D2 must come from the same dense shadow trajectory as D1. Tensor geometry
    cannot establish RoPE semantics; the hook still requires its GPU sentinel.
    Host extraction/digests are diagnostic work, not an online fast path.
    """
    import torch
    if type(completed_depth) is not int or completed_depth not in (1, 2):
        raise ValueError("only completed depths 1 and 2 are valid")
    if (not isinstance(current_k, torch.Tensor) or current_k.ndim != 3
            or current_k.dtype != torch.bfloat16 or current_k.shape[0] < 2
            or any(n == 0 for n in current_k.shape)):
        raise ValueError("exact BF16 [token,kv_head,head_dim] current K required")
    if (not isinstance(source_k_by_id, dict) or not source_k_by_id
            or len(source_k_by_id) > 16
            or any(not isinstance(s, str) or not s for s in source_k_by_id)):
        raise ValueError("1..16 explicitly identified selection states required")
    if not bool(torch.isfinite(current_k).all().item()):
        raise ValueError("nonfinite current K")
    current = current_k.detach().float()
    denominator = current.square().sum((1, 2)).sqrt().clamp_min(1e-12)
    sources = {}
    for sid in sorted(source_k_by_id):
        source = source_k_by_id[sid]
        if (not isinstance(source, torch.Tensor) or source.shape != current_k.shape
                or source.dtype != current_k.dtype or source.device != current_k.device
                or not bool(torch.isfinite(source).all().item())):
            raise ValueError("selection state dtype/device/geometry mismatch")
        drift = (source.detach().float() - current).square().sum((1, 2)).sqrt() / denominator
        if not bool(torch.isfinite(drift).all().item()):
            raise ValueError("nonfinite normalized residual")
        sources[sid] = {"selection_k_digest": _hash_tensor(source),
                        "normalized_k_drifts": drift.cpu().tolist()}
    return {"completed_depth": completed_depth,
            "k_observation_layer_1based": completed_depth + 1,
            "current_k_digest": _hash_tensor(current_k),
            "geometry": list(current_k.shape), "dtype": "bfloat16",
            "sources": sources}


def build_observation(*, provenance, absolute_positions, correctness_eligible_source_ids,
                      depth_observations, evidence_origin="cpu_interface_test"):
    value = {"kind": KIND, "provenance": dict(provenance),
             "absolute_positions": list(absolute_positions),
             "correctness_eligible_source_ids": list(correctness_eligible_source_ids),
             "depth_observations": list(depth_observations),
             "evidence_origin": evidence_origin,
             "trajectory": "dense_full_cohort_shadow",
             "source_freeze_performed": False, "production_admission_allowed": False,
             "gpu_runtime_qualified": False, "paper_evidence": False,
             "locked_test_accessed": False}
    value = _seal(value, "observation_sha256")
    validate_observation(value)
    return value


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_observation(value):
    if not isinstance(value, dict) or value.get("kind") != KIND:
        raise ValueError("unsupported observation format")
    unsigned = {k: v for k, v in value.items() if k != "observation_sha256"}
    if not _digest(value.get("observation_sha256")) or digest_json(unsigned) != value["observation_sha256"]:
        raise ValueError("observation digest mismatch")
    if (value.get("trajectory") != "dense_full_cohort_shadow"
            or value.get("evidence_origin") not in ("cpu_interface_test", "native_hook_diagnostic")
            or any(value.get(k) is not False for k in (
                "source_freeze_performed", "production_admission_allowed",
                "gpu_runtime_qualified", "paper_evidence", "locked_test_accessed"))):
        raise ValueError("shadow observations cannot authorize execution or qualification")
    provenance = value.get("provenance", {})
    if (not isinstance(provenance, dict) or set(provenance) != set(PROVENANCE_FIELDS)
            or any(not isinstance(provenance[k], str) or not provenance[k] for k in PROVENANCE_FIELDS)):
        raise ValueError("complete request/model/partition/code provenance required")
    for key in PROVENANCE_FIELDS:
        if key.endswith(("sha256", "digest")) and not _digest(provenance[key]):
            raise ValueError("invalid provenance digest: " + key)
    if len(provenance["code_commit"]) != 40 or any(c not in "0123456789abcdef" for c in provenance["code_commit"]):
        raise ValueError("exact code commit required")
    positions = value.get("absolute_positions", [])
    if (not isinstance(positions, list) or len(positions) < 2
            or any(type(p) is not int or p < 0 for p in positions)
            or positions != sorted(set(positions))):
        raise ValueError("ordered unique absolute positions required")
    eligible = value.get("correctness_eligible_source_ids", [])
    if (not isinstance(eligible, list) or not 1 <= len(eligible) <= 16
            or any(not isinstance(s, str) or not s for s in eligible)
            or len(set(eligible)) != len(eligible)):
        raise ValueError("explicit immutable correctness-eligible inventory required")
    depths = value.get("depth_observations", [])
    if not isinstance(depths, list) or len(depths) != 2:
        raise ValueError("both full-cohort d1 and d2 observations required")
    cohorts, geometries = [], []
    for depth, record in enumerate(depths, 1):
        if (not isinstance(record, dict) or type(record.get("completed_depth")) is not int
                or record["completed_depth"] != depth
                or type(record.get("k_observation_layer_1based")) is not int
                or record["k_observation_layer_1based"] != depth + 1
                or record.get("dtype") != "bfloat16" or not _digest(record.get("current_k_digest"))):
            raise ValueError("depth/layer or K identity mismatch")
        geometry = record.get("geometry", [])
        if (not isinstance(geometry, list) or len(geometry) != 3
                or any(type(n) is not int or n <= 0 for n in geometry)
                or geometry[0] != len(positions)):
            raise ValueError("K geometry does not cover segment positions")
        geometries.append(geometry)
        sources = record.get("sources", {})
        if not isinstance(sources, dict) or not sources or not set(sources) <= set(eligible):
            raise ValueError("observed cohort must belong to eligible inventory")
        cohorts.append(set(sources))
        for state in sources.values():
            if not isinstance(state, dict) or not _digest(state.get("selection_k_digest")):
                raise ValueError("source selection state digest required")
            drifts = state.get("normalized_k_drifts", [])
            if (not isinstance(drifts, list) or len(drifts) != len(positions)
                    or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in drifts)):
                raise ValueError("finite per-token drift rows required")
    if cohorts[0] != cohorts[1] or geometries[0] != geometries[1]:
        raise ValueError("d2 shadow must cover exactly the entire d1 cohort/geometry")


def replay_observation(value):
    validate_observation(value)
    provenance = value["provenance"]
    positions = value["absolute_positions"]
    eligible_k = len(value["correctness_eligible_source_ids"])
    depths = value["depth_observations"]
    cells = []
    for rho in REFERENCE_TRIM_CANDIDATES:
        scores = [{s: residual_tail_curve(row["normalized_k_drifts"], positions,
                                         ratios=(rho,))[0].residual_mean
                   for s, row in d["sources"].items()} for d in depths]
        for keep in (1., .5):
            shortlist = plan_depth2_shortlist(scores[0], request_binding=digest_json(provenance),
                source_inventory_digest=provenance["source_inventory_digest"],
                reference_trim_ratio=rho, correctness_eligible_k=eligible_k, keep_fraction=keep)
            ranking = shortlist.validate_depth2({s: scores[1][s] for s in shortlist.retained_source_ids},
                request_binding=digest_json(provenance),
                source_inventory_digest=provenance["source_inventory_digest"], reference_trim_ratio=rho)
            audit = audit_depth2_pruning(shortlist, scores[1])
            insufficient = shortlist.insufficient_ranking_coverage
            chosen = None if insufficient else ranking[0]
            margin = None if len(ranking) < 2 else (
                (scores[1][ranking[1]] - scores[1][ranking[0]]) / max(scores[1][ranking[1]], 1e-12))
            cells.append({"reference_trim_ratio": rho, "depth2_keep_fraction": keep,
                "shortlist": asdict(shortlist), "depth2_ranking": list(ranking),
                "depth1_scores": scores[0], "depth2_scores": scores[1],
                "ranked_source_id": chosen, "margin": margin,
                "ranking_status": ("INSUFFICIENT_RANKING_COVERAGE" if insufficient else
                    "SINGLE_CORRECTNESS_ELIGIBLE_SOURCE" if eligible_k == 1 else "OFFLINE_RANKING_ONLY"),
                "depth2_pruning_audit": audit,
                "ranking_scope_complete": shortlist.scope_complete_at_depth2,
                "qa_passed": None, "matched_quality_ttft_ms": None,
                "pruned_execution_comparison_ms": None,
                "source_lock_performed": False, "production_admission_allowed": False})
    return _seal({"kind": "source_policy_offline_replay_v1",
        "observation_sha256": value["observation_sha256"], "provenance": provenance,
        "evidence_origin": value["evidence_origin"], "cells": cells,
        "measured_pruned_execution": False, "gpu_execution_allowed": False,
        "gpu_runtime_qualified": False, "paper_evidence": False,
        "locked_test_accessed": False}, "report_sha256")
