"""Development-only slack repair proposals using the existing joint cost table.

This module does NOT predict per-Source quality, freeze a Profile, transfer KV,
or authorize reuse. It evaluates fixed-ratio continuations before the first
selective layer. The caller supplies explicit development quality support;
production still needs validated policy evidence and a fresh FinalCommit.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import re
import time

from .v8_schema10_cost_provider import (
    EXECUTION_SHAPE_KEY,
    ProfiledJointTimelineEstimator,
    validate_dense_reference,
)
from .v8_schema10_execution import digest_json
from .v8_schema10_profile import SCHEMA10_REPAIR_RATIO_GRID
from .v8_schema7_repair import SourceScoreTrimIndices


def _sha256(value):
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))


def propose_single_segment_slack_repair(
    *,
    estimator: ProfiledJointTimelineEstimator,
    context,
    snapshot,
    current_snapshot,
    frozen_source_variant_id,
    absolute_reuse_eligible,
    ranked_winner_repair_positions,
    base_ratio,
    quality_supported_ratios,
    quality_evidence_sha256,
    dense_reference,
    actual_sunk_ms,
):
    """Return an audited proposal, never a production admission decision.

    All candidate masks come from ONE winner-specific ranking, hence are nested
    at a fixed observation. Selection-score trim indices are not repair input.
    Each ratio builds its own exact complete-future query; missing cells remain
    unsupported. The empirical table maximum is not a probabilistic cost UCB.

    ``quality_supported_ratios`` is caller-supplied development evidence, not
    proof of safety merely because r >= base. Its digest is recorded for audit,
    not verified as a formally frozen Profile by this proposal-only interface.
    ``current_snapshot`` must read the current state, including after queries.
    """
    started = time.perf_counter_ns()
    snapshot.assert_current(current_snapshot())
    if not isinstance(estimator, ProfiledJointTimelineEstimator):
        raise TypeError("slack proposals require the measured joint cost provider")
    if estimator.key_contract != EXECUTION_SHAPE_KEY:
        raise ValueError("slack proposals require execution-shape cost keys")
    if snapshot.runtime_cost_profile_sha != estimator.measurement_digest:
        raise ValueError("snapshot and runtime measurement digest differ")
    if context.scheduler_state_id != snapshot.scheduler_snapshot_id:
        raise ValueError("scheduler snapshot differs from the joint context")
    if not frozen_source_variant_id or type(absolute_reuse_eligible) is not bool:
        raise ValueError("a frozen Source and explicit absolute eligibility are required")
    if not _sha256(quality_evidence_sha256) or not _sha256(estimator.measurement_digest):
        raise ValueError("quality/runtime evidence digests must be SHA256")
    if (isinstance(actual_sunk_ms, bool) or not math.isfinite(actual_sunk_ms)
            or actual_sunk_ms < 0):
        raise ValueError("actual sunk time must be finite and non-negative")

    shape = estimator.shape
    if (len(context.inventory_segment_ids) != 1
            or context.reuse_segment_ids != context.inventory_segment_ids
            or context.dense_fallback_segment_ids or context.committed_segment_ids
            or shape.committed_boundary_by_segment):
        raise ValueError("slack proposals are single-Segment and pre-commit only")
    sid = context.inventory_segment_ids[0]
    if set(shape.positions_by_segment) != {sid}:
        raise ValueError("proposal does not cover the complete Segment inventory")
    boundary = context.boundary_by_segment[sid]
    if shape.completed_depth < 1 or boundary != shape.completed_depth + 1:
        raise ValueError("repair proposal must follow the current dense check depth")
    positions = tuple(shape.positions_by_segment[sid])
    if not positions or tuple(sorted(set(positions))) != positions:
        raise ValueError("canonical Segment positions must be nonempty, sorted and unique")
    if isinstance(ranked_winner_repair_positions, SourceScoreTrimIndices):
        raise TypeError("Source score trim indices are not winner repair rankings")
    ranking = tuple(ranked_winner_repair_positions)
    if (any(type(p) is not int for p in ranking) or len(ranking) != len(positions)
            or set(ranking) != set(positions)):
        raise ValueError("winner ranking must cover only the full non-prefix Segment")
    supported = tuple(quality_supported_ratios)
    if (not supported or len(set(supported)) != len(supported)
            or isinstance(base_ratio, bool) or base_ratio not in SCHEMA10_REPAIR_RATIO_GRID
            or any(isinstance(r, bool) or r not in SCHEMA10_REPAIR_RATIO_GRID for r in supported)
            or base_ratio not in supported):
        raise ValueError("base and quality support must explicitly use the measured ratio grid")
    dense_ms = validate_dense_reference(dense_reference, expected_identity=shape.dense_reference_identity)
    if dense_reference.get("fake_timing") is not False:
        raise ValueError("matched dense reference must explicitly reject fake timing")

    # A copied query-audit list keeps shadow evaluation out of production events.
    query_engine = estimator.for_shape(shape)
    query_engine.query_audit = []
    rows, plans = [], {}
    if absolute_reuse_eligible:
        for ratio in sorted(r for r in supported if r >= base_ratio):
            count = min(len(positions), math.ceil(ratio * len(positions)))
            support = tuple(sorted(ranking[:count]))
            supports = {sid: {layer: support for layer in range(boundary, shape.num_layers + 1)}}
            trial_shape = replace(shape, repair_by_segment_by_layer=supports)
            masks = trial_shape.masks(context)
            mask_digest = digest_json(masks)
            trial_context = replace(context, union_mask_digest=mask_digest)
            lookup = query_engine.for_shape(trial_shape).lookup(trial_context)
            plans[ratio] = supports
            rows.append({
                "ratio": ratio,
                "repair_count": count,
                "effective_ratio": count / len(positions),
                "repair_support_digest": digest_json(supports),
                "union_mask_digest": mask_digest,
                "cost_status": lookup.status,
                "cost_reason": lookup.reason,
                "cost_query_digest": lookup.query_digest,
                "joint_future_ms": lookup.estimate.joint_future_ms if lookup.estimate else None,
            })

    snapshot.assert_current(current_snapshot())
    planning_host_ms = (time.perf_counter_ns() - started) / 1e6
    # No 0.95/1.0 substitute, no residual-as-cost, no sum of layer TTFTs, and
    # no subtraction of already spent probe/preparation time as "free" slack.
    accounted_sunk_ms = actual_sunk_ms + planning_host_ms
    limit_ms = .8 * dense_ms
    for row in rows:
        future = row["joint_future_ms"]
        row["predicted_request_total_ms"] = accounted_sunk_ms + future if future is not None else None
        row["within_gamma_budget"] = future is not None and row["predicted_request_total_ms"] <= limit_ms
    base = next((row for row in rows if row["ratio"] == base_ratio), None)
    # A missing or uneconomic base is not a license to extrapolate to a larger
    # ratio, even if a noisy isolated high-ratio timing looks cheaper.
    selected = None
    if not absolute_reuse_eligible:
        reason = "absolute_reuse_ineligible"
    elif base["cost_status"] != "SUPPORTED":
        reason = "base_cost_unsupported"
    elif not base["within_gamma_budget"]:
        reason = "base_exceeds_gamma_budget"
    else:
        selected = max((row for row in rows if row["within_gamma_budget"]), key=lambda row: row["ratio"])
        reason = "max_supported_ratio_within_gamma_budget"
    report = {
        "proposal_kind": "single_segment_slack_repair_shadow",
        "selected_source_variant_id": frozen_source_variant_id,
        "segment_id": sid,
        "first_selective_reuse_layer": boundary,
        "base_ratio": base_ratio,
        "quality_supported_ratios": sorted(supported),
        "quality_evidence_sha256": quality_evidence_sha256,
        "quality_support_status": "caller_supplied_development_not_certification",
        "runtime_measurement_sha256": estimator.measurement_digest,
        "cost_provenance": dict(estimator.provenance),
        "planner_snapshot": asdict(snapshot),
        "dense_reference_sha256": digest_json(dense_reference),
        "gamma": .8,
        "budget_total_ms": limit_ms,
        "actual_sunk_before_planning_ms": actual_sunk_ms,
        "shadow_planning_host_ms": planning_host_ms,
        "accounted_sunk_ms": accounted_sunk_ms,
        "candidate_observations": rows,
        "query_audit": query_engine.query_audit,
        "reason": reason,
        "recommended_action": "reuse_proposal" if selected else "dense",
        "selected_ratio": selected["ratio"] if selected else None,
        "selected_repair_by_segment_by_layer": plans[selected["ratio"]] if selected else None,
        "predicted_request_total_ms": selected["predicted_request_total_ms"] if selected else None,
        "remaining_gamma_slack_ms": limit_ms - selected["predicted_request_total_ms"] if selected else None,
        "free_slack_assumed": False,
        "production_admission_allowed": False,
        "fresh_final_commit_required": True,
        "gpu_runtime_qualified": False,
        "paper_evidence": False,
    }
    report["report_sha256"] = digest_json(report)
    return report
