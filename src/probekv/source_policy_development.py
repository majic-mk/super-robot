"""CPU-testable policy candidates. No Source freeze, leases, or GPU qualification.

Reference trim is fixed for a run; ratio curves do not adapt the scoring rule
after seeing a winner. A depth-1 shortlist is NOT a Source selection decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Tuple

from .v8_schema10_execution import digest_json


REPAIR_RATIO_CANDIDATES = (.05, .10, .15, .20, .25, .30)
REFERENCE_TRIM_CANDIDATES = (.05, .15)
CACHEBLEND_PINNED_REPAIR_METRIC = "value_squared_l2_pinned_dtype"


def _finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


@dataclass(frozen=True)
class ResidualTailPoint:
    nominal_ratio: float
    trim_count: int
    effective_ratio: float
    remaining_count: int
    residual_mean: float


def residual_tail_curve(drifts, absolute_positions, *, ratios=REPAIR_RATIO_CANDIDATES):
    """One sort plus one suffix scan; returns score statistics, never repair masks.

    These are PRE-repair observations with hypothetical rows removed, not a
    measurement of the resulting Transformer/QA error. r=1 is a separate test.
    """
    values, positions, grid = tuple(drifts), tuple(absolute_positions), tuple(ratios)
    if len(values) < 2 or len(values) != len(positions):
        raise ValueError("Source residual curves require aligned N>=2 rows")
    if (any(type(p) is not int or p < 0 for p in positions)
            or tuple(sorted(set(positions))) != positions):
        raise ValueError("absolute positions must be sorted, unique, nonnegative integers")
    if any(not _finite_nonnegative(v) for v in values):
        raise ValueError("residuals must be finite and nonnegative")
    if (not grid or tuple(sorted(set(grid))) != grid
            or any(isinstance(r, bool) or not math.isfinite(r) or not 0 <= r < 1 for r in grid)):
        raise ValueError("score ratios must be unique ordered values in [0,1)")
    order = sorted(range(len(values)), key=lambda j: (-values[j], positions[j]))
    suffix = [0.] * (len(values) + 1)
    for j in range(len(values) - 1, -1, -1):
        suffix[j] = math.fsum((suffix[j + 1], values[order[j]]))
    if not math.isfinite(suffix[0]):
        raise ValueError("residual accumulation overflow")
    result = []
    for ratio in grid:
        count = min(len(values) - 1, math.ceil(ratio * len(values)))
        result.append(ResidualTailPoint(ratio, count, count / len(values),
                                       len(values) - count, suffix[count] / (len(values) - count)))
    return tuple(result)


def cacheblend_pinned_value_scores(current_v, source_v):
    """Diagnostic mirror of the pinned xformers V-squared-L2 operator.

    Keep input-dtype subtraction/squaring/reduction, as in the pinned patch;
    FP32 normalized V is a different policy and must not share its identity.
    Sorting/ceil and absolute mask ownership remain separate contracts.
    """
    import torch
    if (current_v.shape != source_v.shape or current_v.ndim != 3
            or current_v.numel() == 0 or current_v.dtype != source_v.dtype
            or current_v.device != source_v.device
            or not current_v.is_floating_point()):
        raise ValueError("matched floating-point V geometry/dtype/device required")
    scores = torch.sum((current_v - source_v) ** 2, dim=(1, 2))
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("non-finite pinned repair score")
    return scores


def rank_winner_v_positions(current_v, source_v, absolute_positions, *, metric):
    """Winner-only V ranking; never consumes Source-score trimming indices."""
    import torch
    positions = tuple(absolute_positions)
    if (not positions or any(type(p) is not int or p < 0 for p in positions)
            or positions != tuple(sorted(set(positions))) or len(positions) != current_v.shape[0]):
        raise ValueError("ordered absolute winner Segment rows required")
    if metric == CACHEBLEND_PINNED_REPAIR_METRIC:
        scores = cacheblend_pinned_value_scores(current_v, source_v)
    elif metric == "normalized_v_legacy":
        if (current_v.shape != source_v.shape or current_v.ndim != 3
                or current_v.dtype != source_v.dtype or current_v.device != source_v.device):
            raise ValueError("winner V inputs must match")
        v, old = current_v.float(), source_v.float()
        scores = (v-old).square().sum((1,2)).sqrt()/v.square().sum((1,2)).sqrt().clamp_min(1e-12)
    else:
        raise ValueError("unregistered winner repair metric")
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("nonfinite winner V scores")
    order = scores.argsort(descending=True,stable=True).cpu().tolist()
    return tuple(positions[i] for i in order)


def _scores(rows):
    if not isinstance(rows, Mapping):
        raise ValueError("Source scores must be a mapping with explicit unique IDs")
    values = dict(rows)
    if any(not isinstance(k, str) or not k or not _finite_nonnegative(v)
           for k, v in values.items()):
        raise ValueError("Source IDs and finite nonnegative scores required")
    return values


@dataclass(frozen=True)
class DepthTwoShortlist:
    request_binding: str
    source_inventory_digest: str
    reference_trim_ratio: float
    correctness_eligible_k: int
    depth1_scores: Tuple[Tuple[str, float], ...]
    retained_source_ids: Tuple[str, ...]
    pruned_source_ids: Tuple[str, ...]
    minimum_retained_k: int
    keep_fraction: float
    tie_slack: float

    @property
    def digest(self):
        return digest_json(asdict(self))

    @property
    def scope_complete_at_depth2(self):
        return len(self.retained_source_ids) == self.correctness_eligible_k

    @property
    def insufficient_ranking_coverage(self):
        return self.correctness_eligible_k > 1 and len(self.retained_source_ids) < 2

    def validate_depth2(self, scores: Mapping[str, float], *, request_binding,
                        source_inventory_digest, reference_trim_ratio):
        if (request_binding != self.request_binding
                or source_inventory_digest != self.source_inventory_digest
                or reference_trim_ratio != self.reference_trim_ratio):
            raise ValueError("stale request/inventory or changed reference trim")
        values = _scores(scores)
        if set(values) != set(self.retained_source_ids):
            raise ValueError("depth2 must rescore exactly its retained cohort")
        # Do not turn this ranking into a freeze or claim complete pool mismatch.
        return tuple(sorted(values, key=lambda s: (values[s], s)))


def plan_depth2_shortlist(depth1_scores, *, request_binding, source_inventory_digest,
                          reference_trim_ratio=.15, correctness_eligible_k,
                          keep_fraction=.5, minimum_retained_k=2, tie_slack=1e-6,
                          completed_depth=1):
    """d1 (completed block 1) -> d2. Exact/near-cutoff ties survive pruning.

    A flat d1 representation keeps ALL tied Sources, not an arbitrary half.
    Original eligible count is immutable; pruning cannot invent completeness.
    """
    scores = _scores(depth1_scores)
    if (type(completed_depth) is not int or completed_depth != 1
            or not request_binding or not source_inventory_digest
            or reference_trim_ratio not in REFERENCE_TRIM_CANDIDATES
            or isinstance(reference_trim_ratio, bool)
            or type(correctness_eligible_k) is not int
            or not len(scores) <= correctness_eligible_k <= 16
            or type(minimum_retained_k) is not int or minimum_retained_k < 2
            or isinstance(keep_fraction, bool) or not math.isfinite(keep_fraction)
            or not 0 < keep_fraction <= 1 or not _finite_nonnegative(tie_slack)):
        raise ValueError("invalid bounded development shortlist contract")
    ordered = sorted(scores, key=lambda s: (scores[s], s))
    keep = min(len(ordered), max(minimum_retained_k, math.ceil(len(ordered) * keep_fraction)))
    cutoff = scores[ordered[keep - 1]] if keep else None
    retained = tuple(s for s in ordered if cutoff is not None and scores[s] <= cutoff + tie_slack)
    return DepthTwoShortlist(request_binding, source_inventory_digest, reference_trim_ratio,
        correctness_eligible_k, tuple((s, scores[s]) for s in ordered), retained,
        tuple(s for s in ordered if s not in retained), minimum_retained_k, keep_fraction, tie_slack)


def audit_depth2_pruning(shortlist, full_depth2_scores):
    """Offline shadow only; needs d2 scores for EVERY d1-compared Source."""
    scores = _scores(full_depth2_scores)
    if set(scores) != {s for s, _ in shortlist.depth1_scores}:
        raise ValueError("pruning recall needs complete original cohort d2 evidence")
    if not scores:
        return {"oracle_winner_retained": None, "absolute_residual_regret": None,
                "production_admission_allowed": False}
    best = min(scores.values())
    tied = {s for s in scores if scores[s] == best}
    chosen = min(scores[s] for s in shortlist.retained_source_ids)
    return {"oracle_winner_retained": bool(tied & set(shortlist.retained_source_ids)),
            "absolute_residual_regret": chosen - best,
            "reference_scope": "original_depth1_compared_cohort_not_unseen_sources",
            "depth1_compared_k": len(scores), "depth2_compared_k": len(shortlist.retained_source_ids),
            "scope_complete_at_depth2": shortlist.scope_complete_at_depth2,
            "production_admission_allowed": False}


def development_experiment_spec():
    """Factorized experiment definitions, not executable CUDA task admission."""
    selection = [{"reference_trim_ratio": rho, "depth2_keep_fraction": keep,
                  "minimum_retained_k": 2, "tie_slack": 1e-6}
                 for rho in REFERENCE_TRIM_CANDIDATES for keep in (1., .5)]
    chunking = [{"target_tokens": n, "search_window_tokens": w,
                 "policy": "paragraph_sentence_clause_v2"}
                for n in (512, 1024, 2048) for w in (50, 100)]
    chunking += [{"target_tokens": n, "search_window_tokens": 0, "policy": "fixed_tokens_v1"}
                 for n in (512, 1024, 2048)]
    value = {"kind": "source_policy_candidate_spec_v1", "stage": "no_gpu_development",
        "default_reference_trim_ratio": .15, "reference_candidates": list(REFERENCE_TRIM_CANDIDATES),
        "repair_ratio_candidates": list(REPAIR_RATIO_CANDIDATES), "correctness_endpoint": 1.,
        "default_execution_objective": "efficiency_first",
        "objectives": ["efficiency_first", "quality_within_budget"],
        "repair_metric_contract": CACHEBLEND_PINNED_REPAIR_METRIC,
        "selection_cells": selection, "chunking_cells": chunking,
        "execution_order": ["single_segment_same_backend_source_oracle", "reference_and_cascade_shadow",
                            "winner_ratio_quality_cost", "semantic_chunking_heldout_lengths",
                            "native_prefix_cpu_streaming", "multisegment_then_concurrency"],
        "non_anchor_validation_required": True, "full_depth2_shadow_required": True,
        "native_dense_shadow_capture_entry_implemented": True,
        "online_cascade_opt_in_implemented": True,
        "candidate_specific_unknown_costs_implemented": True,
        "absolute_qualified_future_cost_opt_in_implemented": True,
        "native_winner_raw_v_metric_opt_in_implemented": True,
        "live_reference_candidates_supported_by_existing_profile": [.15],
        "reference_05_requires_new_profile_contract": True,
        "local_cuda_tests_are_a800_qualification": False,
        "model_provenance": None, "quality_profile": None, "runtime_cost_profile": None,
        "live_dispatch_integration_complete": False, "gpu_execution_allowed": False,
        "gpu_runtime_qualified": False, "paper_evidence": False, "locked_test_accessed": False}
    value["spec_sha256"] = digest_json(value)
    return value
