from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .v8_schema7_contracts import RepairMetric, RepairPolicy, RepairSupportState


@dataclass(frozen=True)
class SourceScoreTrimIndices:
    """Score-only rows; deliberately not accepted as a repair support object."""

    absolute_positions: Tuple[int, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.absolute_positions))) != self.absolute_positions:
            raise ValueError("Source-score trim positions must be sorted and unique")


def _normalized_row_drifts(
    current: Sequence[Sequence[float]],
    source: Sequence[Sequence[float]],
    *,
    epsilon: float = 1e-12,
) -> Tuple[float, ...]:
    if epsilon <= 0 or not current or len(current) != len(source):
        raise ValueError("aligned non-empty rows and positive epsilon are required")
    result = []
    for current_row, source_row in zip(current, source):
        if not current_row or len(current_row) != len(source_row):
            raise ValueError("current and Source row geometry differs")
        numerator = math.sqrt(
            sum(
                (float(left) - float(right)) ** 2
                for left, right in zip(current_row, source_row)
            )
        )
        denominator = max(
            math.sqrt(sum(float(value) ** 2 for value in current_row)), epsilon
        )
        result.append(numerator / denominator)
    if any(not math.isfinite(value) for value in result):
        raise ValueError("winner deviation contains a non-finite value")
    return tuple(result)


def winner_repair_drifts(
    *,
    metric: RepairMetric,
    current_k: Sequence[Sequence[float]],
    source_k: Sequence[Sequence[float]],
    current_v: Sequence[Sequence[float]],
    source_v: Sequence[Sequence[float]],
    epsilon: float = 1e-12,
) -> Tuple[float, ...]:
    metric = RepairMetric(metric)
    k = _normalized_row_drifts(current_k, source_k, epsilon=epsilon)
    v = _normalized_row_drifts(current_v, source_v, epsilon=epsilon)
    if len(k) != len(v):
        raise ValueError("winner K/V token rows differ")
    if metric is RepairMetric.WINNER_K_ONLY:
        return k
    if metric is RepairMetric.WINNER_V_ONLY:
        return v
    return tuple(math.sqrt(left * left + right * right) for left, right in zip(k, v))


def source_score_from_k_drifts(
    token_drifts: Sequence[float],
    *,
    trim_ratio: float,
    absolute_positions: Sequence[int],
) -> Tuple[float, SourceScoreTrimIndices]:
    """Return a Source score and score-only trim positions.

    The returned positions are deliberately named/typed as score trimming;
    runtime repair support must be produced by ``build_initial_repair_support``.
    """

    if len(token_drifts) < 2 or not 0 <= trim_ratio < 1:
        raise ValueError("Source scoring requires N>=2 and trim ratio in [0,1)")
    positions = tuple(int(value) for value in absolute_positions)
    if len(positions) != len(token_drifts) or tuple(sorted(set(positions))) != positions:
        raise ValueError("absolute positions must uniquely cover the Source score rows")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in token_drifts):
        raise ValueError("Source score drifts must be finite and non-negative")
    count = min(len(token_drifts) - 1, int(math.ceil(trim_ratio * len(token_drifts))))
    ranking = sorted(
        range(len(token_drifts)),
        key=lambda index: (-float(token_drifts[index]), positions[index]),
    )
    trimmed_local = set(ranking[:count])
    score = sum(
        float(value)
        for index, value in enumerate(token_drifts)
        if index not in trimmed_local
    ) / (len(token_drifts) - count)
    return score, SourceScoreTrimIndices(
        tuple(sorted(positions[index] for index in trimmed_local))
    )


def _top_positions(
    scores: Sequence[float],
    positions: Sequence[int],
    count: int,
) -> Tuple[int, ...]:
    if len(scores) != len(positions) or not 0 <= count <= len(scores):
        raise ValueError("Top-K scores, positions and count are inconsistent")
    ranking = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), int(positions[index])),
    )
    return tuple(sorted(int(positions[index]) for index in ranking[:count]))


def build_initial_repair_support(
    *,
    segment_id: str,
    source_variant_id: str,
    metric: RepairMetric,
    repair_check_completed_depth: int,
    segment_absolute_positions: Sequence[int],
    drift_scores: Sequence[float],
    initial_cap: float = 0.15,
    repair_floor: float = 0.15,
) -> RepairSupportState:
    positions = tuple(int(value) for value in segment_absolute_positions)
    if len(positions) != len(drift_scores) or not positions:
        raise ValueError("initial repair scores must cover the complete Segment")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in drift_scores):
        raise ValueError("repair drift must be finite and non-negative")
    count = min(len(positions), int(math.ceil(initial_cap * len(positions))))
    return RepairSupportState(
        segment_id,
        source_variant_id,
        RepairMetric(metric),
        repair_check_completed_depth,
        repair_check_completed_depth + 1,
        _top_positions(drift_scores, positions, count),
        len(positions),
        initial_cap,
        repair_floor,
    )


def shrink_repair_support(
    parent: RepairSupportState,
    *,
    drift_score_by_absolute_position: Mapping[int, float],
    next_ratio: float,
) -> RepairSupportState:
    if not parent.repair_floor <= next_ratio <= parent.effective_ratio + 1e-12:
        raise ValueError("gradual repair ratio must stay between floor and parent")
    parent_positions = parent.candidate_absolute_positions
    if set(drift_score_by_absolute_position) != set(parent_positions):
        raise ValueError("gradual repair may score only the current support")
    scores = tuple(float(drift_score_by_absolute_position[value]) for value in parent_positions)
    if any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("gradual repair scores must be finite and non-negative")
    count = min(
        len(parent_positions),
        max(
            int(math.ceil(parent.repair_floor * parent.segment_token_count)),
            int(math.ceil(next_ratio * parent.segment_token_count)),
        ),
    )
    child = RepairSupportState(
        parent.segment_id,
        parent.source_variant_id,
        parent.metric,
        parent.producer_completed_depth + 1,
        parent.consumer_layer_1based + 1,
        _top_positions(scores, parent_positions, count),
        parent.segment_token_count,
        parent.initial_cap,
        parent.repair_floor,
        parent.support_digest,
    )
    parent.assert_monotonic_child(child)
    return child


@dataclass(frozen=True)
class LayerRepairPlan:
    ratio: float
    transfer_path: str
    predicted_layer_ms: float
    predicted_load_ms: float
    predicted_repair_ms: float
    predicted_nonoverlap_ms: float


class LoadRecomputeAwareRepairController:
    """Quality-first local plan generator; it never performs request admission."""

    def choose(
        self,
        *,
        parent_ratio: float,
        certified_floor: float,
        repair_ms_by_ratio: Mapping[float, float],
        load_ms_by_path: Mapping[str, float],
        nonoverlap_ms: float = 0.0,
    ) -> LayerRepairPlan:
        if not repair_ms_by_ratio or not load_ms_by_path or nonoverlap_ms < 0:
            raise ValueError("repair controller requires measured candidate costs")
        ratios = sorted(
            {
                float(value)
                for value in repair_ms_by_ratio
                if certified_floor <= float(value) <= parent_ratio + 1e-12
            },
            reverse=True,
        )
        if not ratios:
            raise ValueError("repair controller has no floor-respecting ratio")
        candidates = []
        for ratio in ratios:
            repair_ms = float(repair_ms_by_ratio[ratio])
            if repair_ms < 0:
                raise ValueError("repair timing must be non-negative")
            for path, raw_load in load_ms_by_path.items():
                load_ms = float(raw_load)
                if load_ms < 0:
                    raise ValueError("load timing must be non-negative")
                total = max(load_ms, repair_ms) + nonoverlap_ms
                candidates.append((total, -ratio, path, ratio, load_ms, repair_ms))
        best_total = min(row[0] for row in candidates)
        # Within numerical equality of the fastest critical path, retain the
        # largest repair ratio; then use deterministic path naming.
        viable = [row for row in candidates if row[0] <= best_total + 1e-12]
        total, _, path, ratio, load_ms, repair_ms = min(viable)
        return LayerRepairPlan(
            ratio,
            path,
            total,
            load_ms,
            repair_ms,
            nonoverlap_ms,
        )


@dataclass(frozen=True)
class RepairPolicyStepResult:
    policy: RepairPolicy
    support: RepairSupportState
    layer_plan: Optional[LayerRepairPlan]


class Schema7RepairPolicyExecutor:
    """Keep fixed, static, and load-aware policies as disjoint code paths."""

    def __init__(self, policy: RepairPolicy) -> None:
        self.policy = RepairPolicy(policy)

    def next_support(
        self,
        *,
        parent: RepairSupportState,
        drift_score_by_absolute_position: Mapping[int, float],
        static_next_ratio: Optional[float] = None,
        repair_ms_by_ratio: Optional[Mapping[float, float]] = None,
        load_ms_by_path: Optional[Mapping[str, float]] = None,
        nonoverlap_ms: float = 0.0,
    ) -> RepairPolicyStepResult:
        if self.policy is RepairPolicy.FIXED_15:
            next_ratio = parent.initial_cap
            plan = None
        elif self.policy is RepairPolicy.STATIC_GRADUAL:
            if static_next_ratio is None:
                raise ValueError("static gradual repair requires its frozen next ratio")
            next_ratio = float(static_next_ratio)
            plan = None
        else:
            if repair_ms_by_ratio is None or load_ms_by_path is None:
                raise ValueError("load-aware repair requires RuntimeCostProfile inputs")
            plan = LoadRecomputeAwareRepairController().choose(
                parent_ratio=min(parent.initial_cap, parent.effective_ratio),
                certified_floor=parent.repair_floor,
                repair_ms_by_ratio=repair_ms_by_ratio,
                load_ms_by_path=load_ms_by_path,
                nonoverlap_ms=nonoverlap_ms,
            )
            next_ratio = plan.ratio
        support = shrink_repair_support(
            parent,
            drift_score_by_absolute_position=drift_score_by_absolute_position,
            next_ratio=next_ratio,
        )
        return RepairPolicyStepResult(self.policy, support, plan)


def repair_support_overlap_metrics(
    predicted_positions: Sequence[int],
    oracle_positions: Sequence[int],
) -> Mapping[str, float]:
    predicted = set(int(value) for value in predicted_positions)
    oracle = set(int(value) for value in oracle_positions)
    union = predicted | oracle
    intersection = predicted & oracle
    return {
        "jaccard": (len(intersection) / len(union)) if union else 1.0,
        "oracle_recall": (len(intersection) / len(oracle)) if oracle else 1.0,
        "predicted_precision": (
            len(intersection) / len(predicted) if predicted else (1.0 if not oracle else 0.0)
        ),
    }


def repair_support_oracle_metrics(
    predicted_ranked_positions: Sequence[int],
    oracle_ranked_positions: Sequence[int],
) -> Mapping[str, float]:
    """Jaccard/recall plus deterministic Spearman rank stability.

    Missing positions receive the rank immediately after the corresponding
    list. This keeps the metric defined when a no-reentry support omits a token
    that later becomes important in the full oracle.
    """
    predicted = tuple(int(value) for value in predicted_ranked_positions)
    oracle = tuple(int(value) for value in oracle_ranked_positions)
    if len(set(predicted)) != len(predicted) or len(set(oracle)) != len(oracle):
        raise ValueError("repair oracle rankings must not contain duplicates")
    overlap = dict(repair_support_overlap_metrics(predicted, oracle))
    universe = sorted(set(predicted) | set(oracle))
    if len(universe) < 2:
        overlap["spearman"] = 1.0
        return overlap
    predicted_rank = {value: rank for rank, value in enumerate(predicted, 1)}
    oracle_rank = {value: rank for rank, value in enumerate(oracle, 1)}
    predicted_missing_rank = len(predicted) + 1
    oracle_missing_rank = len(oracle) + 1
    left = [predicted_rank.get(value, predicted_missing_rank) for value in universe]
    right = [oracle_rank.get(value, oracle_missing_rank) for value in universe]

    def average_tie_ranks(values: Sequence[int]) -> Tuple[float, ...]:
        groups = {}
        for index, value in enumerate(values):
            groups.setdefault(value, []).append(index)
        result = [0.0] * len(values)
        for value, indices in groups.items():
            average = sum(
                sorted(values).index(value) + offset + 1
                for offset in range(len(indices))
            ) / len(indices)
            for index in indices:
                result[index] = average
        return tuple(result)

    left_rank = average_tie_ranks(left)
    right_rank = average_tie_ranks(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    covariance = sum(
        (lvalue - left_mean) * (rvalue - right_mean)
        for lvalue, rvalue in zip(left_rank, right_rank)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_rank))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_rank))
    overlap["spearman"] = (
        covariance / (left_scale * right_scale)
        if left_scale > 0 and right_scale > 0
        else (1.0 if left_rank == right_rank else 0.0)
    )
    return overlap
