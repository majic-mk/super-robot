from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple


class PrefetchPolicy(str, Enum):
    P0 = "p0_after_selection"
    P1 = "p1_summary_winner"
    P2 = "p2_speculative_top1"
    P3 = "p3_speculative_top2"
    P4 = "p4_all_sources"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class PrefetchCandidate:
    source_id: str
    selection_probability: float
    kv_bytes: int
    load_ms: float


@dataclass(frozen=True)
class PrefetchDecision:
    policy: PrefetchPolicy
    source_ids: Tuple[str, ...]
    expected_visible_load_ms: float
    transferred_bytes: int
    reason: str


@dataclass(frozen=True)
class LockedSegmentWinner:
    segment_id: str
    source_id: str
    locked_probe_layer: int
    kv_bytes: int
    load_ms: float
    predicted_saved_ms: float

    def __post_init__(self) -> None:
        if not self.segment_id or not self.source_id:
            raise ValueError("locked winner identifiers are required")
        if self.locked_probe_layer < 1:
            raise ValueError("locked probe layer must be 1-based")
        if self.kv_bytes < 0 or self.load_ms < 0:
            raise ValueError("winner load size and time must be non-negative")

    @property
    def value_density(self) -> float:
        return max(0.0, self.predicted_saved_ms) / float(max(1, self.kv_bytes))


@dataclass(frozen=True)
class MultiSegmentPrefetchDecision:
    source_id_by_segment: dict
    dropped_segment_ids: Tuple[str, ...]
    transferred_bytes: int
    expected_visible_load_ms: float


def choose_locked_winner_prefetch(
    winners: Sequence[LockedSegmentWinner],
    hbm_available_bytes: int,
    overlap_ms: float,
) -> MultiSegmentPrefetchDecision:
    """Load at most one already-locked winner per segment under HBM budget."""

    if hbm_available_bytes < 0 or overlap_ms < 0:
        raise ValueError("HBM and overlap budgets must be non-negative")
    segment_ids = [winner.segment_id for winner in winners]
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("only one locked winner is allowed per segment")
    ordered = sorted(
        winners,
        key=lambda winner: (
            winner.locked_probe_layer,
            -winner.value_density,
            winner.segment_id,
            winner.source_id,
        ),
    )
    selected = []
    used = 0
    for winner in ordered:
        if used + winner.kv_bytes > hbm_available_bytes:
            continue
        selected.append(winner)
        used += winner.kv_bytes
    selected_segments = {winner.segment_id for winner in selected}
    dropped = tuple(
        winner.segment_id for winner in ordered
        if winner.segment_id not in selected_segments
    )
    visible = max(
        (max(0.0, winner.load_ms - overlap_ms) for winner in selected),
        default=0.0,
    )
    return MultiSegmentPrefetchDecision(
        source_id_by_segment={
            winner.segment_id: winner.source_id for winner in selected
        },
        dropped_segment_ids=dropped,
        transferred_bytes=used,
        expected_visible_load_ms=visible,
    )


def _fixed_count(policy: PrefetchPolicy, candidate_count: int) -> int:
    if policy in {PrefetchPolicy.P0, PrefetchPolicy.P1}:
        return 0
    if policy is PrefetchPolicy.P2:
        return min(1, candidate_count)
    if policy is PrefetchPolicy.P3:
        return min(2, candidate_count)
    if policy is PrefetchPolicy.P4:
        return candidate_count
    raise ValueError("dynamic policy has no fixed count")


def choose_prefetch(
    policy: PrefetchPolicy,
    candidates: Sequence[PrefetchCandidate],
    hbm_available_bytes: int,
    overlap_ms: float,
    byte_penalty_ms_per_gb: float = 0.15,
    winner_source_id: Optional[str] = None,
) -> PrefetchDecision:
    ordered = sorted(
        candidates, key=lambda candidate: candidate.selection_probability, reverse=True
    )
    if not ordered:
        return PrefetchDecision(policy, (), 0.0, 0, "no candidates")

    def evaluate(count: int):
        selected = tuple(ordered[:count])
        total_bytes = sum(candidate.kv_bytes for candidate in selected)
        if total_bytes > hbm_available_bytes:
            return float("inf"), selected, total_bytes
        prefetched_ids = {candidate.source_id for candidate in selected}
        miss_load = sum(
            candidate.selection_probability
            * (0.0 if candidate.source_id in prefetched_ids else candidate.load_ms)
            for candidate in ordered
        )
        transfer_penalty = (total_bytes / 1_000_000_000.0) * byte_penalty_ms_per_gb
        visible = max(0.0, miss_load - overlap_ms) + transfer_penalty
        return visible, selected, total_bytes

    if policy is not PrefetchPolicy.DYNAMIC:
        if policy in {PrefetchPolicy.P0, PrefetchPolicy.P1} and winner_source_id:
            winner = [
                candidate
                for candidate in ordered
                if candidate.source_id == winner_source_id
            ]
            if not winner:
                raise ValueError("winner_source_id is not a candidate")
            total_bytes = winner[0].kv_bytes
            if total_bytes > hbm_available_bytes:
                return PrefetchDecision(
                    policy, (), 0.0, 0, "HBM budget rejected winner"
                )
            selected = tuple(winner)
            visible = max(0.0, winner[0].load_ms - overlap_ms)
        else:
            count = _fixed_count(policy, len(ordered))
            visible, selected, total_bytes = evaluate(count)
        if visible == float("inf"):
            return PrefetchDecision(policy, (), 0.0, 0, "HBM budget rejected policy")
        return PrefetchDecision(
            policy,
            tuple(candidate.source_id for candidate in selected),
            visible,
            total_bytes,
            "fixed policy",
        )

    options = []
    for count in range(0, len(ordered) + 1):
        visible, selected, total_bytes = evaluate(count)
        options.append((visible, total_bytes, count, selected))
    visible, total_bytes, _, selected = min(
        options, key=lambda item: (item[0], item[1])
    )
    return PrefetchDecision(
        PrefetchPolicy.DYNAMIC,
        tuple(candidate.source_id for candidate in selected),
        visible,
        total_bytes,
        "minimum estimated system cost under HBM budget",
    )
