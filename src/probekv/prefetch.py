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
