"""Shared online/replay decisions; no independent threshold-only selector.

This module performs no tensor I/O. Engine adapters supply *current* checkpoint
observations and use the same session to decide before loading a winner.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .v8_contracts import CandidateCounts, ResidualCandidate
from .v8_schema8_planner import Gate1LocalPlan
from .v8_schema10_selector import Schema10CheckpointSelector


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SelectionCostPolicy:
    mode: str = "end_to_end_aware"
    legacy_fraction: float = 0.05
    final_gamma: float = 0.8

    def __post_init__(self) -> None:
        if self.mode not in {"end_to_end_aware", "legacy_fixed_fraction"}:
            raise ValueError("unknown selection cost policy")
        if self.legacy_fraction != .05 or self.final_gamma != .8:
            raise ValueError("legacy fraction/final gamma differ from the contract")


class SelectionCostLedger:
    """A measured ledger, not an implicit 5% rejection in end-to-end mode.

    A pending plan need not already achieve gamma. Only a certified lower
    bound over *all* feasible joint continuations may prove early futility.
    Resource reservations remain the engine allocator's responsibility.
    """

    def __init__(self, dense_reference_ttft_ms: float, policy: SelectionCostPolicy) -> None:
        if not math.isfinite(dense_reference_ttft_ms) or dense_reference_ttft_ms <= 0:
            raise ValueError("matched dense TTFT must be positive")
        self.dense_reference_ttft_ms = dense_reference_ttft_ms
        self.policy = policy
        self.intervals: list[tuple[int, int]] = []
        self.pending: dict[str, float] = {}
        self.completed: set[str] = set()

    @property
    def actual_active_ms(self) -> float:
        merged: list[list[int]] = []
        for start, end in sorted(self.intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return sum(end - start for start, end in merged) / 1e6

    def may_compare(self, predicted_ms: float, *, resource_available: bool = True) -> bool:
        if not math.isfinite(predicted_ms) or predicted_ms < 0:
            raise ValueError("invalid comparison prediction")
        if not resource_available:
            return False
        if self.policy.mode == "end_to_end_aware":
            return True
        return (self.actual_active_ms + sum(self.pending.values()) + predicted_ms
                <= self.policy.legacy_fraction * self.dense_reference_ttft_ms + 1e-12)

    def reserve(self, event_id: str, predicted_ms: float, *, resource_available: bool = True) -> None:
        if not event_id or event_id in self.pending or event_id in self.completed:
            raise ValueError("duplicate/missing comparison event")
        if not self.may_compare(predicted_ms, resource_available=resource_available):
            raise RuntimeError("comparison not admitted by resource/legacy policy")
        self.pending[event_id] = predicted_ms

    def settle(self, event_id: str, start_ns: int, end_ns: int) -> None:
        if event_id not in self.pending or not 0 <= start_ns <= end_ns:
            raise ValueError("unreserved or invalid timing interval")
        del self.pending[event_id]
        self.completed.add(event_id)
        self.intervals.append((start_ns, end_ns))

    def cancel(self, event_id: str) -> None:
        del self.pending[event_id]

    def observe_shared_interval(self, event_id: str, start_ns: int, end_ns: int) -> None:
        """Account shared probe/metadata once, including a realized overrun.

        These intervals are measured work, not a new comparison admission.
        Overlap with comparison intervals is unioned by actual_active_ms.
        """
        if not event_id or event_id in self.completed or event_id in self.pending:
            raise ValueError("duplicate/missing shared selection event")
        if not 0 <= start_ns <= end_ns:
            raise ValueError("invalid shared selection interval")
        self.completed.add(event_id)
        self.intervals.append((start_ns, end_ns))

    def continuation_proven_infeasible(self, actual_sunk_ms: float, *,
                                       joint_future_lower_ms: float,
                                       complete_scope_lower_bound: bool) -> bool:
        if not all(math.isfinite(x) and x >= 0 for x in (actual_sunk_ms, joint_future_lower_ms)):
            raise ValueError("invalid joint lower bound")
        if not complete_scope_lower_bound:
            return False
        return actual_sunk_ms + joint_future_lower_ms > self.policy.final_gamma * self.dense_reference_ttft_ms


class ProductionSelectionSession:
    """One immutable decision history shared by online execution and replay."""

    def __init__(self, request_id: str, segment_ids: Sequence[str],
                 selector: Schema10CheckpointSelector, ledger: SelectionCostLedger) -> None:
        if not request_id or not segment_ids or len(set(segment_ids)) != len(segment_ids):
            raise ValueError("request requires a complete, unique Segment inventory")
        self.request_id, self.segment_ids = request_id, tuple(segment_ids)
        self.selector, self.ledger = selector, ledger
        self.decisions: dict[str, Any] = {}
        self.previous: dict[str, str | None] = {}
        self.last_depth: dict[str, int] = {}
        self.events: list[dict[str, Any]] = []

    def step(self, segment_id: str, *, completed_depth: int, counts: CandidateCounts,
             candidates: Sequence[ResidualCandidate],
             gate1_plan_by_source: Mapping[str, Gate1LocalPlan]) -> Any:
        if segment_id not in self.segment_ids:
            raise ValueError("Segment outside the complete request inventory")
        if segment_id in self.decisions:
            raise RuntimeError("Source decision is terminal; no reselection after freeze/abstain")
        depths = self.selector.checkpoint_depths
        last = self.last_depth.get(segment_id)
        expected = depths[0] if last is None else depths[depths.index(last) + 1]
        if completed_depth != expected:
            raise ValueError("checkpoint missing, reordered or duplicated")
        decision = self.selector.decide(
            completed_depth=completed_depth, counts=counts, candidates=candidates,
            gate1_plan_by_source=gate1_plan_by_source,
            previous_best_source_variant_id=self.previous.get(segment_id),
        )
        self.last_depth[segment_id] = completed_depth
        self.previous[segment_id] = (min(candidates, key=lambda x: (x.residual_score, x.source_variant_id)).source_variant_id
                                     if candidates else None)
        event = {"request_id": self.request_id, "segment_id": segment_id,
                 "completed_depth": completed_depth, "counts": asdict(counts),
                 "candidates": [asdict(x) for x in candidates],
                 "gate1_plans": {k: asdict(v) for k, v in gate1_plan_by_source.items()},
                 "decision": asdict(decision), "selection_budget_policy": self.ledger.policy.mode}
        event["event_id"] = digest_json(event)
        self.events.append(event)
        if decision.state != "continue_probe":
            self.decisions[segment_id] = decision
        return decision

    @property
    def closed(self) -> bool:
        return len(self.decisions) == len(self.segment_ids)


def replay_selection_events(session: ProductionSelectionSession,
                            events: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    from .v8_schema8_planner import Gate1MarginalLowerBound
    for event in events:
        if event["request_id"] != session.request_id:
            raise ValueError("replay request identity mismatch")
        if event.get("selection_budget_policy") != session.ledger.policy.mode:
            raise ValueError("replay selection cost policy differs")
        unsigned = {k: v for k, v in event.items() if k != "event_id"}
        if event.get("event_id") != digest_json(unsigned):
            raise ValueError("selection event digest mismatch")
        plans = {}
        for source_id, payload in event["gate1_plans"].items():
            value = dict(payload)
            value["marginal_lower_bound"] = Gate1MarginalLowerBound(**value["marginal_lower_bound"])
            plans[source_id] = Gate1LocalPlan(**value)
        result = session.step(
            event["segment_id"], completed_depth=int(event["completed_depth"]),
            counts=CandidateCounts(**event["counts"]),
            candidates=tuple(ResidualCandidate(**row) for row in event["candidates"]),
            gate1_plan_by_source=plans,
        )
        if digest_json(asdict(result)) != digest_json(event["decision"]):
            raise ValueError("recorded decision differs from the production selector")
    if not session.closed:
        raise ValueError("replay is missing terminal Segment decisions")
    return tuple(session.events)
