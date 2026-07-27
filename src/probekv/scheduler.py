from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple


class SchedulerPolicy(str, Enum):
    NO_OVERLAP = "no_overlap"
    A_ONLY = "a_only"
    B_ONLY = "b_only"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class SchedulerScenario:
    load_ms: float
    dense_layer_ms: float
    max_extra_dense_layers: int
    repair_ms: float
    decode_start_ms: float
    other_ready_work_ms: float
    microbatch_ms: float = 0.5

    def __post_init__(self) -> None:
        if min(
            self.load_ms,
            self.dense_layer_ms,
            self.repair_ms,
            self.decode_start_ms,
            self.other_ready_work_ms,
            self.microbatch_ms,
        ) < 0:
            raise ValueError("timings must be non-negative")
        if self.max_extra_dense_layers < 0:
            raise ValueError("max_extra_dense_layers must be non-negative")


@dataclass(frozen=True)
class ScheduleResult:
    policy: SchedulerPolicy
    a_ttft_ms: float
    other_work_completed_ms: float
    useful_a_dense_ms: float
    idle_ms: float
    timeline: Tuple[str, ...]


@dataclass(frozen=True)
class ReadyRequest:
    request_id: str
    ready_at_ms: float
    layer: int
    work_ms: float


@dataclass(frozen=True)
class MultiScheduleResult:
    a_ttft_ms: float
    service_ms_by_request: Dict[str, float]
    elapsed_wait_window_ms: float
    useful_a_dense_ms: float
    jain_fairness: float
    timeline: Tuple[str, ...]


def simulate_schedule(
    policy: SchedulerPolicy,
    scenario: SchedulerScenario,
    hybrid_dense_budget: int = 2,
) -> ScheduleResult:
    time_ms = 0.0
    other_done = 0.0
    useful_dense = 0.0
    timeline: List[str] = []
    dense_cap = scenario.max_extra_dense_layers
    if policy is SchedulerPolicy.HYBRID:
        dense_cap = min(dense_cap, max(0, hybrid_dense_budget))
    elif policy in {SchedulerPolicy.NO_OVERLAP, SchedulerPolicy.B_ONLY}:
        dense_cap = 0

    if policy in {SchedulerPolicy.A_ONLY, SchedulerPolicy.HYBRID}:
        for _ in range(dense_cap):
            if time_ms >= scenario.load_ms:
                break
            step = scenario.dense_layer_ms
            time_ms += step
            useful_dense += step
            timeline.append("A:dense %.3fms" % step)

    if policy in {SchedulerPolicy.B_ONLY, SchedulerPolicy.HYBRID}:
        while (
            time_ms < scenario.load_ms
            and other_done < scenario.other_ready_work_ms
            and scenario.microbatch_ms > 0
        ):
            remaining_load = scenario.load_ms - time_ms
            remaining_work = scenario.other_ready_work_ms - other_done
            step = min(scenario.microbatch_ms, remaining_load, remaining_work)
            time_ms += step
            other_done += step
            timeline.append("B:microbatch %.3fms" % step)

    idle_ms = max(0.0, scenario.load_ms - time_ms)
    if idle_ms:
        time_ms += idle_ms
        timeline.append("idle %.3fms" % idle_ms)
    remaining_repair = max(0.0, scenario.repair_ms - useful_dense)
    if remaining_repair:
        time_ms += remaining_repair
        timeline.append("A:repair %.3fms" % remaining_repair)
    time_ms += scenario.decode_start_ms
    timeline.append("A:decode-start %.3fms" % scenario.decode_start_ms)
    return ScheduleResult(
        policy=policy,
        a_ttft_ms=time_ms,
        other_work_completed_ms=other_done,
        useful_a_dense_ms=useful_dense,
        idle_ms=idle_ms,
        timeline=tuple(timeline),
    )


def simulate_waiting_queue(
    policy: SchedulerPolicy,
    scenario: SchedulerScenario,
    requests: Sequence[ReadyRequest],
    a_layer: int,
    hybrid_dense_budget: int = 2,
    ragged_same_layer_batch: bool = True,
) -> MultiScheduleResult:
    """Event simulation for many ready requests during A's source load.

    Microbatches are capped by the source-ready timestamp, so B/C work cannot
    overrun and delay A. Same-layer requests may share one ragged batch step.
    """
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("request_id values must be unique")
    for request in requests:
        if request.ready_at_ms < 0 or request.work_ms < 0 or request.layer < 0:
            raise ValueError("invalid ready request")
    time_ms = 0.0
    useful_dense = 0.0
    timeline: List[str] = []
    service = {request.request_id: 0.0 for request in requests}
    demand = {request.request_id: request.work_ms for request in requests}
    dense_budget = 0
    if policy is SchedulerPolicy.A_ONLY:
        dense_budget = scenario.max_extra_dense_layers
    elif policy is SchedulerPolicy.HYBRID:
        dense_budget = min(scenario.max_extra_dense_layers, hybrid_dense_budget)
    for _ in range(dense_budget):
        if time_ms >= scenario.load_ms:
            break
        time_ms += scenario.dense_layer_ms
        useful_dense += scenario.dense_layer_ms
        timeline.append("A:dense")

    if policy in {SchedulerPolicy.B_ONLY, SchedulerPolicy.HYBRID}:
        while time_ms < scenario.load_ms:
            ready = [
                request
                for request in requests
                if request.ready_at_ms <= time_ms
                and service[request.request_id] < request.work_ms
            ]
            if not ready:
                future = [
                    request.ready_at_ms
                    for request in requests
                    if request.ready_at_ms > time_ms
                    and service[request.request_id] < request.work_ms
                ]
                if not future:
                    break
                next_time = min(min(future), scenario.load_ms)
                timeline.append("queue-idle")
                time_ms = next_time
                continue
            # Prefer A's current layer, then the least served request.
            ready.sort(
                key=lambda request: (
                    request.layer != a_layer,
                    service[request.request_id] / max(request.work_ms, 1e-12),
                    request.request_id,
                )
            )
            lead = ready[0]
            batch = [lead]
            if ragged_same_layer_batch:
                batch = [request for request in ready if request.layer == lead.layer]
            maximum_step = min(scenario.microbatch_ms, scenario.load_ms - time_ms)
            step = min(
                [maximum_step]
                + [
                    request.work_ms - service[request.request_id]
                    for request in batch
                ]
            )
            if step <= 0:
                break
            time_ms += step
            for request in batch:
                service[request.request_id] += step
            timeline.append(
                "batch:%s" % ",".join(request.request_id for request in batch)
            )

    if time_ms < scenario.load_ms:
        time_ms = scenario.load_ms
    remaining_repair = max(0.0, scenario.repair_ms - useful_dense)
    a_ttft = time_ms + remaining_repair + scenario.decode_start_ms
    fractions = [
        service[request_id] / demand[request_id]
        for request_id in service
        if demand[request_id] > 0
    ]
    fairness = 1.0
    if fractions and sum(value * value for value in fractions) > 0:
        fairness = (sum(fractions) ** 2) / (
            len(fractions) * sum(value * value for value in fractions)
        )
    elif fractions:
        fairness = 0.0
    return MultiScheduleResult(
        a_ttft,
        service,
        scenario.load_ms,
        useful_dense,
        fairness,
        tuple(timeline),
    )
