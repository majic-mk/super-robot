from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple


class SchedulerPolicy(str, Enum):
    NO_OVERLAP = "no_overlap"
    A_ONLY = "a_only"
    B_ONLY = "b_only"
    HYBRID = "hybrid"
    HYBRID_STRICT = "hybrid_strict"
    HYBRID_BOUNDED_OVERRUN = "hybrid_bounded_overrun"


@dataclass(frozen=True)
class SchedulerScenario:
    load_ms: float
    dense_layer_ms: float
    max_extra_dense_layers: int
    repair_ms: float
    decode_start_ms: float
    other_ready_work_ms: float
    microbatch_ms: float = 0.5
    max_post_ready_overrun_ms: float = 0.0
    load_interference_ms: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.load_ms,
            self.dense_layer_ms,
            self.repair_ms,
            self.decode_start_ms,
            self.other_ready_work_ms,
            self.microbatch_ms,
            self.max_post_ready_overrun_ms,
            self.load_interference_ms,
        ) < 0:
            raise ValueError("timings must be non-negative")
        if self.max_extra_dense_layers < 0:
            raise ValueError("max_extra_dense_layers must be non-negative")

    @property
    def source_ready_ms(self) -> float:
        return self.load_ms + self.load_interference_ms


@dataclass(frozen=True)
class ScheduleResult:
    policy: SchedulerPolicy
    a_ttft_ms: float
    other_work_completed_ms: float
    useful_a_dense_ms: float
    idle_ms: float
    timeline: Tuple[str, ...]
    source_ready_ms: float
    scheduled_step_finish_ms: float
    a_resume_ms: float
    post_ready_blocking_ms: float
    hidden_work_ms: float
    useful_other_request_work_ms: float
    load_interference_ms: float
    gpu_busy_ms: float


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
    source_ready_ms: float
    scheduled_step_finish_ms: float
    a_resume_ms: float
    post_ready_blocking_ms: float
    hidden_work_ms: float
    useful_other_request_work_ms: float
    load_interference_ms: float
    gpu_busy_ms: float


def _is_hybrid(policy: SchedulerPolicy) -> bool:
    return policy in {
        SchedulerPolicy.HYBRID,
        SchedulerPolicy.HYBRID_STRICT,
        SchedulerPolicy.HYBRID_BOUNDED_OVERRUN,
    }


def _atomic_step_allowed(
    policy: SchedulerPolicy,
    finish_ms: float,
    source_ready_ms: float,
    max_post_ready_overrun_ms: float,
    overrun_already_used: bool,
) -> bool:
    blocking = max(0.0, finish_ms - source_ready_ms)
    if blocking <= 1e-12:
        return True
    return (
        policy is SchedulerPolicy.HYBRID_BOUNDED_OVERRUN
        and not overrun_already_used
        and blocking <= max_post_ready_overrun_ms + 1e-12
    )


def simulate_schedule(
    policy: SchedulerPolicy,
    scenario: SchedulerScenario,
    hybrid_dense_budget: int = 2,
) -> ScheduleResult:
    time_ms = 0.0
    other_done = 0.0
    useful_dense = 0.0
    hidden_work = 0.0
    wait_busy = 0.0
    overrun_used = False
    timeline: List[str] = []
    source_ready_ms = scenario.source_ready_ms
    dense_cap = scenario.max_extra_dense_layers
    if _is_hybrid(policy):
        dense_cap = min(dense_cap, max(0, hybrid_dense_budget))
    elif policy in {SchedulerPolicy.NO_OVERLAP, SchedulerPolicy.B_ONLY}:
        dense_cap = 0

    if policy is SchedulerPolicy.A_ONLY or _is_hybrid(policy):
        for _ in range(dense_cap):
            if time_ms >= source_ready_ms:
                break
            step = scenario.dense_layer_ms
            finish = time_ms + step
            if not _atomic_step_allowed(
                policy,
                finish,
                source_ready_ms,
                scenario.max_post_ready_overrun_ms,
                overrun_used,
            ):
                break
            hidden_work += min(
                step, max(0.0, source_ready_ms - time_ms)
            )
            time_ms = finish
            wait_busy += step
            useful_dense += step
            timeline.append("A:dense %.3fms" % step)
            if finish > source_ready_ms:
                overrun_used = True
                break

    if policy is SchedulerPolicy.B_ONLY or _is_hybrid(policy):
        while (
            time_ms < source_ready_ms
            and other_done < scenario.other_ready_work_ms
            and scenario.microbatch_ms > 0
        ):
            remaining_work = scenario.other_ready_work_ms - other_done
            step = min(scenario.microbatch_ms, remaining_work)
            finish = time_ms + step
            if not _atomic_step_allowed(
                policy,
                finish,
                source_ready_ms,
                scenario.max_post_ready_overrun_ms,
                overrun_used,
            ):
                break
            hidden_work += min(
                step, max(0.0, source_ready_ms - time_ms)
            )
            time_ms = finish
            wait_busy += step
            other_done += step
            timeline.append("B:microbatch %.3fms" % step)
            if finish > source_ready_ms:
                overrun_used = True
                break

    scheduled_step_finish_ms = time_ms
    idle_ms = max(0.0, source_ready_ms - time_ms)
    if idle_ms:
        time_ms += idle_ms
        timeline.append("idle %.3fms" % idle_ms)
    a_resume_ms = max(source_ready_ms, scheduled_step_finish_ms)
    post_ready_blocking_ms = a_resume_ms - source_ready_ms
    time_ms = a_resume_ms
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
        source_ready_ms=source_ready_ms,
        scheduled_step_finish_ms=scheduled_step_finish_ms,
        a_resume_ms=a_resume_ms,
        post_ready_blocking_ms=post_ready_blocking_ms,
        hidden_work_ms=hidden_work,
        useful_other_request_work_ms=other_done,
        load_interference_ms=scenario.load_interference_ms,
        gpu_busy_ms=(
            wait_busy + remaining_repair + scenario.decode_start_ms
        ),
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

    Strict policies start only complete steps that finish before source-ready.
    The explicit bounded-overrun policy may start one non-preemptible complete
    step whose blocking stays within the configured budget. Same-layer
    requests may share one ragged batch step.
    """
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("request_id values must be unique")
    for request in requests:
        if request.ready_at_ms < 0 or request.work_ms < 0 or request.layer < 0:
            raise ValueError("invalid ready request")
    time_ms = 0.0
    useful_dense = 0.0
    hidden_work = 0.0
    wait_busy = 0.0
    overrun_used = False
    timeline: List[str] = []
    source_ready_ms = scenario.source_ready_ms
    service = {request.request_id: 0.0 for request in requests}
    demand = {request.request_id: request.work_ms for request in requests}
    dense_budget = 0
    if policy is SchedulerPolicy.A_ONLY:
        dense_budget = scenario.max_extra_dense_layers
    elif _is_hybrid(policy):
        dense_budget = min(scenario.max_extra_dense_layers, hybrid_dense_budget)
    for _ in range(dense_budget):
        if time_ms >= source_ready_ms:
            break
        finish = time_ms + scenario.dense_layer_ms
        if not _atomic_step_allowed(
            policy,
            finish,
            source_ready_ms,
            scenario.max_post_ready_overrun_ms,
            overrun_used,
        ):
            break
        hidden_work += min(
            scenario.dense_layer_ms,
            max(0.0, source_ready_ms - time_ms),
        )
        time_ms = finish
        wait_busy += scenario.dense_layer_ms
        useful_dense += scenario.dense_layer_ms
        timeline.append("A:dense")
        if finish > source_ready_ms:
            overrun_used = True
            break

    if policy is SchedulerPolicy.B_ONLY or _is_hybrid(policy):
        while time_ms < source_ready_ms:
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
                next_time = min(min(future), source_ready_ms)
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
            selected_batch = None
            selected_step = 0.0
            for lead in ready:
                batch = [lead]
                if ragged_same_layer_batch:
                    batch = [
                        request
                        for request in ready
                        if request.layer == lead.layer
                    ]
                step = min(
                    [scenario.microbatch_ms]
                    + [
                        request.work_ms - service[request.request_id]
                        for request in batch
                    ]
                )
                if step <= 0:
                    continue
                if _atomic_step_allowed(
                    policy,
                    time_ms + step,
                    source_ready_ms,
                    scenario.max_post_ready_overrun_ms,
                    overrun_used,
                ):
                    selected_batch = batch
                    selected_step = step
                    break
            if selected_batch is None:
                break
            start_ms = time_ms
            time_ms += selected_step
            wait_busy += selected_step
            hidden_work += min(
                selected_step,
                max(0.0, source_ready_ms - start_ms),
            )
            for request in selected_batch:
                service[request.request_id] += selected_step
            timeline.append(
                "batch:%s"
                % ",".join(
                    request.request_id for request in selected_batch
                )
            )
            if time_ms > source_ready_ms:
                overrun_used = True
                break

    scheduled_step_finish_ms = time_ms
    if time_ms < source_ready_ms:
        time_ms = source_ready_ms
    a_resume_ms = max(source_ready_ms, scheduled_step_finish_ms)
    post_ready_blocking_ms = a_resume_ms - source_ready_ms
    time_ms = a_resume_ms
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
        a_ttft_ms=a_ttft,
        service_ms_by_request=service,
        elapsed_wait_window_ms=a_resume_ms,
        useful_a_dense_ms=useful_dense,
        jain_fairness=fairness,
        timeline=tuple(timeline),
        source_ready_ms=source_ready_ms,
        scheduled_step_finish_ms=scheduled_step_finish_ms,
        a_resume_ms=a_resume_ms,
        post_ready_blocking_ms=post_ready_blocking_ms,
        hidden_work_ms=hidden_work,
        useful_other_request_work_ms=sum(service.values()),
        load_interference_ms=scenario.load_interference_ms,
        gpu_busy_ms=(
            wait_busy + remaining_repair + scenario.decode_start_ms
        ),
    )
