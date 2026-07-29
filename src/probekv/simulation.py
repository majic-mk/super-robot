from __future__ import annotations

import random
import statistics
from dataclasses import asdict, replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .config import ExperimentConfig
from .contracts import CandidateBounds
from .cost import DynamicReusePlanner, LayerOption, finalize_execution
from .gates import gate_h1, gate_h2, gate_h4
from .orchestration import (
    ClosedLoopPolicy,
    RefinedCostMeasurement,
    SchedulingFeedback,
    TwoStageReuseController,
)
from .prefetch import (
    PrefetchCandidate,
    PrefetchPolicy,
    choose_prefetch,
)
from .scheduler import (
    ReadyRequest,
    SchedulerPolicy,
    SchedulerScenario,
    simulate_waiting_queue,
)
from .selector import DynamicProbeSelector, ProbePolicy, normalized_oracle_regret
from .statistics import grouped_paired_bootstrap


class _PreparedSimulationRuntime:
    """Deterministic adapter that exercises the production closure state machine."""

    def __init__(self, scheduling, refined) -> None:
        self.scheduling = scheduling
        self.refined = refined
        self.action = None

    def load_and_schedule(self, selection):
        if self.scheduling is None:
            raise RuntimeError("simulation has no selected-source schedule")
        return self.scheduling

    def measure_refined_cost(self, selection, scheduling):
        if self.refined is None:
            raise RuntimeError("simulation has no refined cost")
        return self.refined

    def execute_reuse(self, selection, decision):
        self.action = "reuse"
        return self.action

    def execute_full(self, selection, decision):
        self.action = "full_recompute"
        return self.action


def run_local_simulation(config: ExperimentConfig) -> Dict[str, Any]:
    """Exercise decision logic with deterministic latent costs.

    The result is a software-validation artifact, not empirical model evidence.
    """
    randomizer = random.Random(config.seed)
    selector = DynamicProbeSelector(
        ProbePolicy(
            config.probe_checkpoints,
            config.max_selection_layer,
            config.selector_policy,
            config.gamma,
            config.reuse_ratio_tolerance,
            config.preliminary_economic_filter,
        )
    )
    planner = DynamicReusePlanner(config.gamma)
    rows: List[Dict[str, Any]] = []
    spreads: List[float] = []
    oracle_improvements: List[float] = []
    baseline_regrets: List[float] = []
    probe_regrets: List[float] = []
    regret_reductions: Dict[str, List[float]] = {}
    admitted_reuse: List[float] = []
    admitted_full: List[float] = []
    early_count = 0
    accepted_count = 0
    simulated_ttft: List[float] = []
    simulated_fairness: List[float] = []

    for case_index in range(config.cases):
        case_id = "sim-%04d" % case_index
        full_ms = 115.0 + randomizer.uniform(-5.0, 5.0)
        safe_ratios = [
            min(0.85, max(0.05, randomizer.betavariate(2.0, 5.0)))
            for _ in range(config.online_kmax)
        ]
        # Ensure the harness regularly exercises a meaningful source spread.
        if case_index % 3 == 0 and len(safe_ratios) >= 2:
            safe_ratios[0] = min(safe_ratios[0], 0.12)
            safe_ratios[-1] = max(safe_ratios[-1], 0.35)
        costs = [
            18.0 + 70.0 * ratio + randomizer.uniform(0.0, 4.0)
            for ratio in safe_ratios
        ]
        oracle_index = min(range(len(costs)), key=lambda index: costs[index])
        latest_index = len(costs) - 1
        spreads.append(max(safe_ratios) - min(safe_ratios))
        oracle_improvements.append(
            max(0.0, (costs[latest_index] - costs[oracle_index]) / costs[latest_index])
        )

        bounds_by_layer = {}
        for layer in config.probe_checkpoints:
            width = 16.0 / max(1.0, layer ** 0.8)
            bounds = []
            for source_index, cost in enumerate(costs):
                centre_noise = randomizer.uniform(-width * 0.25, width * 0.25)
                centre = cost + centre_noise
                bounds.append(
                    CandidateBounds(
                        source_id="s%d" % source_index,
                        repair_ratio_upper=min(1.0, safe_ratios[source_index] + 0.02),
                        cost_lower_ms=max(0.0, centre - width),
                        cost_upper_ms=centre + width,
                        quality_covered=True,
                    )
                )
            bounds_by_layer[layer] = tuple(bounds)
        decision = selector.select(
            bounds_by_layer,
            full_recompute_ms=full_ms,
        )
        if not decision.abstained:
            accepted_count += 1
            if decision.probe_layer <= round(config.total_layers * 0.19):
                early_count += 1
            selected_index = int(decision.selected_source_id[1:])
        else:
            selected_index = None

        worst = max(costs)
        baseline_regret = normalized_oracle_regret(
            costs[latest_index], costs[oracle_index], worst
        )
        probe_regret = (
            normalized_oracle_regret(
                costs[selected_index], costs[oracle_index], worst
            )
            if selected_index is not None
            else 1.0
        )
        baseline_regrets.append(baseline_regret)
        probe_regrets.append(probe_regret)
        regret_reductions.setdefault(case_id, []).append(
            baseline_regret - probe_regret
        )

        reuse_plan = None
        layer_options = []
        if selected_index is not None:
            for layer in range(
                decision.probe_layer, min(config.total_layers, 16)
            ):
                ratio = safe_ratios[selected_index]
                repair_ms = (config.total_layers - layer) * 2.7 * ratio
                load_ms = 14.0 + randomizer.uniform(0.0, 5.0)
                overlap_ms = max(
                    0.0, layer - decision.probe_layer
                ) * 2.2
                layer_options.append(
                    LayerOption(
                        layer=layer,
                        repair_ratio_upper=ratio,
                        probe_ms=decision.probe_layer * 0.35,
                        compare_ms=0.35 * config.online_kmax,
                        load_ms=load_ms,
                        overlap_ms=overlap_ms,
                        repair_ms=repair_ms,
                        full_ms=full_ms,
                        buffer_ready=load_ms <= overlap_ms + 12.0,
                    )
                )
            reuse_plan = planner.plan(layer_options)
        provisional_execution = finalize_execution(decision, reuse_plan)
        refined_closure = (
            config.closed_loop_policy
            is ClosedLoopPolicy.TWO_STAGE_REFINED_ADMISSION
        )
        source_load_allowed = (
            selected_index is not None
            if refined_closure
            else provisional_execution.reuse_accepted
        )

        inverse_costs = [1.0 / cost for cost in costs]
        inverse_total = sum(inverse_costs)
        prefetch_candidates = (
            [
                PrefetchCandidate(
                    "s%d" % index,
                    inverse_cost / inverse_total,
                    256_000_000,
                    15.0 + index,
                )
                for index, inverse_cost in enumerate(inverse_costs)
            ]
            if source_load_allowed
            else []
        )
        prefetch = choose_prefetch(
            PrefetchPolicy.DYNAMIC,
            prefetch_candidates,
            hbm_available_bytes=768_000_000,
            overlap_ms=decision.probe_layer * 0.4,
        )
        scheduler_policy = (
            config.scheduler_policy
            if source_load_allowed
            else SchedulerPolicy.NO_OVERLAP
        )
        schedule = simulate_waiting_queue(
            scheduler_policy,
            SchedulerScenario(
                load_ms=(
                    max(6.0, prefetch.expected_visible_load_ms + 4.0)
                    if source_load_allowed
                    else 0.0
                ),
                dense_layer_ms=1.2,
                max_extra_dense_layers=4,
                repair_ms=(
                    min(
                        option.repair_ms for option in layer_options
                    )
                    if source_load_allowed and layer_options
                    else full_ms
                ),
                decode_start_ms=1.0,
                other_ready_work_ms=20.0,
                microbatch_ms=0.5,
                max_post_ready_overrun_ms=(
                    config.max_post_ready_overrun_ms
                ),
                load_interference_ms=config.load_interference_ms,
            ),
            (
                ReadyRequest("b", 0.0, decision.probe_layer, 8.0),
                ReadyRequest("c", 0.5, decision.probe_layer, 8.0),
                ReadyRequest("d", 1.0, decision.probe_layer + 1, 8.0),
            ),
            a_layer=decision.probe_layer,
            hybrid_dense_budget=2,
        )
        closed_loop_result = None
        if refined_closure:
            scheduling_feedback = None
            refined_cost = None
            if selected_index is not None:
                dense_layers = int(
                    round(schedule.useful_a_dense_ms / 1.2)
                )
                evaluated_boundary = min(
                    layer_options[-1].layer,
                    decision.probe_layer + dense_layers,
                )
                selected_option = min(
                    layer_options,
                    key=lambda option: (
                        abs(option.layer - evaluated_boundary),
                        option.layer,
                    ),
                )
                scheduling_feedback = SchedulingFeedback(
                    selected_source_id=decision.selected_source_id,
                    evaluated_reuse_boundary=selected_option.layer,
                    source_ready=True,
                    load_ms=schedule.source_ready_ms,
                    overlap_ms=min(
                        schedule.source_ready_ms,
                        schedule.hidden_work_ms,
                    ),
                    source_ready_ms=schedule.source_ready_ms,
                    scheduled_step_finish_ms=(
                        schedule.scheduled_step_finish_ms
                    ),
                    a_resume_ms=schedule.a_resume_ms,
                    post_ready_blocking_ms=(
                        schedule.post_ready_blocking_ms
                    ),
                    load_interference_ms=schedule.load_interference_ms,
                    useful_a_dense_ms=schedule.useful_a_dense_ms,
                    useful_other_request_work_ms=(
                        schedule.useful_other_request_work_ms
                    ),
                )
                refined_cost = RefinedCostMeasurement(
                    selected_source_id=decision.selected_source_id,
                    evaluated_reuse_boundary=selected_option.layer,
                    repair_ratio_upper=selected_option.repair_ratio_upper,
                    probe_ms=selected_option.probe_ms,
                    compare_ms=selected_option.compare_ms,
                    repair_ms=max(
                        0.0,
                        selected_option.repair_ms
                        - schedule.useful_a_dense_ms,
                    ),
                    full_ms=selected_option.full_ms,
                )
            runtime = _PreparedSimulationRuntime(
                scheduling_feedback, refined_cost
            )
            closed_loop_result = TwoStageReuseController(
                config.gamma, config.closed_loop_policy
            ).execute(decision, runtime)
            execution = closed_loop_result.execution
        elif provisional_execution.reuse_accepted:
            selected_option = next(
                option
                for option in layer_options
                if option.layer == provisional_execution.reuse_layer
            )
            refined_option = replace(
                selected_option,
                load_ms=(
                    selected_option.load_ms
                    + schedule.load_interference_ms
                ),
                post_ready_blocking_ms=(
                    schedule.post_ready_blocking_ms
                ),
                load_interference_ms=schedule.load_interference_ms,
            )
            reuse_plan = planner.plan([refined_option])
            execution = finalize_execution(decision, reuse_plan)
        else:
            execution = provisional_execution
        if execution.reuse_accepted and execution.timing is not None:
            admitted_reuse.append(execution.timing.reuse_total_ms)
            admitted_full.append(execution.timing.full_ms)
        elif source_load_allowed:
            remaining_full = max(
                0.0, full_ms - schedule.useful_a_dense_ms
            )
            schedule = replace(
                schedule,
                a_ttft_ms=(
                    schedule.a_resume_ms
                    + remaining_full
                    + 1.0
                ),
            )
        simulated_ttft.append(schedule.a_ttft_ms)
        simulated_fairness.append(schedule.jain_fairness)

        rows.append(
            {
                "case_id": case_id,
                "safe_ratios": safe_ratios,
                "source_costs_ms": costs,
                "oracle_source": "s%d" % oracle_index,
                "selected_source": decision.selected_source_id,
                "abstained": decision.abstained,
                "selection_reason": decision.selection_reason.value,
                "probe_layer": decision.probe_layer,
                "normalized_regret": probe_regret,
                "reuse_accepted": execution.reuse_accepted,
                "rejection_reason": (
                    execution.rejection_reason.value
                    if execution.rejection_reason is not None
                    else None
                ),
                "execution_mode": execution.execution_mode.value,
                "reuse_layer": execution.reuse_layer,
                "dynamic_prefetch_sources": prefetch.source_ids,
                "dynamic_prefetch_bytes": prefetch.transferred_bytes,
                "hybrid_a_ttft_ms": schedule.a_ttft_ms,
                "hybrid_queue_fairness": schedule.jain_fairness,
                "source_ready_ms": schedule.source_ready_ms,
                "a_resume_ms": schedule.a_resume_ms,
                "post_ready_blocking_ms": (
                    schedule.post_ready_blocking_ms
                ),
                "useful_other_request_work_ms": (
                    schedule.useful_other_request_work_ms
                ),
                "load_interference_ms": schedule.load_interference_ms,
                "closure_policy": config.closed_loop_policy.value,
                "selection_state": execution.selection_state.value,
                "probe_admission_state": decision.admission_state.value,
                "admission_state": execution.admission_state.value,
                "evaluated_reuse_boundary": (
                    closed_loop_result.scheduling.evaluated_reuse_boundary
                    if closed_loop_result is not None
                    and closed_loop_result.scheduling is not None
                    else (
                        reuse_plan.evaluated_layer
                        if reuse_plan is not None
                        else None
                    )
                ),
                "actual_reuse_boundary": (
                    execution.actual_reuse_boundary
                ),
                "evidence_class": "local_simulation",
                "paper_evidence": False,
            }
        )

    improvement_ci = grouped_paired_bootstrap(
        {"all": oracle_improvements},
        iterations=min(2000, max(500, config.cases * 20)),
        seed=config.seed,
    )
    regret_ci = grouped_paired_bootstrap(
        regret_reductions,
        iterations=min(2000, max(500, config.cases * 20)),
        seed=config.seed + 1,
    )
    mean_baseline = statistics.mean(baseline_regrets)
    mean_probe = statistics.mean(probe_regrets)
    h1 = gate_h1(spreads, oracle_improvements, improvement_ci)
    h2 = gate_h2(
        mean_baseline,
        mean_probe,
        regret_ci,
        early_count / float(max(1, accepted_count)),
        overhead_fraction=0.03,
    )
    h4 = gate_h4(admitted_reuse, admitted_full, config.gamma)
    return {
        "summary": {
            "cases": config.cases,
            "accepted": accepted_count,
            "abstained": config.cases - accepted_count,
            "mean_baseline_regret": mean_baseline,
            "mean_probe_regret": mean_probe,
            "admitted_reuse": len(admitted_reuse),
            "mean_hybrid_a_ttft_ms": statistics.mean(simulated_ttft),
            "mean_hybrid_queue_fairness": statistics.mean(simulated_fairness),
            "evidence_class": "local_simulation",
            "paper_evidence": False,
        },
        "gates": [asdict(h1), asdict(h2), asdict(h4)],
        "rows": rows,
    }
