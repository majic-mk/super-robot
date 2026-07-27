from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .statistics import ConfidenceInterval, clopper_pearson_upper_bound


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    summary: str


def gate_h1(
    source_spreads: Sequence[float],
    oracle_improvements: Sequence[float],
    improvement_ci: ConfidenceInterval,
) -> GateResult:
    if not source_spreads or not oracle_improvements:
        return GateResult("H1", False, "missing source-sensitivity evidence")
    spread_fraction = sum(spread >= 0.10 for spread in source_spreads) / float(
        len(source_spreads)
    )
    mean_improvement = sum(oracle_improvements) / len(oracle_improvements)
    passed = (
        spread_fraction >= 0.25
        and mean_improvement >= 0.10
        and improvement_ci.lower > 0.0
    )
    return GateResult(
        "H1",
        passed,
        "spread>=10pp %.1f%%; oracle improvement %.1f%%; CI [%.3f, %.3f]"
        % (
            100.0 * spread_fraction,
            100.0 * mean_improvement,
            improvement_ci.lower,
            improvement_ci.upper,
        ),
    )


def gate_h2(
    baseline_regret: float,
    probe_regret: float,
    regret_reduction_ci: ConfidenceInterval,
    early_exit_fraction: float,
    overhead_fraction: float,
) -> GateResult:
    relative_reduction = (
        (baseline_regret - probe_regret) / baseline_regret
        if baseline_regret > 0
        else 0.0
    )
    passed = (
        baseline_regret > 0.05
        and relative_reduction >= 0.20
        and regret_reduction_ci.lower > 0.0
        and early_exit_fraction >= 0.80
        and overhead_fraction <= 0.05
    )
    return GateResult(
        "H2",
        passed,
        "baseline regret %.3f; ProbeKV %.3f; reduction %.1f%%; early %.1f%%; overhead %.1f%%"
        % (
            baseline_regret,
            probe_regret,
            100.0 * relative_reduction,
            100.0 * early_exit_fraction,
            100.0 * overhead_fraction,
        ),
    )


def gate_h3(
    task_score_difference_ci: ConfidenceInterval,
    tail_violations: int,
    cases: int,
) -> GateResult:
    upper = clopper_pearson_upper_bound(tail_violations, cases, confidence=0.95)
    passed = task_score_difference_ci.lower >= -0.01 and upper <= 0.01
    return GateResult(
        "H3",
        passed,
        "quality lower CI %.4f; tail upper bound %.4f"
        % (task_score_difference_ci.lower, upper),
    )


def gate_h4(reuse_ms: Sequence[float], full_ms: Sequence[float], gamma: float = 0.8) -> GateResult:
    if len(reuse_ms) != len(full_ms) or not reuse_ms:
        return GateResult("H4", False, "missing or unpaired timing evidence")
    violations = sum(
        reuse > gamma * full for reuse, full in zip(reuse_ms, full_ms)
    )
    return GateResult(
        "H4",
        violations == 0,
        "%d/%d admitted requests violate gamma=%.2f"
        % (violations, len(reuse_ms), gamma),
    )


def publication_band(ttft_improvement: float, throughput_improvement: float, p95_improvement: float) -> str:
    if (
        ttft_improvement >= 0.10
        and throughput_improvement >= 0.10
        and p95_improvement >= 0.05
    ):
        return "q1_candidate"
    if ttft_improvement >= 0.05 and throughput_improvement >= 0.05:
        return "q2_candidate"
    if max(ttft_improvement, throughput_improvement) < 0.03:
        return "stop_or_restructure"
    return "insufficient_or_mixed"
