from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean
from typing import Any, Mapping, Sequence

from .v8_schema10_profile import SCHEMA10_TRIM_GRID


def linear_quantile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("quantile requires non-empty values and q in [0,1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("quantile samples must be finite")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_threshold_table(
    observations: Sequence[Mapping[str, Any]], checkpoints: tuple[int, ...]
) -> tuple[list[dict[str, float | int]], dict[tuple[float, int], float]]:
    """Fit score-proxy thresholds only; this is not held-out QA validation."""
    indexed: dict[tuple[str, float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        if not math.isfinite(float(row["residual_score"])) or float(row["residual_score"]) < 0:
            raise ValueError("residual observations must be finite and non-negative")
        indexed[(str(row["case_id"]), float(row["source_residual_trim_ratio"]), int(row["completed_depth"]))].append(row)
    thresholds: dict[tuple[float, int], float] = {}
    output: list[dict[str, float | int]] = []
    deepest = checkpoints[-1]
    case_ids = sorted({str(row["case_id"]) for row in observations})
    for ratio in SCHEMA10_TRIM_GRID:
        oracle = {
            case_id: min(
                indexed[(case_id, ratio, deepest)],
                key=lambda row: (float(row["residual_score"]), str(row["source_id"])),
            )
            for case_id in case_ids
        }
        for depth in checkpoints:
            best = {
                case_id: min(
                    indexed[(case_id, ratio, depth)],
                    key=lambda row: (float(row["residual_score"]), str(row["source_id"])),
                )
                for case_id in case_ids
            }
            candidates = sorted({float(row["residual_score"]) for row in best.values()})
            chosen = 0.0
            for candidate in candidates:
                selected = [row for row in best.values() if float(row["residual_score"]) <= candidate]
                wrong = sum(
                    str(row["source_id"]) != str(oracle[str(row["case_id"])]["source_id"])
                    for row in selected
                )
                if selected and wrong / len(selected) <= 0.05:
                    chosen = candidate
            thresholds[(ratio, depth)] = chosen
            output.append({
                "source_residual_trim_ratio": ratio,
                "completed_depth": depth,
                "upper_residual": chosen,
            })
    return output, thresholds


def build_selection_candidates(
    observations: Sequence[Mapping[str, Any]],
    checkpoints: tuple[int, ...],
    thresholds: Mapping[tuple[float, int], float],
    selection_p95_dense_fraction: float | None,
) -> list[dict[str, Any]]:
    """Read-only score diagnostic, never a certificate for the online selector.

    Online decisions additionally require margins, exact candidate counts and
    Gate1 plans. Use replay_production_d1d2 for that policy, not this surrogate.
    The legacy timing argument is retained for callers but cannot certify a
    request budget, because it may contain generation latency/microbenchmarks.
    """
    indexed: dict[tuple[str, float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        indexed[(str(row["case_id"]), float(row["source_residual_trim_ratio"]), int(row["completed_depth"]))].append(row)
    cases = sorted({str(row["case_id"]) for row in observations})
    output = []
    for ratio in SCHEMA10_TRIM_GRID:
        oracle = {
            case_id: min(
                indexed[(case_id, ratio, checkpoints[-1])],
                key=lambda row: (float(row["residual_score"]), str(row["source_id"])),
            )
            for case_id in cases
        }
        for dispatch, depths in (
            ("d1_only", (1,)),
            ("d1_d2_rescue", (1, 2)),
            ("legacy_multicheckpoint", checkpoints),
        ):
            selected = []
            regrets = []
            for case_id in cases:
                choice = None
                for depth in depths:
                    current = min(
                        indexed[(case_id, ratio, depth)],
                        key=lambda row: (float(row["residual_score"]), str(row["source_id"])),
                    )
                    if float(current["residual_score"]) <= thresholds[(ratio, depth)]:
                        choice = current
                        break
                if choice is None:
                    continue
                selected.append(choice)
                oracle_row = oracle[case_id]
                chosen_at_deep = next(
                    row for row in indexed[(case_id, ratio, checkpoints[-1])]
                    if row["source_id"] == choice["source_id"]
                )
                regrets.append(
                    max(0.0, float(chosen_at_deep["residual_score"]) - float(oracle_row["residual_score"]))
                    / max(float(oracle_row["residual_score"]), 1e-12)
                )
            wrong = sum(
                str(row["source_id"]) != str(oracle[str(row["case_id"])]["source_id"])
                for row in selected
            )
            output.append({
                "evidence_kind": "deep_residual_proxy_diagnostic",
                "profile_freeze_eligible": False,
                "oracle_kind": "deep_residual_score_not_quality_or_cost",
                "dispatch": dispatch,
                "allowed_completed_depths": list(depths),
                "source_residual_trim_ratio": ratio,
                "metrics": {
                    "state_availability": None,
                    "selection_coverage": len(selected) / len(cases),
                    "selected_coverage": len(selected) / len(cases),
                    "wrong_early_lock": wrong / len(selected) if selected else 1.0,
                    "mean_normalized_regret": mean(regrets) if regrets else 1.0,
                    "selection_p95_dense_fraction": None,
                    "illegal_lock_count": None,
                    "budget_admission_violation_count": None,
                },
                "legacy_correctness_passed": None,
            })
    return output


def select_dispatch(
    rows: Sequence[Mapping[str, Any]], *, diagnostic_only: bool = False,
) -> Mapping[str, Any]:
    if diagnostic_only:
        # Explicit reference for forced repair experiments, not Profile freeze.
        reference = [row for row in rows if row["dispatch"] == "legacy_multicheckpoint"
                     and float(row["source_residual_trim_ratio"]) == 0.15]
        if len(reference) != 1:
            raise ValueError("diagnostic reference requires legacy/rho=0.15")
        return reference[0]

    def hard_pass(row: Mapping[str, Any]) -> bool:
        if row.get("evidence_kind") != "production_policy_replay":
            return False
        budget_policy = row.get("selection_budget_policy", "legacy_fixed_fraction")
        if budget_policy not in {"legacy_fixed_fraction", "end_to_end_aware"}:
            return False
        metrics = row["metrics"]
        required = ("state_availability", "selection_coverage", "wrong_early_lock",
                    "mean_normalized_regret", "selection_p95_dense_fraction",
                    "illegal_lock_count", "budget_admission_violation_count")
        if any(metrics.get(key) is None or not math.isfinite(float(metrics[key])) for key in required):
            return False
        return (
            float(metrics["state_availability"]) >= 0.99
            and float(metrics["selection_coverage"]) >= 0.80
            and float(metrics["wrong_early_lock"]) <= 0.05
            and float(metrics["mean_normalized_regret"]) <= 0.10
            and (budget_policy == "end_to_end_aware"
                 or float(metrics["selection_p95_dense_fraction"]) <= 0.05)
            and int(metrics["illegal_lock_count"]) == 0
            and int(metrics["budget_admission_violation_count"]) == 0
            and (
                row["dispatch"] != "legacy_multicheckpoint"
                or bool(row.get("legacy_correctness_passed", False))
            )
        )

    feasible = [row for row in rows if hard_pass(row)]
    if not feasible:
        raise ValueError("no selection dispatch passed its frozen hard Gate")
    feasible.sort(key=lambda row: (
        -float(row["metrics"].get("selected_coverage", 0.0)),
        float(row["metrics"].get("mean_normalized_regret", 1.0)),
        float(row["metrics"].get("selection_p95_dense_fraction", 1.0)),
        len(row["allowed_completed_depths"]),
        str(row["dispatch"]),
    ))
    return feasible[0]


def replay_production_d1d2(selector: Any, checkpoints: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    """Execute the actual online selector over recorded inputs, without I/O.

    Counts and Gate1LocalPlan objects must come from the recorded request, not
    be reconstructed from a residual threshold. No Source is reconsidered
    after a decision-ready/abstain outcome. Legacy replay needs its own actual
    controller and is deliberately not emulated by a threshold-only loop.
    """
    decisions = []
    previous = None
    last_depth = 0
    for step in checkpoints:
        depth = int(step["completed_depth"])
        if depth <= last_depth:
            raise ValueError("replay checkpoints must strictly increase")
        last_depth = depth
        candidates = tuple(step["candidates"])
        decision = selector.decide(
            completed_depth=depth, counts=step["counts"], candidates=candidates,
            gate1_plan_by_source=step["gate1_plan_by_source"],
            previous_best_source_variant_id=previous,
        )
        decisions.append(decision)
        if decision.state != "continue_probe":
            break
        if candidates:
            previous = min(candidates, key=lambda x: (x.residual_score, x.source_variant_id)).source_variant_id
    return tuple(decisions)


def select_case_source(
    observations: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    selected_dispatch: Mapping[str, Any],
    thresholds: Mapping[tuple[float, int], float],
) -> Mapping[str, Any] | None:
    ratio = float(selected_dispatch["source_residual_trim_ratio"])
    for depth in tuple(int(value) for value in selected_dispatch["allowed_completed_depths"]):
        rows = [
            row for row in observations
            if str(row["case_id"]) == case_id
            and float(row["source_residual_trim_ratio"]) == ratio
            and int(row["completed_depth"]) == depth
        ]
        winner = min(rows, key=lambda row: (float(row["residual_score"]), str(row["source_id"])))
        if float(winner["residual_score"]) <= thresholds[(ratio, depth)]:
            return winner
    return None
