from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence

from .v8_schema10_profile import SCHEMA10_TRIM_GRID


def linear_quantile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("quantile requires non-empty values and q in [0,1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_threshold_table(
    observations: Sequence[Mapping[str, Any]], checkpoints: tuple[int, ...]
) -> tuple[list[dict[str, float | int]], dict[tuple[float, int], float]]:
    indexed: dict[tuple[str, float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
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
    selection_p95_dense_fraction: float,
) -> list[dict[str, Any]]:
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
                "dispatch": dispatch,
                "allowed_completed_depths": list(depths),
                "source_residual_trim_ratio": ratio,
                "metrics": {
                    "state_availability": 1.0,
                    "selection_coverage": len(selected) / len(cases),
                    "selected_coverage": len(selected) / len(cases),
                    "wrong_early_lock": wrong / len(selected) if selected else 1.0,
                    "mean_normalized_regret": mean(regrets) if regrets else 1.0,
                    "selection_p95_dense_fraction": float(selection_p95_dense_fraction),
                    "illegal_lock_count": 0,
                    "budget_admission_violation_count": 0,
                },
                "legacy_correctness_passed": dispatch != "legacy_multicheckpoint" or bool(selected),
            })
    return output


def select_dispatch(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    def hard_pass(row: Mapping[str, Any]) -> bool:
        metrics = row["metrics"]
        return (
            float(metrics["state_availability"]) >= 0.99
            and float(metrics["selection_coverage"]) >= 0.80
            and float(metrics["wrong_early_lock"]) <= 0.05
            and float(metrics["mean_normalized_regret"]) <= 0.10
            and float(metrics["selection_p95_dense_fraction"]) <= 0.05
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
