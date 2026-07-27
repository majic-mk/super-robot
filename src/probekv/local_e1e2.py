from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .backend import DeterministicSimulationBackend
from .calibration import (
    CalibratedGradientBoostingIntervalPredictor,
    QuantileGradientBoostingBudgetPredictor,
)
from .config import ExperimentConfig
from .contracts import CandidateBounds, ProbeObservation
from .features import CacheCraftMetadata, cache_craft_style_score, combined_feature_vector
from .io import atomic_write_json, try_write_parquet, write_jsonl
from .labeling import RatioMeasurement, safe_repair_ratio
from .manifest import ManifestCase, manifest_digest, synthetic_manifest
from .resume import StageLedger
from .selector import DynamicProbeSelector, ProbePolicy, normalized_oracle_regret
from .statistics import kendall_tau, spearman_correlation


PRIMARY_REUSE_FRACTION = 0.15
REUSE_FRACTIONS = (0.10, 0.15, 0.22, 0.30, 0.40)
PIPELINE_REVISION = "local-e1e2-v2"


def _stable_unit(*parts: object) -> float:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2 ** 64)


def _latent_safe_ratio(case: ManifestCase, source_id: str, reuse_layer: int) -> float:
    source_index = int(source_id[1:])
    base = 0.07 + 0.45 * _stable_unit(case.case_id, source_id, "latent")
    if "high-prefix" in case.regime:
        base -= 0.035 * (3 - source_index)
    if "same-order" in case.regime:
        base -= 0.018 * (source_index % 2 == 0)
    layer_adjustment = 0.004 * (reuse_layer - round(32 * PRIMARY_REUSE_FRACTION))
    return min(0.75, max(0.03, base + layer_adjustment))


def _metadata(case: ManifestCase, source_id: str) -> CacheCraftMetadata:
    source_index = int(source_id[1:])
    jitter = (_stable_unit(case.case_id, source_id, "metadata") - 0.5) * 0.16
    prefix = (0.78 if "high-prefix" in case.regime else 0.22) + jitter
    order = (0.78 if "same-order" in case.regime else 0.22) - jitter / 2.0
    cci = 0.35 + 0.08 * ((source_index + 1) % 3) + jitter
    cfo = 0.40 + 0.06 * (3 - source_index) - jitter
    clamp = lambda value: min(1.0, max(0.0, value))
    return CacheCraftMetadata(clamp(prefix), clamp(order), clamp(cci), clamp(cfo))


def _probe_observation(
    case: ManifestCase,
    source_id: str,
    layer: int,
    maximum_layer: int,
    target_ratio: float,
) -> ProbeObservation:
    information = math.sqrt(layer / float(maximum_layer))
    amplitude = 0.16 * (1.0 - information) + 0.008

    def noise(channel: str) -> float:
        return (_stable_unit(case.case_id, source_id, layer, channel) - 0.5) * 2.0 * amplitude

    metadata = _metadata(case, source_id)
    k_drift = max(0.0, target_ratio + noise("k"))
    v_drift = max(0.0, target_ratio * 0.92 + noise("v"))
    hidden_drift = max(0.0, target_ratio * 0.78 + noise("h"))
    query_score = max(0.0, target_ratio * 0.65 + noise("q"))
    return ProbeObservation(
        case.case_id,
        source_id,
        layer,
        k_drift,
        v_drift,
        hidden_drift,
        query_score,
        metadata.prefix_overlap,
        metadata.order_score,
        comparison_latency_ms=0.018 + 0.004 * layer,
    )


def _fingerprint(config: ExperimentConfig, cases: Sequence[ManifestCase]) -> str:
    payload = {
        "pipeline": PIPELINE_REVISION,
        "config": asdict(config),
        "manifest_digest": manifest_digest(cases),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()


def _feature(observation: ProbeObservation, case: ManifestCase) -> Tuple[float, ...]:
    return combined_feature_vector(
        observation, _metadata(case, observation.source_id)
    )


def run_local_e1e2(
    config: ExperimentConfig, output: Path, resume: bool = False
) -> Dict[str, Any]:
    """Run a complete synthetic E1/E2 software-validation loop.

    The pipeline exercises the same split, labeling, fitting, calibration,
    dynamic early-exit and audit boundaries planned for the A100 run.  Its
    latent labels and timings are synthetic, so every artifact is explicitly
    marked as non-paper evidence.
    """
    if config.evidence_class != "local_simulation":
        raise ValueError("local E1/E2 requires evidence_class=local_simulation")
    output.mkdir(parents=True, exist_ok=True)
    cases = synthetic_manifest(
        config.cases, config.seed, online_kmax=config.online_kmax
    )
    fingerprint = _fingerprint(config, cases)
    ledger = StageLedger(output / "ledger.json")
    final_outputs = (output / "summary.json", output / "decisions.jsonl")
    if resume and ledger.completed("evaluate", fingerprint, final_outputs):
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        summary["resumed"] = True
        return summary

    case_lookup = {case.case_id: case for case in cases}
    manifest_rows = [
        dict(case.to_row(), evidence_class="local_fixture", paper_evidence=False)
        for case in cases
    ]
    write_jsonl(output / "case_manifest.jsonl", manifest_rows)
    atomic_write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "manifest_digest": manifest_digest(cases),
            "pipeline_fingerprint": fingerprint,
            "cases": len(cases),
            "split_counts": {
                split: sum(case.split == split for case in cases)
                for split in ("train", "calibration", "test")
            },
            "model_signature": "synthetic-reference-v1",
            "evidence_class": "local_fixture",
            "paper_evidence": False,
        },
    )
    ledger.mark_complete(
        "manifest",
        fingerprint,
        (output / "manifest.json", output / "case_manifest.jsonl"),
    )

    primary_layer = max(1, round(config.total_layers * PRIMARY_REUSE_FRACTION))
    reuse_layers = tuple(
        sorted(
            {
                max(1, min(config.total_layers - 1, round(config.total_layers * fraction)))
                for fraction in REUSE_FRACTIONS
            }
        )
    )
    ratio_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []
    primary_labels: Dict[Tuple[str, str], float] = {}
    observation_rows: List[Dict[str, Any]] = []
    observations: Dict[Tuple[str, str, int], ProbeObservation] = {}

    for case in cases:
        for source in case.sources:
            for reuse_layer in reuse_layers:
                threshold = _latent_safe_ratio(case, source.source_id, reuse_layer)
                backend = DeterministicSimulationBackend(
                    total_layers=config.total_layers,
                    safe_ratio_by_source={source.source_id: threshold},
                )
                source_stub = _historical_source_stub(case, source.source_id)
                measurements = []
                results_by_ratio = {}
                for ratio in config.repair_ratios:
                    result = backend.repair(source_stub, reuse_layer, ratio)
                    measurement = RatioMeasurement(
                        ratio,
                        task_score_drop=1.0 - result.quality_score,
                        token_f1=result.token_f1,
                    )
                    measurements.append(measurement)
                    results_by_ratio[ratio] = result
                    ratio_rows.append(
                        {
                            "case_id": case.case_id,
                            "source_id": source.source_id,
                            "split": case.split,
                            "reuse_layer": reuse_layer,
                            "repair_ratio": ratio,
                            "task_score_drop": measurement.task_score_drop,
                            "token_f1": measurement.token_f1,
                            "repair_latency_ms": result.latency_ms,
                            "evidence_class": "local_simulation",
                            "paper_evidence": False,
                        }
                    )
                safe = safe_repair_ratio(measurements)
                if safe is None:
                    raise RuntimeError("full repair must pass the synthetic backend")
                safe_result = results_by_ratio[safe]
                label_rows.append(
                    {
                        "case_id": case.case_id,
                        "source_id": source.source_id,
                        "split": case.split,
                        "reuse_layer": reuse_layer,
                        "safe_repair_ratio": safe,
                        "quality_score": safe_result.quality_score,
                        "token_f1": safe_result.token_f1,
                        "repair_latency_ms": safe_result.latency_ms,
                        "evidence_class": "local_simulation",
                        "paper_evidence": False,
                    }
                )
                if reuse_layer == primary_layer:
                    primary_labels[(case.case_id, source.source_id)] = safe
            target = primary_labels[(case.case_id, source.source_id)]
            for layer in config.probe_checkpoints:
                observation = _probe_observation(
                    case, source.source_id, layer, config.probe_checkpoints[-1], target
                )
                observations[(case.case_id, source.source_id, layer)] = observation
                observation_rows.append(
                    dict(
                        asdict(observation),
                        split=case.split,
                        target_safe_ratio=target,
                        evidence_class="local_simulation",
                        paper_evidence=False,
                    )
                )

    write_jsonl(output / "ratio_measurements.jsonl", ratio_rows)
    write_jsonl(output / "safe_budget_labels.jsonl", label_rows)
    write_jsonl(output / "probe_observations.jsonl", observation_rows)
    try_write_parquet(output / "safe_budget_labels.parquet", label_rows)
    try_write_parquet(output / "probe_observations.parquet", observation_rows)
    ledger.mark_complete(
        "label_and_probe",
        fingerprint,
        (
            output / "ratio_measurements.jsonl",
            output / "safe_budget_labels.jsonl",
            output / "probe_observations.jsonl",
        ),
        {"ratio_rows": len(ratio_rows), "observation_rows": len(observation_rows)},
    )

    upper_models = {}
    interval_models = {}
    model_rows = []
    for layer in config.probe_checkpoints:
        train_pairs = [
            (observation, primary_labels[(observation.case_id, observation.source_id)])
            for key, observation in observations.items()
            if key[2] == layer and case_lookup[observation.case_id].split == "train"
        ]
        calibration_pairs = [
            (observation, primary_labels[(observation.case_id, observation.source_id)])
            for key, observation in observations.items()
            if key[2] == layer and case_lookup[observation.case_id].split == "calibration"
        ]
        train_x = [_feature(obs, case_lookup[obs.case_id]) for obs, _ in train_pairs]
        train_y = [target for _, target in train_pairs]
        calibration_x = [_feature(obs, case_lookup[obs.case_id]) for obs, _ in calibration_pairs]
        calibration_y = [target for _, target in calibration_pairs]
        if not train_x or not calibration_x:
            raise RuntimeError("train and calibration splits are both required")
        upper = QuantileGradientBoostingBudgetPredictor(
            random_state=config.seed + layer
        ).fit(train_x, train_y, calibration_x, calibration_y)
        interval = CalibratedGradientBoostingIntervalPredictor(
            random_state=config.seed + layer
        ).fit(train_x, train_y, calibration_x, calibration_y)
        upper_models[layer] = upper
        interval_models[layer] = interval
        model_rows.append(
            {
                "layer": layer,
                "train_rows": len(train_x),
                "calibration_rows": len(calibration_x),
                "upper_correction": upper.calibrator.correction,
                "interval_radius": interval.calibrator.radius,
                "test_labels_accessed_during_fit": False,
            }
        )
    atomic_write_json(
        output / "calibration_report.json",
        {
            "models": model_rows,
            "thresholds_frozen_before_test": True,
            "feature_order": [
                "k_drift",
                "v_drift",
                "hidden_drift",
                "query_score",
                "prefix_overlap",
                "order_score",
                "cci",
                "cfo",
            ],
            "paper_evidence": False,
        },
    )
    ledger.mark_complete(
        "fit_and_calibrate",
        fingerprint,
        (output / "calibration_report.json",),
        {"layers": len(model_rows)},
    )

    selector = DynamicProbeSelector(
        ProbePolicy(config.probe_checkpoints, config.probe_checkpoints[-1])
    )
    decision_rows: List[Dict[str, Any]] = []
    accepted_regrets = []
    fallback_regrets = []
    cachecraft_regrets = []
    top1 = 0
    accepted = 0
    rank_taus = []
    rank_rhos = []
    spreads = []
    selected_quality_budget_violations = 0
    selected_quality_budget_trials = 0
    interval_violations = 0
    interval_trials = 0
    for case in cases:
        if case.split != "test":
            continue
        actual_ratios = {
            source.source_id: primary_labels[(case.case_id, source.source_id)]
            for source in case.sources
        }
        actual_costs = {
            source_id: 12.0 + 85.0 * ratio
            for source_id, ratio in actual_ratios.items()
        }
        oracle = min(actual_costs, key=actual_costs.get)
        worst_cost = max(actual_costs.values())
        oracle_cost = actual_costs[oracle]
        spreads.append(max(actual_ratios.values()) - min(actual_ratios.values()))
        bounds_by_layer = {}
        layer_predictions = {}
        for layer in config.probe_checkpoints:
            features = [
                _feature(
                    observations[(case.case_id, source.source_id, layer)], case
                )
                for source in case.sources
            ]
            quality_uppers = upper_models[layer].predict_upper(features)
            intervals = interval_models[layer].predict_bounds(features)
            bounds = []
            predictions = []
            for source, quality_upper, interval in zip(
                case.sources, quality_uppers, intervals
            ):
                lower_ratio, interval_upper = interval
                bounds.append(
                    CandidateBounds(
                        source.source_id,
                        quality_upper,
                        12.0 + 85.0 * lower_ratio,
                        12.0 + 85.0 * interval_upper,
                        quality_covered=True,
                    )
                )
                predictions.append((lower_ratio + interval_upper) / 2.0)
            bounds_by_layer[layer] = tuple(bounds)
            layer_predictions[layer] = predictions
        decision = selector.select(bounds_by_layer)
        if decision.abstained:
            selected = None
            selected_cost = 100.0
            selected_quality_upper = None
            quality_budget_violation = False
        else:
            selected = decision.selected_source_id
            selected_cost = actual_costs[selected]
            selected_bound = next(
                bound
                for bound in bounds_by_layer[decision.probe_layer]
                if bound.source_id == selected
            )
            selected_quality_upper = selected_bound.repair_ratio_upper
            quality_budget_violation = (
                actual_ratios[selected] > selected_quality_upper + 1e-12
            )
            selected_quality_budget_trials += 1
            selected_quality_budget_violations += int(quality_budget_violation)
            accepted += 1
            top1 += int(selected == oracle)
            accepted_regrets.append(
                normalized_oracle_regret(selected_cost, oracle_cost, worst_cost)
            )
        fallback_worst = max(worst_cost, 100.0)
        fallback_regret = normalized_oracle_regret(
            selected_cost, oracle_cost, fallback_worst
        )
        fallback_regrets.append(fallback_regret)
        metadata_scores = {
            source.source_id: cache_craft_style_score(_metadata(case, source.source_id))
            for source in case.sources
        }
        cachecraft = min(metadata_scores, key=metadata_scores.get)
        cachecraft_regrets.append(
            normalized_oracle_regret(actual_costs[cachecraft], oracle_cost, worst_cost)
        )
        final_predictions = layer_predictions[decision.probe_layer]
        actual_order = [actual_ratios[source.source_id] for source in case.sources]
        for bound in bounds_by_layer[decision.probe_layer]:
            interval_trials += 1
            actual_cost = actual_costs[bound.source_id]
            interval_violations += int(
                actual_cost < bound.cost_lower_ms - 1e-12
                or actual_cost > bound.cost_upper_ms + 1e-12
            )
        rank_taus.append(kendall_tau(final_predictions, actual_order))
        rank_rhos.append(spearman_correlation(final_predictions, actual_order))
        decision_rows.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "oracle_source": oracle,
                "selected_source": selected,
                "abstained": decision.abstained,
                "decision_reason": decision.reason.value,
                "probe_layer": decision.probe_layer,
                "actual_safe_ratios": actual_ratios,
                "actual_costs_ms": actual_costs,
                "cachecraft_source": cachecraft,
                "selected_quality_ratio_upper": selected_quality_upper,
                "selected_quality_budget_violation": quality_budget_violation,
                "candidate_bounds_by_layer": {
                    str(layer): [asdict(bound) for bound in bounds_by_layer[layer]]
                    for layer in config.probe_checkpoints
                },
                "normalized_regret_with_full_fallback": fallback_regret,
                "evidence_class": "local_simulation",
                "paper_evidence": False,
            }
        )

    test_cases = len(decision_rows)
    summary = {
        "pipeline": PIPELINE_REVISION,
        "pipeline_fingerprint": fingerprint,
        "cases": len(cases),
        "test_cases": test_cases,
        "ratio_measurements": len(ratio_rows),
        "safe_budget_labels": len(label_rows),
        "probe_observations": len(observation_rows),
        "probe_checkpoints": list(config.probe_checkpoints),
        "primary_reuse_layer": primary_layer,
        "coverage": accepted / float(max(1, test_cases)),
        "abstention_rate": (test_cases - accepted) / float(max(1, test_cases)),
        "accepted_top1_accuracy": top1 / float(max(1, accepted)),
        "selected_quality_budget_violations": selected_quality_budget_violations,
        "selected_quality_budget_coverage": 1.0
        - selected_quality_budget_violations
        / float(max(1, selected_quality_budget_trials)),
        "ranking_interval_empirical_coverage": 1.0
        - interval_violations / float(max(1, interval_trials)),
        "accepted_mean_normalized_regret": (
            statistics.mean(accepted_regrets) if accepted_regrets else None
        ),
        "full_fallback_mean_normalized_regret": statistics.mean(fallback_regrets),
        "cachecraft_mean_normalized_regret": statistics.mean(cachecraft_regrets),
        "mean_kendall_tau": statistics.mean(rank_taus),
        "mean_spearman_rho": statistics.mean(rank_rhos),
        "source_spread_ge_10pp_fraction": sum(value >= 0.10 for value in spreads)
        / float(max(1, len(spreads))),
        "thresholds_frozen_before_test": True,
        "labels": "synthetic",
        "evidence_class": "local_simulation",
        "paper_evidence": False,
        "resumed": False,
    }
    write_jsonl(output / "decisions.jsonl", decision_rows)
    try_write_parquet(output / "decisions.parquet", decision_rows)
    atomic_write_json(output / "summary.json", summary)
    ledger.mark_complete("evaluate", fingerprint, final_outputs, summary)
    return summary


def _historical_source_stub(case: ManifestCase, source_id: str):
    from .contracts import HistoricalSource, KVLocation, SourceOrigin

    return HistoricalSource(
        source_id=source_id,
        content_hash=case.content_hash,
        context_id=next(
            source.context_id for source in case.sources if source.source_id == source_id
        ),
        model_signature=case.model_signature,
        token_count=len(case.segment_token_ids),
        exact=True,
        origin=SourceOrigin.FULL_PREFILL,
        kv_location=KVLocation.PINNED_CPU,
    )
