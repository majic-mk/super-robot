"""Executable experiment harnesses. Backends execute policies, not estimates.

The same backend entry serves requests, capacity traces and Gate1 paired A/B.
No experiment in this module can freeze profiles or claim GPU qualification.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
import random
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .v8_schema10_evidence import summarize_measured_coverage, validate_paired_gate1_executions
from .v8_schema10_execution import digest_json


class ExperimentBackend(Protocol):
    capabilities: Mapping[str, Any]

    def reset(self, *, capacity: int, global_byte_budget: int) -> None: ...
    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...
    def execute(self, request: Mapping[str, Any], dispatch: Mapping[str, Any], *,
                arrival_ns: int) -> Mapping[str, Any]: ...
    def finalize_request(self, request: Mapping[str, Any], outcome: Mapping[str, Any]) -> None: ...


def _online(outcome: Mapping[str, Any]) -> None:
    if outcome.get("execution_kind") != "online_policy" or outcome.get("forced_source") is not False:
        raise ValueError("experiment requires actual online policy execution")
    dense_abstention = (outcome.get("final_commit_not_applicable_reason") == "no_frozen_sources"
                        and outcome.get("selected_source_variant_ids") == [])
    cost_unsupported_dense = (
        outcome.get("final_commit_not_applicable_reason") in {"matched_dense_cost_unsupported", "selection_cost_unsupported"}
        and outcome.get("selected_source_variant_ids") == []
        and outcome.get("committed_source_variant_ids") == []
        and any(e.get("kind") == "dense_fallback" and e.get("reason") == outcome.get("final_commit_not_applicable_reason")
                for e in outcome.get("runtime_events", ())))
    freeze_failures = {e.get("source_id") for e in outcome.get("runtime_events", ()) if e.get("kind") == "freeze_failed"}
    freeze_failed_dense = (
        outcome.get("final_commit_not_applicable_reason") == "all_source_freezes_failed"
        and bool(outcome.get("selected_source_variant_ids"))
        and set(outcome["selected_source_variant_ids"]) <= freeze_failures
        and outcome.get("committed_source_variant_ids") == [])
    if outcome.get("final_commit_executed") is not True and not (dense_abstention or cost_unsupported_dense or freeze_failed_dense):
        raise ValueError("actual FinalCommit evidence is missing")
    if not outcome.get("selection_events") and not cost_unsupported_dense:
        raise ValueError("actual Source decision events are missing")
    if not outcome.get("runtime_events"):
        raise ValueError("actual runtime event records are missing")
    if dense_abstention and any(e.get("decision", {}).get("selected_source_variant_id")
                                for e in outcome["selection_events"]):
        raise ValueError("selected Source cannot claim selector-only dense abstention")
    if not math.isfinite(float(outcome["request_ttft_ms"])) or float(outcome["request_ttft_ms"]) <= 0:
        raise ValueError("request TTFT must be measured and positive")


def run_gate1_pairs(backend: ExperimentBackend, requests: Sequence[Mapping[str, Any]],
                   dispatch: Mapping[str, Any], *, seed: int = 20260726) -> list[dict[str, Any]]:
    """Restore an identical pool before EACH arm; randomize arm order.

    Snapshot restoration must cover replicas, LRU, counters and in-flight work,
    not just the set of Source IDs. No materialization from one arm leaks out.
    """
    if backend.capabilities.get("quiescent_snapshot_restore") is not True:
        raise RuntimeError("paired replay requires quiescent complete snapshot restore")
    rng = random.Random(seed)
    results = []
    for request in requests:
        snapshot = backend.snapshot()
        snapshot_sha = digest_json(snapshot)
        modes = ["explicit_barrier", "fused_advisory"]
        rng.shuffle(modes)
        arms = {}
        try:
            for mode in modes:
                backend.restore(snapshot)
                descriptor = backend.snapshot_descriptor() if hasattr(backend, "snapshot_descriptor") else backend.snapshot()
                if digest_json(descriptor) != snapshot_sha:
                    raise RuntimeError("backend did not restore the complete initial pool")
                configuration = {**dispatch, "gate1_mode": mode}
                row = dict(backend.execute(request, configuration, arrival_ns=time.perf_counter_ns()))
                _online(row)
                if row.get("initial_pool_snapshot_sha256") != snapshot_sha:
                    raise ValueError("execution used another initial pool")
                if row.get("dispatch_config") != configuration:
                    raise ValueError("backend did not execute the requested Gate1 arm")
                arms[mode] = row
            metrics = validate_paired_gate1_executions(arms["explicit_barrier"], arms["fused_advisory"])
            results.append({"request_id": request["request_id"], "execution_order": modes,
                            "arms": arms, "metrics": metrics, "paper_evidence": False})
        finally:
            backend.restore(snapshot)
            # Retaining every historical backing snapshot would defeat global
            # byte budgets and SSD eviction during long traces.
            if hasattr(backend, "release_snapshot"):
                backend.release_snapshot(snapshot)
    return results


def run_causal_capacity_traces(backend: ExperimentBackend, requests: Sequence[Mapping[str, Any]],
                              dispatch: Mapping[str, Any], *, global_byte_budget: int,
                              capacities: Sequence[int] = (1, 2, 4, 8, 16)) -> dict[str, Any]:
    if global_byte_budget <= 0 or len(set(capacities)) != len(capacities):
        raise ValueError("coverage needs a fixed global byte budget and unique capacities")
    epochs = [int(q["request_epoch"]) for q in requests]
    if not epochs or epochs != sorted(set(epochs)):
        raise ValueError("coverage requests must be in strict causal order")
    if len({q["request_id"] for q in requests}) != len(requests):
        raise ValueError("duplicate coverage request")
    traces = {}
    for capacity in capacities:
        if not 1 <= capacity <= 16:
            raise ValueError("capacity is outside the canonical Variant contract")
        backend.reset(capacity=capacity, global_byte_budget=global_byte_budget)
        rows = []
        for request in requests:
            result = dict(backend.execute(request, dispatch, arrival_ns=time.perf_counter_ns()))
            _online(result)
            row = dict(result["coverage_event"])
            if row.get("request_id") != request["request_id"] or row.get("request_epoch") != request["request_epoch"]:
                raise ValueError("coverage event differs from the current request")
            if row.get("global_byte_budget") != global_byte_budget:
                raise ValueError("capacity experiment changed the global storage budget")
            # Validate before allowing writes/grace/LRU to affect the next request.
            summarize_measured_coverage({capacity: [row]})
            rows.append(row)
            backend.finalize_request(request, result)
        traces[capacity] = rows
    return {"traces": traces, "curves": summarize_measured_coverage(traces),
            "oracle_future_pool_used": False, "paper_evidence": False}


@dataclass(frozen=True)
class SourceOutcome:
    request_id: str
    source_id: str | None  # None is the dense action, not a fake successful reuse.
    first_reuse_layer: int
    repair_ratio: float
    ttft_ms: float
    answer_f1: float
    source_digest_unchanged: bool
    timing_scope: str
    provenance_sha256: str
    measurement_kind: str = "real_execution"

    def __post_init__(self) -> None:
        if not self.request_id or not all(math.isfinite(x) for x in (self.ttft_ms, self.answer_f1, self.repair_ratio)):
            raise ValueError("Oracle outcomes require finite real measurements")
        if self.ttft_ms <= 0 or not 0 <= self.answer_f1 <= 1 or not 0 <= self.repair_ratio <= 1:
            raise ValueError("invalid Oracle outcome")
        if self.measurement_kind != "real_execution" or self.timing_scope not in {
            "request_ttft", "executor_prefill_start_not_request_arrival"
        }:
            raise ValueError("Oracle requires real, explicitly scoped first-token timing")
        if len(self.provenance_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.provenance_sha256):
            raise ValueError("Oracle requires matched request/model/pool/runtime provenance")


def measured_quality_cost_oracle(outcomes: Sequence[SourceOutcome], *,
                                 chosen_source_id: str | None,
                                 max_answer_f1_drop: float) -> dict[str, Any]:
    if not outcomes or not math.isfinite(max_answer_f1_drop) or not 0 <= max_answer_f1_drop <= 1:
        raise ValueError("Oracle needs a pre-registered quality tolerance")
    if len({x.request_id for x in outcomes}) != 1 or len({x.source_id for x in outcomes}) != len(outcomes):
        raise ValueError("Oracle is one request, one observation per action")
    if len({(x.provenance_sha256, x.timing_scope) for x in outcomes}) != 1:
        raise ValueError("Oracle actions have different timing scope or provenance")
    dense = [x for x in outcomes if x.source_id is None]
    if len(dense) != 1:
        raise ValueError("Oracle requires the measured dense action")
    baseline = dense[0]
    reuse = [x for x in outcomes if x.source_id is not None]
    if len({(x.first_reuse_layer, x.repair_ratio) for x in reuse}) > 1:
        raise ValueError("Source-only Oracle must hold boundary and repair fixed")
    if any(not x.source_digest_unchanged for x in outcomes):
        raise ValueError("corrupt Source outcome cannot enter an Oracle")
    chosen = next((x for x in outcomes if x.source_id == chosen_source_id), None)
    if chosen is None:
        raise ValueError("chosen action was not measured")
    feasible = [x for x in outcomes if baseline.answer_f1 - x.answer_f1 <= max_answer_f1_drop + 1e-12]
    oracle = min(feasible, key=lambda x: (x.ttft_ms, x.source_id is not None, x.source_id or ""))
    chosen_feasible = chosen in feasible
    return {"oracle_kind": "measured_quality_constrained_fixed_repair_source_oracle",
            "oracle_source_id": oracle.source_id, "oracle_ttft_ms": oracle.ttft_ms,
            "chosen_quality_passed": chosen_feasible,
            "timing_scope": baseline.timing_scope,
            "provenance_sha256": baseline.provenance_sha256,
            "normalized_ttft_regret": ((chosen.ttft_ms - oracle.ttft_ms) / baseline.ttft_ms
                                       if chosen_feasible else None),
            "quality_contract_drop": max_answer_f1_drop, "paper_evidence": False}


def run_source_oracle(request_id: str, source_ids: Sequence[str], *, first_reuse_layer: int,
                      repair_ratio: float, chosen_source_id: str | None,
                      max_answer_f1_drop: float,
                      measure: Callable[[str | None, int, float], SourceOutcome]) -> dict[str, Any]:
    if len(source_ids) != len(set(source_ids)) or len(source_ids) > 16:
        raise ValueError("Oracle Source list must be unique and bounded")
    rows = [measure(None, first_reuse_layer, 1.0)]
    # Sequential execution deliberately avoids staging 16 full KV artifacts together.
    rows.extend(measure(s, first_reuse_layer, repair_ratio) for s in source_ids)
    if any(x.request_id != request_id for x in rows):
        raise ValueError("Oracle measurement request mismatch")
    if [x.source_id for x in rows] != [None, *source_ids]:
        raise ValueError("Oracle backend measured another Source")
    if any(x.first_reuse_layer != first_reuse_layer or x.repair_ratio != repair_ratio for x in rows[1:]):
        raise ValueError("Oracle backend changed repair/boundary")
    from dataclasses import asdict
    return {**measured_quality_cost_oracle(rows, chosen_source_id=chosen_source_id,
                                         max_answer_f1_drop=max_answer_f1_drop),
            "outcomes": [asdict(x) for x in rows]}


def run_serving_trace(backend: ExperimentBackend, requests: Sequence[Mapping[str, Any]],
                      dispatch: Mapping[str, Any], *, concurrency: int,
                      slo_ttft_ms: float) -> dict[str, Any]:
    if concurrency < 1 or not math.isfinite(slo_ttft_ms) or slo_ttft_ms <= 0:
        raise ValueError("invalid serving concurrency/SLO")
    if concurrency > int(backend.capabilities.get("max_integrated_concurrency", 1)):
        raise RuntimeError("backend lacks true integrated concurrent requests; copy contention is insufficient")
    offsets = [float(q["arrival_offset_ms"]) for q in requests]
    if not offsets or any(not math.isfinite(x) or x < 0 for x in offsets) or offsets != sorted(offsets):
        raise ValueError("arrival offsets must be finite, non-negative and sorted")
    if len({q["request_id"] for q in requests}) != len(requests):
        raise ValueError("duplicate serving request")
    start = time.perf_counter_ns()
    def execute(q: Mapping[str, Any], arrival: int) -> dict[str, Any]:
        row = dict(backend.execute(q, dispatch, arrival_ns=arrival))
        _online(row)
        if row.get("request_id") != q["request_id"] or row.get("arrival_ns") != arrival:
            raise ValueError("backend omitted queue time or changed request identity")
        if not arrival <= row["first_token_ns"] <= row["completion_ns"]:
            raise ValueError("serving clock endpoints are invalid")
        if abs(row["request_ttft_ms"] - (row["first_token_ns"] - arrival) / 1e6) > 1e-6:
            raise ValueError("reported TTFT excludes client queue/service time")
        backend.finalize_request(q, row)
        service_end = (backend.service_completion_ns(q["request_id"])
                       if hasattr(backend, "service_completion_ns") else row["completion_ns"])
        return {**row, "service_completion_ns": service_end, "failed": False}
    def recorded_execute(q: Mapping[str, Any], arrival: int) -> dict[str, Any]:
        try:
            return execute(q, arrival)
        except Exception as exc:
            # Never silently drop OOMs or policy/evidence failures from load tests.
            return {"request_id": q["request_id"], "arrival_ns": arrival,
                    "completion_ns": time.perf_counter_ns(), "failed": True,
                    "error_type": type(exc).__name__, "error": str(exc)}
    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as workers:
        for request, offset in zip(requests, offsets):
            arrival = start + int(offset * 1e6)
            while time.perf_counter_ns() < arrival:
                time.sleep(max(0.0, min(.05, (arrival - time.perf_counter_ns()) / 1e9)))
            # Timestamp is scheduled arrival, not worker start: queuing counts.
            futures.append(workers.submit(recorded_execute, request, arrival))
        rows = [future.result() for future in as_completed(futures)]
    elapsed = (max(row.get("service_completion_ns", row["completion_ns"]) for row in rows) - start) / 1e9
    successes = [row for row in rows if not row["failed"]]
    good = sum(row["request_ttft_ms"] <= slo_ttft_ms and row.get("quality_passed") is True for row in successes)
    from .v8_schema10_profile_analysis import linear_quantile
    return {"rows": sorted(rows, key=lambda x: x["arrival_ns"]),
            "attempted": len(rows), "completed": len(successes), "failed": len(rows) - len(successes),
            "throughput_requests_per_s": len(successes) / elapsed,
            "goodput_requests_per_s": good / elapsed,
            "ttft_p50_ms": linear_quantile([r["request_ttft_ms"] for r in successes], .5) if successes else None,
            "ttft_p95_ms": linear_quantile([r["request_ttft_ms"] for r in successes], .95) if successes else None,
            "ttft_p99_ms": linear_quantile([r["request_ttft_ms"] for r in successes], .99) if successes else None,
            "ttft_percentiles_scope": "successful_requests_only_report_failure_count_alongside",
            "integrated_concurrent_requests": concurrency > 1, "paper_evidence": False}
