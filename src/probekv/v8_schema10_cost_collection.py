"""Real provisional measurements, collected before online trace admission."""
from __future__ import annotations

import math
import time

from .v8_schema10_execution import digest_json
from .v8_schema10_cost_provider import EXECUTION_SHAPE_KEY, MeasurementKey
from .v8_schema10_event_log import atomic_json


class CudaCostCollector:
    def __init__(self, *, provenance, deadline=math.inf):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("cost collector requires actual CUDA, not fake timing")
        required = {"model", "code", "patch", "gpu", "config", "runtime_profile", "timing_scope"}
        if not required <= provenance.keys() or any(not provenance[k] for k in required):
            raise ValueError("incomplete measured cost provenance")
        self.torch, self.provenance, self.deadline = torch, dict(provenance), deadline
        self.rows, self.joint_rows, self.raw_intervals = [], [], []
        self.started = time.perf_counter_ns()

    def measure(self, *, category, query, operation, reset, warmup=2, repeats=5, joint=False):
        if warmup < 0 or repeats < 1 or not callable(reset):
            raise ValueError("each measurement requires a reset and explicit repeat count")
        if category == "dense_reference" and repeats != 1:
            raise ValueError("matched dense baseline is a preregistered single observation")
        wall, gpu, observations, component_observations = [], [], [], []
        for index in range(warmup + repeats):
            if time.perf_counter() >= self.deadline:
                raise TimeoutError("cost capture deadline; no partial table publication")
            reset()
            self.torch.cuda.synchronize()
            start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
            begin = time.perf_counter_ns()
            start.record()
            result = operation()
            end.record()
            end.synchronize()
            finish = time.perf_counter_ns()
            # first-token callbacks can delimit TTFT even when operation also
            # decodes: never call complete generation wall time "TTFT".
            if category == "dense_reference":
                if not isinstance(result, dict) or not begin <= result.get("first_token_ns", -1) <= finish:
                    raise ValueError("dense reference lacks actual first-token endpoint")
                endpoint = result["first_token_ns"]
            else:
                endpoint = finish
            interval = {"warmup": index < warmup, "host_start_ns": begin, "host_end_ns": endpoint,
                        "cuda_ms": float(start.elapsed_time(end)), "completion_ns": finish}
            self.raw_intervals.append({"category": category, "query": query, **interval})
            if index >= warmup:
                wall.append((endpoint - begin) / 1e6)
                gpu.append(interval["cuda_ms"])
                observations.append(interval)
                if category == "source_local_marginal":
                    parts = result.get("component_lower_ms") if isinstance(result, dict) else None
                    if (not isinstance(parts, dict) or set(parts) != {"support_build", "visible_load", "repair"}
                            or any(not math.isfinite(v) or v < 0 for v in parts.values())
                            or sum(parts.values()) > wall[-1] + 1e-6):
                        raise ValueError("source-local marginal needs measured non-overlapping component evidence")
                    component_observations.append(parts)
        row = {"category": category, "query": query, "provenance": self.provenance,
            "origin": "real_cuda_execution", "fake_timing": False,
            "warmup_excluded": True, "outlier_policy": "none", "cuda_samples_ms": gpu,
            "raw_intervals": observations, "samples_ms": wall}
        if joint:
            row["joint_future_wall_ms_samples"] = wall
        if category == "source_local_marginal":
            row["component_observations_ms"] = component_observations
            row["component_lower_ms"] = {key: min(p[key] for p in component_observations)
                                         for key in ("support_build", "visible_load", "repair")}
        row["row_sha256"] = digest_json(row)
        (self.joint_rows if joint else self.rows).append(row)
        return row

    def publish(self, path, *, expected_cells):
        keys = [(r["category"], digest_json(r["query"])) for r in self.rows + self.joint_rows]
        if len(keys) != len(set(keys)) or set(keys) != set(expected_cells):
            raise ValueError("incomplete/duplicate cost grid; do not publish a partial table")
        payload = {"key_contract": EXECUTION_SHAPE_KEY, "provenance": self.provenance,
            "formal_profile_frozen": False, "rows": self.rows, "joint_rows": self.joint_rows,
            "measurement_session_host_ms": (time.perf_counter_ns() - self.started) / 1e6,
            "raw_intervals_sha256": digest_json(self.raw_intervals), "paper_evidence": False}
        atomic_json(path, payload)
        return payload


STAGES = ("environment", "correctness", "cost_collection", "cost_support_validation",
          "online_trace", "gate1_paired_ab", "source_oracle", "aggregation")


def staged_sentinel_manifest(*, binding, trace_manifest, correctness_jobs, cost_cells, paired_jobs, oracle_jobs):
    if not correctness_jobs or not cost_cells or not paired_jobs or not oracle_jobs:
        raise ValueError("preregister every sentinel stage before GPU execution")
    if trace_manifest["binding"] != binding:
        raise ValueError("trace and stage bindings differ")
    row = {"binding": binding, "stage_order": list(STAGES), "trace_manifest": trace_manifest,
        "correctness_jobs": correctness_jobs, "cost_cells": cost_cells,
        "gate1_paired_jobs": paired_jobs, "source_oracle_jobs": oracle_jobs,
        "formal_profile_frozen": False, "online_trace_execution_allowed": False,
        "paper_evidence": False, "locked_test_accessed": False}
    row["manifest_sha256"] = digest_json(row)
    return row
