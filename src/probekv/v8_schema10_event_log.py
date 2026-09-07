"""Append-only hash-chained raw evidence, with explicit immutable resume."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from threading import RLock

from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest
from .v8_schema10_evidence import validate_disjoint_case_groups


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_events(path, *, binding):
    rows, previous = [], "0" * 64
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ValueError("torn final event; preserve file and use a new output directory")
            row = json.loads(line)
            claimed = row.pop("event_sha256")
            if (row["sequence"] != number or row["previous_sha256"] != previous
                    or row["binding"] != binding or digest_json(row) != claimed):
                raise ValueError("raw event chain, sequence or execution binding differs")
            rows.append({**row, "event_sha256": claimed})
            previous = claimed
    return rows


class OnlineEventLog:
    def __init__(self, path, *, binding, resume=False):
        required = {"code_commit", "patch_sha256", "model_signature", "config_sha256",
                    "initial_state_sha256", "runtime_measurement_sha256", "dispatch"}
        if not required <= binding.keys() or any(not binding[k] for k in required):
            raise ValueError("event log requires exact immutable execution binding")
        self.path, self.binding = Path(path), dict(binding)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        if self.path.exists():
            if not resume:
                raise FileExistsError("never overwrite existing experiment evidence")
            rows = read_events(self.path, binding=self.binding)
            if any(r["kind"] in {"request_failed", "materialization_failed"} for r in rows):
                raise ValueError("failed evidence is immutable; rerun in a new output directory")
            started = [r["request_id"] for r in rows if r["kind"] == "request_started"]
            finalized = [r["request_id"] for r in rows if r["kind"] == "request_finalized"]
            if started != finalized:
                raise ValueError("resume requires a completely finalized successful prefix")
        else:
            self.path.touch(exist_ok=False)
            rows = []
        self.rows = rows

    def append(self, kind, request_id, payload):
        with self.lock:
            row = {"sequence": len(self.rows) + 1, "previous_sha256": self.rows[-1]["event_sha256"] if self.rows else "0" * 64,
                   "binding": self.binding, "kind": kind, "request_id": request_id, "payload": payload}
            row["event_sha256"] = digest_json(row)
            encoded = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self.rows.append(row)
            return row["event_sha256"]


def aggregate_online_events(path, *, expected_file_sha256, binding, fit_rows, validation_rows):
    if file_digest(Path(path)) != expected_file_sha256:
        raise ValueError("evidence file SHA256 mismatch")
    validate_disjoint_case_groups(fit_rows, validation_rows)
    rows = read_events(path, binding=binding)
    started, complete, finalized, failures = {}, {}, set(), []
    for event in rows:
        rid, kind, value = event["request_id"], event["kind"], event["payload"]
        if kind == "request_started":
            if rid in started:
                raise ValueError("duplicate request start")
            started[rid] = value
        elif kind == "request_completed":
            if rid not in started or rid in complete:
                raise ValueError("completion without unique request start")
            if value.get("request_id") != rid:
                raise ValueError("completion request binding differs")
            a, f, c = (value.get(k) for k in ("arrival_ns", "first_token_ns", "completion_ns"))
            if not all(isinstance(x, int) for x in (a, f, c)) or not 0 <= a <= f <= c:
                raise ValueError("missing or unordered timing events")
            ttft = value.get("request_ttft_ms")
            if not isinstance(ttft, (int, float)) or not math.isfinite(ttft) or abs(ttft - (f - a) / 1e6) > 1e-6:
                raise ValueError("TTFT omits queue time or differs from raw endpoints")
            if value.get("dispatch_config") != started[rid]["dispatch"]:
                raise ValueError("executed dispatch differs from request start")
            if value.get("arrival_ns") != started[rid]["arrival_ns"]:
                raise ValueError("completion rewrote the request arrival")
            if value.get("initial_pool_snapshot_sha256") != started[rid]["initial_snapshot_sha256"]:
                raise ValueError("completion used a different initial pool")
            if value.get("code_commit") != binding["code_commit"] or value.get("model_signature") != binding["model_signature"]:
                raise ValueError("completion code/model differs from event binding")
            for item in value.get("selection_events", ()):
                if item.get("event_id") != digest_json({k: v for k, v in item.items() if k != "event_id"}):
                    raise ValueError("bad raw selector event digest")
            if not value.get("runtime_events"):
                raise ValueError("missing runtime events")
            from .v8_schema10_experiments import _online
            _online(value)
            coverage = value.get("coverage_event", {})
            if (coverage.get("request_id") != rid or coverage.get("request_epoch") != started[rid]["request"]["request_epoch"]
                    or coverage.get("actual_ttft_ms") != ttft
                    or coverage.get("selected_variant_ids") != value.get("selected_source_variant_ids")
                    or coverage.get("committed_variant_ids") != value.get("committed_source_variant_ids")):
                raise ValueError("coverage is not derived from the current execution")
            complete[rid] = value
        elif kind == "request_finalized":
            if rid not in complete or rid in finalized or value["outcome_sha256"] != digest_json(complete[rid]):
                raise ValueError("finalization missing a matching outcome")
            finalized.add(rid)
        elif kind.endswith("failed"):
            failures.append({"request_id": rid, "kind": kind, "payload": value})
    pending = sorted(set(started) - finalized)
    measured = list(complete.values())
    if not measured:
        raise ValueError("no actual completed request evidence")
    if any(r.get("evidence_origin") != "real_cuda_execution" for r in measured):
        gpu_samples = False
    else:
        gpu_samples = True
    known_quality = [r for r in measured if isinstance(r.get("quality_passed"), bool)]
    violations = sum(not r["quality_passed"] for r in known_quality) if len(known_quality) == len(measured) else None
    realized = []
    for r in measured:
        if r.get("committed_source_variant_ids"):
            dense = r.get("coverage_event", {}).get("matched_dense_ttft_ms")
            overrun = max(0., r["request_ttft_ms"] - .8 * dense) if isinstance(dense, (int, float)) and math.isfinite(dense) and dense > 0 else None
            claimed = r.get("realized_overrun_ms")
            if claimed is not None and (overrun is None or not math.isfinite(claimed) or abs(claimed - overrun) > 1e-6):
                raise ValueError("claimed realized overrun differs from actual timing")
            realized.append(overrun)
    return {"input_events_sha256": expected_file_sha256, "binding": binding,
            "completed": len(measured), "finalized": len(finalized), "pending": pending,
            "failures": failures, "observed_quality_violations": violations,
            "realized_gamma_overrun_count": (sum(x > 0 for x in realized) if realized and all(x is not None for x in realized) else None),
            "real_cuda_samples": gpu_samples, "raw_event_integrity_passed": True,
            "trace_events_complete": not pending and not failures,
            # Raw trace integrity cannot substitute for independent native
            # Prefix, r=1, QA/Oracle and cost-support prerequisite evidence.
            "sentinel_evidence_complete": False,
            "formal_profile_bundle_frozen": False, "gpu_runtime_qualified": False,
            "h1_h2_execution_allowed": False, "paper_evidence": False, "locked_test_accessed": False}
