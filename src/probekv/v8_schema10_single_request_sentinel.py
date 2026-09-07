"""Preregistered single-request trace entry. No GPU is started by preparation."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import time

from .v8_schema10_event_log import OnlineEventLog, atomic_json
from .v8_schema10_execution import digest_json
from .v8_schema10_experiments import _online
from .v8_schema10_storage import file_digest


def _validate_inputs(trace_set, dispatches, binding):
    for field, length in (("code_commit", 40), ("patch_sha256", 64), ("config_sha256", 64),
                          ("runtime_measurement_sha256", 64), ("tokenizer_hash", 64)):
        value = binding.get(field, "")
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("missing exact sentinel binding: " + field)
    if not binding.get("model_signature") or not binding.get("model_revision") or binding.get("global_byte_budget", 0) <= 0:
        raise ValueError("sentinel must bind model/revision and fixed storage budget")
    for dispatch in dispatches.values():
        if dispatch.get("force_nonpaper_measurement_admission"):
            raise ValueError("production sentinel forbids diagnostic admission bypass")
        if dispatch.get("selection_budget_policy") not in {"end_to_end_aware", "legacy_fixed_fraction"}:
            raise ValueError("selection budget policy must be explicit")
    for requests in trace_set.values():
        for request in requests:
            tokens = request.get("token_ids", ())
            if not tokens or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("sentinel requires actual model token IDs")
            positions, ids = set(), set()
            for s in request["segments"]:
                ps = s.get("positions", ())
                if not s.get("segment_id") or s["segment_id"] in ids or not s.get("content_key"):
                    raise ValueError("invalid/duplicate Segment identity")
                if (not ps or list(ps) != sorted(set(ps)) or min(ps) < 0 or max(ps) >= len(tokens)
                        or positions.intersection(ps) or [tokens[p] for p in ps] != s.get("token_ids")):
                    raise ValueError("Segment token identity/span differs from request")
                positions.update(ps)
                ids.add(s["segment_id"])


def prepare_manifest(*, trace_set, dispatches, binding, backend_factory=None):
    """Source traces are actual preregistered data, never fabricated here."""
    if set(trace_set) != {"1", "2", "5", "37"}:
        raise ValueError("sentinel requires 1/2/5/37 Segment traces")
    if set(dispatches) != {"fast", "legacy"}:
        raise ValueError("freeze distinct FAST and legacy dispatch adapters")
    if dispatches["fast"]["selection_path"] not in {"d1_only", "d1_d2_rescue"} or dispatches["legacy"]["selection_path"] != "legacy_multicheckpoint":
        raise ValueError("dispatch cannot silently substitute FAST for legacy")
    for count, requests in trace_set.items():
        if not requests or len({q["request_id"] for q in requests}) != len(requests):
            raise ValueError("missing/duplicate requests in sentinel trace")
        epochs = [q["request_epoch"] for q in requests]
        if epochs != sorted(set(epochs)):
            raise ValueError("trace epochs must be strictly causal")
        for request in requests:
            if len(request["segments"]) != int(count) or request.get("locked_test_accessed") is not False:
                raise ValueError("trace inventory or development partition binding differs")
            if not request.get("content_group_id") or not request.get("partition_id"):
                raise ValueError("trace must identify its development content-group partition")
    _validate_inputs(trace_set, dispatches, binding)
    jobs = []
    for name, dispatch in sorted(dispatches.items()):
        for k in (1, 4, 16):
            for count in (1, 2, 5):
                jobs.append({"job_id": f"{name}-k{k}-s{count}", "capacity": k, "segments": count,
                             "dispatch": dispatch, "requests": trace_set[str(count)]})
        jobs.append({"job_id": name + "-capacity-s37", "capacity": 16, "segments": 37,
                     "dispatch": dispatch, "requests": trace_set["37"]})
    unsigned = {"protocol_version": 8, "schema_version": 10, "stage": "single_request_trace_sentinel",
        "binding": binding, "backend_factory": backend_factory, "jobs": jobs,
        "max_integrated_concurrency": 1, "max_hours": 4, "max_hourly_yuan": 7.5, "max_total_yuan": 30,
        "formal_profile_bundle_frozen": False, "gpu_runtime_qualified": False,
        "paper_evidence": False, "locked_test_accessed": False,
        "readiness": {"ready_for_single_request_gpu_sentinel": False,
                      "pending": ["real_native_model_adapters", "native_prefix_shadow_binding",
                                  "prerequisite_correctness_and_cost_measurements", "instance_budget_confirmation"]}}
    return {**unsigned, "manifest_sha256": digest_json(unsigned)}


def validate_manifest(manifest):
    if manifest["manifest_sha256"] != digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("preregistered sentinel manifest SHA mismatch")
    if (manifest.get("protocol_version"), manifest.get("schema_version"), manifest.get("max_integrated_concurrency")) != (8, 10, 1):
        raise ValueError("not a schema10 single-request manifest")
    if any(manifest.get(k) is not False for k in ("formal_profile_bundle_frozen", "gpu_runtime_qualified", "paper_evidence", "locked_test_accessed")):
        raise ValueError("a sentinel cannot claim formal qualification")
    if manifest.get("stage") != "single_request_trace_sentinel":
        raise ValueError("wrong execution stage")
    if (manifest.get("max_hours"), manifest.get("max_hourly_yuan"), manifest.get("max_total_yuan")) != (4, 7.5, 30):
        raise ValueError("sentinel time/money boundary changed")
    jobs = manifest.get("jobs", ())
    by_id = {j["job_id"]: j for j in jobs}
    if len(by_id) != 20 or len(jobs) != 20:
        raise ValueError("sentinel matrix is incomplete/duplicated")
    try:
        dispatches = {name: by_id[name + "-k1-s1"]["dispatch"] for name in ("fast", "legacy")}
        traces = {str(n): by_id[f"fast-k1-s{n}"]["requests"] for n in (1, 2, 5)}
        traces["37"] = by_id["fast-capacity-s37"]["requests"]
    except KeyError as exc:
        raise ValueError("sentinel matrix omitted a preregistered cell") from exc
    rebuilt = prepare_manifest(trace_set=traces, dispatches=dispatches, binding=manifest["binding"],
                               backend_factory=manifest.get("backend_factory"))
    if digest_json(rebuilt["jobs"]) != digest_json(jobs):
        raise ValueError("capacity/dispatch replays must use the same immutable traces")


def run_manifest(manifest, *, output_dir, hourly_yuan, instance_confirmed, repository, resume=False):
    started = time.perf_counter()  # includes imports, model loading and preflight
    validate_manifest(manifest)
    if not instance_confirmed or not 0 < hourly_yuan <= 7.5 or hourly_yuan * 4 > 30:
        raise RuntimeError("instance/price/budget confirmation required; no experiment started")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository, text=True).strip()
    if dirty or commit != manifest["binding"]["code_commit"]:
        raise RuntimeError("execution checkout is dirty or differs from frozen SHA")
    factory_path = manifest.get("backend_factory")
    if not factory_path:
        raise RuntimeError("real native runtime factory not connected; do not use diagnostic fixture executor")
    module, name = factory_path.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    backend = factory(manifest)
    if backend.capabilities.get("real_cuda_native_online_backend") is not True:
        raise RuntimeError("backend does not provide a real native CUDA execution path")
    # Runtime factory must fail closed on missing Prefix/r1/mask prerequisites.
    backend.verify_prerequisite_evidence(manifest)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    completed, pending = [], []
    for index, job in enumerate(manifest["jobs"]):
        if time.perf_counter() - started >= 4 * 3600:
            pending.extend(j["job_id"] for j in manifest["jobs"][index:])
            break
        directory = root / job["job_id"]
        directory.mkdir(exist_ok=True)
        result_path = directory / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not resume or result.get("manifest_sha256") != manifest["manifest_sha256"] or result.get("failed") is not False:
                raise RuntimeError("cannot overwrite or reinterpret existing sentinel result")
            if file_digest(directory / "events.jsonl") != result["events_sha256"]:
                raise ValueError("successful-prefix event digest changed")
            completed.append(job["job_id"])
            continue
        if (directory / "events.jsonl").exists():
            raise RuntimeError("incomplete job preserved; restart in a new output directory")
        backend.reset(capacity=job["capacity"], global_byte_budget=manifest["binding"]["global_byte_budget"])
        state = backend.snapshot()
        binding = {**manifest["binding"], "dispatch": digest_json(job["dispatch"]),
                   "initial_state_sha256": digest_json(state), "job_id": job["job_id"]}
        backend.release_snapshot(state)
        backend.event_log = OnlineEventLog(directory / "events.jsonl", binding=binding)
        try:
            for request in job["requests"]:
                if time.perf_counter() - started >= 4 * 3600:
                    raise TimeoutError("4-hour limit; preserve the incomplete job")
                backend.set_session_deadline(started + 4 * 3600)
                outcome = backend.execute(request, job["dispatch"], arrival_ns=time.perf_counter_ns())
                _online(outcome)
                backend.finalize_request(request, outcome)
            result = {"manifest_sha256": manifest["manifest_sha256"], "failed": False,
                      "events_sha256": file_digest(directory / "events.jsonl"), "paper_evidence": False}
            atomic_json(result_path, result)
            completed.append(job["job_id"])
        except Exception as exc:
            atomic_json(result_path, {"manifest_sha256": manifest["manifest_sha256"], "failed": True,
                "error_type": type(exc).__name__, "error": str(exc), "paper_evidence": False})
            raise
    summary = {"completed_jobs": completed, "pending_jobs": pending,
        "single_request_runtime_sentinel_passed": False,  # traces alone are not the full correctness/Oracle gate
        "formal_profile_bundle_frozen": False, "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False, "paper_evidence": False, "locked_test_accessed": False}
    atomic_json(root / "trace_summary.json", summary)
    return summary
