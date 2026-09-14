"""Read only preregistered numbered samples, never profiler/control JSONs."""
import json
import math
from pathlib import Path
import statistics

from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest


def summarize_matched_backend(root, *, repeats):
    root = Path(root)
    if not 1 <= repeats <= 20:
        raise ValueError("invalid backend repeat count")
    evidence = {}
    for name in ("r1-comparison.json", "fixed15-equivalence.json", "source-integrity.json",
                 "boundary-equivalence-1.0.json", "boundary-equivalence-0.15.json"):
        data = json.loads((root / name).read_text())
        if name == "r1-comparison.json":
            passed = bool(data) and all(v.get("passed") is True for v in data.values())
        elif name == "source-integrity.json":
            passed = data.get("unchanged") is True and data.get("before") == data.get("after")
        else:
            passed = data.get("passed") is True
        if not passed:
            raise ValueError("failed prerequisite " + name)
        evidence[name] = file_digest(root / name)
    results = {}
    for label, arms, field in (("setup_inclusive", ("dense", "cacheblend_loop", "probekv"), "first_token_host_ms"),
                               ("boundary_executor", ("cacheblend", "probekv"), "executor_host_ms")):
        rows_by_arm = {}
        for arm in arms:
            rows = []
            for i in range(repeats + 2):
                name = ("boundary-" if label == "boundary_executor" else "") + f"{i:02d}-{arm}.json"
                r = json.loads((root / name).read_text())
                digest = r.pop("raw_observation_sha256", None)
                if digest != digest_json(r):
                    raise ValueError("sample digest mismatch: " + name)
                if (r.get("warmup") != (i < 2) or r.get("repeat") != i or r.get("arm") != arm
                        or r.get("fake_timing") is not False or r.get("origin") != "real_cuda_execution"
                        or r.get("instrumented_timing_not_performance_evidence", False)):
                    raise ValueError("wrong sample identity/timing: " + name)
                value = r.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError("invalid measured latency: " + name)
                if r.get("cached_prefix_tokens") != 0:
                    raise ValueError("matched executor benchmark requires zero Prefix")
                if arm not in ("dense",) and (r.get("diagnostic_repair_ratio") != .15
                        or not r.get("external_repair_mask_sha256")):
                    raise ValueError("missing fixed15 mask binding")
                evidence[name] = file_digest(root / name)
                if i >= 2:
                    rows.append(r)
            rows_by_arm[arm] = rows
        for i in range(repeats):
            same = [rows_by_arm[a][i] for a in arms]
            for key in ("request_tokens_sha256", "sampling_signature", "cached_prefix_tokens"):
                if any(r.get(key) != same[0].get(key) for r in same):
                    raise ValueError("unmatched pair " + key)
            reuse = [r for r in same if r.get("source_id") is not None]
            for key in ("source_id", "external_repair_mask_sha256", "boundary"):
                if any(r.get(key) != reuse[0].get(key) for r in reuse):
                    raise ValueError("unmatched reuse pair " + key)
        results[label] = {arm: {"n": repeats, "mean_ms": statistics.mean(r[field] for r in rows),
            "median_ms": statistics.median(r[field] for r in rows),
            "min_ms": min(r[field] for r in rows), "max_ms": max(r[field] for r in rows),
            "samples_ms": [r[field] for r in rows]} for arm, rows in rows_by_arm.items()}
        cb = "cacheblend_loop" if label == "setup_inclusive" else "cacheblend"
        gaps = [p[field] - c[field] for p, c in zip(rows_by_arm["probekv"], rows_by_arm[cb])]
        results[label]["paired_probekv_minus_cacheblend_ms"] = gaps
        results[label]["paired_mean_gap_ms"] = statistics.mean(gaps)
    return {"comparisons": results, "raw_file_sha256": evidence,
            "backend_equivalence_passed": True, "fixed_repair_ratio": .15,
            "native_prefix_supported": False, "unmodified_cacheblend_upstream": False,
            "selection_executed": False, "planner_executed": False,
            "quality_vs_dense_qualified": False, "paper_evidence": False}
