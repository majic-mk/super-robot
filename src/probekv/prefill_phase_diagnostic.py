"""Correlated profiler phase analysis; never additive performance accounting."""
from collections import defaultdict
import math
from .v8_schema10_hardware_overlap import union_duration


def summarize_prefill_phase(trace, arm):
    events = [e for e in trace.get("traceEvents", ()) if e.get("ph") == "X"]
    if any(not all(math.isfinite(float(e.get(k, float("nan")))) for k in ("ts", "dur"))
           or e["dur"] < 0 for e in events):
        raise ValueError("invalid profiler intervals")
    names = (("cacheblend.native_prefill",) if arm == "cacheblend_loop" else
             ("probekv.compute_layer.1", "probekv.compute_layer.32"))
    if arm not in {"cacheblend_loop", "probekv"}:
        raise ValueError("unknown control arm")
    bounds = []
    for name in names:
        found = [e for e in events if e.get("name") == name and e.get("cat") == "user_annotation"]
        if len(found) != 1:
            raise ValueError("requires exactly one marked Mistral prefill")
        bounds.extend(found)
    start = min(e["ts"] for e in bounds)
    stop = max(e["ts"] + e["dur"] for e in bounds)
    pid, tid = bounds[0]["pid"], bounds[0]["tid"]
    cpu = [e for e in events if e.get("cat") in {"cpu_op", "cuda_runtime"}
           and (e.get("pid"), e.get("tid")) == (pid, tid)
           and start <= e["ts"] and e["ts"] + e["dur"] <= stop]
    externals = {str(e["args"]["External id"]) for e in cpu
                 if "External id" in e.get("args", {})}
    gpu = [e for e in events if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
           and str(e.get("args", {}).get("External id")) in externals]
    kernels = [e for e in gpu if e.get("cat") == "kernel"]
    by_name = defaultdict(list)
    for e in kernels:
        by_name[e.get("name", "")].append(e["dur"])
    calls = defaultdict(list)
    for e in cpu:
        if e["cat"] == "cuda_runtime" and ("Synchronize" in e["name"] or "Memcpy" in e["name"]):
            calls[e["name"]].append(e["dur"])
    return dict(arm=arm, scope="correlated_prefill_only_decode_excluded",
        host_envelope_ms=(stop-start)/1000, hardware_activity_available=bool(gpu),
        kernel_count=len(kernels),
        kernel_union_ms=union_duration((e["ts"], e["ts"]+e["dur"]) for e in kernels)/1000 if gpu else None,
        gpu_activity_union_ms=union_duration((e["ts"], e["ts"]+e["dur"]) for e in gpu)/1000 if gpu else None,
        gpu_activity_span_ms=(max(e["ts"]+e["dur"] for e in gpu)-min(e["ts"] for e in gpu))/1000 if gpu else None,
        kernel_groups=[dict(name=k, count=len(v), duration_sum_ms=sum(v)/1000)
                       for k, v in sorted(by_name.items(), key=lambda kv: -sum(kv[1]))],
        runtime_calls={k: dict(count=len(v), inclusive_host_ms=sum(v)/1000) for k, v in calls.items()},
        host_gpu_components_are_not_additive=True, paper_evidence=False)
