"""CUPTI-backed overlap, not intersections of broad CUDA Event envelopes.

Profiler runs are diagnostics, never matched-performance timing samples.
Only GPU events correlated to explicitly marked pending-copy/prefill ranges
are attributed to layers. Initial eager copies and decode are out of scope.
"""
import math


def union_duration(intervals):
    end = None
    total = 0.0
    for left, right in sorted(intervals):
        if end is None or left >= end:
            total += right - left
        elif right > end:
            total += right - end
        end = max(end if end is not None else right, right)
    return total


def summarize_hardware_overlap(trace):
    events = [e for e in trace.get("traceEvents", ()) if e.get("ph") == "X"]
    for event in events:
        if (not all(math.isfinite(float(event.get(k, float("nan")))) for k in ("ts", "dur"))
                or event["dur"] < 0):
            raise ValueError("invalid hardware trace interval")
    ranges = [e for e in events if e.get("cat") == "user_annotation"
              and e.get("name", "").startswith(("probekv.copy_layer.", "probekv.compute_layer."))]
    owners = {}
    for event in events:
        if event.get("cat") != "cpu_op":
            continue
        external = event.get("args", {}).get("External id")
        if external is None:
            continue
        enclosing = [r for r in ranges if (r.get("pid"), r.get("tid")) == (event.get("pid"), event.get("tid"))
                     and r["ts"] <= event["ts"] and event["ts"] + event["dur"] <= r["ts"] + r["dur"]]
        if enclosing:
            name = min(enclosing, key=lambda r: r["dur"])["name"]
            owners[str(external)] = ("copy" if ".copy_layer." in name else "compute", int(name.rsplit(".", 1)[1]))
    copies, kernels = [], []
    for event in events:
        args = event.get("args", {})
        owner = owners.get(str(args.get("External id")))
        if not owner or args.get("device") is None or args.get("stream") is None:
            continue
        category = event.get("cat", "")
        row = {"layer": owner[1], "start_us": event["ts"], "end_us": event["ts"] + event["dur"],
               "device": args["device"], "stream": args["stream"], "name": event.get("name", "")}
        if owner[0] == "copy" and category == "gpu_memcpy" and "htod" in row["name"].lower():
            copies.append(row)
        elif owner[0] == "compute" and category == "kernel":
            kernels.append(row)
    intersections, pair_intervals = [], {}
    for copy in copies:
        for kernel in kernels:
            if copy["device"] != kernel["device"] or copy["stream"] == kernel["stream"]:
                continue
            left = max(copy["start_us"], kernel["start_us"])
            right = min(copy["end_us"], kernel["end_us"])
            if right > left:
                intersections.append((left, right))
                pair_intervals.setdefault((copy["layer"], kernel["layer"]), []).append((left, right))
    return {"evidence_kind": "correlated_cupti_gpu_activity", "paper_evidence": False,
            "scope": "marked_pending_layer_H2D_vs_marked_prefill_kernels",
            "initial_copy_and_decode_excluded": True,
            "hardware_activity_available": bool(copies and kernels),
            "hardware_copy_kernel_overlap_observed": bool(intersections),
            "attributed_h2d_activity_count": len(copies), "attributed_kernel_count": len(kernels),
            "copy_layers": sorted({r["layer"] for r in copies}),
            "compute_layers": sorted({r["layer"] for r in kernels}),
            "h2d_union_ms": union_duration((r["start_us"], r["end_us"]) for r in copies) / 1000,
            "kernel_union_ms": union_duration((r["start_us"], r["end_us"]) for r in kernels) / 1000,
            "copy_kernel_overlap_union_ms": union_duration(intersections) / 1000,
            "layer_pairs": [{"copy_layer": c, "compute_layer": k,
                             "overlap_union_ms": union_duration(v) / 1000}
                            for (c, k), v in sorted(pair_intervals.items())],
            "raw_attributed_h2d_intervals": copies, "raw_attributed_kernel_intervals": kernels}
