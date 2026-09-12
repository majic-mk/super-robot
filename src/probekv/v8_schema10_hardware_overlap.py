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
        # Kineto may correlate GPU activities directly with cuda_runtime API
        # IDs instead of the surrounding aten operation's External id.
        if event.get("cat") not in {"cpu_op", "cuda_runtime"}:
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
    # Some Kineto builds do not propagate record_function External ids to the
    # asynchronous memcpy activities.  When the trace contains the exact
    # two-transfer-per-layer shape emitted by our BF16 K/V loader, recover the
    # layer association by deterministic transfer order.  This is deliberately
    # conservative: ambiguous counts remain unavailable rather than being
    # guessed.
    # Only layers explicitly marked as pending-copy are expected in this
    # diagnostic.  With a windowed loader the first layer(s) can already be
    # resident and therefore intentionally produce no H2D activity; deriving
    # the count from every compute layer would falsely mark such traces as
    # incomplete (e.g. window=1 has 31 pending layers, not 32).
    marked_copy_layers = {int(r["name"].rsplit(".", 1)[1]) for r in ranges
                          if ".copy_layer." in r.get("name", "")}
    expected_h2d_count = (2 * len(marked_copy_layers) if marked_copy_layers
                          else (2 * len({r["layer"] for r in kernels}) if kernels else 0))
    large_h2d = []
    for event in events:
        if event.get("cat") != "gpu_memcpy" or "htod" not in event.get("name", "").lower():
            continue
        args = event.get("args", {})
        if args.get("device") is None or args.get("stream") is None:
            continue
        if int(args.get("bytes", 0) or 0) < 1_000_000:
            continue
        large_h2d.append((event, args))
    # A partial External-id correlation must never silently shift layer
    # labels.  If the complete large-transfer set has the expected K/V shape,
    # rebuild deterministic attribution from that complete set; otherwise
    # retain the correlated rows but mark attribution incomplete.
    attribution_complete = expected_h2d_count == len(copies) if expected_h2d_count else bool(copies)
    if kernels and len(copies) != expected_h2d_count and len(large_h2d) == expected_h2d_count:
        copies = []
        compute_layers = sorted(marked_copy_layers or {r["layer"] for r in kernels})
        for index, (event, args) in enumerate(large_h2d):
            copies.append({"layer": compute_layers[index // 2], "start_us": event["ts"],
                           "end_us": event["ts"] + event["dur"],
                           "device": args["device"], "stream": args["stream"],
                           "name": event.get("name", "")})
        attribution_complete = True
    elif not copies and kernels:
        compute_layers = sorted(marked_copy_layers or {r["layer"] for r in kernels})
        unowned = []
        for event in events:
            if event.get("cat") != "gpu_memcpy" or "htod" not in event.get("name", "").lower():
                continue
            args = event.get("args", {})
            if args.get("device") is None or args.get("stream") is None:
                continue
            # The layerwise BF16 K/V copies are the large transfers; tiny
            # allocator/setup H2D events must not participate in attribution.
            if int(args.get("bytes", 0) or 0) < 1_000_000:
                continue
            unowned.append((event, args))
        if len(unowned) == 2 * len(compute_layers):
            for index, (event, args) in enumerate(unowned):
                layer = compute_layers[index // 2]
                copies.append({"layer": layer, "start_us": event["ts"],
                               "end_us": event["ts"] + event["dur"],
                               "device": args["device"], "stream": args["stream"],
                               "name": event.get("name", "")})
            attribution_complete = True
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
            "expected_h2d_activity_count": expected_h2d_count,
            "layer_attribution_complete": bool(attribution_complete),
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


if __name__ == "__main__":
    import argparse
    import json
    import subprocess
    from pathlib import Path
    from .v8_schema10_storage import file_digest
    from .v8_schema10_event_log import atomic_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.trace), Path(args.output)
    if output.exists():
        raise ValueError("hardware reanalysis must not overwrite prior evidence")
    result = summarize_hardware_overlap(json.loads(source.read_text()))
    result.update(trace_sha256=file_digest(source),
                  analysis_code_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
                  input_trace_path=str(source.resolve()), instrumented_timing_not_performance_evidence=True)
    atomic_json(output, result)
