"""Contiguous host TTFT accounting, never a sum of CUDA service times."""


def partition_request_wallclock(arrival_ns, first_token_ns, landmarks):
    points = [("arrival", arrival_ns), *landmarks, ("first_token", first_token_ns)]
    if any(isinstance(t, bool) or not isinstance(t, int) for _, t in points):
        raise ValueError("wall-clock timestamps must be integer nanoseconds")
    if len({name for name, _ in points}) != len(points):
        raise ValueError("wall-clock landmark names must be unique")
    rows = []
    for (start_name, start), (end_name, end) in zip(points, points[1:]):
        if end < start:
            raise ValueError("wall-clock landmarks are not monotonic")
        rows.append({"start": start_name, "end": end_name, "start_ns": start,
                     "end_ns": end, "duration_ns": end - start,
                     "duration_ms": (end - start) / 1e6})
    total = first_token_ns - arrival_ns
    accounted = sum(r["duration_ns"] for r in rows)
    assert accounted == total
    return {"intervals": rows, "ttft_ns": total, "accounted_ns": accounted,
            "unaccounted_ns": total - accounted, "ttft_ms": total / 1e6,
            "semantics": "contiguous_host_wallclock_not_cuda_service_times",
            "adds_device_synchronization": False}
