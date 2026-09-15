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


def refine_request_wallclock(partition, events):
    """Split existing host intervals without adding nested CUDA/profile times.

    Simultaneous labels are retained together, never counted once per label.
    Uninstrumented gaps remain gaps, not invented 'GPU compute' or 'waste'.
    """
    rows = partition['intervals']
    if not rows:
        raise ValueError('missing wallclock intervals')
    arrival, first = rows[0]['start_ns'], rows[-1]['end_ns']
    cursor = arrival
    for row in rows:
        if (type(row['start_ns']) is not int or type(row['end_ns']) is not int
                or row['start_ns'] != cursor or row['end_ns'] < cursor
                or row['duration_ns'] != row['end_ns'] - row['start_ns']):
            raise ValueError('corrupt or noncontiguous wallclock partition')
        cursor = row['end_ns']
    if partition['ttft_ns'] != first - arrival:
        raise ValueError('partition TTFT mismatch')
    if partition.get('accounted_ns') != first - arrival or partition.get('unaccounted_ns') != 0:
        raise ValueError('partition accounting mismatch')
    if any(not isinstance(event.get('event_id'), str) or not event['event_id'] for event in events):
        raise ValueError('subinterval ID must be a nonempty string')
    if len({event['event_id'] for event in events}) != len(events):
        raise ValueError('duplicate subinterval ID')
    for event in events:
        start, end = event['start_ns'], event['end_ns']
        if type(start) is not int or type(end) is not int or not arrival <= start <= end <= first:
            raise ValueError('subinterval outside TTFT endpoints')
    refined = []
    for parent in rows:
        start, end = parent['start_ns'], parent['end_ns']
        cuts = sorted({start, end} | {t for event in events for t in (event['start_ns'], event['end_ns'])
                                     if start < t < end})
        for left, right in zip(cuts, cuts[1:]):
            labels = sorted(e['event_id'] for e in events if e['start_ns'] <= left and right <= e['end_ns'])
            refined.append(dict(parent_start=parent['start'], parent_end=parent['end'],
                                start_ns=left, end_ns=right, duration_ns=right-left,
                                duration_ms=(right-left)/1e6, active_event_ids=labels,
                                classification='observed_host_interval' if labels else 'uninstrumented_host_gap'))
    total = sum(row['duration_ns'] for row in refined)
    if total != first - arrival:
        raise ValueError('refined accounting does not close')
    return dict(intervals=refined, ttft_ns=total, accounted_ns=total, unaccounted_ns=0,
                semantics='nonoverlapping_host_partition_with_nested_labels_not_cuda_cost_sum')
