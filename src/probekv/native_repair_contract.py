"""Explicit repair identity for new native runs and historical evidence.

Missing identity is interpreted as V-only ONLY for old manifests/results.
New experiment entry points always write the primary metric explicitly.
The joint formula is ProbeKV's normalized K/V deviation; this name does not
claim bitwise equivalence to an upstream CacheBlend implementation.
"""

PRIMARY_REPAIR_METRIC = "normalized_kv_deviation"
LEGACY_REPAIR_METRIC = "normalized_v_legacy"
REPAIR_METRICS = (PRIMARY_REPAIR_METRIC, LEGACY_REPAIR_METRIC,
                  "value_squared_l2_pinned_dtype")


def historical_repair_metric(runtime):
    metric = runtime.get("repair_metric", LEGACY_REPAIR_METRIC)
    if metric not in REPAIR_METRICS:
        raise ValueError("unknown explicit native winner repair metric")
    return metric


def validate_repair_cost_evidence(runtime, observations):
    metric = historical_repair_metric(runtime)
    for observation in observations:
        observed = observation.get("winner_repair_metric", LEGACY_REPAIR_METRIC)
        if observed != metric:
            raise ValueError("repair metric differs between manifest and actual cost evidence")
    # Preserve historical keys; never reinterpret old V-only timings as K/V.
    return {} if metric == LEGACY_REPAIR_METRIC else {"repair_metric": metric}
