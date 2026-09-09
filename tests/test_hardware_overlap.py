import unittest
from probekv.v8_schema10_hardware_overlap import summarize_hardware_overlap, union_duration


def event(cat, name, ts, dur, external=None, **args):
    return dict(ph="X", cat=cat, name=name, ts=ts, dur=dur, pid=1, tid=1,
                args={"External id": external, **args})


def trace():
    return {"traceEvents": [event("user_annotation", "probekv.copy_layer.3", 0, 2),
        event("cpu_op", "aten::to", 0, 1, 10),
        event("user_annotation", "probekv.compute_layer.2", 3, 2),
        event("cpu_op", "aten::matmul", 3, 1, 11),
        event("gpu_memcpy", "Memcpy HtoD (Pinned -> Device)", 10, 10, 10, device=0, stream=7),
        event("kernel", "matmul", 15, 10, 11, device=0, stream=1)]}


class HardwareOverlapTests(unittest.TestCase):
    def test_requires_correlated_hardware_activity_not_cpu_ranges(self):
        t = trace()
        t["traceEvents"] = t["traceEvents"][:4]
        self.assertFalse(summarize_hardware_overlap(t)["hardware_activity_available"])

    def test_actual_kernel_copy_intersection(self):
        result = summarize_hardware_overlap(trace())
        self.assertTrue(result["hardware_copy_kernel_overlap_observed"])
        self.assertEqual(result["copy_kernel_overlap_union_ms"], .005)
        self.assertEqual(result["layer_pairs"][0]["copy_layer"], 3)

    def test_does_not_sum_duplicate_or_overlapping_intervals(self):
        t = trace()
        t["traceEvents"].append(t["traceEvents"][-1].copy())
        self.assertEqual(summarize_hardware_overlap(t)["copy_kernel_overlap_union_ms"], .005)
        self.assertEqual(union_duration([(1, 5), (2, 3), (4, 7)]), 6)

    def test_different_device_or_missing_correlation_cannot_prove_overlap(self):
        for field, value in (("device", 1), ("External id", 99), ("stream", 7)):
            t = trace()
            t["traceEvents"][-1]["args"][field] = value
            self.assertFalse(summarize_hardware_overlap(t)["hardware_copy_kernel_overlap_observed"])

    def test_invalid_interval_rejected(self):
        t = trace()
        t["traceEvents"][-1]["dur"] = -1
        with self.assertRaises(ValueError):
            summarize_hardware_overlap(t)
