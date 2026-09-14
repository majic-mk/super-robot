import json
import tempfile
import unittest
from pathlib import Path

from scripts.server.aggregate_multisource_capture_matrix import aggregate


class MultiSourceCaptureAggregateTests(unittest.TestCase):
    def test_reports_complementarity_without_claiming_gain(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [
                ({"a": 0.1, "b": 0.2}, {"a": 0.15, "b": 0.1}),
                ({"a": 0.2, "b": 0.1}, {"a": 0.2, "b": 0.1}),
            ]
            for i, (d1, d2) in enumerate(rows):
                child = root / f"r{i}"
                child.mkdir()
                (child / "replay.json").write_text(json.dumps({
                    "gpu_execution_allowed": False,
                    "observation_sha256": f"obs{i}",
                    "cells": [{"reference_trim_ratio": 0.15,
                               "depth2_keep_fraction": 1.0,
                               "depth1_scores": d1, "depth2_scores": d2,
                               "margin": 0.2,
                               "qa_passed": None,
                               "production_admission_allowed": False}],
                }), encoding="utf-8")
            report = aggregate([root / "r0", root / "r1"])
            self.assertEqual(report["capture_count"], 2)
            self.assertEqual(report["source_count_per_capture"], [2])
            self.assertEqual(report["distinct_d1_winners"], 2)
            self.assertEqual(report["d1_d2_disagreement_count"], 1)
            self.assertFalse(report["reuse_gain_measured"])
            self.assertTrue(report["multisource_complementarity_observed"])


if __name__ == "__main__":
    unittest.main()
