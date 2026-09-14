import json
from pathlib import Path
import tempfile
import unittest

from scripts.server.aggregate_position_validation_ab import aggregate
from scripts.server.run_schema10_native_correctness import position_validation_pair_specs
from probekv.v8_schema10_execution import digest_json


def fixture(root):
    request = {"token_ids": [1, 2, 3]}
    specs = position_validation_pair_specs(2)
    (root / "manifest.json").write_text(json.dumps({"specs": specs, "request": request,
        "code_commit": "a" * 40, "patch_sha256": "b" * 64}))
    for spec in specs:
        for name in ("host", "device"):
            ms = (100 if spec["warmup"] else 10) + (2 if name == "device" else 0)
            row = {"origin": "real_cuda_execution", "fake_timing": False,
                "request_tokens_sha256": digest_json(request["token_ids"]),
                "cached_prefix_tokens": 1, "prefix_cache_mode": "native_exact_blocks",
                "sampling_signature": {"temperature": 0}, "diagnostic_completed_depth": 1,
                "token_ids": [4, 5], "source_id": None, "committed_segments": {},
                "layer_rows": [{"layer": 2, "expected_positions": [1, 2]}],
                "diagnostic_start_ns": 100, "first_token_ns": 100 + ms * 1000000,
                "first_token_host_ms": ms}
            (root / ("pair-%02d-%s.json" % (spec["pair"], name))).write_text(json.dumps(row))


class PositionABAggregateTests(unittest.TestCase):
    def test_excludes_warmups_and_binds_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fixture(root)
            report = aggregate(root)
            self.assertEqual(report["host_mean_ms"], 10)
            self.assertEqual(report["paired_mean_delta_ms"], -2)
            self.assertFalse(report["online_gain_proven"])
            self.assertEqual(len(report["rows"][0]["arm_file_sha256"]["host"]), 64)

    def test_mismatched_conditions_rejected(self):
        for field, value in [("cached_prefix_tokens", 0), ("token_ids", [9]),
                             ("first_token_host_ms", 0),
                             ("layer_rows", [{"layer": 2, "expected_positions": [2]}]),
                             ("raw_observation_sha256", "0" * 64)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                root = Path(td); fixture(root)
                p = root / "pair-02-host.json"; row = json.loads(p.read_text())
                row[field] = value; p.write_text(json.dumps(row))
                with self.assertRaises(ValueError): aggregate(root)

    def test_missing_pair_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fixture(root)
            (root / "pair-02-host.json").unlink()
            with self.assertRaises(FileNotFoundError): aggregate(root)
