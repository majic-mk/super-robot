"""CPU interface tests only; these do not generate native/GPU evidence."""
import math
from pathlib import Path
import tempfile
import unittest

from probekv.v8_schema10_contracts import AbsoluteResidualThreshold
from probekv.v8_schema9_contracts import AbsoluteResidualThreshold as OldThreshold
from probekv.v8_schema10_native_factory import verified_model_asset_path
from probekv.v8_schema10_native_correctness import execute_fixed_source_arm, run_combined_native_r1
from probekv.v8_schema10_native_validation import validate_correctness_observation


class ConcreteCorrectnessTests(unittest.TestCase):
    def test_legacy_depths_do_not_use_schema9_d1d2_constraint(self):
        for d in (1, 2, 4, 5, 7, 8):
            self.assertEqual(AbsoluteResidualThreshold(d, .25).completed_depth, d)
        with self.assertRaises(ValueError):
            OldThreshold(8, .25)
        for d, value in ((0, .25), (True, .25), (1, math.nan), (1, math.inf), (1, -.1)):
            with self.assertRaises(ValueError):
                AbsoluteResidualThreshold(d, value)

    def test_asset_path_rejects_traversal_and_missing_files(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / "config.json").write_text("{}")
            self.assertEqual(verified_model_asset_path(path, "config.json"), (path / "config.json").resolve())
            for name in ("../config.json", "missing.json", str(path / "config.json")):
                with self.assertRaises(ValueError):
                    verified_model_asset_path(path, name)

    def test_short_teacher_trace_fails_before_gpu_or_output(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "not-created"
            with self.assertRaises(ValueError):
                run_combined_native_r1(None, request={}, warm_request={}, source_id="s", segment_id="c",
                                      teacher_token_ids=[1] * 30, output_dir=output)
            self.assertFalse(output.exists())

    def test_fixed_source_cost_arm_keeps_repair_and_integrity_modes_explicit(self):
        for ratio in (0, -0.1, 1.1, True):
            with self.assertRaises(ValueError):
                execute_fixed_source_arm(None, request={}, repair_ratio=ratio)
        with self.assertRaisesRegex(ValueError, "meaningful only"):
            execute_fixed_source_arm(None, request={}, repair_ratio=.15)
        with self.assertRaisesRegex(ValueError, "preparation controls require"):
            execute_fixed_source_arm(None, request={}, wait_all_source_layers=True)
        with self.assertRaisesRegex(ValueError, "preparation controls require"):
            execute_fixed_source_arm(None, request={}, commit_source=False)

    def test_selective_absolute_mask_uses_repair_support_not_r1_rows(self):
        row = {"origin": "real_cuda_execution", "fake_timing": False,
               "cached_prefix_tokens": 4, "layer_rows": [
                   {"layer": 1, "active_positions": [4, 5, 6, 7],
                    "expected_positions": [4, 5, 6, 7]},
                   {"layer": 2, "active_positions": [4, 7],
                    "expected_positions": [4, 7]}]}
        self.assertTrue(validate_correctness_observation("absolute_mask", row))
        row["layer_rows"][1]["expected_positions"] = [4, 5, 6, 7]
        with self.assertRaises(ValueError):
            validate_correctness_observation("absolute_mask", row)


if __name__ == "__main__":
    unittest.main()
