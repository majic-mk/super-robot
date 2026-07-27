import unittest

from probekv.labeling import RatioMeasurement, safe_repair_ratio


class SafeRatioTests(unittest.TestCase):
    def test_more_repair_can_turn_failure_into_pass(self):
        measurements = [
            RatioMeasurement(0.10, 0.12, 0.89),
            RatioMeasurement(0.20, 0.08, 0.92),
            RatioMeasurement(0.30, 0.05, 0.95),
        ]
        self.assertEqual(safe_repair_ratio(measurements), 0.20)

    def test_nonmonotonic_isolated_pass_is_not_safe(self):
        measurements = [
            RatioMeasurement(0.10, 0.05, 0.95),
            RatioMeasurement(0.20, 0.12, 0.89),
            RatioMeasurement(0.30, 0.03, 0.96),
        ]
        self.assertEqual(safe_repair_ratio(measurements), 0.30)

    def test_no_safe_ratio_returns_none(self):
        self.assertIsNone(
            safe_repair_ratio([RatioMeasurement(1.0, 0.2, 0.7)])
        )


if __name__ == "__main__":
    unittest.main()
