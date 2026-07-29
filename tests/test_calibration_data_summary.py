import unittest

from probekv.calibration import (
    GroupedSimultaneousConformal,
    CalibratedGradientBoostingIntervalPredictor,
    ConservativeRatioPredictor,
    IsotonicRegressor,
    QuantileGradientBoostingBudgetPredictor,
    SplitConformalUpper,
    SplitConformalInterval,
)
from probekv.data import (
    assert_group_isolation,
    assert_locked_test,
    deterministic_group_split,
)
from probekv.summary import (
    block_pool,
    dequantize_int8,
    mean_absolute_error,
    quantize_int8,
)


class CalibrationTests(unittest.TestCase):
    def test_isotonic_predictions_are_monotonic(self):
        model = IsotonicRegressor().fit([0, 1, 2, 3], [0.1, 0.5, 0.3, 0.9])
        predictions = model.predict([0, 1, 2, 3])
        self.assertEqual(predictions, sorted(predictions))

    def test_conformal_upper_never_decreases_prediction(self):
        calibrator = SplitConformalUpper.fit(
            [0.1, 0.2, 0.3, 0.4], [0.15, 0.28, 0.32, 0.55]
        )
        self.assertGreaterEqual(calibrator.upper(0.2), 0.2)

    def test_conformal_uses_finite_sample_order_statistic(self):
        calibrator = SplitConformalUpper.fit(
            [0.0] * 9,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            miscoverage=0.2,
        )
        # ceil((9+1)*0.8)=8th ordered residual.
        self.assertAlmostEqual(calibrator.correction, 0.8)

    def test_two_sided_conformal_interval_contains_prediction(self):
        calibrator = SplitConformalInterval.fit(
            [0.1, 0.2, 0.3], [0.2, 0.25, 0.5], miscoverage=0.2
        )
        lower, upper = calibrator.bounds(0.4)
        self.assertLessEqual(lower, 0.4)
        self.assertGreaterEqual(upper, 0.4)

    def test_simultaneous_conformal_uses_one_maximum_per_case(self):
        calibrator = GroupedSimultaneousConformal.fit(
            {
                "case-a": [0.1, 0.8, 0.2],
                "case-b": [0.3, 0.4],
                "case-c": [0.2, 0.5],
            },
            miscoverage=0.25,
        )
        self.assertEqual(calibrator.groups, 3)
        self.assertEqual(calibrator.correction, 0.8)

    def test_conservative_predictor_returns_bounded_ratio(self):
        predictor = ConservativeRatioPredictor().fit(
            [0, 1, 2, 3],
            [0.1, 0.2, 0.4, 0.6],
            [0.5, 1.5, 2.5],
            [0.2, 0.4, 0.7],
        )
        self.assertTrue(0 <= predictor.predict_upper(1.2) <= 1)

    def test_quantile_gradient_boosting_combined_predictor(self):
        predictor = QuantileGradientBoostingBudgetPredictor(
            random_state=20260726
        ).fit(
            [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [3, 3]],
            [0.1, 0.2, 0.25, 0.4, 0.7, 0.9],
            [[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]],
            [0.3, 0.6, 0.85],
        )
        values = predictor.predict_upper([[0.5, 0.5], [2.5, 2.5]])
        self.assertEqual(len(values), 2)
        self.assertTrue(all(0 <= value <= 1 for value in values))

    def test_calibrated_interval_predictor(self):
        predictor = CalibratedGradientBoostingIntervalPredictor().fit(
            [[0], [1], [2], [3], [4], [5]],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [[0.5], [2.5], [4.5]],
            [0.15, 0.35, 0.55],
        )
        bounds = predictor.predict_bounds([[1.5], [3.5]])
        self.assertEqual(len(bounds), 2)
        self.assertTrue(all(0 <= lower <= upper <= 1 for lower, upper in bounds))


class DataIsolationTests(unittest.TestCase):
    def test_group_split_is_deterministic(self):
        first = deterministic_group_split("doc-1", 20260726)
        second = deterministic_group_split("doc-1", 20260726)
        self.assertEqual(first, second)

    def test_group_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_group_isolation({"hash-c": ["train", "test"]})

    def test_locked_test_cannot_tune_thresholds(self):
        with self.assertRaisesRegex(ValueError, "invisible"):
            assert_locked_test(["train", "test"], False)
        assert_locked_test(["train", "calibration"], False)


class SummaryTests(unittest.TestCase):
    def test_int8_round_trip_error_is_small(self):
        values = [index / 100.0 - 1.0 for index in range(200)]
        recovered = dequantize_int8(quantize_int8(values))
        self.assertLess(mean_absolute_error(values, recovered), 0.01)

    def test_block_pool_includes_tail(self):
        self.assertEqual(block_pool([1, 3, 5], 2), [2.0, 5.0])


if __name__ == "__main__":
    unittest.main()
