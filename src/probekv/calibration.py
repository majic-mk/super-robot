from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

class IsotonicRegressor:
    """Small dependency-free increasing isotonic regression (PAV)."""

    def __init__(self) -> None:
        self._thresholds: List[float] = []
        self._values: List[float] = []

    def fit(self, features: Sequence[float], targets: Sequence[float]) -> "IsotonicRegressor":
        if len(features) != len(targets) or not features:
            raise ValueError("features and targets must be paired and non-empty")
        ordered = sorted(zip(features, targets), key=lambda item: item[0])
        blocks: List[List[float]] = []
        # Each block stores start_x, end_x, target_sum, weight.
        for feature, target in ordered:
            if blocks and float(feature) == blocks[-1][1]:
                blocks[-1][2] += float(target)
                blocks[-1][3] += 1.0
            else:
                blocks.append([float(feature), float(feature), float(target), 1.0])
            while (
                len(blocks) >= 2
                and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]
            ):
                right = blocks.pop()
                left = blocks.pop()
                blocks.append(
                    [
                        left[0],
                        right[1],
                        left[2] + right[2],
                        left[3] + right[3],
                    ]
                )
        self._thresholds = [block[1] for block in blocks]
        self._values = [block[2] / block[3] for block in blocks]
        return self

    def predict_one(self, feature: float) -> float:
        if not self._thresholds:
            raise RuntimeError("regressor is not fitted")
        index = bisect.bisect_left(self._thresholds, float(feature))
        return self._values[min(index, len(self._values) - 1)]

    def predict(self, features: Sequence[float]) -> List[float]:
        return [self.predict_one(feature) for feature in features]


@dataclass(frozen=True)
class SplitConformalUpper:
    """Calibrate a one-sided upper prediction bound on held-out residuals."""

    miscoverage: float = 0.10
    correction: float = 0.0

    @classmethod
    def fit(
        cls,
        predictions: Sequence[float],
        actual: Sequence[float],
        miscoverage: float = 0.10,
    ) -> "SplitConformalUpper":
        if len(predictions) != len(actual) or len(predictions) == 0:
            raise ValueError("predictions and actual values must be paired")
        if not 0.0 < miscoverage < 1.0:
            raise ValueError("miscoverage must be in (0, 1)")
        residuals = sorted(
            max(0.0, float(target) - float(prediction))
            for prediction, target in zip(predictions, actual)
        )
        # Finite-sample split-conformal quantile: ceil((n+1)*(1-alpha))/n.
        n = len(residuals)
        rank = min(n, math.ceil((n + 1) * (1.0 - miscoverage)))
        return cls(miscoverage, residuals[rank - 1])

    def upper(self, prediction: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return min(upper, max(lower, float(prediction) + self.correction))


class ConservativeRatioPredictor:
    def __init__(self, miscoverage: float = 0.10) -> None:
        self.regressor = IsotonicRegressor()
        self.miscoverage = miscoverage
        self.calibrator = None  # type: SplitConformalUpper

    def fit(
        self,
        train_features: Sequence[float],
        train_targets: Sequence[float],
        calibration_features: Sequence[float],
        calibration_targets: Sequence[float],
    ) -> "ConservativeRatioPredictor":
        self.regressor.fit(train_features, train_targets)
        predictions = self.regressor.predict(calibration_features)
        self.calibrator = SplitConformalUpper.fit(
            predictions, calibration_targets, self.miscoverage
        )
        return self

    def predict_upper(self, feature: float) -> float:
        if self.calibrator is None:
            raise RuntimeError("predictor is not fitted")
        return self.calibrator.upper(self.regressor.predict_one(feature))


class QuantileGradientBoostingBudgetPredictor:
    """Combined-feature predictor prescribed by the experiment contract.

    scikit-learn is imported lazily so the source invariants and simulation
    remain usable in a dependency-free environment.
    """

    def __init__(
        self,
        quantile: float = 0.90,
        miscoverage: float = 0.10,
        random_state: int = 20260726,
    ) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        self.quantile = quantile
        self.miscoverage = miscoverage
        self.random_state = random_state
        self.model = None
        self.calibrator = None

    def fit(
        self,
        train_features,
        train_targets: Sequence[float],
        calibration_features,
        calibration_targets: Sequence[float],
    ) -> "QuantileGradientBoostingBudgetPredictor":
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError as error:
            raise RuntimeError(
                "scikit-learn is required for the combined-feature predictor"
            ) from error
        self.model = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.quantile,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=self.random_state,
        )
        self.model.fit(train_features, train_targets)
        predictions = self.model.predict(calibration_features)
        self.calibrator = SplitConformalUpper.fit(
            predictions, calibration_targets, self.miscoverage
        )
        return self

    def predict_upper(self, features) -> List[float]:
        if self.model is None or self.calibrator is None:
            raise RuntimeError("predictor is not fitted")
        return [
            self.calibrator.upper(float(prediction))
            for prediction in self.model.predict(features)
        ]
