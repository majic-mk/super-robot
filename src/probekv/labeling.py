from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class RatioMeasurement:
    ratio: float
    task_score_drop: float
    token_f1: float

    @property
    def individually_passes(self) -> bool:
        return self.task_score_drop <= 0.10 and self.token_f1 >= 0.90


def safe_repair_ratio(measurements: Sequence[RatioMeasurement]) -> Optional[float]:
    """Return the minimum ratio whose entire measured suffix passes.

    This implements the monotonic envelope: an isolated pass at 10% is not
    considered safe when a higher tested ratio fails. Conversely, a failed 10%
    and passed 20% is expected because more repaired tokens generally improve
    fidelity.
    """
    if not measurements:
        return None
    ordered = sorted(measurements, key=lambda item: item.ratio)
    ratios = [item.ratio for item in ordered]
    if len(ratios) != len(set(ratios)):
        raise ValueError("repair ratios must be unique")
    suffix_passes = True
    result: Optional[float] = None
    for measurement in reversed(ordered):
        suffix_passes = suffix_passes and measurement.individually_passes
        if suffix_passes:
            result = measurement.ratio
    return result
