from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Int8Summary:
    values: Tuple[int, ...]
    scale: float


def quantize_int8(values: Sequence[float]) -> Int8Summary:
    if not values:
        return Int8Summary((), 1.0)
    peak = max(abs(float(value)) for value in values)
    scale = peak / 127.0 if peak > 0 else 1.0
    quantized = tuple(
        max(-127, min(127, int(round(float(value) / scale)))) for value in values
    )
    return Int8Summary(quantized, scale)


def dequantize_int8(summary: Int8Summary) -> List[float]:
    return [value * summary.scale for value in summary.values]


def block_pool(values: Sequence[float], block_size: int = 32) -> List[float]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    result = []
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        result.append(sum(float(value) for value in block) / len(block))
    return result


def mean_absolute_error(reference: Sequence[float], actual: Sequence[float]) -> float:
    if len(reference) != len(actual):
        raise ValueError("sequences must have equal length")
    if not reference:
        return 0.0
    return sum(
        abs(float(left) - float(right)) for left, right in zip(reference, actual)
    ) / len(reference)
