from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


def rope_angles(
    position: int, head_dim: int, theta: float = 10000.0
) -> Tuple[List[float], List[float]]:
    if position < 0:
        raise ValueError("position must be non-negative")
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be a positive even integer")
    cosines: List[float] = []
    sines: List[float] = []
    for pair_index in range(head_dim // 2):
        exponent = (2.0 * pair_index) / head_dim
        angle = position / (theta ** exponent)
        cosines.append(math.cos(angle))
        sines.append(math.sin(angle))
    return cosines, sines


def apply_rope(
    vector: Sequence[float],
    cosines: Sequence[float],
    sines: Sequence[float],
    inverse: bool = False,
) -> List[float]:
    if len(vector) % 2:
        raise ValueError("RoPE vector dimension must be even")
    if len(cosines) != len(vector) // 2 or len(sines) != len(cosines):
        raise ValueError("cos/sin dimensions do not match vector")
    output: List[float] = []
    sign = -1.0 if inverse else 1.0
    for idx, (cosine, sine) in enumerate(zip(cosines, sines)):
        first = float(vector[2 * idx])
        second = float(vector[2 * idx + 1])
        output.append(first * cosine - sign * second * sine)
        output.append(sign * first * sine + second * cosine)
    return output


def relative_l2_error(reference: Sequence[float], actual: Sequence[float]) -> float:
    if len(reference) != len(actual):
        raise ValueError("vectors must have equal length")
    numerator = math.sqrt(
        sum((float(left) - float(right)) ** 2 for left, right in zip(reference, actual))
    )
    denominator = math.sqrt(sum(float(value) ** 2 for value in reference))
    return numerator / max(denominator, 1e-12)
