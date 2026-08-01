from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from .contracts import ProbeObservation


@dataclass(frozen=True)
class CacheCraftMetadata:
    prefix_overlap: float
    order_score: float
    cci: float
    cfo: float

    def __post_init__(self) -> None:
        for value in (
            self.prefix_overlap,
            self.order_score,
            self.cci,
            self.cfo,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("metadata features must be in [0, 1]")

    @classmethod
    def from_cachecraft_components(
        cls,
        prefix_overlap: float,
        order_penalty: float,
        cci: float,
        alpha: float = 1.0,
    ) -> "CacheCraftMetadata":
        if not 0 <= order_penalty <= 1:
            raise ValueError("Cache-Craft order penalty must be in [0, 1]")
        order_similarity = 1.0 - order_penalty
        return cls(
            prefix_overlap=prefix_overlap,
            order_score=order_similarity,
            cci=cci,
            cfo=cache_craft_cfo(
                prefix_overlap, order_penalty, cci, alpha
            ),
        )

    @property
    def adjusted_prefix_overlap(self) -> float:
        return self.prefix_overlap * self.order_score


def cache_craft_cfo(
    prefix_overlap: float,
    order_penalty: float,
    cci: float,
    alpha: float = 1.0,
) -> float:
    """Cache-Craft Eq. 8 and Eq. 12 with an explicit deployment alpha."""

    if not all(0 <= value <= 1 for value in (prefix_overlap, order_penalty, cci)):
        raise ValueError("Cache-Craft inputs must be in [0, 1]")
    if alpha < 0 or not math.isfinite(alpha):
        raise ValueError("Cache-Craft alpha must be finite and non-negative")
    beta_prime = prefix_overlap * (1.0 - order_penalty)
    return min(1.0, alpha * cci * (1.0 - beta_prime))


def cache_craft_style_score(metadata: CacheCraftMetadata) -> float:
    """Legacy v3-v5 weighted heuristic retained for exact reproduction."""
    compatibility = (
        0.30 * metadata.prefix_overlap
        + 0.20 * metadata.order_score
        + 0.25 * metadata.cci
        + 0.25 * metadata.cfo
    )
    return 1.0 - compatibility


def cache_craft_cfo_score(metadata: CacheCraftMetadata) -> float:
    """Protocol-v6 exact stored Cache-Craft CFO baseline."""
    return metadata.cfo


legacy_cache_craft_style_score = cache_craft_style_score


def combined_feature_vector(observation: ProbeObservation, metadata: CacheCraftMetadata):
    return (
        observation.k_drift,
        observation.v_drift,
        observation.hidden_drift,
        observation.query_score,
        metadata.prefix_overlap,
        metadata.order_score,
        metadata.cci,
        metadata.cfo,
    )


def raw_drift_score(observation: ProbeObservation) -> float:
    return math.sqrt(
        observation.k_drift ** 2
        + observation.v_drift ** 2
        + observation.hidden_drift ** 2
    )
