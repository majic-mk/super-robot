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


def cache_craft_style_score(metadata: CacheCraftMetadata) -> float:
    """Metadata-only baseline score; lower predicts lower repair cost."""
    compatibility = (
        0.30 * metadata.prefix_overlap
        + 0.20 * metadata.order_score
        + 0.25 * metadata.cci
        + 0.25 * metadata.cfo
    )
    return 1.0 - compatibility


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
