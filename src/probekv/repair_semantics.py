from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple


@dataclass(frozen=True)
class TokenRegions:
    """Token regions for one ProbeKV repair request.

    ``P`` is the exact current prefix, ``C`` is the only repair-eligible
    repeated segment, and ``S`` is the mandatory dense suffix.  The three
    regions are contiguous and cover the complete prompt.
    """

    prefix_tokens: int
    segment_tokens: int
    suffix_tokens: int

    def validate(self, total_tokens: int | None = None) -> None:
        if self.prefix_tokens < 0:
            raise ValueError("prefix_tokens must be non-negative")
        if self.segment_tokens <= 0:
            raise ValueError("segment_tokens must be positive")
        if self.suffix_tokens < 0:
            raise ValueError("suffix_tokens must be non-negative")
        if total_tokens is not None and self.total_tokens != total_tokens:
            raise ValueError("P/C/S regions do not cover the complete prompt")

    @property
    def segment_start(self) -> int:
        return self.prefix_tokens

    @property
    def segment_end(self) -> int:
        return self.prefix_tokens + self.segment_tokens

    @property
    def suffix_start(self) -> int:
        return self.segment_end

    @property
    def total_tokens(self) -> int:
        return self.prefix_tokens + self.segment_tokens + self.suffix_tokens


@dataclass(frozen=True)
class RepairSelection:
    requested_ratio: float
    eligible_segment_tokens: int
    selected_segment_indices: Tuple[int, ...]
    mandatory_suffix_indices: Tuple[int, ...]

    def validate(self, regions: TokenRegions) -> None:
        regions.validate()
        if not 0.0 <= self.requested_ratio <= 1.0:
            raise ValueError("requested_ratio must be in [0, 1]")
        if self.eligible_segment_tokens != regions.segment_tokens:
            raise ValueError("eligible token count does not match C")
        segment_range = set(range(regions.segment_start, regions.segment_end))
        suffix_range = tuple(range(regions.suffix_start, regions.total_tokens))
        if len(self.selected_segment_indices) != len(
            set(self.selected_segment_indices)
        ):
            raise ValueError("selected C indices must be unique")
        if not set(self.selected_segment_indices).issubset(segment_range):
            raise ValueError("repair selection contains a token outside C")
        if self.mandatory_suffix_indices != suffix_range:
            raise ValueError("mandatory suffix indices do not exactly cover S")
        expected = repaired_segment_token_count(
            regions.segment_tokens, self.requested_ratio
        )
        if len(self.selected_segment_indices) != expected:
            raise ValueError("selected C token count does not match ratio policy")

    @property
    def selected_segment_tokens(self) -> int:
        return len(self.selected_segment_indices)

    @property
    def effective_ratio(self) -> float:
        return self.selected_segment_tokens / float(self.eligible_segment_tokens)

    @property
    def execution_indices(self) -> Tuple[int, ...]:
        return tuple(
            sorted(self.selected_segment_indices + self.mandatory_suffix_indices)
        )

    def to_audit_row(self, regions: TokenRegions) -> Dict[str, Any]:
        self.validate(regions)
        return {
            "requested_ratio": self.requested_ratio,
            "eligible_segment_tokens": self.eligible_segment_tokens,
            "selected_segment_tokens": self.selected_segment_tokens,
            "effective_ratio": self.effective_ratio,
            "mandatory_suffix_tokens": len(self.mandatory_suffix_indices),
            "selected_segment_indices": list(self.selected_segment_indices),
            "mandatory_suffix_indices": list(self.mandatory_suffix_indices),
            "execution_indices": list(self.execution_indices),
        }


def repaired_segment_token_count(segment_tokens: int, ratio: float) -> int:
    """Return the frozen repair count for C, excluding mandatory suffix tokens."""

    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be finite and in [0, 1]")
    if ratio == 0.0:
        return 0
    if ratio == 1.0:
        return segment_tokens
    return int(math.floor(segment_tokens * ratio))


def select_repair_tokens(
    drift_scores: Sequence[float],
    regions: TokenRegions,
    ratio: float,
) -> RepairSelection:
    """Apply CacheBlend-style largest-drift ranking only within C.

    Ties are resolved by the absolute token index.  This makes repeated runs
    deterministic and guarantees nested selected sets for an increasing ratio
    grid when the drift scores are unchanged.
    """

    regions.validate(total_tokens=len(drift_scores))
    count = repaired_segment_token_count(regions.segment_tokens, ratio)
    ranked = sorted(
        range(regions.segment_start, regions.segment_end),
        key=lambda index: (-float(drift_scores[index]), index),
    )
    selection = RepairSelection(
        requested_ratio=float(ratio),
        eligible_segment_tokens=regions.segment_tokens,
        selected_segment_indices=tuple(sorted(ranked[:count])),
        mandatory_suffix_indices=tuple(
            range(regions.suffix_start, regions.total_tokens)
        ),
    )
    selection.validate(regions)
    return selection


def assert_nested_selections(selections: Iterable[RepairSelection]) -> None:
    ordered = sorted(selections, key=lambda selection: selection.requested_ratio)
    previous: set[int] = set()
    for selection in ordered:
        current = set(selection.selected_segment_indices)
        if not previous.issubset(current):
            raise ValueError("repair selections are not nested across ratios")
        previous = current
