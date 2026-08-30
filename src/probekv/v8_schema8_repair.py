from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

from .v8_schema8_contracts import RepairRatioScope


@dataclass(frozen=True)
class SegmentLayerRepairRatio:
    segment_id: str
    layer_1based: int
    first_selective_reuse_layer: int
    ratio: float

    def __post_init__(self) -> None:
        if not self.segment_id or self.first_selective_reuse_layer < 1:
            raise ValueError("repair ratio row requires a Segment and reuse boundary")
        if self.layer_1based < self.first_selective_reuse_layer:
            raise ValueError("repair ratio precedes the Segment reuse boundary")
        if not 0 < self.ratio <= 0.15:
            raise ValueError("schema-v8 online repair ratio must be in (0, 0.15]")

    @property
    def repair_age(self) -> int:
        return self.layer_1based - self.first_selective_reuse_layer


@dataclass(frozen=True)
class MultiSegmentRepairRatioPlan:
    scope: RepairRatioScope
    rows: Tuple[SegmentLayerRepairRatio, ...]
    certified_floor: float
    profile_frozen: bool
    certified_ratio_candidates: Tuple[float, ...] = (0.10, 0.12, 0.15)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", RepairRatioScope(self.scope))
        if not self.rows or not 0 < self.certified_floor <= 0.15:
            raise ValueError("repair ratio plan requires rows and a certified floor")
        keys = tuple((row.segment_id, row.layer_1based) for row in self.rows)
        if len(keys) != len(set(keys)):
            raise ValueError("a Segment may have one ratio per layer")
        candidates = tuple(sorted(set(float(value) for value in self.certified_ratio_candidates)))
        if not candidates or candidates[0] <= 0 or candidates[-1] > 0.15:
            raise ValueError("repair-ratio candidate set must remain in (0, 0.15]")
        by_segment: Dict[str, list[SegmentLayerRepairRatio]] = {}
        for row in self.rows:
            if row.ratio < self.certified_floor - 1e-12:
                raise ValueError("repair ratio fell below the certified floor")
            by_segment.setdefault(row.segment_id, []).append(row)
        for segment_rows in by_segment.values():
            ordered = sorted(segment_rows, key=lambda row: row.layer_1based)
            if any(
                later.ratio > earlier.ratio + 1e-12
                for earlier, later in zip(ordered, ordered[1:])
            ):
                raise ValueError("gradual repair ratio must be non-increasing")

        if self.scope is RepairRatioScope.UNIFORM_FIXED:
            if any(abs(row.ratio - 0.15) > 1e-12 for row in self.rows):
                raise ValueError("fixed15 requires 0.15 for every Segment and layer")
        elif self.scope is RepairRatioScope.SHARED_RELATIVE_SCHEDULE:
            ratio_by_age: Dict[int, float] = {}
            for row in self.rows:
                previous = ratio_by_age.setdefault(row.repair_age, row.ratio)
                if abs(previous - row.ratio) > 1e-12:
                    raise ValueError(
                        "static gradual uses one ratio at each relative repair age"
                    )
        else:
            if not self.profile_frozen:
                raise ValueError("per-Segment adaptive ratios require a frozen Profile")
            if any(
                all(abs(row.ratio - candidate) > 1e-12 for candidate in candidates)
                for row in self.rows
            ):
                raise ValueError("adaptive ratio is outside the certified candidates")

    def ratios_for_layer(self, layer_1based: int) -> Mapping[str, float]:
        return {
            row.segment_id: row.ratio
            for row in self.rows
            if row.layer_1based == layer_1based
        }


def validate_union_repair_ratio_plan(
    *,
    scope: RepairRatioScope,
    rows: Sequence[SegmentLayerRepairRatio],
    certified_floor: float,
    profile_frozen: bool,
    certified_ratio_candidates: Sequence[float] = (0.10, 0.12, 0.15),
) -> MultiSegmentRepairRatioPlan:
    return MultiSegmentRepairRatioPlan(
        scope,
        tuple(rows),
        certified_floor,
        profile_frozen,
        tuple(float(value) for value in certified_ratio_candidates),
    )
