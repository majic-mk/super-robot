from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
class JointRepairRatioCandidate:
    """One bounded request-level ratio vector for a single layer."""

    candidate_id: str
    layer_1based: int
    ratios_by_segment: Tuple[Tuple[str, float], ...]
    aggregate_load_upper_ms: float
    union_repair_upper_ms: float
    nonoverlap_upper_ms: float

    def __post_init__(self) -> None:
        if not self.candidate_id or self.layer_1based < 1:
            raise ValueError("joint repair candidate needs identity and a layer")
        segment_ids = tuple(segment_id for segment_id, _ in self.ratios_by_segment)
        if not segment_ids or len(segment_ids) != len(set(segment_ids)):
            raise ValueError("joint repair candidate must cover unique Segments")
        if any(not 0 < float(ratio) <= 0.15 for _, ratio in self.ratios_by_segment):
            raise ValueError("joint repair candidate ratio must be in (0, 0.15]")
        if min(
            self.aggregate_load_upper_ms,
            self.union_repair_upper_ms,
            self.nonoverlap_upper_ms,
        ) < 0:
            raise ValueError("joint repair candidate costs must be non-negative")

    @property
    def request_layer_critical_path_upper_ms(self) -> float:
        return (
            max(self.aggregate_load_upper_ms, self.union_repair_upper_ms)
            + self.nonoverlap_upper_ms
        )


@dataclass(frozen=True)
class JointRepairRatioDecision:
    """Frozen-profile-backed request-level adaptive ratio choice."""

    layer_1based: int
    selected_candidate_id: str
    ratios_by_segment: Tuple[Tuple[str, float], ...]
    predicted_layer_critical_path_upper_ms: float
    candidate_set_digest: str
    repair_policy_profile_sha256: str
    runtime_cost_profile_sha256: str

    def __post_init__(self) -> None:
        if self.layer_1based < 1 or not self.selected_candidate_id:
            raise ValueError("joint repair decision requires a layer and candidate")
        if self.predicted_layer_critical_path_upper_ms < 0:
            raise ValueError("joint repair decision cost must be non-negative")
        if len(self.candidate_set_digest) != 64:
            raise ValueError("joint repair candidate-set digest must be SHA256")
        if len(self.repair_policy_profile_sha256) != 64:
            raise ValueError("adaptive ratio decision requires RepairPolicyProfile SHA")
        if len(self.runtime_cost_profile_sha256) != 64:
            raise ValueError("adaptive ratio decision requires RuntimeCostProfile SHA")

    def ratio_map(self) -> Mapping[str, float]:
        return dict(self.ratios_by_segment)


def choose_request_level_adaptive_ratio(
    *,
    candidates: Sequence[JointRepairRatioCandidate],
    expected_segment_ids: Sequence[str],
    repair_policy_profile_sha256: str,
    runtime_cost_profile_sha256: str,
) -> JointRepairRatioDecision:
    """Choose one bounded ratio vector using the request union critical path.

    This deliberately does not optimize each Segment independently.  Ties in
    predicted completion time prefer the higher total repair ratio, followed
    by deterministic candidate identity.
    """

    rows = tuple(candidates)
    if not rows:
        raise ValueError("adaptive repair requires bounded joint candidates")
    expected = set(expected_segment_ids)
    if not expected:
        raise ValueError("adaptive repair requires active Segments")
    layer = rows[0].layer_1based
    for row in rows:
        if row.layer_1based != layer:
            raise ValueError("joint ratio candidates must belong to one layer")
        if {segment_id for segment_id, _ in row.ratios_by_segment} != expected:
            raise ValueError("joint ratio candidate Segment inventory differs")
    ordered = sorted(
        rows,
        key=lambda row: (
            row.request_layer_critical_path_upper_ms,
            -sum(float(ratio) for _, ratio in row.ratios_by_segment),
            row.candidate_id,
        ),
    )
    selected = ordered[0]
    digest_payload = [
        {
            "candidate_id": row.candidate_id,
            "layer_1based": row.layer_1based,
            "ratios_by_segment": sorted(row.ratios_by_segment),
            "aggregate_load_upper_ms": row.aggregate_load_upper_ms,
            "union_repair_upper_ms": row.union_repair_upper_ms,
            "nonoverlap_upper_ms": row.nonoverlap_upper_ms,
        }
        for row in sorted(rows, key=lambda row: row.candidate_id)
    ]
    candidate_set_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return JointRepairRatioDecision(
        layer,
        selected.candidate_id,
        tuple(sorted(selected.ratios_by_segment)),
        selected.request_layer_critical_path_upper_ms,
        candidate_set_digest,
        repair_policy_profile_sha256,
        runtime_cost_profile_sha256,
    )


@dataclass(frozen=True)
class MultiSegmentRepairRatioPlan:
    scope: RepairRatioScope
    rows: Tuple[SegmentLayerRepairRatio, ...]
    certified_floor: float
    profile_frozen: bool
    certified_ratio_candidates: Tuple[float, ...] = (0.10, 0.12, 0.15)
    adaptive_joint_decisions: Tuple[JointRepairRatioDecision, ...] = ()

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
            if self.adaptive_joint_decisions:
                raise ValueError("fixed15 cannot carry adaptive joint decisions")
            if any(abs(row.ratio - 0.15) > 1e-12 for row in self.rows):
                raise ValueError("fixed15 requires 0.15 for every Segment and layer")
        elif self.scope is RepairRatioScope.SHARED_RELATIVE_SCHEDULE:
            if self.adaptive_joint_decisions:
                raise ValueError("static gradual cannot carry adaptive joint decisions")
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
            decisions = {
                decision.layer_1based: decision
                for decision in self.adaptive_joint_decisions
            }
            layers = {row.layer_1based for row in self.rows}
            if len(decisions) != len(self.adaptive_joint_decisions) or set(decisions) != layers:
                raise ValueError(
                    "adaptive ratios require one request-level joint decision per layer"
                )
            for layer, decision in decisions.items():
                selected = {
                    row.segment_id: row.ratio
                    for row in self.rows
                    if row.layer_1based == layer
                }
                if selected != decision.ratio_map():
                    raise ValueError(
                        "adaptive ratio rows differ from request-level joint decision"
                    )

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
    adaptive_joint_decisions: Sequence[JointRepairRatioDecision] = (),
) -> MultiSegmentRepairRatioPlan:
    return MultiSegmentRepairRatioPlan(
        scope,
        tuple(rows),
        certified_floor,
        profile_frozen,
        tuple(float(value) for value in certified_ratio_candidates),
        tuple(adaptive_joint_decisions),
    )
