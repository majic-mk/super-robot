from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from .v8_schema8_contracts import RepairRatioScope

if TYPE_CHECKING:
    from .v8_schema8_profile import RepairPolicyProfileV8, RuntimeCostProfileV8


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
        if not 0 < self.ratio <= 1.0:
            raise ValueError("schema-v8 online repair ratio must be in (0, 1]")

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
    template_id: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id or self.layer_1based < 1:
            raise ValueError("joint repair candidate needs identity and a layer")
        segment_ids = tuple(segment_id for segment_id, _ in self.ratios_by_segment)
        if not segment_ids or len(segment_ids) != len(set(segment_ids)):
            raise ValueError("joint repair candidate must cover unique Segments")
        if any(not 0 < float(ratio) <= 1.0 for _, ratio in self.ratios_by_segment):
            raise ValueError("joint repair candidate ratio must be in (0, 1]")
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
    timing_equivalence_tolerance_ms: float = 0.0

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
        if self.timing_equivalence_tolerance_ms < 0:
            raise ValueError("timing-equivalence tolerance must be non-negative")

    def ratio_map(self) -> Mapping[str, float]:
        return dict(self.ratios_by_segment)


@dataclass(frozen=True)
class UniformIOBalanceDecision:
    """One request/layer-wide I/O-balanced repair decision.

    ``io_balance_ratio`` is the largest profiled ratio whose union repair is
    hidden by aggregate winner loading.  ``selected_ratio`` additionally
    applies the quality floor, the 15% conservative reference at the first
    reuse layer, and the no-reentry monotonic bound from the previous layer.
    """

    layer_1based: int
    active_segment_ids: Tuple[str, ...]
    io_balance_ratio: float
    quality_floor: float
    quality_reference_ratio: float
    selected_ratio: float
    aggregate_load_upper_ms: float
    union_repair_upper_ms: float
    nonoverlap_upper_ms: float
    candidate_set_digest: str
    repair_policy_profile_sha256: str
    runtime_cost_profile_sha256: str

    def __post_init__(self) -> None:
        if self.layer_1based < 1 or not self.active_segment_ids:
            raise ValueError("uniform I/O decision requires a layer and Segments")
        if len(self.active_segment_ids) != len(set(self.active_segment_ids)):
            raise ValueError("uniform I/O decision repeats a Segment")
        for value in (
            self.io_balance_ratio,
            self.quality_floor,
            self.quality_reference_ratio,
            self.selected_ratio,
        ):
            if not 0 <= value <= 1:
                raise ValueError("uniform I/O decision ratio is outside [0,1]")
        if self.selected_ratio + 1e-12 < self.quality_floor:
            raise ValueError("uniform I/O decision fell below the quality floor")
        if min(
            self.aggregate_load_upper_ms,
            self.union_repair_upper_ms,
            self.nonoverlap_upper_ms,
        ) < 0:
            raise ValueError("uniform I/O decision costs must be non-negative")
        if any(
            len(value) != 64
            for value in (
                self.candidate_set_digest,
                self.repair_policy_profile_sha256,
                self.runtime_cost_profile_sha256,
            )
        ):
            raise ValueError("uniform I/O decision requires SHA256 provenance")

    @property
    def predicted_layer_critical_path_upper_ms(self) -> float:
        return (
            max(self.aggregate_load_upper_ms, self.union_repair_upper_ms)
            + self.nonoverlap_upper_ms
        )

    def ratio_map(self) -> Mapping[str, float]:
        return {
            segment_id: self.selected_ratio
            for segment_id in self.active_segment_ids
        }


def choose_request_level_adaptive_ratio(
    *,
    candidates: Sequence[JointRepairRatioCandidate],
    expected_segment_ids: Sequence[str],
    repair_policy_profile_sha256: str,
    runtime_cost_profile_sha256: str,
    timing_equivalence_absolute_ms: float = 0.0,
    timing_equivalence_relative: float = 0.0,
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
    if min(timing_equivalence_absolute_ms, timing_equivalence_relative) < 0:
        raise ValueError("timing-equivalence tolerances must be non-negative")
    best_cost = min(row.request_layer_critical_path_upper_ms for row in rows)
    tolerance = max(
        float(timing_equivalence_absolute_ms),
        float(timing_equivalence_relative) * best_cost,
    )
    # If predicted completion is indistinguishable at the frozen profile's
    # resolution, retain more repair work for quality instead of shaving an
    # unmeasurable amount of TTFT.
    equivalent = tuple(
        row
        for row in rows
        if row.request_layer_critical_path_upper_ms <= best_cost + tolerance + 1e-12
    )
    selected = min(
        equivalent,
        key=lambda row: (
            -sum(float(ratio) for _, ratio in row.ratios_by_segment),
            row.candidate_id,
        ),
    )
    digest_payload = [
        {
            "candidate_id": row.candidate_id,
            "layer_1based": row.layer_1based,
            "ratios_by_segment": sorted(row.ratios_by_segment),
            "aggregate_load_upper_ms": row.aggregate_load_upper_ms,
            "union_repair_upper_ms": row.union_repair_upper_ms,
            "nonoverlap_upper_ms": row.nonoverlap_upper_ms,
            "template_id": row.template_id,
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
        tolerance,
    )


class JointLoadRecomputeAwareRepairController:
    """Profile-bound request-level I/O/repair balancing controller.

    The controller never chooses ratios independently per Segment.  A caller
    supplies the bounded candidate vectors measured/profiled for the current
    request layer; this class enforces the certified floor/cap, no-reentry
    monotonicity and profile provenance before choosing the union critical
    path.  Fixed15 and static-gradual remain separate plans and do not call it.
    """

    def __init__(
        self,
        *,
        repair_policy_profile: "RepairPolicyProfileV8",
        runtime_cost_profile: "RuntimeCostProfileV8",
    ) -> None:
        if repair_policy_profile.policy != "load_recompute_aware_gradual":
            raise ValueError("joint adaptive controller requires the adaptive policy")
        if not repair_policy_profile.provenance.frozen:
            raise ValueError("joint adaptive controller requires a frozen repair Profile")
        if not runtime_cost_profile.provenance.frozen:
            raise ValueError("joint adaptive controller requires a frozen runtime Profile")
        if (
            repair_policy_profile.provenance.model_id
            != runtime_cost_profile.provenance.model_id
            or repair_policy_profile.provenance.model_revision
            != runtime_cost_profile.provenance.model_revision
            or repair_policy_profile.provenance.code_commit
            != runtime_cost_profile.provenance.code_commit
        ):
            raise ValueError("repair and runtime Profiles have incompatible provenance")
        self.repair_policy_profile = repair_policy_profile
        self.runtime_cost_profile = runtime_cost_profile
        self.timing_equivalence_absolute_ms = float(
            repair_policy_profile.timing_equivalence_absolute_ms
        )
        self.timing_equivalence_relative = float(
            repair_policy_profile.timing_equivalence_relative
        )

    def choose_layer(
        self,
        *,
        candidates: Sequence[JointRepairRatioCandidate],
        expected_segment_ids: Sequence[str],
        previous_ratio_by_segment: Optional[Mapping[str, float]] = None,
    ) -> JointRepairRatioDecision:
        previous = dict(previous_ratio_by_segment or {})
        expected = set(str(value) for value in expected_segment_ids)
        if previous and set(previous) != expected:
            raise ValueError("previous adaptive ratios differ from active Segments")
        floor = float(self.repair_policy_profile.certified_floor)
        certified = {0.10, 0.12, 0.15}
        allowed_templates = set(
            self.repair_policy_profile.adaptive_candidate_templates
        )
        filtered = []
        for candidate in candidates:
            if candidate.template_id not in allowed_templates:
                raise ValueError(
                    "adaptive candidate is not emitted by a frozen template"
                )
            ratios = dict(candidate.ratios_by_segment)
            if set(ratios) != expected:
                raise ValueError("adaptive candidate Segment inventory differs")
            if any(
                ratio < floor - 1e-12
                or ratio > 0.15 + 1e-12
                or all(abs(ratio - value) > 1e-12 for value in certified)
                for ratio in ratios.values()
            ):
                raise ValueError("adaptive candidate is outside the certified ratio grid")
            if (
                candidate.template_id == "uniform_floor"
                and any(abs(ratio - floor) > 1e-12 for ratio in ratios.values())
            ):
                raise ValueError("uniform-floor template emitted a non-floor ratio")
            if (
                candidate.template_id == "uniform_cap"
                and any(abs(ratio - 0.15) > 1e-12 for ratio in ratios.values())
            ):
                raise ValueError("uniform-cap template emitted a non-cap ratio")
            if any(
                segment_id in previous
                and ratio > previous[segment_id] + 1e-12
                for segment_id, ratio in ratios.items()
            ):
                raise ValueError("adaptive repair ratio may not increase by layer")
            filtered.append(candidate)
        return choose_request_level_adaptive_ratio(
            candidates=tuple(filtered),
            expected_segment_ids=tuple(sorted(expected)),
            repair_policy_profile_sha256=(
                self.repair_policy_profile.profile_sha256
            ),
            runtime_cost_profile_sha256=self.runtime_cost_profile.profile_sha256,
            timing_equivalence_absolute_ms=self.timing_equivalence_absolute_ms,
            timing_equivalence_relative=self.timing_equivalence_relative,
        )

    def build_plan(
        self,
        *,
        candidates_by_layer: Mapping[int, Sequence[JointRepairRatioCandidate]],
        first_selective_reuse_layer_by_segment: Mapping[str, int],
    ) -> "MultiSegmentRepairRatioPlan":
        """Build the executable no-reentry plan in increasing layer order."""

        first_layers = {
            str(segment_id): int(layer)
            for segment_id, layer in first_selective_reuse_layer_by_segment.items()
        }
        if not first_layers:
            raise ValueError("adaptive plan requires reusable Segments")
        missing_boundary_layers = {
            layer for layer in first_layers.values()
            if layer not in {int(value) for value in candidates_by_layer}
        }
        if missing_boundary_layers:
            raise ValueError("adaptive plan misses a Segment's first reuse layer")
        previous = {segment_id: 0.15 for segment_id in first_layers}
        decisions = []
        rows = []
        for layer in sorted(int(value) for value in candidates_by_layer):
            active = tuple(
                sorted(
                    segment_id
                    for segment_id, first_layer in first_layers.items()
                    if layer >= first_layer
                )
            )
            if not active:
                raise ValueError("adaptive candidates precede all reuse boundaries")
            decision = self.choose_layer(
                candidates=candidates_by_layer[layer],
                expected_segment_ids=active,
                previous_ratio_by_segment={
                    segment_id: previous[segment_id] for segment_id in active
                },
            )
            ratios = decision.ratio_map()
            for segment_id in active:
                ratio = float(ratios[segment_id])
                rows.append(
                    SegmentLayerRepairRatio(
                        segment_id,
                        layer,
                        first_layers[segment_id],
                        ratio,
                    )
                )
                previous[segment_id] = ratio
            decisions.append(decision)
        if not rows:
            raise ValueError("adaptive plan has no profiled layers")
        return validate_union_repair_ratio_plan(
            scope=RepairRatioScope.PER_SEGMENT_LOAD_AWARE,
            rows=tuple(rows),
            certified_floor=self.repair_policy_profile.certified_floor,
            profile_frozen=True,
            adaptive_joint_decisions=tuple(decisions),
        )


class RequestLayerUniformIORepairController:
    """Choose one repair fraction for every active Segment in a layer.

    This is the CacheBlend-aligned schema-v8 main candidate.  It profiles the
    request's aggregate next-layer load and union repair curve, derives the
    largest hidden ratio, then applies a frozen quality floor.  The first
    selective layer also retains at least the 15% quality reference.  Later
    layers may keep or shrink that support, never grow it.
    """

    def __init__(
        self,
        *,
        repair_policy_profile: "RepairPolicyProfileV8",
        runtime_cost_profile: "RuntimeCostProfileV8",
    ) -> None:
        if repair_policy_profile.policy != "load_recompute_aware_uniform":
            raise ValueError("uniform I/O controller requires its explicit policy")
        if not repair_policy_profile.provenance.frozen:
            raise ValueError("uniform I/O controller requires a frozen repair Profile")
        if not runtime_cost_profile.provenance.frozen:
            raise ValueError("uniform I/O controller requires a frozen runtime Profile")
        if (
            repair_policy_profile.provenance.model_id
            != runtime_cost_profile.provenance.model_id
            or repair_policy_profile.provenance.model_revision
            != runtime_cost_profile.provenance.model_revision
            or repair_policy_profile.provenance.code_commit
            != runtime_cost_profile.provenance.code_commit
        ):
            raise ValueError("repair and runtime Profiles have incompatible provenance")
        self.repair_policy_profile = repair_policy_profile
        self.runtime_cost_profile = runtime_cost_profile

    @staticmethod
    def _uniform_ratio(candidate: JointRepairRatioCandidate) -> float:
        ratios = {float(value) for _, value in candidate.ratios_by_segment}
        if len(ratios) != 1:
            raise ValueError("request/layer I/O candidate must use one uniform ratio")
        return next(iter(ratios))

    def choose_layer(
        self,
        *,
        candidates: Sequence[JointRepairRatioCandidate],
        expected_segment_ids: Sequence[str],
        previous_uniform_ratio: Optional[float] = None,
    ) -> UniformIOBalanceDecision:
        rows = tuple(candidates)
        expected = tuple(sorted(str(value) for value in expected_segment_ids))
        if not rows or not expected:
            raise ValueError("uniform I/O balancing requires candidates and Segments")
        layer = rows[0].layer_1based
        allowed = tuple(self.repair_policy_profile.io_balance_ratio_candidates)
        by_ratio: Dict[float, JointRepairRatioCandidate] = {}
        aggregate_load = rows[0].aggregate_load_upper_ms
        nonoverlap = rows[0].nonoverlap_upper_ms
        for row in rows:
            if row.layer_1based != layer:
                raise ValueError("uniform I/O candidates must belong to one layer")
            if tuple(sorted(segment_id for segment_id, _ in row.ratios_by_segment)) != expected:
                raise ValueError("uniform I/O candidate Segment inventory differs")
            ratio = self._uniform_ratio(row)
            if all(abs(ratio - value) > 1e-12 for value in allowed):
                raise ValueError("uniform I/O candidate is outside the frozen ratio grid")
            if any(abs(ratio - value) <= 1e-12 for value in by_ratio):
                raise ValueError("uniform I/O profile repeats a ratio")
            if abs(row.aggregate_load_upper_ms - aggregate_load) > 1e-12:
                raise ValueError("aggregate load must not vary with repair ratio")
            if abs(row.nonoverlap_upper_ms - nonoverlap) > 1e-12:
                raise ValueError("non-overlap cost must not vary with repair ratio")
            by_ratio[ratio] = row

        if set(by_ratio) != set(allowed):
            raise ValueError("uniform I/O candidates must cover the frozen ratio grid")
        ordered_curve = sorted(
            (ratio, row.union_repair_upper_ms)
            for ratio, row in by_ratio.items()
        )
        if any(
            later_ms + 1e-12 < earlier_ms
            for (_, earlier_ms), (_, later_ms) in zip(
                ordered_curve, ordered_curve[1:]
            )
        ):
            raise ValueError("profiled union repair curve must be non-decreasing")

        floor = float(self.repair_policy_profile.certified_floor)
        reference = float(self.repair_policy_profile.quality_reference_ratio)
        hidden = [
            ratio
            for ratio, row in by_ratio.items()
            if row.union_repair_upper_ms <= aggregate_load + 1e-12
        ]
        io_ratio = max(hidden, default=0.0)
        target = max(floor, io_ratio)
        if previous_uniform_ratio is None:
            selected_ratio = max(reference, target)
        elif target >= float(previous_uniform_ratio) - 1e-12:
            selected_ratio = float(previous_uniform_ratio)
        else:
            # Gradual filtering takes one frozen-grid step toward the target
            # per layer.  It never jumps below the current I/O/quality target.
            selected_ratio = max(
                value
                for value in allowed
                if target - 1e-12 <= value < float(previous_uniform_ratio) - 1e-12
            )
        selected = next(
            (
                row for ratio, row in by_ratio.items()
                if abs(ratio - selected_ratio) <= 1e-12
            ),
            None,
        )
        if selected is None:
            raise ValueError("frozen I/O grid cannot represent the selected ratio")

        digest_payload = [
            {
                "candidate_id": row.candidate_id,
                "layer_1based": row.layer_1based,
                "ratios_by_segment": sorted(row.ratios_by_segment),
                "aggregate_load_upper_ms": row.aggregate_load_upper_ms,
                "union_repair_upper_ms": row.union_repair_upper_ms,
                "nonoverlap_upper_ms": row.nonoverlap_upper_ms,
            }
            for row in sorted(rows, key=lambda value: value.candidate_id)
        ]
        digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return UniformIOBalanceDecision(
            layer,
            expected,
            io_ratio,
            floor,
            reference,
            selected_ratio,
            selected.aggregate_load_upper_ms,
            selected.union_repair_upper_ms,
            selected.nonoverlap_upper_ms,
            digest,
            self.repair_policy_profile.profile_sha256,
            self.runtime_cost_profile.profile_sha256,
        )

    def build_plan(
        self,
        *,
        candidates_by_layer: Mapping[int, Sequence[JointRepairRatioCandidate]],
        first_selective_reuse_layer_by_segment: Mapping[str, int],
    ) -> "MultiSegmentRepairRatioPlan":
        first_layers = {
            str(segment_id): int(layer)
            for segment_id, layer in first_selective_reuse_layer_by_segment.items()
        }
        if not first_layers:
            raise ValueError("uniform I/O plan requires reusable Segments")
        if len(set(first_layers.values())) != 1:
            raise ValueError(
                "schema-v8 uniform I/O plan requires the common dense-barrier boundary"
            )
        if set(first_layers.values()) - {int(value) for value in candidates_by_layer}:
            raise ValueError("uniform I/O plan misses a Segment's first reuse layer")
        previous: Optional[float] = None
        decisions = []
        plan_rows = []
        for layer in sorted(int(value) for value in candidates_by_layer):
            active = tuple(
                sorted(
                    segment_id
                    for segment_id, first_layer in first_layers.items()
                    if layer >= first_layer
                )
            )
            if not active:
                raise ValueError("uniform I/O candidates precede all reuse boundaries")
            decision = self.choose_layer(
                candidates=candidates_by_layer[layer],
                expected_segment_ids=active,
                previous_uniform_ratio=previous,
            )
            previous = decision.selected_ratio
            plan_rows.extend(
                SegmentLayerRepairRatio(
                    segment_id,
                    layer,
                    first_layers[segment_id],
                    decision.selected_ratio,
                )
                for segment_id in active
            )
            decisions.append(decision)
        return validate_union_repair_ratio_plan(
            scope=RepairRatioScope.REQUEST_LAYER_UNIFORM_IO_BALANCED,
            rows=tuple(plan_rows),
            certified_floor=self.repair_policy_profile.certified_floor,
            profile_frozen=True,
            certified_ratio_candidates=(
                self.repair_policy_profile.io_balance_ratio_candidates
            ),
            uniform_io_decisions=tuple(decisions),
        )


@dataclass(frozen=True)
class MultiSegmentRepairRatioPlan:
    scope: RepairRatioScope
    rows: Tuple[SegmentLayerRepairRatio, ...]
    certified_floor: float
    profile_frozen: bool
    certified_ratio_candidates: Tuple[float, ...] = (0.10, 0.12, 0.15)
    adaptive_joint_decisions: Tuple[JointRepairRatioDecision, ...] = ()
    uniform_io_decisions: Tuple[UniformIOBalanceDecision, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", RepairRatioScope(self.scope))
        if not self.rows or not 0 < self.certified_floor <= 0.15:
            raise ValueError("repair ratio plan requires rows and a certified floor")
        keys = tuple((row.segment_id, row.layer_1based) for row in self.rows)
        if len(keys) != len(set(keys)):
            raise ValueError("a Segment may have one ratio per layer")
        candidates = tuple(sorted(set(float(value) for value in self.certified_ratio_candidates)))
        if not candidates or candidates[0] <= 0 or candidates[-1] > 1.0:
            raise ValueError("repair-ratio candidate set must remain in (0, 1]")
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
            if self.adaptive_joint_decisions or self.uniform_io_decisions:
                raise ValueError("fixed15 cannot carry adaptive joint decisions")
            if any(abs(row.ratio - 0.15) > 1e-12 for row in self.rows):
                raise ValueError("fixed15 requires 0.15 for every Segment and layer")
        elif self.scope is RepairRatioScope.DEVELOPMENT_PROFILE_MEASUREMENT:
            if self.profile_frozen:
                raise ValueError(
                    "development repair measurements cannot masquerade as a frozen Profile"
                )
            if self.adaptive_joint_decisions or self.uniform_io_decisions:
                raise ValueError(
                    "development repair measurements cannot carry runtime decisions"
                )
            if any(
                all(abs(row.ratio - candidate) > 1e-12 for candidate in candidates)
                for row in self.rows
            ):
                raise ValueError(
                    "development repair measurement is outside its declared ratio grid"
                )
        elif self.scope is RepairRatioScope.SHARED_RELATIVE_SCHEDULE:
            if self.adaptive_joint_decisions or self.uniform_io_decisions:
                raise ValueError("static gradual cannot carry adaptive joint decisions")
            if any(row.ratio > 0.15 + 1e-12 for row in self.rows):
                raise ValueError("legacy static gradual remains capped at 0.15")
            ratio_by_age: Dict[int, float] = {}
            for row in self.rows:
                previous = ratio_by_age.setdefault(row.repair_age, row.ratio)
                if abs(previous - row.ratio) > 1e-12:
                    raise ValueError(
                        "static gradual uses one ratio at each relative repair age"
                    )
        elif self.scope is RepairRatioScope.PER_SEGMENT_LOAD_AWARE:
            if self.uniform_io_decisions:
                raise ValueError("legacy per-Segment plan cannot carry uniform decisions")
            if not self.profile_frozen:
                raise ValueError("per-Segment adaptive ratios require a frozen Profile")
            if any(row.ratio > 0.15 + 1e-12 for row in self.rows):
                raise ValueError("legacy per-Segment adaptive ratio exceeds 0.15")
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
        else:
            if self.adaptive_joint_decisions:
                raise ValueError("uniform I/O plan cannot carry per-Segment decisions")
            if not self.profile_frozen:
                raise ValueError("uniform I/O ratios require a frozen Profile")
            decisions = {
                decision.layer_1based: decision
                for decision in self.uniform_io_decisions
            }
            layers = {row.layer_1based for row in self.rows}
            if len(decisions) != len(self.uniform_io_decisions) or set(decisions) != layers:
                raise ValueError("uniform I/O plan requires one decision per layer")
            for layer, decision in decisions.items():
                selected = {
                    row.segment_id: row.ratio
                    for row in self.rows
                    if row.layer_1based == layer
                }
                if selected != decision.ratio_map():
                    raise ValueError("uniform I/O rows differ from their decision")
                if len(set(selected.values())) != 1:
                    raise ValueError("active Segments must share one ratio per layer")

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
    uniform_io_decisions: Sequence[UniformIOBalanceDecision] = (),
) -> MultiSegmentRepairRatioPlan:
    return MultiSegmentRepairRatioPlan(
        scope,
        tuple(rows),
        certified_floor,
        profile_frozen,
        tuple(float(value) for value in certified_ratio_candidates),
        tuple(adaptive_joint_decisions),
        tuple(uniform_io_decisions),
    )
