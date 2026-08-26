from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from .v6_contracts import RegionKind, RequestSpec


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


def repaired_segment_token_count(
    segment_tokens: int, ratio: float, rounding_policy: str = "floor"
) -> int:
    """Return repair count for C; v6=floor and conservative v7=ceil."""

    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be finite and in [0, 1]")
    if ratio == 0.0:
        return 0
    if ratio == 1.0:
        return segment_tokens
    if rounding_policy == "floor":
        return int(math.floor(segment_tokens * ratio))
    if rounding_policy == "ceil":
        return min(segment_tokens, int(math.ceil(segment_tokens * ratio)))
    raise ValueError("rounding_policy must be floor or ceil")


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


@dataclass(frozen=True)
class MultiSegmentRepairSelection:
    requested_ratios: Mapping[str, float]
    selected_indices_by_segment: Mapping[str, Tuple[int, ...]]
    dense_indices: Tuple[int, ...]
    execution_indices: Tuple[int, ...]
    union_mask_digest: str
    rounding_policy: str = "floor"

    def validate(self, request: RequestSpec) -> None:
        request.validate()
        segments = {segment.segment_id: segment for segment in request.segments}
        if set(self.requested_ratios) != set(self.selected_indices_by_segment):
            raise ValueError("ratio and repair-index segment sets disagree")
        if not set(self.requested_ratios).issubset(segments):
            raise ValueError("repair selection references an unknown segment")
        selected_union = set()
        for segment_id, ratio in self.requested_ratios.items():
            if not 0 <= ratio <= 1:
                raise ValueError("segment repair ratio must be in [0, 1]")
            segment = segments[segment_id]
            indices = self.selected_indices_by_segment[segment_id]
            if len(indices) != len(set(indices)):
                raise ValueError("segment repair indices must be unique")
            allowed = set(range(segment.token_start, segment.token_end))
            if not set(indices).issubset(allowed):
                raise ValueError("repair index lies outside its segment")
            expected = repaired_segment_token_count(
                segment.token_count, ratio, self.rounding_policy
            )
            if len(indices) != expected:
                raise ValueError("segment repair count does not match ratio")
            selected_union.update(indices)
        if len(self.dense_indices) != len(set(self.dense_indices)):
            raise ValueError("dense token indices must be unique")
        expected_dense = set()
        accepted = set(self.requested_ratios)
        for region in request.regions:
            if region.kind is RegionKind.PREFIX_EXACT:
                continue
            if (
                region.kind is RegionKind.REUSE_CANDIDATE
                and region.segment_id in accepted
            ):
                continue
            expected_dense.update(range(region.start, region.end))
        if set(self.dense_indices) != expected_dense:
            raise ValueError(
                "dense mask must cover every non-accepted region and suffix"
            )
        if selected_union & set(self.dense_indices):
            raise ValueError("repair and dense token indices must be disjoint")
        expected_execution = tuple(sorted(selected_union | set(self.dense_indices)))
        if self.execution_indices != expected_execution:
            raise ValueError("union execution mask does not match its regions")
        if any(index < request.exact_prefix_tokens for index in self.execution_indices):
            raise ValueError("exact prefix token entered the repair mask")
        payload = json.dumps(
            list(self.execution_indices), separators=(",", ":")
        ).encode("ascii")
        expected_digest = hashlib.sha256(payload).hexdigest()
        if self.union_mask_digest != expected_digest:
            raise ValueError("union repair mask digest mismatch")


def select_multisegment_repair_tokens(
    drift_scores: Sequence[float],
    request: RequestSpec,
    requested_ratios: Mapping[str, float],
    rounding_policy: str = "floor",
) -> MultiSegmentRepairSelection:
    """Build one absolute-position union mask for a v6 request.

    Only accepted segment IDs appear in ``requested_ratios``. Every other
    non-prefix region is dense. CacheBlend's stable largest-drift ordering is
    applied independently inside each accepted segment.
    """

    request.validate()
    if len(drift_scores) != len(request.token_ids):
        raise ValueError("drift scores must cover the complete request")
    segments = {segment.segment_id: segment for segment in request.segments}
    if not set(requested_ratios).issubset(segments):
        raise ValueError("ratio supplied for an unknown segment")
    selected: Dict[str, Tuple[int, ...]] = {}
    dense = set()
    for region in request.regions:
        if region.kind is RegionKind.PREFIX_EXACT:
            continue
        if (
            region.kind is RegionKind.REUSE_CANDIDATE
            and region.segment_id in requested_ratios
        ):
            segment_id = str(region.segment_id)
            ratio = float(requested_ratios[segment_id])
            count = repaired_segment_token_count(
                region.token_count, ratio, rounding_policy
            )
            ranked = sorted(
                range(region.start, region.end),
                key=lambda index: (-float(drift_scores[index]), index),
            )
            selected[segment_id] = tuple(sorted(ranked[:count]))
        else:
            dense.update(range(region.start, region.end))
    selected_union = set(
        index for indices in selected.values() for index in indices
    )
    execution = tuple(sorted(selected_union | dense))
    payload = json.dumps(list(execution), separators=(",", ":")).encode("ascii")
    result = MultiSegmentRepairSelection(
        requested_ratios=dict(requested_ratios),
        selected_indices_by_segment=selected,
        dense_indices=tuple(sorted(dense)),
        execution_indices=execution,
        union_mask_digest=hashlib.sha256(payload).hexdigest(),
        rounding_policy=rounding_policy,
    )
    result.validate(request)
    return result


def assert_nested_multisegment_selections(
    selections: Iterable[MultiSegmentRepairSelection],
    segment_id: str,
) -> None:
    relevant = [
        selection for selection in selections
        if segment_id in selection.requested_ratios
    ]
    relevant.sort(key=lambda item: item.requested_ratios[segment_id])
    previous = set()
    for selection in relevant:
        current = set(selection.selected_indices_by_segment[segment_id])
        if not previous.issubset(current):
            raise ValueError("multi-segment repair selections are not nested")
        previous = current


@dataclass(frozen=True)
class StaggeredMultiSegmentRepairPlan:
    """Layer-indexed execution masks for per-segment reuse boundaries."""

    requested_ratios: Mapping[str, float]
    boundary_by_segment: Mapping[str, int]
    selected_indices_by_segment_layer: Mapping[
        str, Mapping[int, Tuple[int, ...]]
    ]
    execution_indices_by_layer: Mapping[int, Tuple[int, ...]]
    union_mask_digest_by_layer: Mapping[int, str]
    total_layers: int
    rounding_policy: str = "floor"

    def validate(self, request: RequestSpec) -> None:
        request.validate()
        if self.total_layers < 1:
            raise ValueError("staggered repair requires positive total_layers")
        segments = {segment.segment_id: segment for segment in request.segments}
        accepted = set(self.requested_ratios)
        if accepted != set(self.boundary_by_segment):
            raise ValueError("ratio and boundary segment sets disagree")
        if accepted != set(self.selected_indices_by_segment_layer):
            raise ValueError("repair selections omit an accepted segment")
        if not accepted.issubset(segments):
            raise ValueError("staggered repair references an unknown segment")
        expected_layers = set(range(1, self.total_layers + 1))
        if set(self.execution_indices_by_layer) != expected_layers:
            raise ValueError("staggered execution masks must cover every layer")
        if set(self.union_mask_digest_by_layer) != expected_layers:
            raise ValueError("staggered mask digests must cover every layer")
        for segment_id in accepted:
            ratio = float(self.requested_ratios[segment_id])
            boundary = int(self.boundary_by_segment[segment_id])
            if not 0 <= ratio <= 1:
                raise ValueError("segment repair ratio must be in [0, 1]")
            if not 1 <= boundary <= self.total_layers:
                raise ValueError("segment boundary lies outside model layers")
            selected_by_layer = self.selected_indices_by_segment_layer[segment_id]
            expected_active_layers = set(range(boundary, self.total_layers + 1))
            if set(selected_by_layer) != expected_active_layers:
                raise ValueError("segment repair masks do not match its boundary")
            segment = segments[segment_id]
            allowed = set(range(segment.token_start, segment.token_end))
            expected_count = repaired_segment_token_count(
                segment.token_count, ratio, self.rounding_policy
            )
            for indices in selected_by_layer.values():
                if len(indices) != len(set(indices)):
                    raise ValueError("segment repair indices must be unique")
                if len(indices) != expected_count:
                    raise ValueError("segment repair count does not match ratio")
                if not set(indices).issubset(allowed):
                    raise ValueError("repair index lies outside its segment")

        for layer in range(1, self.total_layers + 1):
            expected = set()
            for region in request.regions:
                if region.kind is RegionKind.PREFIX_EXACT:
                    continue
                segment_id = str(region.segment_id) if region.segment_id else None
                if (
                    region.kind is RegionKind.REUSE_CANDIDATE
                    and segment_id in accepted
                    and layer >= self.boundary_by_segment[segment_id]
                ):
                    expected.update(
                        self.selected_indices_by_segment_layer[segment_id][layer]
                    )
                else:
                    expected.update(range(region.start, region.end))
            observed = self.execution_indices_by_layer[layer]
            if observed != tuple(sorted(expected)):
                raise ValueError("layer union mask does not match staggered regions")
            if any(index < request.exact_prefix_tokens for index in observed):
                raise ValueError("exact prefix token entered a staggered repair mask")
            payload = json.dumps(list(observed), separators=(",", ":")).encode(
                "ascii"
            )
            if self.union_mask_digest_by_layer[layer] != hashlib.sha256(
                payload
            ).hexdigest():
                raise ValueError("staggered union mask digest mismatch")


def select_staggered_multisegment_repair_tokens(
    drift_scores_by_layer: Mapping[int, Sequence[float]],
    request: RequestSpec,
    requested_ratios: Mapping[str, float],
    boundary_by_segment: Mapping[str, int],
    total_layers: int,
    rounding_policy: str = "floor",
) -> StaggeredMultiSegmentRepairPlan:
    """Build absolute-position union masks without a Segment-count ceiling."""

    request.validate()
    if total_layers < 1:
        raise ValueError("total_layers must be positive")
    expected_layers = set(range(1, total_layers + 1))
    if set(drift_scores_by_layer) != expected_layers:
        raise ValueError("drift scores must cover every model layer")
    if set(requested_ratios) != set(boundary_by_segment):
        raise ValueError("ratio and boundary segment sets disagree")
    segments = {segment.segment_id: segment for segment in request.segments}
    if not set(requested_ratios).issubset(segments):
        raise ValueError("ratio supplied for an unknown segment")
    selected: Dict[str, Dict[int, Tuple[int, ...]]] = {
        segment_id: {} for segment_id in requested_ratios
    }
    execution_by_layer: Dict[int, Tuple[int, ...]] = {}
    digests: Dict[int, str] = {}
    for layer in range(1, total_layers + 1):
        scores = tuple(float(value) for value in drift_scores_by_layer[layer])
        if len(scores) != len(request.token_ids):
            raise ValueError("drift scores must cover the complete request")
        execution = set()
        for region in request.regions:
            if region.kind is RegionKind.PREFIX_EXACT:
                continue
            segment_id = str(region.segment_id) if region.segment_id else None
            if (
                region.kind is RegionKind.REUSE_CANDIDATE
                and segment_id in requested_ratios
                and layer >= int(boundary_by_segment[segment_id])
            ):
                count = repaired_segment_token_count(
                    region.token_count,
                    float(requested_ratios[segment_id]),
                    rounding_policy,
                )
                ranked = sorted(
                    range(region.start, region.end),
                    key=lambda index: (-scores[index], index),
                )
                indices = tuple(sorted(ranked[:count]))
                selected[segment_id][layer] = indices
                execution.update(indices)
            else:
                execution.update(range(region.start, region.end))
        ordered = tuple(sorted(execution))
        execution_by_layer[layer] = ordered
        payload = json.dumps(list(ordered), separators=(",", ":")).encode(
            "ascii"
        )
        digests[layer] = hashlib.sha256(payload).hexdigest()
    result = StaggeredMultiSegmentRepairPlan(
        requested_ratios=dict(requested_ratios),
        boundary_by_segment={
            segment_id: int(layer)
            for segment_id, layer in boundary_by_segment.items()
        },
        selected_indices_by_segment_layer={
            segment_id: dict(by_layer) for segment_id, by_layer in selected.items()
        },
        execution_indices_by_layer=execution_by_layer,
        union_mask_digest_by_layer=digests,
        total_layers=total_layers,
        rounding_policy=rounding_policy,
    )
    result.validate(request)
    return result
