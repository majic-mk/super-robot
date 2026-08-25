from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Tuple

from .backend import RepairBackend, RepairResult
from .contracts import HistoricalSource, KVLocation
from .repair_semantics import repaired_segment_token_count
from .repair_semantics import (
    MultiSegmentRepairSelection,
    StaggeredMultiSegmentRepairPlan,
)
from .v6_contracts import RequestSpec


@dataclass(frozen=True)
class RuntimeRepairMeasurement:
    quality_score: float
    token_f1: float
    latency_ms: float
    source_digest_before: str
    source_digest_after: str
    requested_ratio: float = 0.0
    eligible_segment_tokens: int = 0
    selected_segment_tokens: int = 0
    effective_ratio: float = 0.0
    mandatory_suffix_tokens: int = 0
    reuse_start_layer: int = 1
    repair_gpu_ms: float = 0.0
    repair_host_ms: float = 0.0
    output_token_ids: Tuple[int, ...] = ()
    output_hash: str = ""
    output_text: str = ""


class CacheBlendRuntime(Protocol):
    """Narrow shim implemented inside the pinned CacheBlend fork.

    Keeping this protocol free of vLLM tensor types lets all orchestration and
    invariants be tested locally. Only the shim implementation is stack-specific.
    """

    def stage_canonical_source(
        self, source: HistoricalSource, target: KVLocation
    ) -> float:
        ...

    def selective_repair(
        self, source: HistoricalSource, start_layer: int, ratio: float
    ) -> RuntimeRepairMeasurement:
        ...

    def dense_remaining_ms(self, token_count: int, start_layer: int) -> float:
        ...

    def provenance(self) -> Mapping[str, Any]:
        ...


class CacheBlendBackend(RepairBackend):
    """Validated adapter around a pinned CacheBlend runtime shim."""

    def __init__(self, runtime: CacheBlendRuntime, total_layers: int) -> None:
        if total_layers <= 0:
            raise ValueError("total_layers must be positive")
        self.runtime = runtime
        self.total_layers = total_layers

    def prepare_source(self, source: HistoricalSource, target: KVLocation) -> float:
        source.validate_canonical()
        latency = float(self.runtime.stage_canonical_source(source, target))
        if latency < 0:
            raise ValueError("source preparation latency must be non-negative")
        return latency

    def repair(
        self, source: HistoricalSource, start_layer: int, ratio: float
    ) -> RepairResult:
        source.validate_canonical()
        if not 1 <= start_layer <= self.total_layers:
            raise ValueError("invalid start_layer")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be in [0, 1]")
        measurement = self.runtime.selective_repair(source, start_layer, ratio)
        if measurement.source_digest_before != measurement.source_digest_after:
            raise RuntimeError("CacheBlend runtime mutated a canonical source")
        if measurement.latency_ms < 0:
            raise ValueError("repair latency must be non-negative")
        if not 0.0 <= measurement.quality_score <= 1.0:
            raise ValueError("quality score must be in [0, 1]")
        if not 0.0 <= measurement.token_f1 <= 1.0:
            raise ValueError("token F1 must be in [0, 1]")
        eligible = measurement.eligible_segment_tokens or source.token_count
        selected = measurement.selected_segment_tokens
        expected_selected = repaired_segment_token_count(eligible, ratio)
        if selected != expected_selected:
            raise ValueError(
                "runtime selected %d C tokens, expected %d"
                % (selected, expected_selected)
            )
        effective = selected / float(eligible)
        if abs(measurement.effective_ratio - effective) > 1e-12:
            raise ValueError("runtime effective ratio does not match token counts")
        if abs(measurement.requested_ratio - ratio) > 1e-12:
            raise ValueError("runtime requested ratio does not match request")
        if measurement.reuse_start_layer != start_layer:
            raise ValueError("runtime reuse layer does not match request")
        timing_values = (
            measurement.repair_gpu_ms,
            measurement.repair_host_ms,
            measurement.latency_ms,
        )
        if any(value < 0 for value in timing_values):
            raise ValueError("repair timings must be non-negative")
        return RepairResult(
            measurement.quality_score,
            measurement.token_f1,
            measurement.latency_ms,
            effective,
            requested_ratio=ratio,
            eligible_segment_tokens=eligible,
            selected_segment_tokens=selected,
            effective_ratio=effective,
            mandatory_suffix_tokens=measurement.mandatory_suffix_tokens,
            reuse_start_layer=start_layer,
            repair_gpu_ms=measurement.repair_gpu_ms,
            repair_host_ms=measurement.repair_host_ms,
            source_digest_before=measurement.source_digest_before,
            source_digest_after=measurement.source_digest_after,
            output_token_ids=measurement.output_token_ids,
            output_hash=measurement.output_hash,
            output_text=measurement.output_text,
        )

    def full_remaining(self, token_count: int, start_layer: int) -> float:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        if not 1 <= start_layer <= self.total_layers:
            raise ValueError("invalid start_layer")
        latency = float(self.runtime.dense_remaining_ms(token_count, start_layer))
        if latency < 0:
            raise ValueError("dense latency must be non-negative")
        return latency

    def provenance(self) -> Mapping[str, Any]:
        record = dict(self.runtime.provenance())
        required = {
            "cacheblend_commit",
            "cacheblend_patch_sha256",
            "cacheblend_tree",
            "vllm",
            "torch",
            "cuda",
        }
        missing = sorted(required - set(record))
        if missing:
            raise ValueError("missing CacheBlend provenance: %s" % ", ".join(missing))
        return record


@dataclass(frozen=True)
class MultiSourceStageMeasurement:
    selected_source_ids: Mapping[str, str]
    load_start_ms_by_segment: Mapping[str, float]
    ready_ms_by_segment: Mapping[str, float]
    layer_ready_ms_by_segment: Mapping[str, Mapping[int, float]]
    transferred_bytes_by_segment: Mapping[str, int]
    digest_before_by_segment: Mapping[str, str]
    digest_after_by_segment: Mapping[str, str]


@dataclass(frozen=True)
class SegmentRuntimeRepairMeasurement:
    segment_id: str
    source_id: str
    requested_ratio: float
    eligible_segment_tokens: int
    selected_segment_indices: Tuple[int, ...]
    effective_ratio: float
    source_digest_before: str
    source_digest_after: str


@dataclass(frozen=True)
class MultiSegmentRepairMeasurement:
    request_id: str
    reuse_start_layer: int
    segments: Tuple[SegmentRuntimeRepairMeasurement, ...]
    union_execution_indices: Tuple[int, ...]
    union_mask_digest: str
    repair_gpu_ms: float
    repair_host_ms: float
    output_token_ids: Tuple[int, ...] = ()
    output_hash: str = ""
    output_text: str = ""
    teacher_forced_logit_relative_l2: float = 0.0
    teacher_forced_logit_positions: int = 0
    dense_reference_token_ids: Tuple[int, ...] = ()
    causal_mask_mode: str = "unspecified"
    rope_alignment_mode: str = "unspecified"


@dataclass(frozen=True)
class StaggeredMultiSegmentRepairMeasurement:
    request_id: str
    boundary_by_segment: Mapping[str, int]
    selected_indices_by_segment_layer: Mapping[
        str, Mapping[int, Tuple[int, ...]]
    ]
    execution_indices_by_layer: Mapping[int, Tuple[int, ...]]
    union_mask_digest_by_layer: Mapping[int, str]
    digest_before_by_segment: Mapping[str, str]
    digest_after_by_segment: Mapping[str, str]
    repair_gpu_ms: float
    repair_host_ms: float
    output_token_ids: Tuple[int, ...] = ()
    dense_reference_token_ids: Tuple[int, ...] = ()
    teacher_forced_logit_relative_l2: float = 0.0
    teacher_forced_logit_positions: int = 0
    causal_mask_mode: str = "unspecified"
    rope_alignment_mode: str = "unspecified"


class MultiSegmentCacheBlendRuntime(Protocol):
    """Pinned-stack interface for v6 multi-region execution."""

    def stage_canonical_sources(
        self,
        request: RequestSpec,
        sources_by_segment: Mapping[str, HistoricalSource],
        target: KVLocation,
    ) -> MultiSourceStageMeasurement:
        ...

    def execute_multisegment_prefill(
        self,
        request: RequestSpec,
        sources_by_segment: Mapping[str, HistoricalSource],
        start_layer: int,
        repair_selection: MultiSegmentRepairSelection,
    ) -> MultiSegmentRepairMeasurement:
        ...

    def execute_staggered_multisegment_prefill(
        self,
        request: RequestSpec,
        sources_by_segment: Mapping[str, HistoricalSource],
        repair_plan: StaggeredMultiSegmentRepairPlan,
    ) -> StaggeredMultiSegmentRepairMeasurement:
        ...

    def dense_remaining_profile(
        self, request: RequestSpec, start_layer: int
    ) -> float:
        ...

    def provenance(self) -> Mapping[str, Any]:
        ...


class MultiSegmentCacheBlendBackend:
    """Invariant-checking adapter for CacheBlend's v6 runtime shim."""

    def __init__(
        self, runtime: MultiSegmentCacheBlendRuntime, total_layers: int
    ) -> None:
        if total_layers <= 0:
            raise ValueError("total_layers must be positive")
        self.runtime = runtime
        self.total_layers = total_layers

    @staticmethod
    def _validate_sources(
        request: RequestSpec,
        sources_by_segment: Mapping[str, HistoricalSource],
    ) -> None:
        request.validate()
        segments = {segment.segment_id: segment for segment in request.segments}
        if not set(sources_by_segment).issubset(segments):
            raise ValueError("Source mapping contains an unknown segment")
        for segment_id, source in sources_by_segment.items():
            source.validate_canonical()
            segment = segments[segment_id]
            if source not in segment.sources:
                raise ValueError("Source is not a canonical variant of its segment")

    def prepare_sources(
        self,
        request: RequestSpec,
        sources_by_segment: Mapping[str, HistoricalSource],
        target: KVLocation = KVLocation.GPU,
    ) -> MultiSourceStageMeasurement:
        self._validate_sources(request, sources_by_segment)
        measurement = self.runtime.stage_canonical_sources(
            request, sources_by_segment, target
        )
        expected_ids = {
            segment_id: source.source_id
            for segment_id, source in sources_by_segment.items()
        }
        if dict(measurement.selected_source_ids) != expected_ids:
            raise ValueError("runtime staged a different Source set")
        required_segments = set(expected_ids)
        mappings = (
            measurement.load_start_ms_by_segment,
            measurement.ready_ms_by_segment,
            measurement.layer_ready_ms_by_segment,
            measurement.transferred_bytes_by_segment,
            measurement.digest_before_by_segment,
            measurement.digest_after_by_segment,
        )
        if any(set(mapping) != required_segments for mapping in mappings):
            raise ValueError("multi-source staging audit is incomplete")
        for segment_id in required_segments:
            if measurement.load_start_ms_by_segment[segment_id] < 0:
                raise ValueError("load start must be non-negative")
            if measurement.ready_ms_by_segment[segment_id] < (
                measurement.load_start_ms_by_segment[segment_id]
            ):
                raise ValueError("Source ready cannot precede load start")
            layer_ready = measurement.layer_ready_ms_by_segment[segment_id]
            if not layer_ready:
                raise ValueError("layer-wise Source readiness is required")
            layers = tuple(sorted(int(layer) for layer in layer_ready))
            if layers[0] < 1 or layers[-1] > self.total_layers:
                raise ValueError("layer-wise readiness uses invalid layers")
            times = tuple(float(layer_ready[layer]) for layer in layers)
            if any(
                value < measurement.load_start_ms_by_segment[segment_id]
                for value in times
            ) or tuple(sorted(times)) != times:
                raise ValueError("layer-wise Source readiness is not monotonic")
            if times[-1] > measurement.ready_ms_by_segment[segment_id] + 1e-9:
                raise ValueError("full Source ready time precedes a layer")
            if measurement.transferred_bytes_by_segment[segment_id] < 0:
                raise ValueError("transferred bytes must be non-negative")
            if measurement.digest_before_by_segment[segment_id] != (
                measurement.digest_after_by_segment[segment_id]
            ):
                raise RuntimeError("staging mutated a canonical Source")
        return measurement

    def repair_request(
        self,
        request: RequestSpec,
        sources_by_segment: Mapping[str, HistoricalSource],
        start_layer: int,
        repair_selection: MultiSegmentRepairSelection,
    ) -> MultiSegmentRepairMeasurement:
        self._validate_sources(request, sources_by_segment)
        repair_selection.validate(request)
        if not 1 <= start_layer <= self.total_layers:
            raise ValueError("invalid reuse start layer")
        if set(sources_by_segment) != set(repair_selection.requested_ratios):
            raise ValueError("every reused segment requires exactly one Source")
        result = self.runtime.execute_multisegment_prefill(
            request, sources_by_segment, start_layer, repair_selection
        )
        if result.request_id != request.request_id:
            raise ValueError("runtime returned another request")
        if result.reuse_start_layer != start_layer:
            raise ValueError("runtime changed the common reuse boundary")
        if result.union_execution_indices != repair_selection.execution_indices:
            raise ValueError("runtime changed the union repair mask")
        if result.union_mask_digest != repair_selection.union_mask_digest:
            raise ValueError("runtime union mask digest mismatch")
        if min(
            result.repair_gpu_ms,
            result.repair_host_ms,
            result.teacher_forced_logit_relative_l2,
        ) < 0:
            raise ValueError("runtime measurements must be non-negative")
        if result.teacher_forced_logit_positions < 0:
            raise ValueError("logit comparison count must be non-negative")
        if result.causal_mask_mode != "absolute_query_positions":
            raise ValueError("multi-region repair requires absolute causal rows")
        if result.rope_alignment_mode != "pre_rope_derotate_rerotate":
            raise ValueError("multi-region repair requires explicit RoPE alignment")
        measured = {item.segment_id: item for item in result.segments}
        if set(measured) != set(sources_by_segment):
            raise ValueError("runtime omitted a reused segment")
        segments = {segment.segment_id: segment for segment in request.segments}
        for segment_id, item in measured.items():
            source = sources_by_segment[segment_id]
            segment = segments[segment_id]
            if item.source_id != source.source_id:
                raise ValueError("runtime repaired a different Source")
            if item.source_digest_before != item.source_digest_after:
                raise RuntimeError("repair mutated a canonical Source")
            if item.requested_ratio != repair_selection.requested_ratios[segment_id]:
                raise ValueError("runtime changed a segment repair ratio")
            if item.eligible_segment_tokens != segment.token_count:
                raise ValueError("runtime used an incorrect ratio denominator")
            expected_indices = repair_selection.selected_indices_by_segment[
                segment_id
            ]
            if item.selected_segment_indices != expected_indices:
                raise ValueError("runtime changed selected repair tokens")
            expected_ratio = len(expected_indices) / float(segment.token_count)
            if abs(item.effective_ratio - expected_ratio) > 1e-12:
                raise ValueError("runtime effective ratio is inconsistent")
        all_candidate_ids = {segment.segment_id for segment in request.segments}
        all_r_one = (
            set(repair_selection.requested_ratios) == all_candidate_ids
            and all(
                abs(float(ratio) - 1.0) <= 1e-12
                for ratio in repair_selection.requested_ratios.values()
            )
        )
        if all_r_one:
            if not result.dense_reference_token_ids:
                raise ValueError("r=1 validation requires dense reference tokens")
            if result.output_token_ids != result.dense_reference_token_ids:
                raise RuntimeError("r=1 output differs from dense recomputation")
            if result.teacher_forced_logit_relative_l2 > 1e-4:
                raise RuntimeError("r=1 logit fidelity exceeds 1e-4")
            if result.teacher_forced_logit_positions < 32:
                raise ValueError("r=1 validation requires the first 32 logits")
        return result

    def dense_remaining_profile(
        self, request: RequestSpec, start_layer: int
    ) -> float:
        request.validate()
        if not 1 <= start_layer <= self.total_layers:
            raise ValueError("invalid dense profile start layer")
        latency = float(self.runtime.dense_remaining_profile(request, start_layer))
        if latency < 0:
            raise ValueError("dense profile latency must be non-negative")
        return latency

    def repair_request_staggered(
        self,
        request: RequestSpec,
        sources_by_segment: Mapping[str, HistoricalSource],
        repair_plan: StaggeredMultiSegmentRepairPlan,
    ) -> StaggeredMultiSegmentRepairMeasurement:
        """Validate the A/C layer-indexed data-plane contract.

        This adapter intentionally exposes no fixed Segment-count limit.  The
        pinned runtime must echo every per-layer absolute union mask and keep
        every canonical Source immutable.
        """

        self._validate_sources(request, sources_by_segment)
        repair_plan.validate(request)
        if repair_plan.total_layers != self.total_layers:
            raise ValueError("staggered repair plan uses another model depth")
        if set(sources_by_segment) != set(repair_plan.requested_ratios):
            raise ValueError("every staggered segment requires one locked Source")
        result = self.runtime.execute_staggered_multisegment_prefill(
            request, sources_by_segment, repair_plan
        )
        if result.request_id != request.request_id:
            raise ValueError("runtime returned another request")
        if dict(result.boundary_by_segment) != dict(
            repair_plan.boundary_by_segment
        ):
            raise ValueError("runtime changed staggered boundaries")
        if {
            segment_id: dict(by_layer)
            for segment_id, by_layer in result.selected_indices_by_segment_layer.items()
        } != {
            segment_id: dict(by_layer)
            for segment_id, by_layer in repair_plan.selected_indices_by_segment_layer.items()
        }:
            raise ValueError("runtime changed staggered repair tokens")
        if dict(result.execution_indices_by_layer) != dict(
            repair_plan.execution_indices_by_layer
        ):
            raise ValueError("runtime changed staggered union masks")
        if dict(result.union_mask_digest_by_layer) != dict(
            repair_plan.union_mask_digest_by_layer
        ):
            raise ValueError("runtime staggered union-mask digest mismatch")
        required = set(sources_by_segment)
        if set(result.digest_before_by_segment) != required or set(
            result.digest_after_by_segment
        ) != required:
            raise ValueError("staggered Source digest audit is incomplete")
        if any(
            result.digest_before_by_segment[segment_id]
            != result.digest_after_by_segment[segment_id]
            for segment_id in required
        ):
            raise RuntimeError("staggered repair mutated a canonical Source")
        if min(
            result.repair_gpu_ms,
            result.repair_host_ms,
            result.teacher_forced_logit_relative_l2,
        ) < 0:
            raise ValueError("staggered runtime measurements must be non-negative")
        if result.causal_mask_mode != "absolute_query_positions_per_layer":
            raise ValueError("staggered repair requires per-layer absolute causal rows")
        if result.rope_alignment_mode != "pre_rope_derotate_rerotate":
            raise ValueError("staggered repair requires explicit RoPE alignment")
        all_r_one = (
            required == {segment.segment_id for segment in request.segments}
            and all(
                abs(float(ratio) - 1.0) <= 1e-12
                for ratio in repair_plan.requested_ratios.values()
            )
        )
        if all_r_one:
            if not result.dense_reference_token_ids:
                raise ValueError("staggered r=1 requires dense reference tokens")
            if result.output_token_ids != result.dense_reference_token_ids:
                raise RuntimeError("staggered r=1 output differs from dense")
            if result.teacher_forced_logit_relative_l2 > 1e-4:
                raise RuntimeError("staggered r=1 logit fidelity exceeds 1e-4")
            if result.teacher_forced_logit_positions < 32:
                raise ValueError("staggered r=1 requires the first 32 logits")
        return result

    def provenance(self) -> Mapping[str, Any]:
        record = dict(self.runtime.provenance())
        required = {
            "cacheblend_commit",
            "cacheblend_patch_sha256",
            "cacheblend_tree",
            "patch_mode",
            "vllm",
            "torch",
            "cuda",
        }
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(
                "missing CacheBlend provenance: %s" % ", ".join(missing)
            )
        if record["patch_mode"] not in {
            "probekv_v6_multiregion",
            "probekv_v6_staggered_runtime",
        }:
            raise ValueError("v6 requires an explicit multi-region patch mode")
        return record
