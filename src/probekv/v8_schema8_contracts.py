from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Tuple


V8_SCHEMA8_PROTOCOL_VERSION = 8
V8_SCHEMA8_VERSION = 8
V8_SCHEMA8_PATCH_MODE = "probekv_v8_gradual_barrier_tiered_lru"


class RepairRatioScope(str, Enum):
    UNIFORM_FIXED = "uniform_fixed"
    SHARED_RELATIVE_SCHEDULE = "shared_relative_schedule"
    PER_SEGMENT_LOAD_AWARE = "per_segment_load_aware"
    REQUEST_LAYER_UNIFORM_IO_BALANCED = "request_layer_uniform_io_balanced"
    DEVELOPMENT_PROFILE_MEASUREMENT = "development_profile_measurement"


class BarrierResolution(str, Enum):
    SOURCE_FROZEN = "source_frozen"
    ABSTAINED_DENSE = "abstained_dense"


@dataclass(frozen=True)
class DenseSelectionBarrierDecision:
    resolved_completed_depth_by_segment: Mapping[str, int]
    resolution_by_segment: Mapping[str, BarrierResolution]
    barrier_completed_depth: int
    first_selective_reuse_layer: int
    d2_rescue_segment_ids: Tuple[str, ...]
    nonpaper_measurement_only: bool = False

    def __post_init__(self) -> None:
        if not self.resolved_completed_depth_by_segment:
            raise ValueError("selection barrier requires a Segment inventory")
        if set(self.resolution_by_segment) != set(
            self.resolved_completed_depth_by_segment
        ):
            raise ValueError("every Segment requires an explicit barrier resolution")
        object.__setattr__(
            self,
            "resolution_by_segment",
            {
                key: BarrierResolution(value)
                for key, value in self.resolution_by_segment.items()
            },
        )
        if self.nonpaper_measurement_only:
            if any(
                int(value) < 1
                for value in self.resolved_completed_depth_by_segment.values()
            ):
                raise ValueError("development measurement depth must be positive")
            if self.barrier_completed_depth < 1:
                raise ValueError("development measurement barrier depth must be positive")
        else:
            if any(
                value not in {1, 2}
                for value in self.resolved_completed_depth_by_segment.values()
            ):
                raise ValueError("schema-v8 selection must resolve at d=1 or d=2")
            if self.barrier_completed_depth not in {1, 2}:
                raise ValueError("schema-v8 barrier must close at d=1 or d=2")
        if self.first_selective_reuse_layer != self.barrier_completed_depth + 1:
            raise ValueError("selective reuse must begin after the dense barrier")
        if set(self.d2_rescue_segment_ids) - set(self.resolved_completed_depth_by_segment):
            raise ValueError("barrier rescue set is outside the inventory")

    @property
    def reuse_segment_ids(self) -> Tuple[str, ...]:
        return tuple(
            segment_id
            for segment_id in self.resolved_completed_depth_by_segment
            if self.resolution_by_segment[segment_id]
            is BarrierResolution.SOURCE_FROZEN
        )

    @property
    def dense_segment_ids(self) -> Tuple[str, ...]:
        return tuple(
            segment_id
            for segment_id in self.resolved_completed_depth_by_segment
            if self.resolution_by_segment[segment_id]
            is BarrierResolution.ABSTAINED_DENSE
        )


def schema8_no_gpu_gate(*, runtime_source_ready: bool = False) -> Mapping[str, object]:
    return {
        "protocol_version": 8,
        "schema_version": 8,
        "schema8_local_implementation_complete": True,
        "artifact_preparation_ready": bool(runtime_source_ready),
        "gpu_rental_ready_for_schema8_sentinel": bool(runtime_source_ready),
        "selector_depth_profile_frozen": False,
        "repair_policy_profile_frozen": False,
        "runtime_cost_profile_frozen": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "failures": ([] if runtime_source_ready else ["runtime_source_audit_required"]),
    }
