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


@dataclass(frozen=True)
class DenseSelectionBarrierDecision:
    resolved_completed_depth_by_segment: Mapping[str, int]
    barrier_completed_depth: int
    first_selective_reuse_layer: int
    d2_rescue_segment_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.resolved_completed_depth_by_segment:
            raise ValueError("selection barrier requires a Segment inventory")
        if any(value not in {1, 2} for value in self.resolved_completed_depth_by_segment.values()):
            raise ValueError("schema-v8 selection must resolve at d=1 or d=2")
        if self.barrier_completed_depth not in {1, 2}:
            raise ValueError("schema-v8 barrier must close at d=1 or d=2")
        if self.first_selective_reuse_layer != self.barrier_completed_depth + 1:
            raise ValueError("selective reuse must begin after the dense barrier")
        if set(self.d2_rescue_segment_ids) - set(self.resolved_completed_depth_by_segment):
            raise ValueError("barrier rescue set is outside the inventory")


def schema8_no_gpu_gate() -> Mapping[str, object]:
    return {
        "protocol_version": 8,
        "schema_version": 8,
        "schema8_local_implementation_complete": True,
        "artifact_preparation_ready": True,
        "gpu_rental_ready_for_schema8_sentinel": True,
        "selector_depth_profile_frozen": False,
        "repair_policy_profile_frozen": False,
        "runtime_cost_profile_frozen": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "failures": [],
    }
