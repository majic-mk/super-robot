from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple


V8_SCHEMA9_PROTOCOL_VERSION = 8
V8_SCHEMA9_VERSION = 9
V8_SCHEMA9_PATCH_MODE = "probekv_v8_absolute_residual_variant_admission"


class DenseKVProvenance(str, Enum):
    """How the working KV that might become canonical was produced."""

    DENSE_EXACT = "dense_exact"
    SELECTIVE_REPAIR = "selective_repair"
    R1_REPAIR_EQUIVALENT = "r1_repair_equivalent"
    UNKNOWN = "unknown"


class VariantMaterializationReason(str, Enum):
    CONTENT_MISS = "content_miss"
    ABSOLUTE_RESIDUAL_MISMATCH = "absolute_residual_mismatch"
    EXPLICIT_EXPLORATION = "explicit_exploration"
    ECONOMIC_REJECTION = "economic_rejection"
    RUNTIME_REJECTION = "runtime_rejection"
    BUDGET_TRUNCATED = "budget_truncated"


class VariantMaterializationState(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class AbsoluteResidualThreshold:
    completed_depth: int
    upper_residual: float

    def __post_init__(self) -> None:
        if self.completed_depth not in {1, 2}:
            raise ValueError("schema9 online threshold must be for d1/d2")
        if self.upper_residual < 0:
            raise ValueError("absolute residual threshold must be non-negative")


@dataclass(frozen=True)
class VariantMaterializationDecision:
    state: VariantMaterializationState
    reason: VariantMaterializationReason
    selection_scope_complete: bool
    best_residual: Optional[float]
    absolute_threshold: Optional[float]
    dense_kv_provenance: DenseKVProvenance
    existing_variant_count: int
    materialization_budget_ms: float
    estimated_materialization_ms: float
    replacement_source_variant_id: Optional[str] = None
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", VariantMaterializationState(self.state))
        object.__setattr__(self, "reason", VariantMaterializationReason(self.reason))
        object.__setattr__(
            self, "dense_kv_provenance", DenseKVProvenance(self.dense_kv_provenance)
        )
        if self.existing_variant_count < 0:
            raise ValueError("existing Variant count must be non-negative")
        if min(self.materialization_budget_ms, self.estimated_materialization_ms) < 0:
            raise ValueError("materialization costs must be non-negative")
        if self.best_residual is not None and self.best_residual < 0:
            raise ValueError("best residual must be non-negative")
        if self.absolute_threshold is not None and self.absolute_threshold < 0:
            raise ValueError("absolute threshold must be non-negative")
        if self.state is VariantMaterializationState.ADMITTED:
            if self.dense_kv_provenance is not DenseKVProvenance.DENSE_EXACT:
                raise ValueError("only exact dense KV may be materialized")
            if self.rejection_reason:
                raise ValueError("admitted materialization cannot have a rejection")
        elif not self.rejection_reason:
            raise ValueError("rejected materialization requires an audit reason")


def schema9_no_gpu_gate(*, artifact_preparation_ready: bool) -> Mapping[str, object]:
    return {
        "protocol_version": V8_SCHEMA9_PROTOCOL_VERSION,
        "schema_version": V8_SCHEMA9_VERSION,
        "schema9_local_implementation_complete": True,
        "artifact_preparation_ready": bool(artifact_preparation_ready),
        "gpu_rental_ready_for_schema9_profile_freeze": bool(
            artifact_preparation_ready
        ),
        "variant_admission_profile_frozen": False,
        "selection_depth_profile_frozen": False,
        "repair_policy_profile_frozen": False,
        "runtime_cost_profile_frozen": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "failures": [] if artifact_preparation_ready else [
            "schema9_artifact_preparation_required"
        ],
    }
