from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from .v8_schema9_contracts import AbsoluteResidualThreshold, DenseKVProvenance


V8_SCHEMA10_PROTOCOL_VERSION = 8
V8_SCHEMA10_VERSION = 10
V8_SCHEMA10_PATCH_MODE = "probekv_v8_variant_growth_counterfactual"


class VariantMaterializationReasonV10(str, Enum):
    CONTENT_MISS = "content_miss"
    COMPLETE_SCOPE_ABSOLUTE_MISMATCH = "complete_scope_absolute_mismatch"
    BUDGET_TRUNCATED_EXPLORATION = "budget_truncated_exploration"
    ECONOMIC_REJECTION = "economic_rejection"
    RUNTIME_REJECTION = "runtime_rejection"


class VariantMaterializationStateV10(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"


class Gate1Mode(str, Enum):
    EXPLICIT_BARRIER = "explicit_barrier"
    FUSED_ADVISORY = "fused_advisory"


@dataclass(frozen=True)
class VariantMaterializationDecisionV10:
    state: VariantMaterializationStateV10
    reason: VariantMaterializationReasonV10
    selection_scope_complete: bool
    context_novelty_proven: bool
    best_residual: Optional[float]
    absolute_threshold: Optional[float]
    dense_kv_provenance: DenseKVProvenance
    existing_variant_count: int
    materialization_budget_ms: float
    estimated_materialization_ms: float
    replacement_source_variant_id: Optional[str] = None
    replacement_budget_ms: float = 0.0
    estimated_replacement_ms: float = 0.0
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", VariantMaterializationStateV10(self.state))
        object.__setattr__(self, "reason", VariantMaterializationReasonV10(self.reason))
        object.__setattr__(
            self, "dense_kv_provenance", DenseKVProvenance(self.dense_kv_provenance)
        )
        if self.existing_variant_count < 0:
            raise ValueError("existing Variant count must be non-negative")
        if min(
            self.materialization_budget_ms,
            self.estimated_materialization_ms,
            self.replacement_budget_ms,
            self.estimated_replacement_ms,
        ) < 0:
            raise ValueError("materialization costs must be non-negative")
        if self.context_novelty_proven and self.reason not in {
            VariantMaterializationReasonV10.CONTENT_MISS,
            VariantMaterializationReasonV10.COMPLETE_SCOPE_ABSOLUTE_MISMATCH,
        }:
            raise ValueError("exploration may not claim context novelty")
        if self.state is VariantMaterializationStateV10.ADMITTED:
            if self.dense_kv_provenance is not DenseKVProvenance.DENSE_EXACT:
                raise ValueError("only exact dense KV may be materialized")
            if self.rejection_reason:
                raise ValueError("admitted materialization cannot have a rejection")
        elif not self.rejection_reason:
            raise ValueError("rejected materialization requires an audit reason")


def schema10_no_gpu_gate(*, artifact_preparation_ready: bool) -> Mapping[str, object]:
    return {
        "protocol_version": V8_SCHEMA10_PROTOCOL_VERSION,
        "schema_version": V8_SCHEMA10_VERSION,
        "schema10_local_implementation_complete": True,
        "artifact_preparation_ready": bool(artifact_preparation_ready),
        "gpu_rental_ready_for_schema10_profile_freeze": bool(
            artifact_preparation_ready
        ),
        "variant_admission_profile_frozen": False,
        "preparation_policy_profile_frozen": False,
        "selection_depth_profile_frozen": False,
        "repair_policy_profile_frozen": False,
        "runtime_cost_profile_frozen": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "full_h1_started": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "failures": []
        if artifact_preparation_ready
        else ["schema10_artifact_preparation_required"],
    }


__all__ = [
    "AbsoluteResidualThreshold",
    "DenseKVProvenance",
    "Gate1Mode",
    "V8_SCHEMA10_PATCH_MODE",
    "VariantMaterializationDecisionV10",
    "VariantMaterializationReasonV10",
    "VariantMaterializationStateV10",
    "schema10_no_gpu_gate",
]
