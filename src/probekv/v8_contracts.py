from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from .contracts import KVLocation


V8_PROTOCOL_VERSION = 8
V8_LEGACY_SCHEMA_VERSION = 4
V8_SCHEMA_VERSION = 5


def stable_v8_digest(domain: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"domain": domain, **dict(payload)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class InsufficientRankingPolicy(str, Enum):
    ABSTAIN_DENSE = "abstain_dense"
    CFO_TOP1_FALLBACK = "cfo_top1_fallback"


class SelectionScope(str, Enum):
    FULL_CORRECTNESS_SET = "full_correctness_set"
    CFO_BUDGET_TRUNCATED = "cfo_budget_truncated"
    SINGLE_CORRECTNESS_ELIGIBLE = "single_correctness_eligible"
    INSUFFICIENT_RANKING_COVERAGE = "insufficient_ranking_coverage"


class ResidualLockReason(str, Enum):
    NONE = "none"
    SINGLE_CORRECTNESS_ELIGIBLE_SOURCE = "single_correctness_eligible_source"
    STABLE_MARGIN_EARLY_EXIT = "stable_margin_early_exit"
    STRONG_MARGIN_EARLY_EXIT = "strong_margin_early_exit"
    MAX_DEPTH_RESIDUAL_COST = "max_depth_residual_cost"
    CFO_TOP1_FALLBACK = "cfo_top1_fallback"


class ResidualSelectionState(str, Enum):
    PENDING = "pending"
    SELECTOR_DECISION_READY = "selector_decision_ready"
    SOURCE_FROZEN = "locked"
    LOCKED = "locked"
    ABSTAINED = "abstained"


class AbstainReason(str, Enum):
    NONE = "none"
    NO_CORRECTNESS_ELIGIBLE_SOURCE = "no_correctness_eligible_source"
    SELECTION_STATE_UNAVAILABLE = "selection_state_unavailable"
    COMPARISON_BUDGET_EXHAUSTED = "comparison_budget_exhausted"
    INSUFFICIENT_RANKING_COVERAGE = "insufficient_ranking_coverage"
    PRELIMINARY_ECONOMIC_REJECTION = "preliminary_economic_rejection"
    MAX_DEPTH_NO_ECONOMIC_CANDIDATE = "max_depth_no_economic_candidate"


@dataclass(frozen=True)
class SourceSelectionState:
    selection_state_id: str
    source_variant_id: str
    completed_depth: int
    k_observation_layer_1based: int
    token_count: int
    num_kv_heads: int
    head_dim: int
    parent_source_state_digest: str
    logical_digest: str
    dtype: str = "bfloat16"
    k_semantics: str = "pre_rope"

    def __post_init__(self) -> None:
        if not all(
            (
                self.selection_state_id,
                self.source_variant_id,
                self.parent_source_state_digest,
                self.logical_digest,
            )
        ):
            raise ValueError("complete SourceSelectionState identity is required")
        if self.completed_depth < 0:
            raise ValueError("completed_depth must be non-negative")
        if self.k_observation_layer_1based != self.completed_depth + 1:
            raise ValueError("K observation layer must be completed_depth + 1")
        if min(self.token_count, self.num_kv_heads, self.head_dim) <= 0:
            raise ValueError("selection-state geometry must be positive")
        if self.dtype != "bfloat16" or self.k_semantics != "pre_rope":
            raise ValueError("v8 SelectionState must be exact BF16 pre-RoPE K")


@dataclass(frozen=True)
class SelectionStateReplica:
    state_replica_id: str
    selection_state_id: str
    tier: KVLocation
    generation: int
    placement_epoch: int
    locator: str
    layout_signature: str
    logical_digest: str
    bytes_digest: str
    size_bytes: int
    is_backing: bool = False
    healthy: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", KVLocation(self.tier))
        if not all(
            (
                self.state_replica_id,
                self.selection_state_id,
                self.locator,
                self.layout_signature,
                self.logical_digest,
                self.bytes_digest,
            )
        ):
            raise ValueError("complete SelectionStateReplica identity is required")
        if min(self.generation, self.placement_epoch) < 1 or self.size_bytes < 0:
            raise ValueError("invalid SelectionStateReplica generation or size")
        if self.is_backing and self.tier is KVLocation.GPU:
            raise ValueError("GPU comparison scratch cannot be a persistent backing")


@dataclass(frozen=True)
class SelectorPolicyProfile:
    profile_id: str
    model_math_signature: str
    selection_execution_policy: str
    checkpoint_depths: Tuple[int, ...]
    max_completed_depth: int
    eta: float
    eta_strong: float
    residual_band_relative_tolerance: float
    residual_band_numeric_slack: float = 1e-6
    fixed_repair_ratio: float = 0.15
    profile_sha256: str = ""
    frozen: bool = False
    profile_freeze_contract_sha256: str = ""
    profile_freeze_runtime_cost_profile_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.profile_id or not self.model_math_signature:
            raise ValueError("profile identity is required")
        if not self.checkpoint_depths:
            raise ValueError("at least one online checkpoint is required")
        if tuple(sorted(set(self.checkpoint_depths))) != self.checkpoint_depths:
            raise ValueError("checkpoint depths must be sorted and unique")
        if self.checkpoint_depths[0] < 1:
            raise ValueError("d=0 is a negative control, not an online checkpoint")
        if self.checkpoint_depths[-1] != self.max_completed_depth:
            raise ValueError("the maximum depth must be an explicit checkpoint")
        if not 0 <= self.eta <= self.eta_strong <= 1:
            raise ValueError("invalid early-exit margins")
        if not 0 <= self.residual_band_relative_tolerance <= 1:
            raise ValueError("invalid residual band")
        if self.residual_band_numeric_slack != 1e-6:
            raise ValueError("v8 freezes numeric residual slack at 1e-6")
        if self.fixed_repair_ratio != 0.15:
            raise ValueError("v8 freezes the online repair ratio at 0.15")
        if self.frozen and not self.profile_sha256:
            raise ValueError("a frozen profile must carry its SHA256")
        if self.frozen and bool(self.profile_freeze_contract_sha256) != bool(
            self.profile_freeze_runtime_cost_profile_sha256
        ):
            raise ValueError("schema-v5 frozen Profile bindings must be complete")


@dataclass(frozen=True)
class RuntimeCostProfile:
    runtime_cost_profile_id: str
    model_math_signature: str
    selection_execution_policy: str
    gpu_uuid: str
    hardware_compatibility_signature: str
    comparison_batch_upper_ms: Mapping[int, float]
    profile_sha256: str
    code_commit: str
    cacheblend_patch_sha256: str
    schema_version: int = V8_SCHEMA_VERSION
    protocol_version: int = V8_PROTOCOL_VERSION
    cuda_event_timing: bool = True
    fake_timing: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != V8_SCHEMA_VERSION or self.protocol_version != 8:
            raise ValueError("RuntimeCostProfile must use v8 schema-v5")
        if not all(
            (
                self.runtime_cost_profile_id,
                self.model_math_signature,
                self.selection_execution_policy,
                self.gpu_uuid,
                self.hardware_compatibility_signature,
                self.profile_sha256,
                self.code_commit,
                self.cacheblend_patch_sha256,
            )
        ):
            raise ValueError("RuntimeCostProfile identity is incomplete")
        if self.selection_execution_policy not in {
            "causal_commit_wait",
            "immediate_staggered_closed_loop",
        }:
            raise ValueError("unsupported RuntimeCostProfile policy")
        if tuple(sorted(self.comparison_batch_upper_ms)) != tuple(
            self.comparison_batch_upper_ms
        ):
            raise ValueError("comparison batch curve must be ordered")
        if any(key not in {1, 2, 4, 8, 16} for key in self.comparison_batch_upper_ms):
            raise ValueError("comparison batch curve used an unsupported K")
        if any(float(value) < 0 for value in self.comparison_batch_upper_ms.values()):
            raise ValueError("RuntimeCostProfile timings must be non-negative")
        if not self.cuda_event_timing or self.fake_timing:
            raise ValueError("RuntimeCostProfile requires real CUDA timing")


@dataclass(frozen=True)
class CandidateCounts:
    stored_k: int
    correctness_eligible_k: int
    selection_state_available_k: int
    metadata_ranked_k: int
    compared_k: int

    def __post_init__(self) -> None:
        values = (
            self.stored_k,
            self.correctness_eligible_k,
            self.selection_state_available_k,
            self.metadata_ranked_k,
            self.compared_k,
        )
        if min(values) < 0:
            raise ValueError("candidate counts must be non-negative")
        if not (
            self.stored_k
            >= self.correctness_eligible_k
            >= self.selection_state_available_k
            >= self.metadata_ranked_k
            >= self.compared_k
        ):
            raise ValueError("candidate counts must form a non-increasing chain")

    @property
    def uncompared_correctness_eligible_k(self) -> int:
        return self.correctness_eligible_k - self.compared_k


@dataclass(frozen=True)
class ResidualCandidate:
    source_variant_id: str
    residual_score: float
    predicted_future_upper_ms: float
    metadata_rank: int

    def __post_init__(self) -> None:
        if not self.source_variant_id:
            raise ValueError("Source Variant identity is required")
        if min(self.residual_score, self.predicted_future_upper_ms) < 0:
            raise ValueError("residual and predicted costs must be non-negative")
        if self.metadata_rank < 0:
            raise ValueError("metadata rank must be non-negative")


@dataclass(frozen=True)
class ResidualSelectionDecision:
    state: ResidualSelectionState
    completed_depth: int
    counts: CandidateCounts
    selected_source_variant_id: Optional[str] = None
    lock_reason: ResidualLockReason = ResidualLockReason.NONE
    abstain_reason: AbstainReason = AbstainReason.NONE
    selection_scope: SelectionScope = SelectionScope.FULL_CORRECTNESS_SET
    best_source_variant_id: Optional[str] = None
    runner_up_source_variant_id: Optional[str] = None
    margin_defined: bool = False
    margin_value: Optional[float] = None
    current_state_ranking_performed: bool = False
    predicted_total_upper_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.completed_depth < 0:
            raise ValueError("completed depth must be non-negative")
        if self.margin_defined != (self.margin_value is not None):
            raise ValueError("undefined margin must be represented by null")
        if self.state in {
            ResidualSelectionState.SELECTOR_DECISION_READY,
            ResidualSelectionState.SOURCE_FROZEN,
        }:
            if not self.selected_source_variant_id:
                raise ValueError("selector decision requires a Source Variant")
            if self.lock_reason is ResidualLockReason.NONE:
                raise ValueError("selector decision requires a reason")
            if self.abstain_reason is not AbstainReason.NONE:
                raise ValueError("selector decision cannot have an abstain reason")
        elif self.selected_source_variant_id is not None:
            raise ValueError("only locked selection may expose a selected Source")
        if self.state is ResidualSelectionState.ABSTAINED and self.abstain_reason is AbstainReason.NONE:
            raise ValueError("abstention requires an explicit reason")

    @property
    def source_frozen(self) -> bool:
        return self.state is ResidualSelectionState.SOURCE_FROZEN

    @property
    def gate1_passed(self) -> bool:
        return self.source_frozen

    @property
    def provisional_source_variant_id(self) -> Optional[str]:
        if self.state in {
            ResidualSelectionState.SELECTOR_DECISION_READY,
            ResidualSelectionState.SOURCE_FROZEN,
        }:
            return self.selected_source_variant_id
        return self.best_source_variant_id
