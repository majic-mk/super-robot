from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


V8_SCHEMA7_PROTOCOL_VERSION = 8
V8_SCHEMA7_VERSION = 7
V8_SCHEMA7_PATCH_MODE = "probekv_v8_winner_gradual_streaming"


def stable_schema7_digest(domain: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"domain": domain, **dict(payload)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SourceSelectionDepthPolicy(str, Enum):
    D1_ONLY = "d1_only"
    D1_D2_RESCUE = "d1_d2_rescue"
    LEGACY_MULTICHECKPOINT = "legacy_multicheckpoint"
    DEEP_FULL_CANDIDATE_ORACLE = "deep_full_candidate_oracle"


class RepairMetric(str, Enum):
    WINNER_K_ONLY = "winner_k_only"
    WINNER_V_ONLY = "winner_v_only"
    WINNER_KV_NORMALIZED = "winner_kv_normalized"


class RepairPolicy(str, Enum):
    FIXED_15 = "fixed_15"
    STATIC_GRADUAL = "static_gradual"
    LOAD_RECOMPUTE_AWARE_GRADUAL = "load_recompute_aware_gradual"


class TransferPath(str, Enum):
    GPU_RESIDENT = "gpu_resident"
    CPU_PINNED_TO_GPU = "cpu_pinned_to_gpu"
    SSD_STAGED_TO_GPU = "ssd_staged_to_gpu"
    SSD_GDS_TO_GPU = "ssd_gds_to_gpu"


class IntegrityVerificationMode(str, Enum):
    QUALIFICATION_FULL = "qualification_full"
    ONLINE_IMMUTABLE = "online_immutable"
    ONLINE_SAMPLED = "online_sampled"


@dataclass(frozen=True)
class Schema7ProfileProvenance:
    profile_id: str
    profile_kind: str
    code_commit: str
    cacheblend_patch_sha256: str
    model_id: str
    model_revision: str
    tokenizer_hash: str
    gpu_uuid: str = ""
    measurement_sha256: str = ""
    protocol_version: int = V8_SCHEMA7_PROTOCOL_VERSION
    schema_version: int = V8_SCHEMA7_VERSION
    frozen: bool = False

    def __post_init__(self) -> None:
        if (self.protocol_version, self.schema_version) != (8, 7):
            raise ValueError("schema-v7 Profile provenance version mismatch")
        if self.profile_kind not in {
            "selection_depth", "repair_policy", "runtime_cost"
        }:
            raise ValueError("unknown schema-v7 Profile kind")
        if not all(
            (
                self.profile_id,
                self.code_commit,
                self.cacheblend_patch_sha256,
                self.model_id,
                self.model_revision,
                self.tokenizer_hash,
            )
        ):
            raise ValueError("schema-v7 Profile provenance is incomplete")
        if self.frozen and not all((self.gpu_uuid, self.measurement_sha256)):
            raise ValueError("a frozen schema-v7 Profile requires GPU measurements")


@dataclass(frozen=True)
class RepairSupportState:
    segment_id: str
    source_variant_id: str
    metric: RepairMetric
    producer_completed_depth: int
    consumer_layer_1based: int
    candidate_absolute_positions: Tuple[int, ...]
    segment_token_count: int
    initial_cap: float
    repair_floor: float
    parent_support_digest: str = ""
    support_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", RepairMetric(self.metric))
        positions = tuple(int(value) for value in self.candidate_absolute_positions)
        object.__setattr__(self, "candidate_absolute_positions", positions)
        if not self.segment_id or not self.source_variant_id:
            raise ValueError("repair support requires Segment and frozen Source")
        if self.producer_completed_depth < 1:
            raise ValueError("repair support requires a completed dense check layer")
        if self.consumer_layer_1based != self.producer_completed_depth + 1:
            raise ValueError("repair support must be consumed by the next layer")
        if self.segment_token_count <= 0:
            raise ValueError("repair support Segment size must be positive")
        if tuple(sorted(set(positions))) != positions:
            raise ValueError("repair support positions must be sorted and unique")
        if len(positions) > math.ceil(self.initial_cap * self.segment_token_count):
            raise ValueError("repair support exceeds the initial cap")
        if not 0 < self.repair_floor <= self.initial_cap <= 1:
            raise ValueError("invalid repair cap/floor")
        minimum = math.ceil(self.repair_floor * self.segment_token_count)
        if len(positions) < minimum:
            raise ValueError("repair support fell below the configured floor")
        payload = {
            "segment_id": self.segment_id,
            "source_variant_id": self.source_variant_id,
            "metric": self.metric.value,
            "producer_completed_depth": self.producer_completed_depth,
            "consumer_layer_1based": self.consumer_layer_1based,
            "positions": list(positions),
            "segment_token_count": self.segment_token_count,
            "initial_cap": self.initial_cap,
            "repair_floor": self.repair_floor,
            "parent_support_digest": self.parent_support_digest,
        }
        expected = stable_schema7_digest("repair-support", payload)
        if self.support_digest and self.support_digest != expected:
            raise ValueError("repair support digest mismatch")
        object.__setattr__(self, "support_digest", expected)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_absolute_positions)

    @property
    def effective_ratio(self) -> float:
        return self.candidate_count / float(self.segment_token_count)

    def assert_monotonic_child(self, child: "RepairSupportState") -> None:
        if (
            child.segment_id != self.segment_id
            or child.source_variant_id != self.source_variant_id
            or child.metric is not self.metric
        ):
            raise RuntimeError("repair support child changed logical identity")
        if child.parent_support_digest != self.support_digest:
            raise RuntimeError("repair support parent digest is stale")
        if child.consumer_layer_1based != self.consumer_layer_1based + 1:
            raise RuntimeError("repair support child skipped a Transformer layer")
        if not set(child.candidate_absolute_positions).issubset(
            self.candidate_absolute_positions
        ):
            raise RuntimeError("gradual repair cannot reintroduce a removed token")


@dataclass(frozen=True)
class RepairCheckBoundary:
    source_selection_completed_depth: int
    repair_check_completed_depth: int
    first_selective_reuse_layer: int

    def __post_init__(self) -> None:
        if self.source_selection_completed_depth < 1:
            raise ValueError("online Source selection cannot lock at d=0")
        if self.repair_check_completed_depth < self.source_selection_completed_depth:
            raise ValueError("repair check cannot precede Source selection")
        if self.first_selective_reuse_layer != self.repair_check_completed_depth + 1:
            raise ValueError("first reuse layer must follow the dense repair check")


@dataclass(frozen=True)
class FinalCommitDecision:
    accepted_ready_segment_ids: Tuple[str, ...]
    rejected_ready_segment_ids: Tuple[str, ...]
    untouched_segment_ids: Tuple[str, ...]
    request_total_ms: float
    dense_reference_total_ms: float
    planner_snapshot: Any
    reason_by_segment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        accepted = set(self.accepted_ready_segment_ids)
        rejected = set(self.rejected_ready_segment_ids)
        untouched = set(self.untouched_segment_ids)
        if accepted & rejected or accepted & untouched or rejected & untouched:
            raise ValueError("FinalCommitAdmission sets must be disjoint")
        if min(self.request_total_ms, self.dense_reference_total_ms) < 0:
            raise ValueError("FinalCommitAdmission costs must be non-negative")
        if set(self.reason_by_segment) - (accepted | rejected | untouched):
            raise ValueError("FinalCommitAdmission reason references an unknown Segment")


@dataclass(frozen=True)
class IntegrityAudit:
    mode: IntegrityVerificationMode
    expected_artifact_digest: str
    source_digest_before: str
    source_digest_after: str
    destination_replica_digest: str
    per_request_full_digest_verified: bool
    digest_bytes: int = 0
    source_before_host_ms: float = 0.0
    source_after_host_ms: float = 0.0
    destination_host_ms: float = 0.0
    sampled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", IntegrityVerificationMode(self.mode))
        if not self.expected_artifact_digest:
            raise ValueError("integrity audit requires the Artifact digest")
        if min(
            self.digest_bytes,
            self.source_before_host_ms,
            self.source_after_host_ms,
            self.destination_host_ms,
        ) < 0:
            raise ValueError("integrity audit measurements must be non-negative")
        if self.mode is IntegrityVerificationMode.QUALIFICATION_FULL:
            if not self.per_request_full_digest_verified:
                raise ValueError("qualification must perform full digest verification")
            if not (
                self.source_digest_before
                == self.source_digest_after
                == self.destination_replica_digest
                == self.expected_artifact_digest
            ):
                raise RuntimeError("qualification detected Source/Replica corruption")
        elif self.mode is IntegrityVerificationMode.ONLINE_IMMUTABLE:
            if self.per_request_full_digest_verified:
                raise ValueError("online immutable mode must not claim a full digest")
            if any(
                (
                    self.source_digest_before,
                    self.source_digest_after,
                    self.destination_replica_digest,
                )
            ):
                raise ValueError("online immutable mode cannot expose fabricated digests")


def schema7_no_gpu_gate() -> Mapping[str, Any]:
    return {
        "protocol_version": 8,
        "schema_version": 7,
        "schema7_local_implementation_complete": True,
        "artifact_preparation_ready": True,
        "gpu_rental_ready_for_schema7_sentinel": True,
        "selector_depth_profile_frozen": False,
        "repair_policy_profile_frozen": False,
        "runtime_cost_profile_frozen": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "failures": [],
    }
