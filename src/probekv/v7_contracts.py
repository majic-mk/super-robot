from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from .contracts import KVLocation


def _stable_digest(domain: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"domain": domain, **dict(payload)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelMathSignature:
    model_id: str
    weights_revision: str
    architecture: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    rope_scaling: Any = None
    sliding_window: Optional[int] = None

    def __post_init__(self) -> None:
        if not all((self.model_id, self.weights_revision, self.architecture)):
            raise ValueError("model-math identity fields are required")
        if min(
            self.num_layers,
            self.num_attention_heads,
            self.num_kv_heads,
            self.head_dim,
        ) <= 0:
            raise ValueError("model-math geometry must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope theta must be positive")

    def digest(self) -> str:
        return _stable_digest("probekv-v7-model-math", self.__dict__)


@dataclass(frozen=True)
class TokenizerSignature:
    tokenizer_id: str
    revision: str
    tokenizer_hash: str
    special_tokens_hash: str

    def __post_init__(self) -> None:
        if not all(self.__dict__.values()):
            raise ValueError("tokenizer identity fields are required")

    def digest(self) -> str:
        return _stable_digest("probekv-v7-tokenizer", self.__dict__)


class SourceVariantState(str, Enum):
    ACTIVE = "active"
    RETIRED = "retired"
    EVICTED = "evicted"


class ArtifactState(str, Enum):
    CREATING = "creating"
    HEALTHY = "healthy"
    CORRUPT = "corrupt"
    DELETING = "deleting"
    DELETED = "deleted"


class ReplicaState(str, Enum):
    ALLOCATING = "allocating"
    READY = "ready"
    LEASED = "leased"
    COPYING = "copying"
    EXECUTING = "executing"
    EVICTING = "evicting"
    DELETED = "deleted"


@dataclass(frozen=True)
class SourceVariantIdentity:
    reuse_content_key: str
    historical_prefix_digest: str
    position_ids_digest: str
    occurrence_id: str
    model_math_signature: str
    origin: str = "full_prefill"

    def __post_init__(self) -> None:
        if not all(
            (
                self.reuse_content_key,
                self.historical_prefix_digest,
                self.position_ids_digest,
                self.occurrence_id,
                self.model_math_signature,
            )
        ):
            raise ValueError("complete Source Variant provenance is required")
        if self.origin != "full_prefill":
            raise ValueError("only full-prefill state may become a Source Variant")

    @property
    def source_variant_id(self) -> str:
        return _stable_digest("probekv-v7-source-variant", self.__dict__)


@dataclass
class CanonicalKVArtifact:
    artifact_id: str
    source_variant_id: str
    generation: int
    parent_source_state_digest: str
    artifact_logical_digest: str
    artifact_bytes_digest: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str = "bfloat16"
    k_semantics: str = "pre_rope"
    v_semantics: str = "raw"
    serialization_version: str = "probekv-kv-v1"
    state: ArtifactState = ArtifactState.HEALTHY

    def __post_init__(self) -> None:
        if not all(
            (
                self.artifact_id,
                self.source_variant_id,
                self.parent_source_state_digest,
                self.artifact_logical_digest,
                self.artifact_bytes_digest,
            )
        ):
            raise ValueError("canonical Artifact identity and digests are required")
        if self.generation < 1 or min(
            self.num_layers, self.num_kv_heads, self.head_dim
        ) <= 0:
            raise ValueError("invalid Artifact generation or geometry")
        if (
            self.dtype != "bfloat16"
            or self.k_semantics != "pre_rope"
            or self.v_semantics != "raw"
        ):
            raise ValueError("v7 supports one lossless BF16 pre-RoPE Artifact")


@dataclass
class ReplicaLocator:
    value: str
    layout_signature: str
    placement_epoch: int = 1

    def __post_init__(self) -> None:
        if not self.value or not self.layout_signature or self.placement_epoch < 1:
            raise ValueError("valid Replica locator is required")


@dataclass
class PhysicalReplica:
    replica_id: str
    artifact_id: str
    generation: int
    tier: KVLocation
    logical_digest: str
    bytes_digest: str
    size_bytes: int
    locator: ReplicaLocator
    # A backing Replica keeps the canonical Artifact resident.  Copies made
    # from it are transient hot/staging Replicas unless explicitly promoted.
    is_backing: bool = False
    state: ReplicaState = ReplicaState.READY
    derived_from_replica_id: Optional[str] = None
    lease_count: int = 0
    copy_in_flight: int = 0
    execution_in_flight: int = 0

    def __post_init__(self) -> None:
        self.tier = KVLocation(self.tier)
        if not all(
            (self.replica_id, self.artifact_id, self.logical_digest, self.bytes_digest)
        ):
            raise ValueError("Replica identity and digests are required")
        if self.generation < 1 or self.size_bytes < 0:
            raise ValueError("invalid Replica generation or size")

    @property
    def busy(self) -> bool:
        return bool(
            self.lease_count or self.copy_in_flight or self.execution_in_flight
        )


@dataclass(frozen=True)
class PredictedAccessPlan:
    access_plan_id: str
    source_variant_id: str
    artifact_id: str
    artifact_generation: int
    replica_id: str
    replica_generation: int
    placement_epoch: int
    pool_snapshot_id: int
    scheduler_snapshot_id: int
    profile_version: str
    visible_load_upper_ms: float
    post_ready_blocking_upper_ms: float
    interference_upper_ms: float
    repair_selection_upper_ms: float
    repair_upper_ms: float
    remaining_upper_ms: float

    def __post_init__(self) -> None:
        if not all(
            (
                self.access_plan_id,
                self.source_variant_id,
                self.artifact_id,
                self.replica_id,
                self.profile_version,
            )
        ):
            raise ValueError("complete predicted access-plan identity is required")
        if min(
            self.artifact_generation,
            self.replica_generation,
            self.placement_epoch,
        ) < 1:
            raise ValueError("access plan generations must be positive")
        if min(self.component_upper_ms) < 0:
            raise ValueError("predicted access costs must be non-negative")

    @property
    def component_upper_ms(self) -> Tuple[float, ...]:
        return (
            self.visible_load_upper_ms,
            self.post_ready_blocking_upper_ms,
            self.interference_upper_ms,
            self.repair_selection_upper_ms,
            self.repair_upper_ms,
            self.remaining_upper_ms,
        )

    @property
    def future_cost_upper_ms(self) -> float:
        return sum(self.component_upper_ms)
