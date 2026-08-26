from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

from .contracts import KVLocation
from .v7_contracts import (
    ArtifactState,
    PredictedAccessPlan,
    ReplicaState,
)
from .v7_source_pool import StoredSourceVariant, V7SourcePool


class SourceVariantIneligibility(str, Enum):
    CONTENT_MISMATCH = "content_mismatch"
    MODEL_MISMATCH = "model_mismatch"
    NON_CANONICAL_ORIGIN = "non_canonical_origin"
    PREFIX_PROVENANCE_MISSING = "prefix_provenance_missing"
    POSITION_PROVENANCE_MISSING = "position_provenance_missing"
    SOURCE_DIGEST_INVALID = "source_digest_invalid"
    SOURCE_NOT_ACTIVE = "source_not_active"


class CalibrationIneligibility(str, Enum):
    NAMESPACE_MISMATCH = "namespace_mismatch"
    LENGTH_OUT_OF_SUPPORT = "length_out_of_support"
    SOURCE_COUNT_OUT_OF_SUPPORT = "source_count_out_of_support"
    PROBE_LAYER_OUT_OF_SUPPORT = "probe_layer_out_of_support"
    SUMMARY_FORMAT_OUT_OF_SUPPORT = "summary_format_out_of_support"
    FEATURE_OUT_OF_ENVELOPE = "feature_out_of_envelope"


class ArtifactIncompatibility(str, Enum):
    MISSING = "artifact_missing"
    UNHEALTHY = "artifact_unhealthy"
    PARENT_DIGEST_MISMATCH = "parent_digest_mismatch"
    LOGICAL_DIGEST_MISSING = "logical_digest_missing"
    FORMAT_MISMATCH = "format_mismatch"
    GEOMETRY_MISMATCH = "geometry_mismatch"


class RuntimeIneligibility(str, Enum):
    NO_FEASIBLE_REPLICA = "no_feasible_replica"
    PREVIEW_STALE = "preview_stale"
    REPLICA_BUSY = "replica_busy"
    REPLICA_CORRUPT = "replica_corrupt"
    ECONOMIC_REJECTION = "economic_rejection"


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibrationEnvelope:
    namespace: str
    min_segment_tokens: int
    max_segment_tokens: int
    max_sources: int
    max_probe_layer: int
    summary_formats: Tuple[str, ...]
    max_feature_norm: float

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("calibration namespace is required")
        if not 0 < self.min_segment_tokens <= self.max_segment_tokens:
            raise ValueError("invalid calibration length envelope")
        if self.max_sources < 1 or self.max_probe_layer < 1:
            raise ValueError("invalid calibration count/layer envelope")
        if not self.summary_formats or self.max_feature_norm <= 0:
            raise ValueError("invalid calibration feature envelope")


@dataclass(frozen=True)
class ReplicaTierProfile:
    visible_load_upper_ms: float
    post_ready_blocking_upper_ms: float
    interference_upper_ms: float

    def __post_init__(self) -> None:
        if min(self.__dict__.values()) < 0:
            raise ValueError("Replica tier profiles must be non-negative")


@dataclass(frozen=True)
class AccessPreview:
    source_variant_id: str
    artifact_compatible: bool
    artifact_reasons: Tuple[str, ...]
    plans: Tuple[PredictedAccessPlan, ...]
    pool_snapshot_id: int

    @property
    def best_plan(self) -> Optional[PredictedAccessPlan]:
        return min(self.plans, key=lambda plan: plan.future_cost_upper_ms, default=None)


def evaluate_source_variant(
    variant: StoredSourceVariant,
    *,
    expected_content_key: str,
    expected_model_math_signature: str,
    expected_source_digest: Optional[str] = None,
) -> EligibilityResult:
    reasons = []
    identity = variant.identity
    if identity.reuse_content_key != expected_content_key:
        reasons.append(SourceVariantIneligibility.CONTENT_MISMATCH.value)
    if identity.model_math_signature != expected_model_math_signature:
        reasons.append(SourceVariantIneligibility.MODEL_MISMATCH.value)
    if identity.origin != "full_prefill":
        reasons.append(SourceVariantIneligibility.NON_CANONICAL_ORIGIN.value)
    if not identity.historical_prefix_digest:
        reasons.append(SourceVariantIneligibility.PREFIX_PROVENANCE_MISSING.value)
    if not identity.position_ids_digest:
        reasons.append(SourceVariantIneligibility.POSITION_PROVENANCE_MISSING.value)
    if not variant.canonical_source_state_digest or (
        expected_source_digest is not None
        and variant.canonical_source_state_digest != expected_source_digest
    ):
        reasons.append(SourceVariantIneligibility.SOURCE_DIGEST_INVALID.value)
    if variant.state.value != "active":
        reasons.append(SourceVariantIneligibility.SOURCE_NOT_ACTIVE.value)
    return EligibilityResult(not reasons, tuple(reasons))


def evaluate_calibration(
    *,
    canonicalizer_namespace: str,
    segment_tokens: int,
    source_count: int,
    probe_layer: int,
    summary_format: str,
    feature_norm: float,
    envelope: CalibrationEnvelope,
) -> EligibilityResult:
    reasons = []
    if canonicalizer_namespace != envelope.namespace:
        reasons.append(CalibrationIneligibility.NAMESPACE_MISMATCH.value)
    if not envelope.min_segment_tokens <= segment_tokens <= envelope.max_segment_tokens:
        reasons.append(CalibrationIneligibility.LENGTH_OUT_OF_SUPPORT.value)
    if not 1 <= source_count <= envelope.max_sources:
        reasons.append(CalibrationIneligibility.SOURCE_COUNT_OUT_OF_SUPPORT.value)
    if not 1 <= probe_layer <= envelope.max_probe_layer:
        reasons.append(CalibrationIneligibility.PROBE_LAYER_OUT_OF_SUPPORT.value)
    if summary_format not in envelope.summary_formats:
        reasons.append(CalibrationIneligibility.SUMMARY_FORMAT_OUT_OF_SUPPORT.value)
    if feature_norm < 0 or feature_norm > envelope.max_feature_norm:
        reasons.append(CalibrationIneligibility.FEATURE_OUT_OF_ENVELOPE.value)
    return EligibilityResult(not reasons, tuple(reasons))


def evaluate_artifact(
    variant: StoredSourceVariant,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> EligibilityResult:
    reasons = []
    artifact = variant.artifact
    if artifact is None:
        reasons.append(ArtifactIncompatibility.MISSING.value)
    else:
        if artifact.state is not ArtifactState.HEALTHY:
            reasons.append(ArtifactIncompatibility.UNHEALTHY.value)
        if artifact.parent_source_state_digest != variant.canonical_source_state_digest:
            reasons.append(ArtifactIncompatibility.PARENT_DIGEST_MISMATCH.value)
        if not artifact.artifact_logical_digest:
            reasons.append(ArtifactIncompatibility.LOGICAL_DIGEST_MISSING.value)
        if (
            artifact.dtype != "bfloat16"
            or artifact.k_semantics != "pre_rope"
            or artifact.v_semantics != "raw"
        ):
            reasons.append(ArtifactIncompatibility.FORMAT_MISMATCH.value)
        if (
            artifact.num_layers != num_layers
            or artifact.num_kv_heads != num_kv_heads
            or artifact.head_dim != head_dim
        ):
            reasons.append(ArtifactIncompatibility.GEOMETRY_MISMATCH.value)
    return EligibilityResult(not reasons, tuple(reasons))


def preview_replica_access(
    pool: V7SourcePool,
    variant: StoredSourceVariant,
    *,
    scheduler_snapshot_id: int,
    profile_version: str,
    tier_profiles: Mapping[KVLocation, ReplicaTierProfile],
    repair_selection_upper_ms: float,
    repair_upper_ms: float,
    remaining_upper_ms: float,
    expected_geometry: Tuple[int, int, int],
) -> AccessPreview:
    artifact_result = evaluate_artifact(
        variant,
        num_layers=expected_geometry[0],
        num_kv_heads=expected_geometry[1],
        head_dim=expected_geometry[2],
    )
    snapshot = pool.snapshot_id
    if not artifact_result.eligible or variant.artifact is None:
        return AccessPreview(
            variant.source_variant_id,
            False,
            artifact_result.reasons,
            (),
            snapshot,
        )
    plans = []
    artifact = variant.artifact
    for replica in variant.healthy_replicas:
        profile = tier_profiles.get(replica.tier)
        if profile is None or replica.state not in {ReplicaState.READY, ReplicaState.LEASED}:
            continue
        if replica.logical_digest != artifact.artifact_logical_digest:
            continue
        plan_id = hashlib.sha256(
            (
                "%s|%s|%d|%d|%d|%s"
                % (
                    variant.source_variant_id,
                    replica.replica_id,
                    replica.generation,
                    replica.locator.placement_epoch,
                    scheduler_snapshot_id,
                    profile_version,
                )
            ).encode("utf-8")
        ).hexdigest()
        plans.append(
            PredictedAccessPlan(
                access_plan_id=plan_id,
                source_variant_id=variant.source_variant_id,
                artifact_id=artifact.artifact_id,
                artifact_generation=artifact.generation,
                replica_id=replica.replica_id,
                replica_generation=replica.generation,
                placement_epoch=replica.locator.placement_epoch,
                pool_snapshot_id=snapshot,
                scheduler_snapshot_id=scheduler_snapshot_id,
                profile_version=profile_version,
                visible_load_upper_ms=profile.visible_load_upper_ms,
                post_ready_blocking_upper_ms=profile.post_ready_blocking_upper_ms,
                interference_upper_ms=profile.interference_upper_ms,
                repair_selection_upper_ms=repair_selection_upper_ms,
                repair_upper_ms=repair_upper_ms,
                remaining_upper_ms=remaining_upper_ms,
            )
        )
    return AccessPreview(
        variant.source_variant_id,
        True,
        (),
        tuple(sorted(plans, key=lambda plan: (plan.future_cost_upper_ms, plan.replica_id))),
        snapshot,
    )


def bind_or_replan_same_source(
    pool: V7SourcePool,
    variant: StoredSourceVariant,
    preview: AccessPreview,
    *,
    model_math_signature: str,
    tier_profiles: Mapping[KVLocation, ReplicaTierProfile],
    scheduler_snapshot_id: int,
    profile_version: str,
    expected_geometry: Tuple[int, int, int],
    repair_selection_upper_ms: float,
    repair_upper_ms: float,
    remaining_upper_ms: float,
) -> tuple[Optional[PredictedAccessPlan], Optional[object], int]:
    """Bind the preview or replan only within the already locked Source."""
    if preview.source_variant_id != variant.source_variant_id:
        raise ValueError("preview belongs to another Source Variant")
    attempts = 0
    for plan in preview.plans:
        try:
            replica = pool.bind_replica(
                model_math_signature,
                variant.identity.reuse_content_key,
                variant.source_variant_id,
                plan.replica_id,
                artifact_generation=plan.artifact_generation,
                replica_generation=plan.replica_generation,
                placement_epoch=plan.placement_epoch,
            )
            return plan, replica, attempts
        except (KeyError, RuntimeError):
            attempts += 1
    refreshed = preview_replica_access(
        pool,
        variant,
        scheduler_snapshot_id=scheduler_snapshot_id,
        profile_version=profile_version,
        tier_profiles=tier_profiles,
        repair_selection_upper_ms=repair_selection_upper_ms,
        repair_upper_ms=repair_upper_ms,
        remaining_upper_ms=remaining_upper_ms,
        expected_geometry=expected_geometry,
    )
    for plan in refreshed.plans:
        try:
            replica = pool.bind_replica(
                model_math_signature,
                variant.identity.reuse_content_key,
                variant.source_variant_id,
                plan.replica_id,
                artifact_generation=plan.artifact_generation,
                replica_generation=plan.replica_generation,
                placement_epoch=plan.placement_epoch,
            )
            return plan, replica, attempts + 1
        except (KeyError, RuntimeError):
            attempts += 1
    return None, None, attempts
