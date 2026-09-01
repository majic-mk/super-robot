from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .v8_schema10_contracts import AbsoluteResidualThreshold, Gate1Mode


SCHEMA10_TRIM_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class VariantAdmissionProfileV10:
    code_commit: str
    cacheblend_patch_sha256: str
    model_id: str
    model_revision: str
    tokenizer_hash: str
    source_residual_trim_ratio: float
    thresholds: Tuple[AbsoluteResidualThreshold, ...]
    materialization_budget_fraction: float = 0.02
    replacement_policy: str = "per_content_variant_lru_full_scope_only"
    replacement_budget_fraction: float = 0.01
    exploration_quota_per_content: int = 2
    probation_comparison_observations: int = 2
    probation_lookup_opportunities: int = 2
    max_protected_probation_per_content: int = 2
    max_variants_per_content: int = 16
    canonical_source_provenance: str = "dense_exact"
    development_partition_sha256: str = ""
    frozen: bool = False
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        if not all(
            (
                self.code_commit,
                self.cacheblend_patch_sha256,
                self.model_id,
                self.model_revision,
                self.tokenizer_hash,
            )
        ):
            raise ValueError("schema10 VariantAdmissionProfile provenance is incomplete")
        if self.source_residual_trim_ratio not in SCHEMA10_TRIM_GRID:
            raise ValueError("source residual trim ratio is outside the schema10 grid")
        depths = tuple(row.completed_depth for row in self.thresholds)
        if depths != (1, 2):
            raise ValueError("schema10 thresholds must cover d1 and d2 exactly")
        if not (
            0 <= self.materialization_budget_fraction <= 0.05
            and 0 <= self.replacement_budget_fraction <= 0.05
        ):
            raise ValueError("materialization/replacement budgets must be within 5%")
        if self.replacement_policy != "per_content_variant_lru_full_scope_only":
            raise ValueError("schema10 replacement policy changed")
        if not 1 <= self.max_variants_per_content <= 16:
            raise ValueError("schema10 supports 1-16 Variants per content")
        if not 0 <= self.exploration_quota_per_content <= self.max_variants_per_content:
            raise ValueError("exploration quota exceeds the Variant limit")
        if min(
            self.probation_comparison_observations,
            self.probation_lookup_opportunities,
            self.max_protected_probation_per_content,
        ) <= 0:
            raise ValueError("schema10 probation settings must be positive")
        if self.canonical_source_provenance != "dense_exact":
            raise ValueError("canonical Sources require exact dense prefill")
        if self.frozen:
            if not self.development_partition_sha256:
                raise ValueError("frozen Profile needs development provenance")
            if self.profile_sha256 != _digest(self.payload(include_sha=False)):
                raise ValueError("VariantAdmissionProfile SHA differs")

    def threshold_for_depth(self, completed_depth: int) -> float:
        for row in self.thresholds:
            if row.completed_depth == completed_depth:
                return row.upper_residual
        raise KeyError("VariantAdmissionProfile has no threshold for this depth")

    def payload(self, *, include_sha: bool = True) -> Mapping[str, Any]:
        payload = {
            "protocol_version": 8,
            "schema_version": 10,
            "code_commit": self.code_commit,
            "cacheblend_patch_sha256": self.cacheblend_patch_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_hash": self.tokenizer_hash,
            "source_residual_trim_ratio": self.source_residual_trim_ratio,
            "thresholds": [row.__dict__ for row in self.thresholds],
            "materialization_budget_fraction": self.materialization_budget_fraction,
            "replacement_policy": self.replacement_policy,
            "replacement_budget_fraction": self.replacement_budget_fraction,
            "exploration_quota_per_content": self.exploration_quota_per_content,
            "probation_comparison_observations": self.probation_comparison_observations,
            "probation_lookup_opportunities": self.probation_lookup_opportunities,
            "max_protected_probation_per_content": self.max_protected_probation_per_content,
            "max_variants_per_content": self.max_variants_per_content,
            "canonical_source_provenance": self.canonical_source_provenance,
            "development_partition_sha256": self.development_partition_sha256,
            "frozen": self.frozen,
        }
        if include_sha:
            payload["profile_sha256"] = self.profile_sha256
        return payload


@dataclass(frozen=True)
class PreparationPolicyProfile:
    code_commit: str
    model_id: str
    runtime_policy: str
    gate1_mode: Gate1Mode
    atomic_preparation_reservation_required: bool = True
    final_commit_admission_required: bool = True
    mean_overhead_limit_fraction: float = 0.005
    p95_overhead_limit_fraction: float = 0.01
    transferred_bytes_limit_fraction: float = 0.01
    development_partition_sha256: str = ""
    frozen: bool = False
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate1_mode", Gate1Mode(self.gate1_mode))
        if not self.code_commit or not self.model_id or not self.runtime_policy:
            raise ValueError("PreparationPolicyProfile provenance is incomplete")
        if not (
            self.atomic_preparation_reservation_required
            and self.final_commit_admission_required
        ):
            raise ValueError("schema10 cannot remove reservation/final admission")
        if min(
            self.mean_overhead_limit_fraction,
            self.p95_overhead_limit_fraction,
            self.transferred_bytes_limit_fraction,
        ) < 0:
            raise ValueError("PreparationPolicy thresholds must be non-negative")
        if self.frozen:
            if not self.development_partition_sha256:
                raise ValueError("frozen preparation Profile needs development data")
            if self.profile_sha256 != _digest(self.payload(include_sha=False)):
                raise ValueError("PreparationPolicyProfile SHA differs")

    def payload(self, *, include_sha: bool = True) -> Mapping[str, Any]:
        payload = {
            "protocol_version": 8,
            "schema_version": 10,
            "code_commit": self.code_commit,
            "model_id": self.model_id,
            "runtime_policy": self.runtime_policy,
            "gate1_mode": self.gate1_mode.value,
            "atomic_preparation_reservation_required": self.atomic_preparation_reservation_required,
            "final_commit_admission_required": self.final_commit_admission_required,
            "mean_overhead_limit_fraction": self.mean_overhead_limit_fraction,
            "p95_overhead_limit_fraction": self.p95_overhead_limit_fraction,
            "transferred_bytes_limit_fraction": self.transferred_bytes_limit_fraction,
            "development_partition_sha256": self.development_partition_sha256,
            "frozen": self.frozen,
        }
        if include_sha:
            payload["profile_sha256"] = self.profile_sha256
        return payload


def _freeze(profile_type: type, kwargs: Mapping[str, Any]) -> Any:
    base = dict(kwargs)
    base.pop("profile_sha256", None)
    if not base.get("frozen", False):
        return profile_type(**base)
    provisional = profile_type(**{**base, "frozen": False})
    payload = dict(provisional.payload(include_sha=False))
    payload["frozen"] = True
    return profile_type(**{**base, "frozen": True, "profile_sha256": _digest(payload)})


def build_variant_admission_profile_v10(**kwargs: Any) -> VariantAdmissionProfileV10:
    return _freeze(VariantAdmissionProfileV10, kwargs)


def build_preparation_policy_profile(**kwargs: Any) -> PreparationPolicyProfile:
    return _freeze(PreparationPolicyProfile, kwargs)
