from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from .v8_schema10_contracts import AbsoluteResidualThreshold, Gate1Mode


SCHEMA10_TRIM_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)
SCHEMA10_REPAIR_RATIO_GRID = (0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00)
SCHEMA10_MODEL_CHECKPOINTS = {
    "mistral": (1, 2, 4, 5, 8),
    "qwen": (1, 2, 4, 5, 7),
}


@dataclass(frozen=True)
class AbsoluteResidualThresholdPointV10:
    completed_depth: int
    source_residual_trim_ratio: float
    upper_residual: float

    def __post_init__(self) -> None:
        if self.completed_depth < 1:
            raise ValueError("absolute residual threshold depth must be positive")
        if self.source_residual_trim_ratio not in SCHEMA10_TRIM_GRID:
            raise ValueError("absolute residual threshold trim ratio is outside grid")
        if not math.isfinite(self.upper_residual) or self.upper_residual < 0:
            raise ValueError("absolute residual threshold must be finite and non-negative")


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
    threshold_table: Tuple[AbsoluteResidualThresholdPointV10, ...] = ()
    materialization_budget_fraction: float = 0.02
    replacement_policy: str = "per_content_variant_lru_full_scope_only"
    replacement_budget_fraction: float = 0.01
    exploration_quota_per_content: int = 2
    probation_comparison_observations: int = 2
    probation_lookup_opportunities: int = 2
    max_protected_probation_per_content: int = 2
    verification_comparisons: int = 0
    probation_grace_lookup_opportunities: int = 0
    probation_grace_capacity_per_content: int = 0
    max_variants_per_content: int = 16
    canonical_source_provenance: str = "dense_exact"
    development_partition_sha256: str = ""
    development_case_manifest_sha256: str = ""
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
        if tuple(sorted(set(depths))) != depths or not {1, 2}.issubset(depths):
            raise ValueError("schema10 selected thresholds must include d1/d2 uniquely")
        table = self.threshold_table or tuple(
            AbsoluteResidualThresholdPointV10(
                row.completed_depth,
                self.source_residual_trim_ratio,
                row.upper_residual,
            )
            for row in self.thresholds
        )
        object.__setattr__(self, "threshold_table", table)
        keys = tuple(
            (row.source_residual_trim_ratio, row.completed_depth) for row in table
        )
        if len(set(keys)) != len(keys):
            raise ValueError("schema10 threshold table contains duplicate cells")
        selected_lookup = {
            row.completed_depth: row.upper_residual
            for row in table
            if row.source_residual_trim_ratio == self.source_residual_trim_ratio
        }
        if any(
            selected_lookup.get(row.completed_depth) != row.upper_residual
            for row in self.thresholds
        ):
            raise ValueError("selected thresholds differ from the threshold table")
        model_key = (
            "mistral" if "mistral" in self.model_id.lower()
            else "qwen" if "qwen" in self.model_id.lower()
            else ""
        )
        if self.frozen and model_key:
            expected = {
                (ratio, depth)
                for ratio in SCHEMA10_TRIM_GRID
                for depth in SCHEMA10_MODEL_CHECKPOINTS[model_key]
            }
            if set(keys) != expected:
                raise ValueError(
                    "frozen schema10 threshold table does not cover every rho/depth"
                )
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
        aliases = (
            ("verification_comparisons", self.probation_comparison_observations),
            (
                "probation_grace_lookup_opportunities",
                self.probation_lookup_opportunities,
            ),
            (
                "probation_grace_capacity_per_content",
                self.max_protected_probation_per_content,
            ),
        )
        for name, canonical in aliases:
            supplied = int(getattr(self, name))
            if supplied not in {0, canonical}:
                raise ValueError("schema10 probation grace aliases disagree")
            object.__setattr__(self, name, canonical)
        if self.canonical_source_provenance != "dense_exact":
            raise ValueError("canonical Sources require exact dense prefill")
        if self.frozen:
            if not all(
                (
                    self.development_partition_sha256,
                    self.development_case_manifest_sha256,
                )
            ):
                raise ValueError("frozen Profile needs development provenance")
            if self.profile_sha256 != _digest(self.payload(include_sha=False)):
                raise ValueError("VariantAdmissionProfile SHA differs")

    def threshold_for_depth(
        self, completed_depth: int, source_residual_trim_ratio: float | None = None
    ) -> float:
        ratio = (
            self.source_residual_trim_ratio
            if source_residual_trim_ratio is None
            else float(source_residual_trim_ratio)
        )
        for row in self.threshold_table:
            if (
                row.completed_depth == completed_depth
                and row.source_residual_trim_ratio == ratio
            ):
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
            "threshold_table": [row.__dict__ for row in self.threshold_table],
            "materialization_budget_fraction": self.materialization_budget_fraction,
            "replacement_policy": self.replacement_policy,
            "replacement_budget_fraction": self.replacement_budget_fraction,
            "exploration_quota_per_content": self.exploration_quota_per_content,
            "verification_comparisons": self.verification_comparisons,
            "probation_grace_lookup_opportunities": self.probation_grace_lookup_opportunities,
            "probation_grace_capacity_per_content": self.probation_grace_capacity_per_content,
            "max_variants_per_content": self.max_variants_per_content,
            "canonical_source_provenance": self.canonical_source_provenance,
            "development_partition_sha256": self.development_partition_sha256,
            "development_case_manifest_sha256": self.development_case_manifest_sha256,
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
    paired_mean_error_limit_fraction: float = 0.005
    paired_p95_error_limit_fraction: float = 0.01
    paired_observations: int = 0
    paired_mean_error_fraction: float = 0.0
    paired_p95_error_fraction: float = 0.0
    development_partition_sha256: str = ""
    development_case_manifest_sha256: str = ""
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
            self.paired_mean_error_limit_fraction,
            self.paired_p95_error_limit_fraction,
            self.paired_mean_error_fraction,
            self.paired_p95_error_fraction,
        ) < 0:
            raise ValueError("PreparationPolicy thresholds must be non-negative")
        if self.paired_observations < 0:
            raise ValueError("paired Gate1 observation count must be non-negative")
        if self.frozen and self.gate1_mode is Gate1Mode.FUSED_ADVISORY and (
            self.paired_observations < 18
            or self.paired_mean_error_fraction > self.paired_mean_error_limit_fraction
            or self.paired_p95_error_fraction > self.paired_p95_error_limit_fraction
        ):
            raise ValueError("fused Gate1 lacks calibrated paired A/B evidence")
        if self.frozen:
            if not all(
                (
                    self.development_partition_sha256,
                    self.development_case_manifest_sha256,
                )
            ):
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
            "paired_mean_error_limit_fraction": self.paired_mean_error_limit_fraction,
            "paired_p95_error_limit_fraction": self.paired_p95_error_limit_fraction,
            "paired_observations": self.paired_observations,
            "paired_mean_error_fraction": self.paired_mean_error_fraction,
            "paired_p95_error_fraction": self.paired_p95_error_fraction,
            "development_partition_sha256": self.development_partition_sha256,
            "development_case_manifest_sha256": self.development_case_manifest_sha256,
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


@dataclass(frozen=True)
class SelectionDepthProfileV10:
    code_commit: str
    cacheblend_patch_sha256: str
    model_id: str
    model_revision: str
    tokenizer_hash: str
    selected_dispatch: str
    allowed_completed_depths: Tuple[int, ...]
    source_residual_trim_ratio: float
    reference_repair_policy: str = "fixed_15"
    reference_gate1_mode: str = "explicit_barrier"
    metrics: Mapping[str, float | int] = None  # type: ignore[assignment]
    development_partition_sha256: str = ""
    development_case_manifest_sha256: str = ""
    measurement_sha256: str = ""
    gpu_uuid: str = ""
    frozen: bool = False
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", dict(self.metrics or {}))
        if self.selected_dispatch not in {
            "d1_only", "d1_d2_rescue", "legacy_multicheckpoint"
        }:
            raise ValueError("invalid schema10 selection dispatch")
        if self.source_residual_trim_ratio not in SCHEMA10_TRIM_GRID:
            raise ValueError("selection depth Profile trim ratio is outside grid")
        if self.reference_repair_policy != "fixed_15" or self.reference_gate1_mode != "explicit_barrier":
            raise ValueError("selection Profile must use the frozen reference path")
        expected = {
            "d1_only": (1,),
            "d1_d2_rescue": (1, 2),
        }.get(self.selected_dispatch)
        if expected is not None and self.allowed_completed_depths != expected:
            raise ValueError("fast dispatch and completed depths disagree")
        if self.frozen and not all(
            (
                self.development_partition_sha256,
                self.development_case_manifest_sha256,
                self.measurement_sha256,
                self.gpu_uuid,
            )
        ):
            raise ValueError("frozen selection Profile requires real GPU provenance")
        if self.frozen and self.profile_sha256 != _digest(self.payload(include_sha=False)):
            raise ValueError("SelectionDepthProfile SHA differs")

    def payload(self, *, include_sha: bool = True) -> Mapping[str, Any]:
        value = {
            "protocol_version": 8,
            "schema_version": 10,
            "code_commit": self.code_commit,
            "cacheblend_patch_sha256": self.cacheblend_patch_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_hash": self.tokenizer_hash,
            "selected_dispatch": self.selected_dispatch,
            "allowed_completed_depths": list(self.allowed_completed_depths),
            "source_residual_trim_ratio": self.source_residual_trim_ratio,
            "reference_repair_policy": self.reference_repair_policy,
            "reference_gate1_mode": self.reference_gate1_mode,
            "metrics": dict(self.metrics),
            "development_partition_sha256": self.development_partition_sha256,
            "development_case_manifest_sha256": self.development_case_manifest_sha256,
            "measurement_sha256": self.measurement_sha256,
            "gpu_uuid": self.gpu_uuid,
            "frozen": self.frozen,
        }
        if include_sha:
            value["profile_sha256"] = self.profile_sha256
        return value


@dataclass(frozen=True)
class RepairPolicyProfileV10:
    code_commit: str
    cacheblend_patch_sha256: str
    model_id: str
    model_revision: str
    tokenizer_hash: str
    policy: str
    certified_floor: float
    shared_ratio_by_age: Mapping[int, float]
    no_reentry_oracle_recall: float
    observed_development_violations: int
    development_request_units: int
    one_sided_95_upper_bound: float
    quality_tail_rate_1pct_certified: bool = False
    ratio_grid: Tuple[float, ...] = SCHEMA10_REPAIR_RATIO_GRID
    development_partition_sha256: str = ""
    development_case_manifest_sha256: str = ""
    measurement_sha256: str = ""
    gpu_uuid: str = ""
    frozen: bool = False
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared_ratio_by_age", dict(self.shared_ratio_by_age))
        if self.policy not in {"fixed_15", "static_gradual", "load_recompute_aware_uniform"}:
            raise ValueError("invalid schema10 repair policy")
        if self.ratio_grid != SCHEMA10_REPAIR_RATIO_GRID:
            raise ValueError("schema10 repair Profile grid is incomplete")
        if self.certified_floor not in {0.10, 0.12, 0.15}:
            raise ValueError("repair floor is outside the frozen grid")
        if self.development_request_units < 1 or not 0 <= self.observed_development_violations <= self.development_request_units:
            raise ValueError("invalid development violation counts")
        if not 0 <= self.one_sided_95_upper_bound <= 1:
            raise ValueError("invalid quality confidence bound")
        if self.quality_tail_rate_1pct_certified:
            raise ValueError("90-case Profile data cannot certify a 1% tail rate")
        if self.observed_development_violations != 0:
            raise ValueError("repair Profile requires zero observed development violations")
        if self.policy != "fixed_15" and self.no_reentry_oracle_recall < 0.95:
            raise ValueError("gradual repair failed the no-reentry recall Gate")
        if any(float(value) not in self.ratio_grid for value in self.shared_ratio_by_age.values()):
            raise ValueError("repair schedule uses an unprofiled ratio")
        if self.frozen and not all(
            (
                self.development_partition_sha256,
                self.development_case_manifest_sha256,
                self.measurement_sha256,
                self.gpu_uuid,
            )
        ):
            raise ValueError("frozen repair Profile requires real GPU provenance")
        if self.frozen and self.profile_sha256 != _digest(self.payload(include_sha=False)):
            raise ValueError("RepairPolicyProfile SHA differs")

    def payload(self, *, include_sha: bool = True) -> Mapping[str, Any]:
        value = {
            "protocol_version": 8,
            "schema_version": 10,
            "code_commit": self.code_commit,
            "cacheblend_patch_sha256": self.cacheblend_patch_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_hash": self.tokenizer_hash,
            "policy": self.policy,
            "certified_floor": self.certified_floor,
            "shared_ratio_by_age": dict(self.shared_ratio_by_age),
            "no_reentry_oracle_recall": self.no_reentry_oracle_recall,
            "observed_development_violations": self.observed_development_violations,
            "development_request_units": self.development_request_units,
            "one_sided_95_upper_bound": self.one_sided_95_upper_bound,
            "quality_tail_rate_1pct_certified": False,
            "ratio_grid": list(self.ratio_grid),
            "development_partition_sha256": self.development_partition_sha256,
            "development_case_manifest_sha256": self.development_case_manifest_sha256,
            "measurement_sha256": self.measurement_sha256,
            "gpu_uuid": self.gpu_uuid,
            "frozen": self.frozen,
        }
        if include_sha:
            value["profile_sha256"] = self.profile_sha256
        return value


@dataclass(frozen=True)
class RuntimeCostProfileV10:
    code_commit: str
    cacheblend_patch_sha256: str
    model_id: str
    model_revision: str
    tokenizer_hash: str
    category_measurements: Mapping[str, Tuple[Mapping[str, Any], ...]]
    joint_anchor_measurements: Tuple[Mapping[str, Any], ...]
    factorized: bool = True
    cartesian_product_used: bool = False
    development_case_manifest_sha256: str = ""
    measurement_sha256: str = ""
    gpu_uuid: str = ""
    frozen: bool = False
    profile_sha256: str = ""

    REQUIRED_CATEGORIES = frozenset({
        "selection", "transfer", "repair", "scheduler"
    })

    def __post_init__(self) -> None:
        normalized = {
            str(key): tuple(dict(row) for row in rows)
            for key, rows in self.category_measurements.items()
        }
        object.__setattr__(self, "category_measurements", normalized)
        object.__setattr__(
            self, "joint_anchor_measurements",
            tuple(dict(row) for row in self.joint_anchor_measurements),
        )
        if set(normalized) != self.REQUIRED_CATEGORIES:
            raise ValueError("factorized RuntimeCostProfile categories are incomplete")
        if not self.factorized or self.cartesian_product_used:
            raise ValueError("schema10 forbids a Cartesian RuntimeCostProfile")
        repair_ratios = {
            float(row["repair_ratio"]) for row in normalized["repair"]
        }
        if repair_ratios != set(SCHEMA10_REPAIR_RATIO_GRID):
            raise ValueError("RuntimeCostProfile repair grid is incomplete")
        all_rows = [row for rows in normalized.values() for row in rows]
        all_rows.extend(self.joint_anchor_measurements)
        if self.frozen and any(
            row.get("cuda_event_timing") is not True or row.get("fake_timing") is True
            for row in all_rows
        ):
            raise ValueError("frozen RuntimeCostProfile requires real CUDA timing")
        if self.frozen and any(
            row.get("joint_path_measured") is not True
            or (
                int(row.get("concurrency", 1)) > 1
                and row.get("concurrency_contention_measured") is not True
            )
            for row in self.joint_anchor_measurements
        ):
            raise ValueError(
                "frozen RuntimeCostProfile joint anchors lack real path/contention evidence"
            )
        if self.frozen and not all(
            (
                self.development_case_manifest_sha256,
                self.measurement_sha256,
                self.gpu_uuid,
            )
        ):
            raise ValueError("frozen RuntimeCostProfile requires measurement provenance")
        if self.frozen and self.profile_sha256 != _digest(self.payload(include_sha=False)):
            raise ValueError("RuntimeCostProfile SHA differs")

    def payload(self, *, include_sha: bool = True) -> Mapping[str, Any]:
        value = {
            "protocol_version": 8,
            "schema_version": 10,
            "code_commit": self.code_commit,
            "cacheblend_patch_sha256": self.cacheblend_patch_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_hash": self.tokenizer_hash,
            "factorized": self.factorized,
            "cartesian_product_used": self.cartesian_product_used,
            "category_measurements": {
                key: list(rows) for key, rows in self.category_measurements.items()
            },
            "joint_anchor_measurements": list(self.joint_anchor_measurements),
            "development_case_manifest_sha256": self.development_case_manifest_sha256,
            "measurement_sha256": self.measurement_sha256,
            "gpu_uuid": self.gpu_uuid,
            "frozen": self.frozen,
        }
        if include_sha:
            value["profile_sha256"] = self.profile_sha256
        return value


PROFILE_FREEZE_ORDER = (
    "reference_runtime",
    "selection_depth",
    "variant_admission",
    "repair_policy",
    "runtime_cost",
    "preparation_policy",
    "final_consistency",
)


def validate_schema10_profile_freeze_order(events: Sequence[str]) -> None:
    if tuple(events) != PROFILE_FREEZE_ORDER:
        raise ValueError("schema10 Profile freeze order differs from the contract")


def build_selection_depth_profile_v10(**kwargs: Any) -> SelectionDepthProfileV10:
    return _freeze(SelectionDepthProfileV10, kwargs)


def build_repair_policy_profile_v10(**kwargs: Any) -> RepairPolicyProfileV10:
    return _freeze(RepairPolicyProfileV10, kwargs)


def build_runtime_cost_profile_v10(**kwargs: Any) -> RuntimeCostProfileV10:
    return _freeze(RuntimeCostProfileV10, kwargs)
