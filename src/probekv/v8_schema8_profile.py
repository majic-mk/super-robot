from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Tuple


def schema8_profile_digest(domain: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"domain": domain, **dict(payload)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class Schema8ProfileProvenance:
    profile_kind: str
    code_commit: str
    cacheblend_patch_sha256: str
    model_id: str
    model_revision: str
    tokenizer_hash: str
    gpu_uuid: str = ""
    measurement_sha256: str = ""
    frozen: bool = False

    def __post_init__(self) -> None:
        if self.profile_kind not in {
            "selection_depth", "repair_policy", "runtime_cost"
        }:
            raise ValueError("unknown schema-v8 Profile kind")
        if not all(
            (
                self.code_commit,
                self.cacheblend_patch_sha256,
                self.model_id,
                self.model_revision,
                self.tokenizer_hash,
            )
        ):
            raise ValueError("schema-v8 Profile provenance is incomplete")
        if self.frozen and not all((self.gpu_uuid, self.measurement_sha256)):
            raise ValueError("frozen schema-v8 Profile requires real GPU measurements")


@dataclass(frozen=True)
class SelectionDepthProfileV8:
    provenance: Schema8ProfileProvenance
    allowed_completed_depths: Tuple[int, ...]
    source_score_trim_ratio: float
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        if self.provenance.profile_kind != "selection_depth":
            raise ValueError("wrong schema-v8 Profile kind")
        if self.allowed_completed_depths not in {(1,), (1, 2)}:
            raise ValueError("schema-v8 selection depth must be d1 or d1+d2")
        if self.source_score_trim_ratio not in {0.10, 0.15}:
            raise ValueError("selection trim ratio is outside the frozen grid")
        if self.provenance.frozen and self.profile_sha256 != schema8_profile_digest(
            "schema8-selection-depth", self.payload()
        ):
            raise ValueError("frozen SelectionDepthProfile SHA differs")

    def payload(self) -> Mapping[str, Any]:
        return {
            "protocol_version": 8,
            "schema_version": 8,
            "provenance": self.provenance.__dict__,
            "allowed_completed_depths": list(self.allowed_completed_depths),
            "source_score_trim_ratio": self.source_score_trim_ratio,
        }


@dataclass(frozen=True)
class RepairPolicyProfileV8:
    provenance: Schema8ProfileProvenance
    policy: str
    scope: str
    certified_floor: float
    shared_ratio_by_age: Mapping[int, float]
    no_reentry_oracle_recall: float
    minimum_no_reentry_recall: float
    adaptive_candidate_templates: Tuple[str, ...] = ()
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        if self.provenance.profile_kind != "repair_policy":
            raise ValueError("wrong schema-v8 Profile kind")
        expected_scope = {
            "fixed_15": "uniform_fixed",
            "static_gradual": "shared_relative_schedule",
            "load_recompute_aware_gradual": "per_segment_load_aware",
        }.get(self.policy)
        if expected_scope != self.scope:
            raise ValueError("repair policy and ratio scope disagree")
        allowed_templates = {
            "uniform_floor",
            "uniform_cap",
            "shared_age_schedule",
            "single_segment_quality_priority",
            "single_segment_load_priority",
        }
        if self.policy == "load_recompute_aware_gradual":
            if (
                not self.adaptive_candidate_templates
                or len(self.adaptive_candidate_templates) > 8
                or len(set(self.adaptive_candidate_templates))
                != len(self.adaptive_candidate_templates)
                or not set(self.adaptive_candidate_templates).issubset(allowed_templates)
            ):
                raise ValueError(
                    "adaptive repair requires a bounded frozen candidate-template set"
                )
        elif self.adaptive_candidate_templates:
            raise ValueError("non-adaptive repair cannot carry adaptive templates")
        if self.certified_floor not in {0.10, 0.12, 0.15}:
            raise ValueError("repair floor is outside the development grid")
        ratios = [float(self.shared_ratio_by_age[key]) for key in sorted(self.shared_ratio_by_age)]
        if any(not self.certified_floor <= value <= 0.15 for value in ratios):
            raise ValueError("repair schedule exceeds its floor/cap")
        if any(right > left + 1e-12 for left, right in zip(ratios, ratios[1:])):
            raise ValueError("repair schedule must be non-increasing")
        if not 0 <= self.no_reentry_oracle_recall <= 1:
            raise ValueError("oracle recall is outside [0,1]")
        if self.provenance.frozen and self.policy != "fixed_15" and (
            self.no_reentry_oracle_recall < self.minimum_no_reentry_recall
        ):
            raise ValueError("gradual repair failed the no-reentry recall Gate")
        if self.provenance.frozen and self.profile_sha256 != schema8_profile_digest(
            "schema8-repair-policy", self.payload()
        ):
            raise ValueError("frozen RepairPolicyProfile SHA differs")

    def payload(self) -> Mapping[str, Any]:
        return {
            "protocol_version": 8,
            "schema_version": 8,
            "provenance": self.provenance.__dict__,
            "policy": self.policy,
            "scope": self.scope,
            "certified_floor": self.certified_floor,
            "shared_ratio_by_age": dict(self.shared_ratio_by_age),
            "no_reentry_oracle_recall": self.no_reentry_oracle_recall,
            "minimum_no_reentry_recall": self.minimum_no_reentry_recall,
            "adaptive_candidate_templates": list(self.adaptive_candidate_templates),
        }


@dataclass(frozen=True)
class RuntimeCostProfileV8:
    provenance: Schema8ProfileProvenance
    category_measurements: Mapping[str, Tuple[Mapping[str, Any], ...]]
    profile_sha256: str = ""

    REQUIRED_CATEGORIES = frozenset(
        {
            "selection_and_gate1",
            "winner_layerwise_load",
            "dense_barrier_joint_timeline",
            "repair_and_union_mask",
            "interference_and_scheduler",
        }
    )

    def __post_init__(self) -> None:
        if self.provenance.profile_kind != "runtime_cost":
            raise ValueError("wrong schema-v8 Profile kind")
        if set(self.category_measurements) != self.REQUIRED_CATEGORIES:
            raise ValueError("schema-v8 RuntimeCostProfile categories are incomplete")
        if self.provenance.frozen and any(
            row.get("cuda_event_timing") is not True
            for rows in self.category_measurements.values()
            for row in rows
        ):
            raise ValueError("frozen RuntimeCostProfile requires real CUDA timing")
        if self.provenance.frozen and self.profile_sha256 != schema8_profile_digest(
            "schema8-runtime-cost", self.payload()
        ):
            raise ValueError("frozen RuntimeCostProfile SHA differs")

    def payload(self) -> Mapping[str, Any]:
        return {
            "protocol_version": 8,
            "schema_version": 8,
            "provenance": self.provenance.__dict__,
            "category_measurements": {
                key: list(value) for key, value in self.category_measurements.items()
            },
        }


def build_selection_depth_profile_v8(
    *,
    provenance: Schema8ProfileProvenance,
    allowed_completed_depths: Tuple[int, ...],
    source_score_trim_ratio: float,
) -> SelectionDepthProfileV8:
    payload = {
        "protocol_version": 8,
        "schema_version": 8,
        "provenance": provenance.__dict__,
        "allowed_completed_depths": list(allowed_completed_depths),
        "source_score_trim_ratio": source_score_trim_ratio,
    }
    return SelectionDepthProfileV8(
        provenance,
        allowed_completed_depths,
        source_score_trim_ratio,
        schema8_profile_digest("schema8-selection-depth", payload)
        if provenance.frozen else "",
    )


def build_repair_policy_profile_v8(
    *,
    provenance: Schema8ProfileProvenance,
    policy: str,
    scope: str,
    certified_floor: float,
    shared_ratio_by_age: Mapping[int, float],
    no_reentry_oracle_recall: float,
    minimum_no_reentry_recall: float,
    adaptive_candidate_templates: Tuple[str, ...] = (),
) -> RepairPolicyProfileV8:
    payload = {
        "protocol_version": 8,
        "schema_version": 8,
        "provenance": provenance.__dict__,
        "policy": policy,
        "scope": scope,
        "certified_floor": certified_floor,
        "shared_ratio_by_age": dict(shared_ratio_by_age),
        "no_reentry_oracle_recall": no_reentry_oracle_recall,
        "minimum_no_reentry_recall": minimum_no_reentry_recall,
        "adaptive_candidate_templates": list(adaptive_candidate_templates),
    }
    return RepairPolicyProfileV8(
        provenance,
        policy,
        scope,
        certified_floor,
        shared_ratio_by_age,
        no_reentry_oracle_recall,
        minimum_no_reentry_recall,
        adaptive_candidate_templates,
        schema8_profile_digest("schema8-repair-policy", payload)
        if provenance.frozen else "",
    )


def build_runtime_cost_profile_v8(
    *,
    provenance: Schema8ProfileProvenance,
    category_measurements: Mapping[str, Tuple[Mapping[str, Any], ...]],
) -> RuntimeCostProfileV8:
    payload = {
        "protocol_version": 8,
        "schema_version": 8,
        "provenance": provenance.__dict__,
        "category_measurements": {
            key: list(value) for key, value in category_measurements.items()
        },
    }
    return RuntimeCostProfileV8(
        provenance,
        category_measurements,
        schema8_profile_digest("schema8-runtime-cost", payload)
        if provenance.frozen else "",
    )
