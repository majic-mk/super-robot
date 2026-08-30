from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .v8_schema7_contracts import (
    V8_SCHEMA7_PROTOCOL_VERSION,
    V8_SCHEMA7_VERSION,
    Schema7ProfileProvenance,
    SourceSelectionDepthPolicy,
    RepairMetric,
    RepairPolicy,
    stable_schema7_digest,
)


@dataclass(frozen=True)
class SelectionDepthProfile:
    provenance: Schema7ProfileProvenance
    policy: SourceSelectionDepthPolicy
    checkpoint_depths: Tuple[int, ...]
    source_score_trim_ratio: float
    eta: float
    eta_strong: float
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", SourceSelectionDepthPolicy(self.policy))
        if self.provenance.profile_kind != "selection_depth":
            raise ValueError("wrong Profile provenance kind")
        if tuple(sorted(set(self.checkpoint_depths))) != self.checkpoint_depths:
            raise ValueError("selection depths must be sorted and unique")
        if self.checkpoint_depths[0] < 1:
            raise ValueError("d=0 is a negative control only")
        if self.source_score_trim_ratio not in {0.10, 0.15}:
            raise ValueError("schema-v7 trim ratio is outside the development grid")
        if not 0 <= self.eta <= self.eta_strong <= 1:
            raise ValueError("invalid selection margins")
        self._check_sha()

    def _check_sha(self) -> None:
        if self.provenance.frozen:
            expected = stable_schema7_digest("selection-depth-profile", self.payload())
            if self.profile_sha256 != expected:
                raise ValueError("frozen SelectionDepthProfile SHA differs")

    def payload(self) -> Mapping[str, Any]:
        return {
            "protocol_version": V8_SCHEMA7_PROTOCOL_VERSION,
            "schema_version": V8_SCHEMA7_VERSION,
            "provenance": self.provenance.__dict__,
            "policy": self.policy.value,
            "checkpoint_depths": list(self.checkpoint_depths),
            "source_score_trim_ratio": self.source_score_trim_ratio,
            "eta": self.eta,
            "eta_strong": self.eta_strong,
        }


@dataclass(frozen=True)
class RepairPolicyProfile:
    provenance: Schema7ProfileProvenance
    policy: RepairPolicy
    metric: RepairMetric
    initial_repair_cap: float
    repair_floor: float
    ratio_by_layer: Mapping[int, float]
    no_reentry_oracle_recall: float
    minimum_no_reentry_recall: float
    profile_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", RepairPolicy(self.policy))
        object.__setattr__(self, "metric", RepairMetric(self.metric))
        if self.provenance.profile_kind != "repair_policy":
            raise ValueError("wrong Profile provenance kind")
        if self.initial_repair_cap != 0.15:
            raise ValueError("schema-v7 repair cap is 0.15")
        if self.repair_floor not in {0.10, 0.12, 0.15}:
            raise ValueError("repair floor was not in the development grid")
        if any(
            not self.repair_floor <= float(ratio) <= self.initial_repair_cap
            for ratio in self.ratio_by_layer.values()
        ):
            raise ValueError("layer repair ratio is outside floor/cap")
        ordered = [self.ratio_by_layer[key] for key in sorted(self.ratio_by_layer)]
        if any(right > left + 1e-12 for left, right in zip(ordered, ordered[1:])):
            raise ValueError("gradual repair ratios must not increase")
        if not 0 <= self.no_reentry_oracle_recall <= 1:
            raise ValueError("oracle recall must lie in [0,1]")
        if not 0 <= self.minimum_no_reentry_recall <= 1:
            raise ValueError("minimum oracle recall must lie in [0,1]")
        if (
            self.provenance.frozen
            and self.policy is not RepairPolicy.FIXED_15
            and self.no_reentry_oracle_recall < self.minimum_no_reentry_recall
        ):
            raise ValueError("gradual repair cannot freeze below its oracle recall gate")
        if self.provenance.frozen:
            expected = stable_schema7_digest("repair-policy-profile", self.payload())
            if self.profile_sha256 != expected:
                raise ValueError("frozen RepairPolicyProfile SHA differs")

    def payload(self) -> Mapping[str, Any]:
        return {
            "protocol_version": V8_SCHEMA7_PROTOCOL_VERSION,
            "schema_version": V8_SCHEMA7_VERSION,
            "provenance": self.provenance.__dict__,
            "policy": self.policy.value,
            "metric": self.metric.value,
            "initial_repair_cap": self.initial_repair_cap,
            "repair_floor": self.repair_floor,
            "ratio_by_layer": dict(self.ratio_by_layer),
            "no_reentry_oracle_recall": self.no_reentry_oracle_recall,
            "minimum_no_reentry_recall": self.minimum_no_reentry_recall,
        }


@dataclass(frozen=True)
class RuntimeCostProfileV7:
    provenance: Schema7ProfileProvenance
    category_measurements: Mapping[str, Tuple[Mapping[str, Any], ...]]
    profile_sha256: str = ""

    REQUIRED_CATEGORIES = frozenset(
        {
            "comparison_batch",
            "selection_state_transfer",
            "full_kv_tier_load",
            "dense_remaining_joint",
            "repair",
            "union_mask_remaining",
            "interference",
            "scheduler_blocking",
        }
    )

    def __post_init__(self) -> None:
        if self.provenance.profile_kind != "runtime_cost":
            raise ValueError("wrong Profile provenance kind")
        if set(self.category_measurements) != self.REQUIRED_CATEGORIES:
            raise ValueError("RuntimeCostProfile categories are incomplete")
        if self.provenance.frozen:
            expected = stable_schema7_digest("runtime-cost-profile", self.payload())
            if self.profile_sha256 != expected:
                raise ValueError("frozen RuntimeCostProfile SHA differs")

    def payload(self) -> Mapping[str, Any]:
        return {
            "protocol_version": V8_SCHEMA7_PROTOCOL_VERSION,
            "schema_version": V8_SCHEMA7_VERSION,
            "provenance": self.provenance.__dict__,
            "category_measurements": {
                key: list(value) for key, value in self.category_measurements.items()
            },
        }
