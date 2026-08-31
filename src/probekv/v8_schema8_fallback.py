from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple

from .v8_schema8_profile import SelectionDepthProfileV8


class SelectionRuntimePath(str, Enum):
    D1_ONLY_FAST = "d1_only_fast"
    D1_D2_RESCUE_FAST = "d1_d2_rescue_fast"
    LEGACY_MULTICHECKPOINT_THREE_GATE = "legacy_multicheckpoint_three_gate"
    FULL_DENSE_SAFE_FALLBACK = "full_dense_safe_fallback"


@dataclass(frozen=True)
class FastSelectionQualification:
    """Pre-registered evidence deciding whether schema-v8 fast selection is usable."""

    state_availability: float
    selection_coverage: float
    early_resolution_rate_at_completed_depth5: float
    wrong_early_lock_rate: float
    mean_stable_normalized_oracle_regret: float
    selection_critical_path_p95_fraction: float
    selection_budget_realized_overrun_rate: float
    illegal_lock_count: int
    budget_admission_violation_count: int

    def __post_init__(self) -> None:
        for value in (
            self.state_availability,
            self.selection_coverage,
            self.early_resolution_rate_at_completed_depth5,
            self.wrong_early_lock_rate,
            self.mean_stable_normalized_oracle_regret,
            self.selection_critical_path_p95_fraction,
            self.selection_budget_realized_overrun_rate,
        ):
            if not 0 <= value <= 1:
                raise ValueError("fast-selection metric is outside [0,1]")
        if min(self.illegal_lock_count, self.budget_admission_violation_count) < 0:
            raise ValueError("fast-selection failure counts must be non-negative")

    def passed(self) -> bool:
        return (
            self.state_availability >= 0.99 - 1e-12
            and self.selection_coverage >= 0.80 - 1e-12
            and self.early_resolution_rate_at_completed_depth5 >= 0.80 - 1e-12
            and self.wrong_early_lock_rate <= 0.05 + 1e-12
            and self.mean_stable_normalized_oracle_regret <= 0.10 + 1e-12
            and self.selection_critical_path_p95_fraction <= 0.05 + 1e-12
            and self.selection_budget_realized_overrun_rate <= 0.05 + 1e-12
            and self.illegal_lock_count == 0
            and self.budget_admission_violation_count == 0
        )

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, object]) -> "FastSelectionQualification":
        return cls(
            state_availability=float(metrics["state_availability"]),
            selection_coverage=float(metrics["selection_coverage"]),
            early_resolution_rate_at_completed_depth5=float(
                metrics["early_resolution_rate_at_completed_depth5"]
            ),
            wrong_early_lock_rate=float(metrics["wrong_early_lock_rate"]),
            mean_stable_normalized_oracle_regret=float(
                metrics["mean_stable_normalized_oracle_regret"]
            ),
            selection_critical_path_p95_fraction=float(
                metrics["selection_critical_path_p95_fraction"]
            ),
            selection_budget_realized_overrun_rate=float(
                metrics["selection_budget_realized_overrun_rate"]
            ),
            illegal_lock_count=int(metrics["illegal_lock_count"]),
            budget_admission_violation_count=int(
                metrics["budget_admission_violation_count"]
            ),
        )


@dataclass(frozen=True)
class SelectionRuntimeDispatch:
    path: SelectionRuntimePath
    checkpoint_depths: Tuple[int, ...]
    runtime_schema_version: int
    runtime_contract: str
    selection_execution_policy: str
    fast_feature_set_enabled: bool
    enabled_feature_set: Tuple[str, ...]
    reason: str
    required_gate_names: Tuple[str, ...]


@dataclass(frozen=True)
class H2SelectionCandidate:
    path: SelectionRuntimePath
    checkpoint_depths: Tuple[int, ...]
    reason: str


def _legacy_depths(model_family: str) -> Tuple[int, ...]:
    family = model_family.lower()
    if "mistral" in family:
        return (1, 2, 4, 5, 8)
    if "qwen" in family:
        return (1, 2, 4, 5, 7)
    raise ValueError("selection dispatcher supports only Mistral/Qwen")


def _profile_matches_model(
    profile: Optional[SelectionDepthProfileV8], model_family: str
) -> bool:
    if profile is None:
        return False
    family = model_family.lower()
    profile_model = profile.provenance.model_id.lower()
    return (
        ("mistral" in family and "mistral" in profile_model)
        or ("qwen" in family and "qwen" in profile_model)
    )


def resolve_h2_selection_candidate(
    *,
    model_family: str,
    selection_profile: Optional[SelectionDepthProfileV8],
    fast_selection_qualification: Optional[FastSelectionQualification],
    legacy_three_gate_runtime_qualified: bool,
) -> H2SelectionCandidate:
    """Freeze the H2 candidate without depending on the later repair Profile."""

    legacy_depths = _legacy_depths(model_family)
    if (
        selection_profile is not None
        and selection_profile.provenance.frozen
        and _profile_matches_model(selection_profile, model_family)
        and fast_selection_qualification is not None
        and fast_selection_qualification.passed()
    ):
        path = (
            SelectionRuntimePath.D1_ONLY_FAST
            if selection_profile.allowed_completed_depths == (1,)
            else SelectionRuntimePath.D1_D2_RESCUE_FAST
        )
        return H2SelectionCandidate(
            path,
            selection_profile.allowed_completed_depths,
            "fast_selection_depth_candidate_passed_H2",
        )
    if legacy_three_gate_runtime_qualified:
        return H2SelectionCandidate(
            SelectionRuntimePath.LEGACY_MULTICHECKPOINT_THREE_GATE,
            legacy_depths,
            "fast_selection_failed_H2_use_qualified_legacy_candidate",
        )
    return H2SelectionCandidate(
        SelectionRuntimePath.FULL_DENSE_SAFE_FALLBACK,
        (),
        "no_qualified_selection_candidate",
    )


def resolve_selection_runtime_path(
    *,
    model_family: str,
    selection_profile: Optional[SelectionDepthProfileV8],
    fast_selection_qualification: Optional[FastSelectionQualification],
    repair_policy_profile_frozen: bool,
    runtime_cost_profile_frozen: bool,
    schema8_runtime_qualified: bool,
    legacy_three_gate_runtime_qualified: bool,
    legacy_selection_execution_policy: str = "causal_commit_wait",
    h2_candidate: Optional[H2SelectionCandidate] = None,
) -> SelectionRuntimeDispatch:
    """Resolve one whole-run protocol; never mix fast and legacy per request.

    d1/d2 may enable the schema-v8 dense-barrier, uniform-I/O repair and
    streamlined FinalCommit path only when all Profile/runtime prerequisites
    pass.  Otherwise the dispatcher selects the preserved legacy multi-depth
    selector and Predicted/Refined three-gate closure.  If that legacy runtime
    is not qualified either, correctness takes precedence and execution is
    fully dense.
    """

    legacy_depths = _legacy_depths(model_family)
    if legacy_selection_execution_policy not in {
        "causal_commit_wait",
        "immediate_staggered_closed_loop",
    }:
        raise ValueError("legacy fallback requires an explicit A/C policy")

    profile_model_matches = _profile_matches_model(selection_profile, model_family)
    if h2_candidate is not None and h2_candidate.path is SelectionRuntimePath.FULL_DENSE_SAFE_FALLBACK:
        legacy_three_gate_runtime_qualified = False
    h2_allows_fast = h2_candidate is None or h2_candidate.path in {
        SelectionRuntimePath.D1_ONLY_FAST,
        SelectionRuntimePath.D1_D2_RESCUE_FAST,
    }

    fast_ready = (
        selection_profile is not None
        and profile_model_matches
        and selection_profile.provenance.frozen
        and fast_selection_qualification is not None
        and fast_selection_qualification.passed()
        and repair_policy_profile_frozen
        and runtime_cost_profile_frozen
        and schema8_runtime_qualified
        and h2_allows_fast
    )
    if fast_ready:
        if selection_profile.allowed_completed_depths == (1,):
            path = SelectionRuntimePath.D1_ONLY_FAST
        elif selection_profile.allowed_completed_depths == (1, 2):
            path = SelectionRuntimePath.D1_D2_RESCUE_FAST
        else:  # guarded by SelectionDepthProfileV8, kept defensive here
            raise ValueError("fast selection Profile contains another depth policy")
        if h2_candidate is not None and (
            h2_candidate.path is not path
            or h2_candidate.checkpoint_depths
            != selection_profile.allowed_completed_depths
        ):
            raise RuntimeError("final runtime dispatch differs from frozen H2 candidate")
        return SelectionRuntimeDispatch(
            path,
            selection_profile.allowed_completed_depths,
            8,
            "schema8_dense_barrier_final_commit",
            "dense_selection_barrier",
            True,
            (
                "dense_selection_barrier",
                "d1_d2_detached_winner_prefetch",
                "streamlined_final_commit_admission",
                "request_layer_uniform_io_balanced_repair",
            ),
            "all_fast_path_profiles_and_runtime_gates_passed",
            (
                "selection_depth_profile",
                "repair_policy_profile",
                "runtime_cost_profile",
                "schema8_runtime_qualification",
            ),
        )

    if legacy_three_gate_runtime_qualified:
        return SelectionRuntimeDispatch(
            SelectionRuntimePath.LEGACY_MULTICHECKPOINT_THREE_GATE,
            legacy_depths,
            7,
            "legacy_predicted_gate2_refined_gate3",
            legacy_selection_execution_policy,
            False,
            (),
            "fast_path_unavailable_use_preserved_legacy_protocol",
            ("legacy_multicheckpoint_three_gate_qualification",),
        )

    return SelectionRuntimeDispatch(
        SelectionRuntimePath.FULL_DENSE_SAFE_FALLBACK,
        (),
        8,
        "full_dense",
        "full_dense",
        False,
        (),
        "neither_fast_nor_legacy_reuse_runtime_is_qualified",
        (),
    )


def selection_dispatch_audit(row: SelectionRuntimeDispatch) -> Mapping[str, object]:
    payload = {
        "protocol_version": 8,
        "schema_version": 8,
        "stage": "schema8_final_runtime_dispatch",
        "selection_runtime_path": row.path.value,
        "checkpoint_depths": list(row.checkpoint_depths),
        "runtime_schema_version": row.runtime_schema_version,
        "runtime_contract": row.runtime_contract,
        "selection_execution_policy": row.selection_execution_policy,
        "fast_feature_set_enabled": row.fast_feature_set_enabled,
        "enabled_feature_set": list(row.enabled_feature_set),
        "reason": row.reason,
        "required_gate_names": list(row.required_gate_names),
    }
    payload["selection_runtime_dispatch_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def h2_selection_candidate_audit(row: H2SelectionCandidate) -> Mapping[str, object]:
    payload = {
        "protocol_version": 8,
        "schema_version": 8,
        "stage": "schema8_h2_selection_candidate",
        "selection_runtime_path_candidate": row.path.value,
        "checkpoint_depths": list(row.checkpoint_depths),
        "reason": row.reason,
        "final_runtime_dispatch_frozen": False,
    }
    payload["h2_selection_candidate_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


__all__ = [
    "FastSelectionQualification",
    "H2SelectionCandidate",
    "SelectionRuntimeDispatch",
    "SelectionRuntimePath",
    "resolve_selection_runtime_path",
    "resolve_h2_selection_candidate",
    "h2_selection_candidate_audit",
    "selection_dispatch_audit",
]
