from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import CostAccountingPolicy, InterferenceAccountingMode
from .orchestration import ClosedLoopPolicy
from .scheduler import SchedulerPolicy
from .selector import SelectorPolicy, default_probe_checkpoints
from .source_store import ReplicaEvictionPolicy, SourceEvictionPolicy
from .v6_contracts import SelectionExecutionPolicy


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    evidence_class: str
    seed: int
    cases: int
    total_layers: int
    online_kmax: int
    gamma: float
    probe_checkpoints: Tuple[int, ...]
    max_selection_layer: int
    selector_policy: SelectorPolicy
    reuse_ratio_tolerance: float
    preliminary_economic_filter: bool
    scheduler_policy: SchedulerPolicy
    max_post_ready_overrun_ms: float
    load_interference_ms: float
    cost_accounting_policy: CostAccountingPolicy
    closed_loop_policy: ClosedLoopPolicy
    source_eviction_policy: SourceEvictionPolicy
    replica_eviction_policy: ReplicaEvictionPolicy
    fixed_resident_sources: bool
    runtime_backend: str
    require_async_source_loading: bool
    require_layer_resumable_prefill: bool
    require_cuda_event_timing: bool
    repair_ratios: Tuple[float, ...]
    output_dir: str
    protocol_version: int = 0
    max_stored_variants_per_content: int = 4
    min_variants_per_retained_content: int = 1
    candidate_compare_policy: str = "legacy_fixed_k"
    max_compared_variants_per_segment: int = 4
    probe_compare_budget_fraction: float = 0.05
    segment_planning_policy: str = "single_target_segment"
    max_detected_segments: Optional[int] = 1
    boundary_policy: str = "single_segment"
    partial_reuse_enabled: bool = False
    joint_quality_policy: str = "per_segment"
    source_pool_policy: str = "legacy_per_content"
    summary_format: str = "exact_bf16"
    model_serving_mode: str = "single"
    model_soft_quota_fraction: float = 0.8
    selection_execution_policy: SelectionExecutionPolicy = (
        SelectionExecutionPolicy.LEGACY_COMMON_AFTER_SELECTION
    )
    calibration_policy_match_required: bool = False
    legacy_online_kmax_present: bool = True
    interference_accounting_mode: InterferenceAccountingMode = (
        InterferenceAccountingMode.INCLUDED_IN_LOAD
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        total_layers = int(raw.get("total_layers", 32))
        protocol_version = int(raw.get("protocol_version", 0))
        checkpoints = tuple(
            int(value)
            for value in raw.get(
                "probe_checkpoints", default_probe_checkpoints(total_layers)
            )
        )
        result = cls(
            name=str(raw["name"]),
            evidence_class=str(raw.get("evidence_class", "local_simulation")),
            seed=int(raw.get("seed", 20260726)),
            cases=int(raw.get("cases", 50)),
            total_layers=total_layers,
            online_kmax=int(raw.get("online_kmax", 4)),
            gamma=float(raw.get("gamma", 0.8)),
            probe_checkpoints=checkpoints,
            max_selection_layer=int(
                raw.get("max_selection_layer", checkpoints[-1])
            ),
            selector_policy=SelectorPolicy(
                str(raw.get("selector_policy", "strict_interval"))
            ),
            reuse_ratio_tolerance=float(
                raw.get("reuse_ratio_tolerance", 0.02)
            ),
            preliminary_economic_filter=bool(
                raw.get("preliminary_economic_filter", False)
            ),
            scheduler_policy=SchedulerPolicy(
                str(raw.get("scheduler_policy", "hybrid_strict"))
            ),
            max_post_ready_overrun_ms=float(
                raw.get("max_post_ready_overrun_ms", 0.0)
            ),
            load_interference_ms=float(
                raw.get("load_interference_ms", 0.0)
            ),
            cost_accounting_policy=CostAccountingPolicy(
                str(raw.get("cost_accounting_policy", "legacy_aggregate"))
            ),
            closed_loop_policy=ClosedLoopPolicy(
                str(
                    raw.get(
                        "closed_loop_policy",
                        "legacy_pre_schedule_admission",
                    )
                )
            ),
            source_eviction_policy=SourceEvictionPolicy(
                str(
                    raw.get(
                        "source_eviction_policy",
                        "reject_when_full",
                    )
                )
            ),
            replica_eviction_policy=ReplicaEvictionPolicy(
                str(
                    raw.get(
                        "replica_eviction_policy",
                        "reject_when_full",
                    )
                )
            ),
            fixed_resident_sources=bool(
                raw.get("fixed_resident_sources", False)
            ),
            runtime_backend=str(
                raw.get("runtime_backend", "simulation")
            ),
            require_async_source_loading=bool(
                raw.get("require_async_source_loading", False)
            ),
            require_layer_resumable_prefill=bool(
                raw.get("require_layer_resumable_prefill", False)
            ),
            require_cuda_event_timing=bool(
                raw.get("require_cuda_event_timing", False)
            ),
            repair_ratios=tuple(float(value) for value in raw["repair_ratios"]),
            output_dir=str(raw.get("output_dir", "artifacts/local_smoke")),
            protocol_version=protocol_version,
            max_stored_variants_per_content=int(
                raw.get("max_stored_variants_per_content", 4)
            ),
            min_variants_per_retained_content=int(
                raw.get("min_variants_per_retained_content", 1)
            ),
            candidate_compare_policy=str(
                raw.get("candidate_compare_policy", "legacy_fixed_k")
            ),
            max_compared_variants_per_segment=int(
                raw.get("max_compared_variants_per_segment", 4)
            ),
            probe_compare_budget_fraction=float(
                raw.get("probe_compare_budget_fraction", 0.05)
            ),
            segment_planning_policy=str(
                raw.get("segment_planning_policy", "single_target_segment")
            ),
            max_detected_segments=(
                None
                if raw.get("max_detected_segments", 1) is None
                else int(raw.get("max_detected_segments", 1))
            ),
            boundary_policy=str(raw.get("boundary_policy", "single_segment")),
            partial_reuse_enabled=bool(raw.get("partial_reuse_enabled", False)),
            joint_quality_policy=str(
                raw.get("joint_quality_policy", "per_segment")
            ),
            source_pool_policy=str(
                raw.get("source_pool_policy", "legacy_per_content")
            ),
            summary_format=str(raw.get("summary_format", "exact_bf16")),
            model_serving_mode=str(raw.get("model_serving_mode", "single")),
            model_soft_quota_fraction=float(
                raw.get("model_soft_quota_fraction", 0.8)
            ),
            selection_execution_policy=SelectionExecutionPolicy(
                str(
                    raw.get(
                        "selection_execution_policy",
                        "legacy_common_after_selection",
                    )
                )
            ),
            calibration_policy_match_required=bool(
                raw.get("calibration_policy_match_required", False)
            ),
            legacy_online_kmax_present="online_kmax" in raw,
            interference_accounting_mode=InterferenceAccountingMode(
                str(
                    raw.get(
                        "interference_accounting_mode",
                        (
                            "explicit_penalty"
                            if protocol_version == 6
                            else "included_in_load"
                        ),
                    )
                )
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.evidence_class not in {
            "local_simulation",
            "server_pilot",
            "paper_measurement",
        }:
            raise ValueError("unsupported evidence_class")
        if self.cases <= 0:
            raise ValueError("cases must be positive")
        if self.protocol_version == 6:
            self._validate_v6()
        elif not 1 <= self.online_kmax <= 4:
            raise ValueError("online_kmax must be in [1, 4]")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not self.probe_checkpoints:
            raise ValueError("probe checkpoints required")
        maximum_probe = max(1, int(self.total_layers * 0.25))
        if self.probe_checkpoints[-1] > maximum_probe:
            raise ValueError("L_probe_max exceeds 25% of total layers")
        if not 1 <= self.max_selection_layer <= maximum_probe:
            raise ValueError("max_selection_layer exceeds the probe ceiling")
        if self.probe_checkpoints[-1] > self.max_selection_layer:
            raise ValueError("checkpoint exceeds max_selection_layer")
        if (
            self.selector_policy is not SelectorPolicy.STRICT_INTERVAL
            and self.max_selection_layer not in self.probe_checkpoints
        ):
            raise ValueError(
                "final selector policy requires a max-layer checkpoint"
            )
        if not 0 <= self.reuse_ratio_tolerance <= 1:
            raise ValueError("reuse_ratio_tolerance must be in [0, 1]")
        if self.max_post_ready_overrun_ms < 0:
            raise ValueError("max_post_ready_overrun_ms must be non-negative")
        if self.load_interference_ms < 0:
            raise ValueError("load_interference_ms must be non-negative")
        if (
            self.cost_accounting_policy
            is CostAccountingPolicy.UNIFIED_COMPONENTS_V1
            and self.closed_loop_policy
            is not ClosedLoopPolicy.TWO_STAGE_REFINED_ADMISSION
        ):
            raise ValueError(
                "unified cost accounting requires two-stage refined admission"
            )
        if (
            self.cost_accounting_policy
            is CostAccountingPolicy.UNIFIED_COMPONENTS_V1
            and not self.preliminary_economic_filter
        ):
            raise ValueError(
                "unified cost accounting requires preliminary economic filtering"
            )
        if (
            self.scheduler_policy
            is not SchedulerPolicy.HYBRID_BOUNDED_OVERRUN
            and self.max_post_ready_overrun_ms > 0
        ):
            raise ValueError(
                "only hybrid_bounded_overrun may use a positive overrun budget"
            )
        if any(not 0 <= ratio <= 1 for ratio in self.repair_ratios):
            raise ValueError("repair ratios must be in [0, 1]")
        if self.runtime_backend not in {
            "simulation",
            "cacheblend_case_runner",
            "cacheblend_closed_loop",
            "cacheblend_multisegment_closed_loop",
        }:
            raise ValueError("unsupported runtime_backend")
        if self.runtime_backend in {
            "cacheblend_closed_loop",
            "cacheblend_multisegment_closed_loop",
        }:
            if (
                self.closed_loop_policy
                is not ClosedLoopPolicy.TWO_STAGE_REFINED_ADMISSION
            ):
                raise ValueError(
                    "CacheBlend closed loop requires refined admission"
                )
            if (
                self.cost_accounting_policy
                is not CostAccountingPolicy.UNIFIED_COMPONENTS_V1
            ):
                raise ValueError(
                    "CacheBlend closed loop requires unified cost accounting"
                )
            if not (
                self.require_async_source_loading
                and self.require_layer_resumable_prefill
                and self.require_cuda_event_timing
            ):
                raise ValueError(
                    "CacheBlend closed loop must require async loading, "
                    "resumable prefill, and CUDA event timing"
                )
        elif any(
            (
                self.require_async_source_loading,
                self.require_layer_resumable_prefill,
                self.require_cuda_event_timing,
            )
        ):
            raise ValueError(
                "runtime capability requirements are valid only for "
                "a CacheBlend closed-loop backend"
            )

    def _validate_v6(self) -> None:
        if self.legacy_online_kmax_present:
            raise ValueError("v6 forbids legacy online_kmax")
        if not 1 <= self.min_variants_per_retained_content <= (
            self.max_stored_variants_per_content
        ):
            raise ValueError("invalid v6 retained-variant bounds")
        if self.min_variants_per_retained_content != 1:
            raise ValueError("v6 retained content must keep exactly one variant")
        if self.max_stored_variants_per_content != 16:
            raise ValueError("v6 max stored variants must be 16")
        if not 1 <= self.max_compared_variants_per_segment <= 16:
            raise ValueError("v6 comparison maximum must be in [1, 16]")
        if self.candidate_compare_policy != "all_within_request_budget":
            raise ValueError("v6 requires budgeted all-candidate comparison")
        if not 0 < self.probe_compare_budget_fraction <= 1:
            raise ValueError("invalid probe/compare request budget")
        if self.segment_planning_policy != "all_exact_nonprefix":
            raise ValueError("v6 must plan every exact non-prefix segment")
        if self.max_detected_segments is not None:
            raise ValueError("v6 cannot impose a hard detected-segment cap")
        policy_pairs = {
            "common": SelectionExecutionPolicy.LEGACY_COMMON_AFTER_SELECTION,
            "causal_staggered": SelectionExecutionPolicy.CAUSAL_COMMIT_WAIT,
            "immediate_staggered": (
                SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP
            ),
        }
        if self.boundary_policy not in policy_pairs:
            raise ValueError("unsupported v6 boundary policy")
        if policy_pairs[self.boundary_policy] is not self.selection_execution_policy:
            raise ValueError(
                "boundary and selection-execution policies are inconsistent"
            )
        if (
            self.selection_execution_policy
            is not SelectionExecutionPolicy.LEGACY_COMMON_AFTER_SELECTION
            and not self.calibration_policy_match_required
        ):
            raise ValueError("A/C policies require execution-matched calibration")
        if not self.partial_reuse_enabled:
            raise ValueError("v6 requires partial segment reuse")
        if self.joint_quality_policy != "simultaneous_conformal":
            raise ValueError("v6 requires request-level simultaneous coverage")
        if self.source_pool_policy != "global_hard_model_soft":
            raise ValueError("v6 requires the global Source pool")
        if (
            self.interference_accounting_mode
            is not InterferenceAccountingMode.EXPLICIT_PENALTY
        ):
            raise ValueError("v6 requires explicit interference accounting")
        if self.source_eviction_policy not in {
            SourceEvictionPolicy.VALUE_DENSITY_V1,
            SourceEvictionPolicy.CACHE_CRAFT_FR,
        }:
            raise ValueError("v6 requires an explicit global eviction policy")
        if self.summary_format not in {
            "exact_bf16",
            "per_head_int8",
            "block_pooled_fp16",
        }:
            raise ValueError("unsupported v6 summary format")
        if self.model_serving_mode not in {"single", "multi"}:
            raise ValueError("model_serving_mode must be single or multi")
        if not 0 < self.model_soft_quota_fraction <= 1:
            raise ValueError("model soft quota fraction must be in (0, 1]")
        if self.evidence_class != "local_simulation" and (
            self.runtime_backend != "cacheblend_multisegment_closed_loop"
        ):
            raise ValueError(
                "v6 server runs require the explicit multi-segment runtime"
            )


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        return ExperimentConfig.from_mapping(json.load(handle))
