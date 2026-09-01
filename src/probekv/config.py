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
    artifact_policy: str = "legacy_unspecified"
    max_artifacts_per_source_variant: int = 0
    canonical_kv_dtype: str = ""
    canonical_k_semantics: str = ""
    canonical_v_semantics: str = ""
    canonical_kv_lossless: bool = False
    lossy_full_kv_artifacts_enabled: bool = False
    replica_policy: str = "legacy_tier_map"
    max_replicas_per_artifact_per_tier: int = 0
    replica_tiers: Tuple[str, ...] = ()
    canonicalizer_version: str = ""
    target_tokens: int = 512
    min_tokens: int = 128
    max_tokens: int = 640
    alignment_quantum: int = 16
    search_window_tokens: int = 64
    alignment_policy: str = "soft"
    tail_policy: str = "semantic_rebalance"
    padding: bool = False
    source_selector_type: str = "legacy"
    learned_selector_enabled: bool = True
    quality_predictor_enabled: bool = True
    fixed_repair_ratio: float = 0.15
    min_compared_variants_for_multisource: int = 2
    insufficient_ranking_policy: str = "abstain_dense"
    early_exit_margin: float = 0.30
    strong_early_exit_margin: float = 0.60
    residual_band_relative_tolerance: float = 0.05
    residual_band_numeric_slack: float = 1e-6
    selector_profile_status: str = "legacy"
    selector_profile_sha256: str = ""
    selection_depth_profile_status: str = "legacy"
    selection_depth_profile_sha256: str = ""
    repair_policy_profile_status: str = "legacy"
    repair_policy_profile_sha256: str = ""
    runtime_cost_profile_status: str = "legacy"
    runtime_cost_profile_sha256: str = ""
    v8_execution_phase: str = "online_main"
    v8_schema_version: int = 5
    runtime_patch_mode: str = ""
    source_selection_metric: str = "residual_k_pre_rope"
    source_selection_depth_policy: str = "legacy_multicheckpoint"
    source_score_trim_ratio: float = 0.15
    source_score_trim_ratio_candidates: Tuple[float, ...] = (0.10, 0.15)
    repair_metric: str = "winner_v_only"
    repair_policy: str = "fixed_15"
    initial_repair_cap: float = 0.15
    repair_floor: float = 0.15
    repair_floor_candidates: Tuple[float, ...] = (0.10, 0.12, 0.15)
    repair_quality_reference_ratio: float = 0.15
    io_balance_ratio_candidates: Tuple[float, ...] = (
        0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0,
    )
    repair_reentry_policy: str = "none"
    integrity_verification_mode: str = "online_immutable"
    integrity_sampling_rate: float = 0.001
    integrity_sampling_seed: int = 20260726
    integrity_sample_layers: int = 4
    integrity_sample_rows_per_layer: int = 16
    pinned_staging_pool_bytes: int = 2 * 1024 ** 3
    gate1_gamma: float = 0.8
    backing_tier_policy: str = "legacy"
    cpu_eviction_policy: str = "legacy"
    ssd_eviction_policy: str = "legacy"
    repair_ratio_scope: str = "legacy"
    selection_runtime_fallback_policy: str = "legacy_multicheckpoint_three_gate"
    legacy_fallback_gate_required: bool = True
    variant_admission_profile_status: str = "legacy"
    variant_admission_profile_sha256: str = ""
    absolute_residual_admission_enabled: bool = False
    absolute_residual_threshold_by_depth: Tuple[Tuple[int, float], ...] = ()
    materialize_on_content_miss: bool = False
    materialize_on_complete_residual_mismatch: bool = False
    require_full_candidate_coverage_for_mismatch: bool = True
    variant_materialization_budget_fraction: float = 0.02
    variant_replacement_policy: str = "per_content_variant_lru_full_scope_only"
    variant_replacement_budget_fraction: float = 0.01
    canonical_variant_provenance: str = "dense_exact"
    source_residual_trim_ratio: float = 0.15
    source_residual_trim_ratio_candidates: Tuple[float, ...] = (
        0.10, 0.15, 0.20, 0.25, 0.30,
    )
    materialize_on_budget_truncated_exploration: bool = False
    exploration_quota_per_content: int = 0
    probation_comparison_observations: int = 2
    probation_lookup_opportunities: int = 2
    max_protected_probation_per_content: int = 2
    preparation_policy_profile_status: str = "legacy"
    preparation_policy_profile_sha256: str = ""
    gate1_mode: str = "explicit_barrier"
    atomic_preparation_reservation_required: bool = True
    final_commit_admission_required: bool = True
    legacy_source_score_trim_ratio_present: bool = False

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
            artifact_policy=str(
                raw.get("artifact_policy", "legacy_unspecified")
            ),
            max_artifacts_per_source_variant=int(
                raw.get("max_artifacts_per_source_variant", 0)
            ),
            canonical_kv_dtype=str(raw.get("canonical_kv_dtype", "")),
            canonical_k_semantics=str(
                raw.get("canonical_k_semantics", "")
            ),
            canonical_v_semantics=str(
                raw.get("canonical_v_semantics", "")
            ),
            canonical_kv_lossless=bool(
                raw.get("canonical_kv_lossless", False)
            ),
            lossy_full_kv_artifacts_enabled=bool(
                raw.get("lossy_full_kv_artifacts_enabled", False)
            ),
            replica_policy=str(raw.get("replica_policy", "legacy_tier_map")),
            max_replicas_per_artifact_per_tier=int(
                raw.get("max_replicas_per_artifact_per_tier", 0)
            ),
            replica_tiers=tuple(str(value) for value in raw.get("replica_tiers", [])),
            canonicalizer_version=str(raw.get("canonicalizer_version", "")),
            target_tokens=int(raw.get("target_tokens", 512)),
            min_tokens=int(raw.get("min_tokens", 128)),
            max_tokens=int(raw.get("max_tokens", 640)),
            alignment_quantum=int(raw.get("alignment_quantum", 16)),
            search_window_tokens=int(raw.get("search_window_tokens", 64)),
            alignment_policy=str(raw.get("alignment_policy", "soft")),
            tail_policy=str(raw.get("tail_policy", "semantic_rebalance")),
            padding=bool(raw.get("padding", False)),
            source_selector_type=str(raw.get("source_selector_type", "legacy")),
            learned_selector_enabled=bool(raw.get("learned_selector_enabled", True)),
            quality_predictor_enabled=bool(raw.get("quality_predictor_enabled", True)),
            fixed_repair_ratio=float(raw.get("fixed_repair_ratio", 0.15)),
            min_compared_variants_for_multisource=int(
                raw.get("min_compared_variants_for_multisource", 2)
            ),
            insufficient_ranking_policy=str(
                raw.get("insufficient_ranking_policy", "abstain_dense")
            ),
            early_exit_margin=float(raw.get("early_exit_margin", 0.30)),
            strong_early_exit_margin=float(
                raw.get("strong_early_exit_margin", 0.60)
            ),
            residual_band_relative_tolerance=float(
                raw.get("residual_band_relative_tolerance", 0.05)
            ),
            residual_band_numeric_slack=float(
                raw.get("residual_band_numeric_slack", 1e-6)
            ),
            selector_profile_status=str(
                raw.get("selector_profile_status", "legacy")
            ),
            selector_profile_sha256=str(
                raw.get("selector_profile_sha256", "")
            ),
            selection_depth_profile_status=str(
                raw.get("selection_depth_profile_status", "legacy")
            ),
            selection_depth_profile_sha256=str(
                raw.get("selection_depth_profile_sha256", "")
            ),
            repair_policy_profile_status=str(
                raw.get("repair_policy_profile_status", "legacy")
            ),
            repair_policy_profile_sha256=str(
                raw.get("repair_policy_profile_sha256", "")
            ),
            runtime_cost_profile_status=str(
                raw.get("runtime_cost_profile_status", "legacy")
            ),
            runtime_cost_profile_sha256=str(
                raw.get("runtime_cost_profile_sha256", "")
            ),
            v8_execution_phase=str(raw.get("v8_execution_phase", "online_main")),
            v8_schema_version=int(raw.get("v8_schema_version", 5)),
            runtime_patch_mode=str(raw.get("runtime_patch_mode", "")),
            source_selection_metric=str(
                raw.get("source_selection_metric", "residual_k_pre_rope")
            ),
            source_selection_depth_policy=str(
                raw.get("source_selection_depth_policy", "legacy_multicheckpoint")
            ),
            source_score_trim_ratio=float(raw.get("source_score_trim_ratio", 0.15)),
            source_score_trim_ratio_candidates=tuple(
                float(value)
                for value in raw.get(
                    "source_score_trim_ratio_candidates", [0.10, 0.15]
                )
            ),
            repair_metric=str(raw.get("repair_metric", "winner_v_only")),
            repair_policy=str(raw.get("repair_policy", "fixed_15")),
            initial_repair_cap=float(raw.get("initial_repair_cap", 0.15)),
            repair_floor=float(raw.get("repair_floor", 0.15)),
            repair_floor_candidates=tuple(
                float(value)
                for value in raw.get(
                    "repair_floor_candidates", [0.10, 0.12, 0.15]
                )
            ),
            repair_quality_reference_ratio=float(
                raw.get("repair_quality_reference_ratio", 0.15)
            ),
            io_balance_ratio_candidates=tuple(
                float(value)
                for value in raw.get(
                    "io_balance_ratio_candidates",
                    [0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0],
                )
            ),
            repair_reentry_policy=str(raw.get("repair_reentry_policy", "none")),
            integrity_verification_mode=str(
                raw.get("integrity_verification_mode", "online_immutable")
            ),
            integrity_sampling_rate=float(raw.get("integrity_sampling_rate", 0.001)),
            integrity_sampling_seed=int(raw.get("integrity_sampling_seed", 20260726)),
            integrity_sample_layers=int(raw.get("integrity_sample_layers", 4)),
            integrity_sample_rows_per_layer=int(
                raw.get("integrity_sample_rows_per_layer", 16)
            ),
            pinned_staging_pool_bytes=int(
                raw.get("pinned_staging_pool_bytes", 2 * 1024 ** 3)
            ),
            gate1_gamma=float(raw.get("gate1_gamma", 0.8)),
            backing_tier_policy=str(raw.get("backing_tier_policy", "legacy")),
            cpu_eviction_policy=str(raw.get("cpu_eviction_policy", "legacy")),
            ssd_eviction_policy=str(raw.get("ssd_eviction_policy", "legacy")),
            repair_ratio_scope=str(raw.get("repair_ratio_scope", "legacy")),
            selection_runtime_fallback_policy=str(
                raw.get(
                    "selection_runtime_fallback_policy",
                    "legacy_multicheckpoint_three_gate",
                )
            ),
            legacy_fallback_gate_required=bool(
                raw.get("legacy_fallback_gate_required", True)
            ),
            variant_admission_profile_status=str(
                raw.get("variant_admission_profile_status", "legacy")
            ),
            variant_admission_profile_sha256=str(
                raw.get("variant_admission_profile_sha256", "")
            ),
            absolute_residual_admission_enabled=bool(
                raw.get("absolute_residual_admission_enabled", False)
            ),
            absolute_residual_threshold_by_depth=tuple(
                (int(depth), float(value))
                for depth, value in raw.get(
                    "absolute_residual_threshold_by_depth", []
                )
            ),
            materialize_on_content_miss=bool(
                raw.get("materialize_on_content_miss", False)
            ),
            materialize_on_complete_residual_mismatch=bool(
                raw.get("materialize_on_complete_residual_mismatch", False)
            ),
            require_full_candidate_coverage_for_mismatch=bool(
                raw.get("require_full_candidate_coverage_for_mismatch", True)
            ),
            variant_materialization_budget_fraction=float(
                raw.get("variant_materialization_budget_fraction", 0.02)
            ),
            variant_replacement_policy=str(
                raw.get(
                    "variant_replacement_policy",
                    "per_content_variant_lru_full_scope_only",
                )
            ),
            variant_replacement_budget_fraction=float(
                raw.get("variant_replacement_budget_fraction", 0.01)
            ),
            canonical_variant_provenance=str(
                raw.get("canonical_variant_provenance", "dense_exact")
            ),
            source_residual_trim_ratio=float(
                raw.get("source_residual_trim_ratio", 0.15)
            ),
            source_residual_trim_ratio_candidates=tuple(
                float(value)
                for value in raw.get(
                    "source_residual_trim_ratio_candidates",
                    [0.10, 0.15, 0.20, 0.25, 0.30],
                )
            ),
            materialize_on_budget_truncated_exploration=bool(
                raw.get("materialize_on_budget_truncated_exploration", False)
            ),
            exploration_quota_per_content=int(
                raw.get("exploration_quota_per_content", 0)
            ),
            probation_comparison_observations=int(
                raw.get("probation_comparison_observations", 2)
            ),
            probation_lookup_opportunities=int(
                raw.get("probation_lookup_opportunities", 2)
            ),
            max_protected_probation_per_content=int(
                raw.get("max_protected_probation_per_content", 2)
            ),
            preparation_policy_profile_status=str(
                raw.get("preparation_policy_profile_status", "legacy")
            ),
            preparation_policy_profile_sha256=str(
                raw.get("preparation_policy_profile_sha256", "")
            ),
            gate1_mode=str(raw.get("gate1_mode", "explicit_barrier")),
            atomic_preparation_reservation_required=bool(
                raw.get("atomic_preparation_reservation_required", True)
            ),
            final_commit_admission_required=bool(
                raw.get("final_commit_admission_required", True)
            ),
            legacy_source_score_trim_ratio_present=(
                "source_score_trim_ratio" in raw
                or "source_score_trim_ratio_candidates" in raw
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
        elif self.protocol_version == 7:
            self._validate_v7()
        elif self.protocol_version == 8:
            self._validate_v8()
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
            "cacheblend_v7_closed_loop",
            "cacheblend_v8_training_free",
            "cacheblend_v8_schema6_joint",
            "cacheblend_v8_schema7_gradual_streaming",
            "cacheblend_v8_schema8_gradual_barrier",
        }:
            raise ValueError("unsupported runtime_backend")
        if self.runtime_backend in {
            "cacheblend_closed_loop",
            "cacheblend_multisegment_closed_loop",
            "cacheblend_v7_closed_loop",
            "cacheblend_v8_training_free",
            "cacheblend_v8_schema6_joint",
            "cacheblend_v8_schema7_gradual_streaming",
            "cacheblend_v8_schema8_gradual_barrier",
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

    def _validate_v7(self) -> None:
        if self.legacy_online_kmax_present:
            raise ValueError("v7 forbids legacy online_kmax")
        if self.max_stored_variants_per_content != 16:
            raise ValueError("v7 stores at most 16 variants per content")
        if self.min_variants_per_retained_content != 1:
            raise ValueError("v7 retained content minimum must be one variant")
        if not 1 <= self.max_compared_variants_per_segment <= 16:
            raise ValueError("v7 comparison maximum must be in [1, 16]")
        if self.candidate_compare_policy != "all_within_request_budget":
            raise ValueError("v7 requires budgeted all-candidate comparison")
        if self.probe_compare_budget_fraction != 0.05:
            raise ValueError("v7 freezes the probe/compare budget at 5%")
        if self.segment_planning_policy != "all_exact_nonprefix":
            raise ValueError("v7 must plan every exact non-prefix Segment")
        if self.max_detected_segments is not None:
            raise ValueError("v7 cannot impose a detected-Segment cap")
        if self.boundary_policy != "per_segment_staggered":
            raise ValueError("v7 requires per-Segment staggered boundaries")
        if self.selection_execution_policy not in {
            SelectionExecutionPolicy.CAUSAL_COMMIT_WAIT,
            SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP,
        }:
            raise ValueError("v7 supports only the frozen A/C policies")
        if not self.calibration_policy_match_required:
            raise ValueError("v7 A/C policies require matched calibration")
        if not self.partial_reuse_enabled:
            raise ValueError("v7 requires partial reuse")
        if self.joint_quality_policy != "simultaneous_conformal":
            raise ValueError("v7 requires simultaneous request-level quality")
        if self.source_pool_policy != "global_hard_model_soft":
            raise ValueError("v7 requires the global Source pool")
        if self.artifact_policy != "single_canonical_lossless":
            raise ValueError("v7 requires one canonical lossless Artifact")
        if self.max_artifacts_per_source_variant != 1:
            raise ValueError("v7 permits exactly one Artifact per Source Variant")
        if (
            self.canonical_kv_dtype != "bfloat16"
            or self.canonical_k_semantics != "pre_rope"
            or self.canonical_v_semantics != "raw"
            or not self.canonical_kv_lossless
            or self.lossy_full_kv_artifacts_enabled
        ):
            raise ValueError("v7 canonical Artifact must be lossless BF16 pre-RoPE")
        if self.replica_policy != "one_backing_plus_transient_hot":
            raise ValueError("v7 requires the frozen Replica policy")
        if self.max_replicas_per_artifact_per_tier != 1:
            raise ValueError("v7 permits one Replica per Artifact per tier")
        if tuple(self.replica_tiers) != ("gpu", "pinned_cpu", "ssd"):
            raise ValueError("v7 Replica tiers must be gpu/pinned_cpu/ssd")
        if self.canonicalizer_version != "semantic_block_v1":
            raise ValueError("v7 requires semantic_block_v1")
        if (self.target_tokens, self.min_tokens, self.max_tokens) != (512, 128, 640):
            raise ValueError("v7 canonical token defaults changed")
        if self.alignment_quantum != 16 or self.search_window_tokens != 64:
            raise ValueError("v7 canonical alignment defaults changed")
        if (
            self.alignment_policy != "soft"
            or self.tail_policy != "semantic_rebalance"
            or self.padding
        ):
            raise ValueError("v7 canonical semantic policy changed")
        if (
            self.interference_accounting_mode
            is not InterferenceAccountingMode.EXPLICIT_PENALTY
        ):
            raise ValueError("v7 requires explicit interference accounting")
        if self.evidence_class != "local_simulation" and (
            self.runtime_backend != "cacheblend_v7_closed_loop"
        ):
            raise ValueError("v7 server runs require the explicit v7 runtime")

    def _validate_v8(self) -> None:
        if self.v8_schema_version not in {5, 6, 7, 8, 9, 10}:
            raise ValueError("v8 runtime schema must be 5, 6, 7, 8, 9 or 10")
        if self.legacy_online_kmax_present:
            raise ValueError("v8 forbids legacy online_kmax")
        if self.selector_policy is not SelectorPolicy.RESIDUAL_K_DRIFT_ARGMIN:
            raise ValueError("v8 requires training-free residual-K selection")
        if self.source_selector_type != "training_free":
            raise ValueError("v8 Source selector must be training_free")
        if self.learned_selector_enabled or self.quality_predictor_enabled:
            raise ValueError("v8 forbids learned selectors and quality predictors")
        if self.fixed_repair_ratio != 0.15:
            raise ValueError("v8 freezes the main online repair ratio at 0.15")
        diagnostic_grid = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0)
        if self.v8_execution_phase == "online_main":
            if tuple(self.repair_ratios) != (0.15,):
                raise ValueError("v8 online main path uses only repair ratio 0.15")
        elif self.v8_execution_phase == "h1_offline_diagnostic":
            if tuple(self.repair_ratios) != diagnostic_grid:
                raise ValueError("v8 H1 diagnostic requires the frozen endpoint grid")
        else:
            raise ValueError("unsupported v8 execution phase")
        if self.max_stored_variants_per_content != 16:
            raise ValueError("v8 stores at most 16 variants per content")
        if self.min_variants_per_retained_content != 1:
            raise ValueError("v8 retained content minimum must be one variant")
        if not 1 <= self.max_compared_variants_per_segment <= 16:
            raise ValueError("v8 comparison maximum must be in [1, 16]")
        if self.min_compared_variants_for_multisource != 2:
            raise ValueError("v8 multi-Source ranking requires two comparisons")
        if self.insufficient_ranking_policy not in {
            "abstain_dense", "cfo_top1_fallback"
        }:
            raise ValueError("unsupported v8 insufficient-ranking policy")
        if self.candidate_compare_policy != "all_within_request_budget":
            raise ValueError("v8 requires budgeted candidate comparison")
        if self.probe_compare_budget_fraction != 0.05:
            raise ValueError("v8 freezes the selection budget at 5%")
        if not 0 <= self.early_exit_margin <= self.strong_early_exit_margin <= 1:
            raise ValueError("invalid v8 early-exit margins")
        if not 0 <= self.residual_band_relative_tolerance <= 1:
            raise ValueError("invalid residual-band tolerance")
        if self.residual_band_numeric_slack != 1e-6:
            raise ValueError("v8 freezes the residual numeric slack")
        if self.segment_planning_policy != "all_exact_nonprefix":
            raise ValueError("v8 must plan every exact non-prefix Segment")
        if self.max_detected_segments is not None:
            raise ValueError("v8 cannot impose a detected-Segment cap")
        if self.boundary_policy != "per_segment_staggered":
            raise ValueError("v8 requires per-Segment staggered boundaries")
        allowed_execution_policies = {
            SelectionExecutionPolicy.CAUSAL_COMMIT_WAIT,
            SelectionExecutionPolicy.IMMEDIATE_STAGGERED_CLOSED_LOOP,
        }
        if self.v8_schema_version in {8, 9, 10}:
            allowed_execution_policies.add(
                SelectionExecutionPolicy.DENSE_SELECTION_BARRIER
            )
        if self.selection_execution_policy not in allowed_execution_policies:
            raise ValueError("unsupported v8 selection-execution policy")
        if self.calibration_policy_match_required:
            raise ValueError("v8 has no online calibration eligibility")
        if self.joint_quality_policy != "fixed_repair_offline_audit":
            raise ValueError("v8 requires fixed-repair offline quality audit")
        if not self.partial_reuse_enabled:
            raise ValueError("v8 requires partial reuse")
        if self.source_pool_policy != "global_hard_model_soft":
            raise ValueError("v8 requires the global Source pool")
        if self.summary_format != "selection_k_bf16":
            raise ValueError("v8 compares exact BF16 selection-layer K states")
        if self.artifact_policy != "single_canonical_lossless":
            raise ValueError("v8 requires one canonical lossless Artifact")
        if self.max_artifacts_per_source_variant != 1:
            raise ValueError("v8 permits one Artifact per Source Variant")
        if (
            self.canonical_kv_dtype != "bfloat16"
            or self.canonical_k_semantics != "pre_rope"
            or self.canonical_v_semantics != "raw"
            or not self.canonical_kv_lossless
            or self.lossy_full_kv_artifacts_enabled
        ):
            raise ValueError("v8 canonical Artifact must be lossless BF16 pre-RoPE")
        if self.replica_policy != "one_backing_plus_transient_hot":
            raise ValueError("v8 requires one backing plus transient hot Replicas")
        if self.max_replicas_per_artifact_per_tier != 1:
            raise ValueError("v8 permits one Replica per tier")
        if tuple(self.replica_tiers) != ("gpu", "pinned_cpu", "ssd"):
            raise ValueError("v8 Replica tiers must be gpu/pinned_cpu/ssd")
        if self.canonicalizer_version != "semantic_block_v1":
            raise ValueError("v8 requires semantic_block_v1")
        if (self.target_tokens, self.min_tokens, self.max_tokens) != (512, 128, 640):
            raise ValueError("v8 canonical token defaults changed")
        if self.alignment_quantum != 16 or self.search_window_tokens != 64:
            raise ValueError("v8 canonical alignment defaults changed")
        if self.alignment_policy != "soft" or self.tail_policy != "semantic_rebalance" or self.padding:
            raise ValueError("v8 canonical semantic policy changed")
        if self.interference_accounting_mode is not InterferenceAccountingMode.EXPLICIT_PENALTY:
            raise ValueError("v8 requires explicit interference accounting")
        if self.v8_schema_version < 8:
            if self.selector_profile_status not in {
                "local_test_only", "unfrozen", "frozen"
            }:
                raise ValueError("invalid v8 selector profile status")
            if self.selector_profile_status == "frozen" and not self.selector_profile_sha256:
                raise ValueError("a frozen v8 profile requires a SHA256")
        elif self.selector_profile_status != "legacy" or self.selector_profile_sha256:
            raise ValueError("schema-v8+ cannot reuse the legacy SelectorProfile fields")
        if self.v8_schema_version == 7:
            self._validate_v8_schema7()
        if self.v8_schema_version == 8:
            self._validate_v8_schema8()
        if self.v8_schema_version == 9:
            self._validate_v8_schema9()
        if self.v8_schema_version == 10:
            self._validate_v8_schema10()
        if self.evidence_class != "local_simulation":
            required_backend = (
                "cacheblend_v8_schema7_gradual_streaming"
                if self.v8_schema_version == 7
                else (
                    "cacheblend_v8_schema10_variant_growth_counterfactual"
                    if self.v8_schema_version == 10
                    else (
                        "cacheblend_v8_schema9_absolute_variant_admission"
                        if self.v8_schema_version == 9
                        else (
                            "cacheblend_v8_schema8_gradual_barrier"
                            if self.v8_schema_version == 8
                            else (
                                "cacheblend_v8_schema6_joint"
                                if self.v8_schema_version == 6
                                else "cacheblend_v8_training_free"
                            )
                        )
                    )
                )
            )
            if self.runtime_backend != required_backend:
                raise ValueError(
                    "v8 server runs require the explicit schema-selected runtime"
                )

    def _validate_v8_schema7(self) -> None:
        if self.runtime_patch_mode != "probekv_v8_winner_gradual_streaming":
            raise ValueError("schema-v7 requires its explicit runtime patch mode")
        if self.source_selection_metric != "residual_k_pre_rope":
            raise ValueError("schema-v7 Source selection must use pre-RoPE Residual-K")
        depth_policies = {
            "d1_only",
            "d1_d2_rescue",
            "legacy_multicheckpoint",
            "deep_full_candidate_oracle",
        }
        if self.source_selection_depth_policy not in depth_policies:
            raise ValueError("unsupported schema-v7 Source-depth policy")
        expected_checkpoints = {
            "d1_only": (1,),
            "d1_d2_rescue": (1, 2),
            "legacy_multicheckpoint": (
                (1, 2, 4, 5, 8) if self.total_layers == 32 else (1, 2, 4, 5, 7)
            ),
        }
        expected = expected_checkpoints.get(self.source_selection_depth_policy)
        if expected is not None and self.probe_checkpoints != expected:
            raise ValueError("schema-v7 checkpoints differ from the selected depth policy")
        if tuple(self.source_score_trim_ratio_candidates) != (0.10, 0.15):
            raise ValueError("schema-v7 Source-score trim candidates changed")
        if self.source_score_trim_ratio not in self.source_score_trim_ratio_candidates:
            raise ValueError("Source-score trim ratio is outside its candidate set")
        if self.repair_metric not in {
            "winner_k_only", "winner_v_only", "winner_kv_normalized"
        }:
            raise ValueError("unsupported schema-v7 winner repair metric")
        if self.repair_policy not in {
            "fixed_15", "static_gradual", "load_recompute_aware_gradual"
        }:
            raise ValueError("unsupported schema-v7 repair policy")
        if self.initial_repair_cap != 0.15:
            raise ValueError("schema-v7 initial repair cap must remain 0.15")
        if tuple(self.repair_floor_candidates) != (0.10, 0.12, 0.15):
            raise ValueError("schema-v7 repair-floor candidates changed")
        if self.repair_floor not in self.repair_floor_candidates:
            raise ValueError("schema-v7 repair floor is outside its candidate set")
        if not self.repair_floor <= self.initial_repair_cap:
            raise ValueError("repair floor exceeds the initial cap")
        if self.repair_reentry_policy != "none":
            raise ValueError("schema-v7 main candidates forbid repair-support re-entry")
        if self.integrity_verification_mode not in {
            "qualification_full", "online_immutable", "online_sampled"
        }:
            raise ValueError("unsupported schema-v7 integrity mode")
        if not 0 <= self.integrity_sampling_rate <= 1:
            raise ValueError("integrity sampling rate must be in [0, 1]")
        if min(
            self.integrity_sampling_seed,
            self.integrity_sample_layers,
            self.integrity_sample_rows_per_layer,
            self.pinned_staging_pool_bytes,
        ) <= 0:
            raise ValueError("schema-v7 integrity/staging settings must be positive")

    def _validate_v8_schema8(self) -> None:
        if self.runtime_patch_mode != "probekv_v8_gradual_barrier_tiered_lru":
            raise ValueError("schema-v8 requires its explicit runtime patch mode")
        if self.source_selection_metric != "residual_k_pre_rope":
            raise ValueError("schema-v8 Source selection must use pre-RoPE Residual-K")
        if self.source_selection_depth_policy not in {"d1_only", "d1_d2_rescue"}:
            raise ValueError("schema-v8 online main path is limited to d1/d2")
        expected = (
            (1,)
            if self.source_selection_depth_policy == "d1_only"
            else (1, 2)
        )
        if self.probe_checkpoints != expected:
            raise ValueError("schema-v8 checkpoints differ from its d1/d2 policy")
        if (
            self.selection_execution_policy
            is not SelectionExecutionPolicy.DENSE_SELECTION_BARRIER
        ):
            raise ValueError("schema-v8 requires one dense selection barrier")
        if self.gate1_gamma != 1.0:
            raise ValueError("schema-v8 Gate1 must reject only non-positive savings")
        if self.gamma != 0.8:
            raise ValueError("schema-v8 FinalCommitAdmission freezes gamma at 0.8")
        if self.backing_tier_policy != "cpu_preferred_single_backing":
            raise ValueError("schema-v8 requires CPU-preferred single backing")
        if self.cpu_eviction_policy != "exclusive_backing_lru":
            raise ValueError("schema-v8 CPU eviction must use exclusive-backing LRU")
        if self.ssd_eviction_policy != "exclusive_backing_lru":
            raise ValueError("schema-v8 SSD eviction must use exclusive-backing LRU")
        if (
            self.selection_runtime_fallback_policy
            != "legacy_multicheckpoint_three_gate"
            or self.legacy_fallback_gate_required is not True
        ):
            raise ValueError(
                "schema-v8 must preserve the qualified legacy multi-checkpoint fallback"
            )
        expected_scope = {
            "fixed_15": "uniform_fixed",
            "static_gradual": "shared_relative_schedule",
            "load_recompute_aware_gradual": "per_segment_load_aware",
            "load_recompute_aware_uniform": "request_layer_uniform_io_balanced",
        }.get(self.repair_policy)
        if expected_scope is None or self.repair_ratio_scope != expected_scope:
            raise ValueError("schema-v8 repair policy and ratio scope disagree")
        if self.initial_repair_cap != 0.15:
            raise ValueError(
                "schema-v8 legacy initial-cap field must remain 0.15; "
                "the uniform I/O policy uses repair_quality_reference_ratio instead"
            )
        if self.repair_quality_reference_ratio != 0.15:
            raise ValueError("schema-v8 quality reference ratio must remain 0.15")
        io_grid = tuple(self.io_balance_ratio_candidates)
        if (
            not io_grid
            or tuple(sorted(set(io_grid))) != io_grid
            or io_grid[0] <= 0
            or io_grid[-1] > 1.0
        ):
            raise ValueError("schema-v8 I/O balance ratio grid is invalid")
        if self.repair_policy == "load_recompute_aware_uniform" and (
            self.repair_floor not in io_grid
            or self.repair_quality_reference_ratio not in io_grid
        ):
            raise ValueError("uniform I/O policy grid must contain floor/reference")
        if self.repair_reentry_policy != "none":
            raise ValueError("schema-v8 forbids repair-support re-entry")
        if self.integrity_verification_mode not in {
            "qualification_full", "online_immutable", "online_sampled"
        }:
            raise ValueError("schema-v8 integrity mode is invalid")
        if self.pinned_staging_pool_bytes <= 0:
            raise ValueError("schema-v8 pinned staging pool must be positive")
        profile_pairs = (
            (
                "selection-depth",
                self.selection_depth_profile_status,
                self.selection_depth_profile_sha256,
            ),
            (
                "repair-policy",
                self.repair_policy_profile_status,
                self.repair_policy_profile_sha256,
            ),
            (
                "runtime-cost",
                self.runtime_cost_profile_status,
                self.runtime_cost_profile_sha256,
            ),
        )
        for name, status, sha256 in profile_pairs:
            if status not in {"unfrozen", "frozen"}:
                raise ValueError("invalid schema-v8 %s Profile status" % name)
            if status == "frozen" and len(sha256) != 64:
                raise ValueError("frozen schema-v8 %s Profile requires SHA256" % name)
            if status != "frozen" and sha256:
                raise ValueError("unfrozen schema-v8 %s Profile cannot carry SHA" % name)
        if (
            self.repair_policy in {
                "load_recompute_aware_gradual",
                "load_recompute_aware_uniform",
            }
            and self.v8_execution_phase == "online_main"
            and self.repair_policy_profile_status != "frozen"
        ):
            raise ValueError("adaptive schema-v8 repair requires a frozen RepairPolicyProfile")

    def _validate_v8_schema9(self) -> None:
        if (
            self.runtime_patch_mode
            != "probekv_v8_absolute_residual_variant_admission"
        ):
            raise ValueError("schema9 requires its explicit runtime patch mode")
        if self.source_selection_metric != "residual_k_pre_rope":
            raise ValueError("schema9 Source selection must use pre-RoPE Residual-K")
        if self.source_selection_depth_policy not in {"d1_only", "d1_d2_rescue"}:
            raise ValueError("schema9 online main path is limited to d1/d2")
        expected = (
            (1,)
            if self.source_selection_depth_policy == "d1_only"
            else (1, 2)
        )
        if self.probe_checkpoints != expected:
            raise ValueError("schema9 checkpoints differ from its d1/d2 policy")
        if (
            self.selection_execution_policy
            is not SelectionExecutionPolicy.DENSE_SELECTION_BARRIER
        ):
            raise ValueError("schema9 preserves the qualified dense selection barrier")
        if not self.absolute_residual_admission_enabled:
            raise ValueError("schema9 requires absolute residual admission")
        thresholds = tuple(self.absolute_residual_threshold_by_depth)
        if tuple(depth for depth, _ in thresholds) != tuple(
            sorted(set(depth for depth, _ in thresholds))
        ) or any(depth not in {1, 2} or value < 0 for depth, value in thresholds):
            raise ValueError("schema9 absolute residual thresholds are invalid")
        if set(depth for depth, _ in thresholds) != set(expected):
            raise ValueError("schema9 needs one threshold per online depth")
        if tuple(self.source_score_trim_ratio_candidates) != (
            0.0, 0.10, 0.15, 0.30, 0.40
        ):
            raise ValueError("schema9 Source trim development grid changed")
        if self.source_score_trim_ratio not in self.source_score_trim_ratio_candidates:
            raise ValueError("schema9 Source trim ratio is outside the grid")
        if not (
            self.materialize_on_content_miss
            and self.materialize_on_complete_residual_mismatch
            and self.require_full_candidate_coverage_for_mismatch
        ):
            raise ValueError("schema9 exact-dense materialization guards changed")
        if not 0 <= self.variant_materialization_budget_fraction <= 0.05:
            raise ValueError("schema9 materialization budget must be within 5%")
        if self.canonical_variant_provenance != "dense_exact":
            raise ValueError("schema9 canonical Variant must come from exact dense prefill")
        if self.variant_admission_profile_status not in {"unfrozen", "frozen"}:
            raise ValueError("invalid schema9 VariantAdmissionProfile status")
        if self.variant_admission_profile_status == "frozen":
            if len(self.variant_admission_profile_sha256) != 64:
                raise ValueError("frozen VariantAdmissionProfile requires SHA256")
        elif self.variant_admission_profile_sha256:
            raise ValueError("unfrozen VariantAdmissionProfile cannot carry SHA")
        if self.gate1_gamma != 1.0 or self.gamma != 0.8:
            raise ValueError("schema9 freezes Gate1=1.0 and final admission=0.8")
        if (
            self.selection_runtime_fallback_policy
            != "legacy_multicheckpoint_three_gate"
            or not self.legacy_fallback_gate_required
        ):
            raise ValueError("schema9 must preserve the qualified schema8 fallback")
        expected_scope = {
            "fixed_15": "uniform_fixed",
            "static_gradual": "shared_relative_schedule",
            "load_recompute_aware_gradual": "per_segment_load_aware",
            "load_recompute_aware_uniform": "request_layer_uniform_io_balanced",
        }.get(self.repair_policy)
        if expected_scope is None or self.repair_ratio_scope != expected_scope:
            raise ValueError("schema9 repair policy and ratio scope disagree")
        for name, status, sha256 in (
            (
                "selection-depth",
                self.selection_depth_profile_status,
                self.selection_depth_profile_sha256,
            ),
            (
                "repair-policy",
                self.repair_policy_profile_status,
                self.repair_policy_profile_sha256,
            ),
            (
                "runtime-cost",
                self.runtime_cost_profile_status,
                self.runtime_cost_profile_sha256,
            ),
        ):
            if status not in {"unfrozen", "frozen"}:
                raise ValueError("invalid schema9 %s Profile status" % name)
            if status == "frozen" and len(sha256) != 64:
                raise ValueError("frozen schema9 %s Profile requires SHA256" % name)
            if status != "frozen" and sha256:
                raise ValueError("unfrozen schema9 %s Profile cannot carry SHA" % name)

    def _validate_v8_schema10(self) -> None:
        if self.runtime_patch_mode != "probekv_v8_variant_growth_counterfactual":
            raise ValueError("schema10 requires its explicit runtime patch mode")
        if self.source_selection_metric != "residual_k_pre_rope":
            raise ValueError("schema10 Source selection must use pre-RoPE Residual-K")
        if self.source_selection_depth_policy not in {"d1_only", "d1_d2_rescue"}:
            raise ValueError("schema10 online main path is limited to d1/d2")
        expected = (1,) if self.source_selection_depth_policy == "d1_only" else (1, 2)
        if self.probe_checkpoints != expected:
            raise ValueError("schema10 checkpoints differ from its d1/d2 policy")
        if (
            self.selection_execution_policy
            is not SelectionExecutionPolicy.DENSE_SELECTION_BARRIER
        ):
            raise ValueError("schema10 preserves the dense selection barrier")
        if not self.absolute_residual_admission_enabled:
            raise ValueError("schema10 requires absolute residual admission")
        thresholds = tuple(self.absolute_residual_threshold_by_depth)
        if set(depth for depth, _ in thresholds) != set(expected) or any(
            depth not in {1, 2} or value < 0 for depth, value in thresholds
        ):
            raise ValueError("schema10 absolute residual thresholds are invalid")
        expected_grid = (0.10, 0.15, 0.20, 0.25, 0.30)
        if self.legacy_source_score_trim_ratio_present:
            raise ValueError("schema10 rejects schema9 source_score_trim_ratio fields")
        if tuple(self.source_residual_trim_ratio_candidates) != expected_grid:
            raise ValueError("schema10 residual trim grid changed")
        if self.source_residual_trim_ratio not in expected_grid:
            raise ValueError("schema10 residual trim ratio is outside its grid")
        if not (
            self.materialize_on_content_miss
            and self.materialize_on_complete_residual_mismatch
            and self.require_full_candidate_coverage_for_mismatch
            and self.materialize_on_budget_truncated_exploration
        ):
            raise ValueError("schema10 materialization policy is incomplete")
        if not 0 <= self.variant_materialization_budget_fraction <= 0.05:
            raise ValueError("schema10 materialization budget must be within 5%")
        if (
            self.variant_replacement_policy
            != "per_content_variant_lru_full_scope_only"
            or not 0 <= self.variant_replacement_budget_fraction <= 0.05
        ):
            raise ValueError("schema10 replacement policy/budget is invalid")
        if self.canonical_variant_provenance != "dense_exact":
            raise ValueError("schema10 canonical Variant must be exact dense")
        if not 0 < self.exploration_quota_per_content <= 16:
            raise ValueError("schema10 exploration quota is invalid")
        if min(
            self.probation_comparison_observations,
            self.probation_lookup_opportunities,
            self.max_protected_probation_per_content,
        ) <= 0:
            raise ValueError("schema10 probation controls must be positive")
        if self.gate1_mode not in {"explicit_barrier", "fused_advisory"}:
            raise ValueError("schema10 Gate1 mode is invalid")
        if not (
            self.atomic_preparation_reservation_required
            and self.final_commit_admission_required
        ):
            raise ValueError("schema10 cannot bypass reservation/final admission")
        if self.gate1_gamma != 1.0 or self.gamma != 0.8:
            raise ValueError("schema10 freezes Gate1=1.0 and final admission=0.8")
        if (
            self.selection_runtime_fallback_policy
            != "legacy_multicheckpoint_three_gate"
            or not self.legacy_fallback_gate_required
        ):
            raise ValueError("schema10 must preserve the schema8 legacy fallback")
        for name, status, sha256 in (
            ("variant-admission", self.variant_admission_profile_status, self.variant_admission_profile_sha256),
            ("preparation-policy", self.preparation_policy_profile_status, self.preparation_policy_profile_sha256),
            ("selection-depth", self.selection_depth_profile_status, self.selection_depth_profile_sha256),
            ("repair-policy", self.repair_policy_profile_status, self.repair_policy_profile_sha256),
            ("runtime-cost", self.runtime_cost_profile_status, self.runtime_cost_profile_sha256),
        ):
            if status not in {"unfrozen", "frozen"}:
                raise ValueError("invalid schema10 %s Profile status" % name)
            if status == "frozen" and len(sha256) != 64:
                raise ValueError("frozen schema10 %s Profile requires SHA256" % name)
            if status != "frozen" and sha256:
                raise ValueError("unfrozen schema10 %s Profile cannot carry SHA" % name)


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        return ExperimentConfig.from_mapping(json.load(handle))
