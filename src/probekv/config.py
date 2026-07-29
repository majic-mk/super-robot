from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .contracts import CostAccountingPolicy
from .orchestration import ClosedLoopPolicy
from .scheduler import SchedulerPolicy
from .selector import SelectorPolicy, default_probe_checkpoints
from .source_store import ReplicaEvictionPolicy, SourceEvictionPolicy


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

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        total_layers = int(raw.get("total_layers", 32))
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
        if not 1 <= self.online_kmax <= 4:
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
        }:
            raise ValueError("unsupported runtime_backend")
        if self.runtime_backend == "cacheblend_closed_loop":
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
                "cacheblend_closed_loop"
            )


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        return ExperimentConfig.from_mapping(json.load(handle))
