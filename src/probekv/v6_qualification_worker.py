from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Sequence, Tuple

from .v6_a800_jobs import V6A800Job


@dataclass(frozen=True)
class QualificationJobResult:
    job_id: str
    passed: bool
    cuda_event_timing: bool
    gpu_ms: float
    host_ms: float
    r1_dense_token_ids_equal: bool = True
    teacher_forced_logit_relative_l2: float = 0.0
    canonical_source_digests_unchanged: bool = True
    # Added for protocol v7.  The default preserves schema compatibility with
    # historical v6 JSONL rows, which predate the one-Artifact contract.
    artifact_digests_unchanged: bool = True
    absolute_union_mask_verified: bool = True
    error: str = ""

    def __post_init__(self) -> None:
        if not self.job_id or self.gpu_ms < 0 or self.host_ms < 0:
            raise ValueError("invalid qualification result")

    def to_row(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "QualificationJobResult":
        return cls(**dict(row))


class QualificationExecutor(Protocol):
    concrete_engine_hook: bool
    adapter_name: str

    def capabilities(self) -> Mapping[str, bool]:
        ...

    def execute(self, job: V6A800Job) -> QualificationJobResult:
        ...


def dry_dispatch(jobs: Sequence[V6A800Job], adapter_name: str) -> Dict[str, Any]:
    identifiers = tuple(job.job_id for job in jobs)
    if len(jobs) != 140 or len(set(identifiers)) != 140:
        raise ValueError("qualification dry dispatch requires 140 unique jobs")
    return {
        "paper_evidence": False,
        "adapter_name": adapter_name,
        "planned": 140,
        "job_ids": identifiers,
        "executed": 0,
        "gpu_runtime_qualified": False,
    }


def dispatch_qualification(
    jobs: Sequence[V6A800Job],
    executor: QualificationExecutor,
) -> Tuple[QualificationJobResult, ...]:
    """Execute the immutable matrix; fake/non-CUDA executors cannot pass."""

    dry_dispatch(jobs, executor.adapter_name)
    if executor.concrete_engine_hook is not True:
        raise RuntimeError("qualification executor is not the concrete engine")
    required = (
        "async_multisource_loading", "layer_resumable_prefill",
        "layer_indexed_union_repair_masks", "per_segment_staggered_boundaries",
        "causal_commit_wait_execution",
        "immediate_staggered_closed_loop_execution",
        "policy_conditioned_probe_state", "cuda_event_timing",
    )
    capabilities = executor.capabilities()
    missing = [name for name in required if capabilities.get(name) is not True]
    if missing:
        raise RuntimeError("qualification executor lacks: %s" % ", ".join(missing))
    results = tuple(executor.execute(job) for job in jobs)
    validate_qualification_results(jobs, results)
    return results


def validate_qualification_results(
    jobs: Sequence[V6A800Job],
    results: Sequence[QualificationJobResult],
) -> None:
    """Validate a complete immutable result sequence, including resumed runs."""

    if tuple(result.job_id for result in results) != tuple(job.job_id for job in jobs):
        raise RuntimeError("qualification results changed immutable job order")
    for result in results:
        if not result.cuda_event_timing:
            raise RuntimeError("host/fake timing cannot qualify the GPU runtime")
        if result.gpu_ms <= 0:
            raise RuntimeError("CUDA qualification requires positive GPU time")
        if not result.passed:
            raise RuntimeError(
                "qualification job failed: %s: %s"
                % (result.job_id, result.error or "unspecified failure")
            )
        if not result.r1_dense_token_ids_equal:
            raise RuntimeError("r=1 output differs from dense reference")
        if result.teacher_forced_logit_relative_l2 > 1e-4:
            raise RuntimeError("teacher-forced logit relative-L2 exceeds 1e-4")
        if not result.canonical_source_digests_unchanged:
            raise RuntimeError("qualification mutated a canonical Source")
        if not result.artifact_digests_unchanged:
            raise RuntimeError("qualification mutated a canonical Artifact")
        if not result.absolute_union_mask_verified:
            raise RuntimeError("absolute-position union mask was not verified")
