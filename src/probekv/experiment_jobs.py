from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .backend import DeterministicSimulationBackend
from .contracts import HistoricalSource, KVLocation, SourceOrigin
from .manifest import ManifestCase


class ResultStatus(str, Enum):
    COMPLETED = "completed"
    PROCESS_CRASH = "process_crash"
    GPU_RESET = "gpu_reset"
    OOM = "oom"
    DATA_ERROR = "data_error"
    TRANSIENT_IO = "transient_io"


RETRYABLE_STATUSES = {
    ResultStatus.PROCESS_CRASH,
    ResultStatus.GPU_RESET,
    ResultStatus.TRANSIENT_IO,
}


@dataclass(frozen=True)
class E1Job:
    job_id: str
    case_id: str
    dataset: str
    split: str
    construction: str
    case_digest: str
    content_hash: str
    model_signature: str
    source_id: str
    source_context_id: str
    segment_tokens: int
    reuse_layer: int
    repair_ratio: float
    seed: int

    def validate(self, total_layers: Optional[int] = None) -> None:
        if not self.job_id or not self.case_id or not self.source_id:
            raise ValueError("job, case and source identifiers are required")
        if len(self.case_digest) != 64:
            raise ValueError("case_digest must be a SHA-256 digest")
        if self.segment_tokens <= 0:
            raise ValueError("segment_tokens must be positive")
        if self.reuse_layer < 1:
            raise ValueError("reuse_layer must be positive and 1-based")
        if total_layers is not None and self.reuse_layer >= total_layers:
            raise ValueError("reuse_layer must leave at least one layer")
        if not 0.0 <= self.repair_ratio <= 1.0:
            raise ValueError("repair_ratio must be in [0, 1]")

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "E1Job":
        result = cls(
            job_id=str(row["job_id"]),
            case_id=str(row["case_id"]),
            dataset=str(row["dataset"]),
            split=str(row["split"]),
            construction=str(row["construction"]),
            case_digest=str(row["case_digest"]),
            content_hash=str(row["content_hash"]),
            model_signature=str(row["model_signature"]),
            source_id=str(row["source_id"]),
            source_context_id=str(row["source_context_id"]),
            segment_tokens=int(row["segment_tokens"]),
            reuse_layer=int(row["reuse_layer"]),
            repair_ratio=float(row["repair_ratio"]),
            seed=int(row["seed"]),
        )
        result.validate()
        return result


def _job_id(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _case_digest(case: ManifestCase) -> str:
    payload = json.dumps(
        case.to_row(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _anchor_case_ids(
    cases: Sequence[ManifestCase], seed: int, fraction: float
) -> set:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("anchor fraction must be in [0, 1]")
    strata: Dict[Tuple[str, str, str], List[ManifestCase]] = {}
    for case in cases:
        strata.setdefault(
            (case.dataset, case.split, case.construction), []
        ).append(case)
    selected = set()
    for key, members in strata.items():
        ordered = sorted(
            members,
            key=lambda case: hashlib.sha256(
                ("%d:%s:%s" % (seed, key, case.case_id)).encode("utf-8")
            ).hexdigest(),
        )
        count = int(round(len(ordered) * fraction))
        if fraction > 0 and ordered:
            count = max(1, count)
        selected.update(case.case_id for case in ordered[:count])
    return selected


def generate_e1_jobs(
    cases: Sequence[ManifestCase],
    total_layers: int,
    repair_ratios: Sequence[float],
    seed: int = 20260726,
    include_splits: Sequence[str] = ("pilot", "train"),
    anchor_fraction: float = 0.20,
    reuse_fractions: Sequence[float] = (0.10, 0.15, 0.22, 0.30, 0.40),
) -> List[E1Job]:
    if total_layers <= 1:
        raise ValueError("total_layers must exceed one")
    ratios = tuple(sorted(set(float(value) for value in repair_ratios)))
    if not ratios or ratios[0] != 0.0 or ratios[-1] != 1.0:
        raise ValueError("repair ratio grid must include exact endpoints 0 and 1")
    selected_cases = [case for case in cases if case.split in set(include_splits)]
    anchors = _anchor_case_ids(selected_cases, seed, anchor_fraction)
    primary_layer = max(1, min(total_layers - 1, round(total_layers * 0.15)))
    all_anchor_layers = tuple(
        sorted(
            {
                max(1, min(total_layers - 1, round(total_layers * fraction)))
                for fraction in reuse_fractions
            }
        )
    )
    jobs = []
    for case in sorted(selected_cases, key=lambda item: item.case_id):
        case_digest = _case_digest(case)
        layers = all_anchor_layers if case.case_id in anchors else (primary_layer,)
        for source in case.sources:
            for layer in layers:
                for ratio in ratios:
                    payload = {
                        "case_id": case.case_id,
                        "dataset": case.dataset,
                        "split": case.split,
                        "construction": case.construction,
                        "case_digest": case_digest,
                        "content_hash": case.content_hash,
                        "segment_tokens": len(case.segment_token_ids),
                        "source_id": source.source_id,
                        "source_context_id": source.context_id,
                        "reuse_layer": layer,
                        "repair_ratio": ratio,
                        "model_signature": case.model_signature,
                        "seed": seed,
                    }
                    job = E1Job(
                        job_id=_job_id(payload),
                        case_id=case.case_id,
                        dataset=case.dataset,
                        split=case.split,
                        construction=case.construction,
                        case_digest=case_digest,
                        content_hash=case.content_hash,
                        model_signature=case.model_signature,
                        source_id=source.source_id,
                        source_context_id=source.context_id,
                        segment_tokens=len(case.segment_token_ids),
                        reuse_layer=layer,
                        repair_ratio=ratio,
                        seed=seed,
                    )
                    job.validate(total_layers)
                    jobs.append(job)
    identifiers = [job.job_id for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("E1 job identifiers collided")
    return jobs


def job_shard(job_id: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    return int(job_id[:16], 16) % shard_count


def select_job_shard(
    jobs: Sequence[E1Job], shard_index: int, shard_count: int
) -> List[E1Job]:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [job for job in jobs if job_shard(job.job_id, shard_count) == shard_index]


def group_e1_jobs(
    jobs: Sequence[E1Job],
) -> List[Tuple[Tuple[str, str, int], Tuple[E1Job, ...]]]:
    """Group work so a canonical source is staged once for an entire ratio grid."""

    grouped: Dict[Tuple[str, str, int], List[E1Job]] = {}
    for job in jobs:
        job.validate()
        key = (job.case_id, job.source_id, job.reuse_layer)
        grouped.setdefault(key, []).append(job)
    result = []
    for key, members in sorted(grouped.items()):
        ordered = tuple(sorted(members, key=lambda item: item.repair_ratio))
        ratios = [member.repair_ratio for member in ordered]
        if len(ratios) != len(set(ratios)):
            raise ValueError("group contains duplicate repair ratios")
        result.append((key, ordered))
    return result


@dataclass(frozen=True)
class E1Result:
    job_id: str
    attempt: int
    status: ResultStatus
    quality_score: Optional[float] = None
    task_score_drop: Optional[float] = None
    token_f1: Optional[float] = None
    repair_latency_ms: Optional[float] = None
    full_remaining_ms: Optional[float] = None
    requested_ratio: Optional[float] = None
    eligible_segment_tokens: Optional[int] = None
    selected_segment_tokens: Optional[int] = None
    effective_ratio: Optional[float] = None
    mandatory_suffix_tokens: Optional[int] = None
    prefix_tokens: Optional[int] = None
    selected_segment_indices: Tuple[int, ...] = ()
    reuse_start_layer: Optional[int] = None
    repair_gpu_ms: Optional[float] = None
    repair_host_ms: Optional[float] = None
    full_remaining_gpu_ms: Optional[float] = None
    full_remaining_host_ms: Optional[float] = None
    source_digest_before: Optional[str] = None
    source_digest_after: Optional[str] = None
    output_token_ids: Tuple[int, ...] = ()
    output_hash: str = ""
    output_text: str = ""
    full_output_token_ids: Tuple[int, ...] = ()
    full_output_hash: str = ""
    output_ids_exact_full: bool = False
    logit_relative_l2: Optional[float] = None
    logit_trace_mode: str = "not_recorded"
    logit_positions_compared: int = 0
    source_k_representation: str = "unspecified"
    rope_alignment_mode: str = "unspecified"
    causal_mask_mode: str = "unspecified"
    full_timing_scope: str = "remaining_layers"
    repair_timing_scope: str = "unspecified"
    timing_warmup_runs: int = 0
    timing_measurement_runs: int = 1
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    code_commit: str = ""
    environment_hash: str = ""
    model_revision: str = ""
    cacheblend_commit: str = ""
    cacheblend_patch_sha256: str = ""
    cacheblend_tree: str = ""
    vllm_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    gpu_uuid: str = ""
    finished_at_utc: str = ""
    evidence_class: str = "local_simulation"
    paper_evidence: bool = False

    def validate(self) -> None:
        if not self.job_id or self.attempt < 0:
            raise ValueError("result job_id and non-negative attempt are required")
        if self.status is ResultStatus.COMPLETED:
            required = (
                self.quality_score,
                self.task_score_drop,
                self.token_f1,
                self.repair_latency_ms,
                self.full_remaining_ms,
                self.requested_ratio,
                self.eligible_segment_tokens,
                self.selected_segment_tokens,
                self.effective_ratio,
                self.mandatory_suffix_tokens,
                self.prefix_tokens,
                self.reuse_start_layer,
                self.repair_gpu_ms,
                self.repair_host_ms,
                self.full_remaining_gpu_ms,
                self.full_remaining_host_ms,
                self.source_digest_before,
                self.source_digest_after,
            )
            if any(value is None for value in required):
                raise ValueError("completed result is missing measurements")
            if not 0.0 <= float(self.quality_score) <= 1.0:
                raise ValueError("quality_score must be in [0, 1]")
            if not 0.0 <= float(self.token_f1) <= 1.0:
                raise ValueError("token_f1 must be in [0, 1]")
            numeric = (
                self.quality_score,
                self.task_score_drop,
                self.token_f1,
                self.repair_latency_ms,
                self.full_remaining_ms,
                self.requested_ratio,
                self.effective_ratio,
                self.repair_gpu_ms,
                self.repair_host_ms,
                self.full_remaining_gpu_ms,
                self.full_remaining_host_ms,
            )
            if not all(math.isfinite(float(value)) for value in numeric):
                raise ValueError("completed measurements must be finite")
            if not -1.0 <= float(self.task_score_drop) <= 1.0:
                raise ValueError("task_score_drop must be in [-1, 1]")
            if float(self.repair_latency_ms) < 0 or float(self.full_remaining_ms) < 0:
                raise ValueError("latencies must be non-negative")
            if any(
                float(value) < 0
                for value in (
                    self.repair_gpu_ms,
                    self.repair_host_ms,
                    self.full_remaining_gpu_ms,
                    self.full_remaining_host_ms,
                )
            ):
                raise ValueError("GPU and host timings must be non-negative")
            if not 0.0 <= float(self.requested_ratio) <= 1.0:
                raise ValueError("requested_ratio must be in [0, 1]")
            if not 0.0 <= float(self.effective_ratio) <= 1.0:
                raise ValueError("effective_ratio must be in [0, 1]")
            if int(self.eligible_segment_tokens) <= 0:
                raise ValueError("eligible_segment_tokens must be positive")
            if not 0 <= int(self.selected_segment_tokens) <= int(
                self.eligible_segment_tokens
            ):
                raise ValueError("selected_segment_tokens is outside C")
            expected_effective = int(self.selected_segment_tokens) / float(
                self.eligible_segment_tokens
            )
            if abs(float(self.effective_ratio) - expected_effective) > 1e-12:
                raise ValueError("effective_ratio does not match token counts")
            if int(self.mandatory_suffix_tokens) < 0:
                raise ValueError("mandatory_suffix_tokens must be non-negative")
            if int(self.prefix_tokens) < 0:
                raise ValueError("prefix_tokens must be non-negative")
            if self.selected_segment_indices:
                if len(self.selected_segment_indices) != int(
                    self.selected_segment_tokens
                ):
                    raise ValueError(
                        "selected_segment_indices count does not match"
                    )
                lower = int(self.prefix_tokens)
                upper = lower + int(self.eligible_segment_tokens)
                if any(
                    not lower <= index < upper
                    for index in self.selected_segment_indices
                ):
                    raise ValueError(
                        "selected_segment_indices contains a token outside C"
                    )
            if int(self.reuse_start_layer) < 1:
                raise ValueError("reuse_start_layer must be 1-based")
            if self.source_digest_before != self.source_digest_after:
                raise ValueError("canonical source was mutated")
            if self.logit_relative_l2 is not None and (
                not math.isfinite(float(self.logit_relative_l2))
                or float(self.logit_relative_l2) < 0
            ):
                raise ValueError("logit_relative_l2 must be finite and non-negative")
            if not self.full_timing_scope:
                raise ValueError("full_timing_scope is required")
            if not self.repair_timing_scope:
                raise ValueError("repair_timing_scope is required")
            if self.timing_warmup_runs < 0 or self.timing_measurement_runs < 1:
                raise ValueError("invalid timing repetition counts")
            if self.logit_positions_compared < 0:
                raise ValueError("logit_positions_compared must be non-negative")
            if self.full_output_hash and self.output_ids_exact_full != (
                self.output_token_ids == self.full_output_token_ids
            ):
                raise ValueError(
                    "output_ids_exact_full does not match token IDs"
                )
        elif not self.error_type:
            raise ValueError("failed result must retain an error_type")
        if self.evidence_class not in {
            "local_simulation",
            "server_pilot",
            "paper_measurement",
        }:
            raise ValueError("unsupported result evidence_class")
        if self.evidence_class == "server_pilot":
            if self.paper_evidence:
                raise ValueError("server_pilot results can never be paper evidence")
            if self.status is ResultStatus.COMPLETED:
                required_provenance = (
                    self.output_hash,
                    self.full_output_hash,
                    self.source_k_representation,
                    self.rope_alignment_mode,
                    self.causal_mask_mode,
                    self.code_commit,
                    self.environment_hash,
                    self.model_revision,
                    self.cacheblend_commit,
                    self.cacheblend_patch_sha256,
                    self.cacheblend_tree,
                    self.vllm_version,
                    self.torch_version,
                    self.cuda_version,
                    self.gpu_uuid,
                    self.finished_at_utc,
                )
                if any(not value for value in required_provenance):
                    raise ValueError(
                        "server_pilot result is missing runtime provenance"
                    )
                if len(self.selected_segment_indices) != int(
                    self.selected_segment_tokens
                ):
                    raise ValueError(
                        "server_pilot must audit every selected C token index"
                    )
        if self.paper_evidence and (
            not self.code_commit or not self.environment_hash or not self.finished_at_utc
        ):
            raise ValueError("paper result requires code, environment and timestamp provenance")
        if self.paper_evidence and self.evidence_class != "paper_measurement":
            raise ValueError("paper evidence requires paper_measurement class")

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["status"] = self.status.value
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "E1Result":
        result = cls(
            job_id=str(row["job_id"]),
            attempt=int(row.get("attempt", 0)),
            status=ResultStatus(str(row["status"])),
            quality_score=_optional_float(row.get("quality_score")),
            task_score_drop=_optional_float(row.get("task_score_drop")),
            token_f1=_optional_float(row.get("token_f1")),
            repair_latency_ms=_optional_float(row.get("repair_latency_ms")),
            full_remaining_ms=_optional_float(row.get("full_remaining_ms")),
            requested_ratio=_optional_float(row.get("requested_ratio")),
            eligible_segment_tokens=_optional_int(row.get("eligible_segment_tokens")),
            selected_segment_tokens=_optional_int(row.get("selected_segment_tokens")),
            effective_ratio=_optional_float(row.get("effective_ratio")),
            mandatory_suffix_tokens=_optional_int(row.get("mandatory_suffix_tokens")),
            prefix_tokens=_optional_int(row.get("prefix_tokens", 0)),
            selected_segment_indices=tuple(
                int(value)
                for value in row.get("selected_segment_indices", ())
            ),
            reuse_start_layer=_optional_int(row.get("reuse_start_layer")),
            repair_gpu_ms=_optional_float(row.get("repair_gpu_ms")),
            repair_host_ms=_optional_float(row.get("repair_host_ms")),
            full_remaining_gpu_ms=_optional_float(row.get("full_remaining_gpu_ms")),
            full_remaining_host_ms=_optional_float(row.get("full_remaining_host_ms")),
            source_digest_before=_optional_string(row.get("source_digest_before")),
            source_digest_after=_optional_string(row.get("source_digest_after")),
            output_token_ids=tuple(
                int(value) for value in row.get("output_token_ids", ())
            ),
            output_hash=str(row.get("output_hash", "")),
            output_text=str(row.get("output_text", "")),
            full_output_token_ids=tuple(
                int(value) for value in row.get("full_output_token_ids", ())
            ),
            full_output_hash=str(row.get("full_output_hash", "")),
            output_ids_exact_full=bool(
                row.get("output_ids_exact_full", False)
            ),
            logit_relative_l2=_optional_float(row.get("logit_relative_l2")),
            logit_trace_mode=str(
                row.get("logit_trace_mode", "not_recorded")
            ),
            logit_positions_compared=int(
                row.get("logit_positions_compared", 0)
            ),
            source_k_representation=str(
                row.get("source_k_representation", "unspecified")
            ),
            rope_alignment_mode=str(
                row.get("rope_alignment_mode", "unspecified")
            ),
            causal_mask_mode=str(
                row.get("causal_mask_mode", "unspecified")
            ),
            full_timing_scope=str(
                row.get("full_timing_scope", "remaining_layers")
            ),
            repair_timing_scope=str(
                row.get("repair_timing_scope", "unspecified")
            ),
            timing_warmup_runs=int(row.get("timing_warmup_runs", 0)),
            timing_measurement_runs=int(
                row.get("timing_measurement_runs", 1)
            ),
            error_type=_optional_string(row.get("error_type")),
            error_message=_optional_string(row.get("error_message")),
            code_commit=str(row.get("code_commit", "")),
            environment_hash=str(row.get("environment_hash", "")),
            model_revision=str(row.get("model_revision", "")),
            cacheblend_commit=str(row.get("cacheblend_commit", "")),
            cacheblend_patch_sha256=str(
                row.get("cacheblend_patch_sha256", "")
            ),
            cacheblend_tree=str(row.get("cacheblend_tree", "")),
            vllm_version=str(row.get("vllm_version", "")),
            torch_version=str(row.get("torch_version", "")),
            cuda_version=str(row.get("cuda_version", "")),
            gpu_uuid=str(row.get("gpu_uuid", "")),
            finished_at_utc=str(row.get("finished_at_utc", "")),
            evidence_class=str(row.get("evidence_class", "local_simulation")),
            paper_evidence=bool(row.get("paper_evidence", False)),
        )
        result.validate()
        return result


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _optional_string(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def merge_e1_results(
    jobs: Sequence[E1Job], results: Sequence[E1Result]
) -> Tuple[List[E1Result], Dict[str, Any]]:
    expected = {job.job_id: job for job in jobs}
    if len(expected) != len(jobs):
        raise ValueError("duplicate expected jobs cannot be merged")
    grouped: Dict[str, List[E1Result]] = {}
    for result in results:
        result.validate()
        grouped.setdefault(result.job_id, []).append(result)
    unexpected = sorted(set(grouped) - set(expected))
    latest: List[E1Result] = []
    duplicate_attempt_rows = 0
    duplicate_job_ids = []
    duplicate_conflicts = []
    selected_attempt_by_job: Dict[str, Dict[str, Any]] = {}
    ignored_attempts: Dict[str, List[Dict[str, Any]]] = {}
    for job_id, attempts in sorted(grouped.items()):
        by_attempt: Dict[int, List[E1Result]] = {}
        for result in attempts:
            by_attempt.setdefault(result.attempt, []).append(result)
        duplicate_numbers = sorted(
            attempt for attempt, rows in by_attempt.items() if len(rows) > 1
        )
        if duplicate_numbers:
            duplicate_job_ids.append(job_id)
        for attempt in duplicate_numbers:
            rows = by_attempt[attempt]
            duplicate_attempt_rows += len(rows) - 1
            duplicate_conflicts.append(
                {
                    "job_id": job_id,
                    "attempt": attempt,
                    "rows": len(rows),
                    "reason": "duplicate attempt number is not uniquely resolvable",
                }
            )
        selected = max(
            attempts,
            key=lambda result: (
                result.attempt,
                result.finished_at_utc,
                json.dumps(result.to_row(), sort_keys=True),
            ),
        )
        latest.append(selected)
        selected_attempt_by_job[job_id] = {
            "attempt": selected.attempt,
            "status": selected.status.value,
            "finished_at_utc": selected.finished_at_utc,
        }
        ignored = [
            {
                "attempt": result.attempt,
                "status": result.status.value,
                "finished_at_utc": result.finished_at_utc,
                "reason": (
                    "older attempt"
                    if result.attempt < selected.attempt
                    else "duplicate row lost deterministic tie-break"
                ),
            }
            for result in attempts
            if result is not selected
        ]
        if ignored:
            ignored_attempts[job_id] = ignored
    latest.sort(key=lambda result: result.job_id)
    latest_map = {result.job_id: result for result in latest}
    missing = sorted(set(expected) - set(latest_map))
    resolved = sorted(set(expected) & set(latest_map))
    expected_latest = [
        latest_map[job_id] for job_id in sorted(expected) if job_id in latest_map
    ]
    failed = sorted(
        result.job_id
        for result in expected_latest
        if result.status is not ResultStatus.COMPLETED
    )
    status_counts = {
        status.value: sum(result.status is status for result in expected_latest)
        for status in ResultStatus
    }
    retryable = sorted(
        result.job_id
        for result in expected_latest
        if result.status in RETRYABLE_STATUSES
    )
    completed = [
        result
        for result in expected_latest
        if result.status is ResultStatus.COMPLETED
    ]
    provenance_fields = (
        "code_commit",
        "environment_hash",
        "model_revision",
        "cacheblend_commit",
        "cacheblend_patch_sha256",
        "cacheblend_tree",
        "gpu_uuid",
    )
    provenance_complete = bool(completed) and all(
        result.code_commit and result.environment_hash and result.finished_at_utc
        for result in completed
    )
    provenance_consistent = provenance_complete and all(
        len(
            {
                getattr(result, field)
                for result in completed
                if getattr(result, field)
            }
        )
        <= 1
        for field in provenance_fields
    )
    run_environment_valid = provenance_complete and provenance_consistent
    result_set_complete = (
        bool(expected)
        and not missing
        and not unexpected
        and not duplicate_conflicts
        and not failed
        and len(expected_latest) == len(expected)
        and all(
            result.status is ResultStatus.COMPLETED
            for result in expected_latest
        )
    )
    publication_ready = (
        run_environment_valid
        and result_set_complete
        and bool(expected_latest)
        and all(result.paper_evidence for result in expected_latest)
    )
    audit = {
        "expected_jobs": len(expected),
        "expected_job_ids": sorted(expected),
        "input_result_rows": len(results),
        "latest_result_rows": len(latest),
        "resolved_job_ids": resolved,
        "missing_job_ids": missing,
        "unexpected_job_ids": unexpected,
        "failed_job_ids": failed,
        "duplicate_job_ids": duplicate_job_ids,
        "duplicate_conflicts": duplicate_conflicts,
        "duplicate_attempt_rows": duplicate_attempt_rows,
        "selected_attempt_by_job": selected_attempt_by_job,
        "ignored_attempts": ignored_attempts,
        "status_counts": status_counts,
        "retryable_job_ids": retryable,
        "all_accounted": not missing and not unexpected and not duplicate_conflicts,
        "all_completed": result_set_complete,
        "run_environment_valid": run_environment_valid,
        "result_set_complete": result_set_complete,
        "publication_ready": publication_ready,
        "paper_evidence": publication_ready,
    }
    return latest, audit


def resumable_e1_jobs(
    jobs: Sequence[E1Job], results: Sequence[E1Result]
) -> Tuple[List[E1Job], Dict[str, int]]:
    """Return new/retryable jobs and the attempt number each must use.

    Completed jobs and terminal OOM/data failures remain recorded and are not
    silently rerun.  Process crashes, GPU resets and transient I/O failures are
    eligible for a new attempt.  Existing rows from another shard are rejected.
    """
    expected = {job.job_id: job for job in jobs}
    if len(expected) != len(jobs):
        raise ValueError("duplicate jobs cannot be resumed")
    grouped: Dict[str, List[E1Result]] = {}
    for result in results:
        result.validate()
        if result.job_id not in expected:
            raise ValueError("resume file contains a job outside this shard")
        grouped.setdefault(result.job_id, []).append(result)
    pending = []
    attempt_by_job = {}
    for job in jobs:
        attempts = grouped.get(job.job_id, [])
        if not attempts:
            pending.append(job)
            attempt_by_job[job.job_id] = 0
            continue
        attempt_numbers = [result.attempt for result in attempts]
        if len(attempt_numbers) != len(set(attempt_numbers)):
            raise ValueError("resume file contains duplicate attempt numbers")
        latest = max(
            attempts, key=lambda result: (result.attempt, result.finished_at_utc)
        )
        if latest.status in RETRYABLE_STATUSES:
            pending.append(job)
            attempt_by_job[job.job_id] = latest.attempt + 1
    return pending, attempt_by_job


def simulate_e1_results(
    jobs: Sequence[E1Job],
    total_layers: int = 32,
    attempt_by_job: Optional[Mapping[str, int]] = None,
) -> List[E1Result]:
    """Exercise the exact job/result path with non-paper deterministic data."""
    results = []
    for job in jobs:
        latent_payload = "%s:%s:%d" % (
            job.case_id,
            job.source_id,
            job.reuse_layer,
        )
        latent = int.from_bytes(
            hashlib.sha256(latent_payload.encode("utf-8")).digest()[:8], "big"
        ) / float(2 ** 64)
        threshold = 0.05 + 0.55 * latent
        backend = DeterministicSimulationBackend(
            total_layers=total_layers,
            safe_ratio_by_source={job.source_id: threshold},
        )
        source = HistoricalSource(
            job.source_id,
            job.content_hash,
            job.source_context_id,
            job.model_signature,
            job.segment_tokens,
            True,
            SourceOrigin.FULL_PREFILL,
            KVLocation.PINNED_CPU,
        )
        measurement = backend.repair(source, job.reuse_layer, job.repair_ratio)
        full_remaining = backend.full_remaining(
            job.segment_tokens, job.reuse_layer
        )
        digest = hashlib.sha256(
            (job.content_hash + job.source_context_id).encode("utf-8")
        ).hexdigest()
        output_hash = hashlib.sha256(
            ("%s:%.8f" % (job.job_id, measurement.token_f1)).encode("utf-8")
        ).hexdigest()
        results.append(
            E1Result(
                job_id=job.job_id,
                attempt=(attempt_by_job or {}).get(job.job_id, 0),
                status=ResultStatus.COMPLETED,
                quality_score=measurement.quality_score,
                task_score_drop=1.0 - measurement.quality_score,
                token_f1=measurement.token_f1,
                repair_latency_ms=measurement.latency_ms,
                full_remaining_ms=full_remaining,
                requested_ratio=measurement.requested_ratio,
                eligible_segment_tokens=measurement.eligible_segment_tokens,
                selected_segment_tokens=measurement.selected_segment_tokens,
                effective_ratio=measurement.effective_ratio,
                mandatory_suffix_tokens=measurement.mandatory_suffix_tokens,
                prefix_tokens=0,
                reuse_start_layer=measurement.reuse_start_layer,
                repair_gpu_ms=measurement.repair_gpu_ms,
                repair_host_ms=measurement.repair_host_ms,
                full_remaining_gpu_ms=full_remaining,
                full_remaining_host_ms=full_remaining,
                source_digest_before=digest,
                source_digest_after=digest,
                output_hash=output_hash,
                code_commit="local-simulation",
                environment_hash="local-simulation",
                finished_at_utc="deterministic",
                evidence_class="local_simulation",
                paper_evidence=False,
            )
        )
    return results
