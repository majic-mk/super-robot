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
    source_digest_before: Optional[str] = None
    source_digest_after: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    code_commit: str = ""
    environment_hash: str = ""
    finished_at_utc: str = ""
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
            )
            if not all(math.isfinite(float(value)) for value in numeric):
                raise ValueError("completed measurements must be finite")
            if not -1.0 <= float(self.task_score_drop) <= 1.0:
                raise ValueError("task_score_drop must be in [-1, 1]")
            if float(self.repair_latency_ms) < 0 or float(self.full_remaining_ms) < 0:
                raise ValueError("latencies must be non-negative")
            if self.source_digest_before != self.source_digest_after:
                raise ValueError("canonical source was mutated")
        elif not self.error_type:
            raise ValueError("failed result must retain an error_type")
        if self.paper_evidence and (
            not self.code_commit or not self.environment_hash or not self.finished_at_utc
        ):
            raise ValueError("paper result requires code, environment and timestamp provenance")

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
            source_digest_before=_optional_string(row.get("source_digest_before")),
            source_digest_after=_optional_string(row.get("source_digest_after")),
            error_type=_optional_string(row.get("error_type")),
            error_message=_optional_string(row.get("error_message")),
            code_commit=str(row.get("code_commit", "")),
            environment_hash=str(row.get("environment_hash", "")),
            finished_at_utc=str(row.get("finished_at_utc", "")),
            paper_evidence=bool(row.get("paper_evidence", False)),
        )
        result.validate()
        return result


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _optional_string(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def merge_e1_results(
    jobs: Sequence[E1Job], results: Sequence[E1Result]
) -> Tuple[List[E1Result], Dict[str, Any]]:
    expected = {job.job_id: job for job in jobs}
    grouped: Dict[str, List[E1Result]] = {}
    for result in results:
        result.validate()
        grouped.setdefault(result.job_id, []).append(result)
    unexpected = sorted(set(grouped) - set(expected))
    latest = []
    duplicate_attempt_rows = 0
    for job_id, attempts in grouped.items():
        attempt_numbers = [result.attempt for result in attempts]
        duplicate_attempt_rows += len(attempt_numbers) - len(set(attempt_numbers))
        latest.append(
            max(attempts, key=lambda result: (result.attempt, result.finished_at_utc))
        )
    latest.sort(key=lambda result: result.job_id)
    latest_map = {result.job_id: result for result in latest}
    missing = sorted(set(expected) - set(latest_map))
    status_counts = {
        status.value: sum(result.status is status for result in latest)
        for status in ResultStatus
    }
    retryable = sorted(
        result.job_id
        for result in latest
        if result.status in RETRYABLE_STATUSES
    )
    audit = {
        "expected_jobs": len(expected),
        "input_result_rows": len(results),
        "latest_result_rows": len(latest),
        "missing_job_ids": missing,
        "unexpected_job_ids": unexpected,
        "duplicate_attempt_rows": duplicate_attempt_rows,
        "status_counts": status_counts,
        "retryable_job_ids": retryable,
        "all_accounted": not missing and not unexpected and duplicate_attempt_rows == 0,
        "all_completed": (
            not missing
            and not unexpected
            and duplicate_attempt_rows == 0
            and all(result.status is ResultStatus.COMPLETED for result in latest)
        ),
        "paper_evidence": bool(latest)
        and all(result.paper_evidence for result in latest),
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
        digest = hashlib.sha256(
            (job.content_hash + job.source_context_id).encode("utf-8")
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
                full_remaining_ms=backend.full_remaining(
                    job.segment_tokens, job.reuse_layer
                ),
                source_digest_before=digest,
                source_digest_after=digest,
                code_commit="local-simulation",
                environment_hash="local-simulation",
                finished_at_utc="deterministic",
                paper_evidence=False,
            )
        )
    return results
