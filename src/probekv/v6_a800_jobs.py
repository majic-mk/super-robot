from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Sequence, Tuple


class V6A800JobKind(str, Enum):
    CORRECTNESS = "correctness"
    SUMMARY_GENERATION = "summary_generation"
    CANDIDATE_COMPARE = "candidate_compare"
    MULTISEGMENT_COMPARE = "multisegment_compare"
    MULTISOURCE_LOAD = "multisource_load"
    UNION_REPAIR = "union_repair"


@dataclass(frozen=True)
class V6A800Job:
    job_id: str
    kind: V6A800JobKind
    segment_count: int
    stored_variants: int
    compared_variants: int
    probe_layer: int
    repair_ratio: float
    warmups: int
    repeats: int
    paper_evidence: bool = False

    def __post_init__(self) -> None:
        if min(
            self.segment_count,
            self.stored_variants,
            self.compared_variants,
            self.probe_layer,
            self.warmups,
            self.repeats,
        ) < 0:
            raise ValueError("A800 v6 job dimensions must be non-negative")
        if self.segment_count < 1 or self.repeats < 1:
            raise ValueError("A800 v6 jobs require segments and repeats")
        if self.probe_layer < 1:
            raise ValueError("A800 v6 probe layers are 1-based")
        if self.compared_variants > self.stored_variants:
            raise ValueError("compared variants exceed stored variants")
        if not 0 <= self.repair_ratio <= 1:
            raise ValueError("repair ratio must be in [0, 1]")
        if self.paper_evidence:
            raise ValueError("v6 A800 bring-up jobs are non-paper pilots")

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["kind"] = self.kind.value
        return row


def _job(kind: V6A800JobKind, **values: Any) -> V6A800Job:
    identity = json.dumps(
        {"kind": kind.value, **values}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return V6A800Job(
        job_id="v6-%s-%s" % (kind.value, hashlib.sha256(identity).hexdigest()[:16]),
        kind=kind,
        **values,
    )


def build_v6_a800_jobs(
    raw: Mapping[str, Any]
) -> Tuple[V6A800Job, ...]:
    correctness_segments = tuple(int(v) for v in raw["correctness_segments"])
    correctness_variants = tuple(int(v) for v in raw["correctness_variants"])
    profile_segments = tuple(int(v) for v in raw["profile_segments"])
    compare_k = tuple(int(v) for v in raw["compare_k"])
    layers = tuple(int(v) for v in raw["probe_layers"])
    ratios = tuple(float(v) for v in raw["repair_ratios"])
    warmups = int(raw.get("warmups", 20))
    repeats = int(raw.get("repeats", 100))
    jobs = []
    common = {"warmups": warmups, "repeats": repeats}
    for segments in correctness_segments:
        for variants in correctness_variants:
            for ratio in (0.0, 1.0):
                jobs.append(
                    _job(
                        V6A800JobKind.CORRECTNESS,
                        segment_count=segments,
                        stored_variants=variants,
                        compared_variants=variants,
                        probe_layer=layers[-1],
                        repair_ratio=ratio,
                        **common,
                    )
                )
    for segments in profile_segments:
        for layer in layers:
            jobs.append(
                _job(
                    V6A800JobKind.SUMMARY_GENERATION,
                    segment_count=segments,
                    stored_variants=1,
                    compared_variants=0,
                    probe_layer=layer,
                    repair_ratio=0.0,
                    **common,
                )
            )
        jobs.append(
            _job(
                V6A800JobKind.MULTISEGMENT_COMPARE,
                segment_count=segments,
                stored_variants=4,
                compared_variants=4,
                probe_layer=layers[-1],
                repair_ratio=0.0,
                **common,
            )
        )
        jobs.append(
            _job(
                V6A800JobKind.MULTISOURCE_LOAD,
                segment_count=segments,
                stored_variants=1,
                compared_variants=0,
                probe_layer=layers[-1],
                repair_ratio=0.0,
                **common,
            )
        )
        for ratio in ratios:
            jobs.append(
                _job(
                    V6A800JobKind.UNION_REPAIR,
                    segment_count=segments,
                    stored_variants=1,
                    compared_variants=0,
                    probe_layer=layers[-1],
                    repair_ratio=ratio,
                    **common,
                )
            )
    for candidates in compare_k:
        for layer in layers:
            jobs.append(
                _job(
                    V6A800JobKind.CANDIDATE_COMPARE,
                    segment_count=1,
                    stored_variants=candidates,
                    compared_variants=candidates,
                    probe_layer=layer,
                    repair_ratio=0.0,
                    **common,
                )
            )
    identifiers = [job.job_id for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("v6 A800 job IDs are not unique")
    return tuple(jobs)


def v6_a800_job_digest(jobs: Sequence[V6A800Job]) -> str:
    payload = json.dumps(
        [job.to_row() for job in jobs],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
