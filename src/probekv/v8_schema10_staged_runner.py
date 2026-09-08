"""Fail-closed staged runner for the schema10 single-request session.

This module is intentionally model- and vLLM-independent.  The server runner
supplies callbacks backed by the real native factory.  Keeping orchestration
here makes it impossible for the post-measurement trace command to silently
stand in for the prerequisite correctness and cost stages.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

from .v8_schema10_execution import digest_json
from .v8_schema10_event_log import atomic_json


STAGE_ORDER = (
    "environment", "correctness", "cost_collection", "cost_support_validation",
    "online_trace", "gate1_paired_ab", "source_oracle", "aggregation",
)


@dataclass(frozen=True)
class StageCallback:
    name: str
    run: Callable[[], Mapping]


def _validate_row(stage: str, row: Mapping) -> None:
    if not isinstance(row, Mapping):
        raise ValueError(f"stage {stage} did not return an object")
    # Environment may contain only immutable audits.  Every execution stage
    # must carry real evidence; a hand-filled `passed` flag is never enough.
    if stage not in {"environment", "aggregation"}:
        if row.get("origin") != "real_cuda_execution" or row.get("fake_timing") is not False:
            raise ValueError(f"stage {stage} lacks real CUDA provenance")
        if not row.get("raw_observation_sha256") and not row.get("row_sha256"):
            raise ValueError(f"stage {stage} lacks an immutable observation digest")


def run_staged_session(*, callbacks: Mapping[str, Callable[[], Mapping]],
                       output_path: str | Path, binding: Mapping,
                       resume: Mapping | None = None) -> Mapping:
    """Run all preregistered stages in order and atomically publish evidence.

    Callbacks are executed once in order.  A failed or missing stage prevents
    publication.  ``resume`` is accepted only as an immutable successful
    prefix with the same binding; callers must not skip an unfinished stage.
    """
    if set(callbacks) != set(STAGE_ORDER) or not isinstance(binding, Mapping):
        raise ValueError("the staged session requires every preregistered stage")
    binding_digest = digest_json(dict(binding))
    if resume is not None:
        if resume.get("binding_sha256") != binding_digest:
            raise ValueError("resume binding differs from this code/model session")
        completed = tuple(resume.get("completed_stages", ()))
        if completed != STAGE_ORDER[: len(completed)]:
            raise ValueError("resume is not a successful immutable prefix")
        # A successful prefix is only a hint for audit; the native callbacks
        # still rerun unless they explicitly implement a verified checkpoint.
        if len(completed) != 0:
            raise ValueError("live backend checkpoints are not supported by this runner")
    rows = {}
    for stage in STAGE_ORDER:
        callback = callbacks[stage]
        if not callable(callback):
            raise TypeError(f"stage {stage} callback is not callable")
        row = callback()
        _validate_row(stage, row)
        rows[stage] = dict(row)
    payload = {
        "schema_version": 10,
        "stage_order": list(STAGE_ORDER),
        "binding": dict(binding),
        "binding_sha256": binding_digest,
        "completed_stages": list(STAGE_ORDER),
        "rows": rows,
        "formal_profile_frozen": False,
        "gpu_runtime_qualified": False,
        "online_trace_execution_allowed": True,
        "paper_evidence": False,
        "locked_test_accessed": False,
    }
    payload["session_sha256"] = digest_json(payload)
    atomic_json(Path(output_path), payload)
    return payload

