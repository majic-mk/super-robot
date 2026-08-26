from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Dict, Mapping, Sequence, Tuple

from .v6_a800_jobs import V6A800Job, build_v6_a800_jobs


def build_v7_a800_jobs(raw: Mapping[str, Any]) -> Tuple[V6A800Job, ...]:
    if raw.get("protocol_version") != 7 or raw.get("paper_evidence") is not False:
        raise ValueError("v7 A800 qualification must remain non-paper")
    if raw.get("artifact_policy") != "single_canonical_lossless":
        raise ValueError("v7 jobs require one canonical Artifact")
    if raw.get("repair_rounding_policy") != "ceil":
        raise ValueError("v7 jobs require conservative repair rounding")
    v6_input = dict(raw)
    v6_input["protocol_version"] = 6
    jobs = build_v6_a800_jobs(v6_input)
    return tuple(replace(job, job_id=job.job_id.replace("v6-", "v7-", 1)) for job in jobs)


def v7_a800_job_digest(jobs: Sequence[V6A800Job]) -> str:
    payload = json.dumps(
        [job.to_row() for job in jobs], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_v7_a800_job_manifest(
    jobs: Sequence[V6A800Job],
    *,
    jobs_sha256: str,
    code_commit: str,
    git_clean: bool,
    config_sha256: str,
    contract_sha256: str,
    server_lock_sha256: str,
    model_id: str,
    model_revision: str,
    tokenizer_hash: str,
    adapter_name: str,
    runtime_backend: str,
    cacheblend_commit: str,
    cacheblend_patch_mode: str,
    cacheblend_patch_sha256: str,
    cacheblend_tree: str,
) -> Dict[str, Any]:
    values = locals()
    required = [
        key for key, value in values.items()
        if key not in {"jobs", "git_clean"} and not str(value).strip()
    ]
    if required:
        raise ValueError("v7 job provenance is missing: %s" % ", ".join(required))
    if len(jobs) != 140 or any(job.paper_evidence for job in jobs):
        raise ValueError("v7 requires the frozen 140 non-paper jobs")
    return {
        "schema_version": 3,
        "protocol_version": 7,
        "evidence_class": "server_pilot",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "jobs": len(jobs),
        "job_digest": v7_a800_job_digest(jobs),
        "jobs_sha256": jobs_sha256,
        "code_commit": code_commit,
        "git_clean": bool(git_clean),
        "config_sha256": config_sha256,
        "contract_sha256": contract_sha256,
        "server_lock_sha256": server_lock_sha256,
        "model": {
            "model_id": model_id,
            "revision": model_revision,
            "tokenizer_hash": tokenizer_hash,
            "adapter_name": adapter_name,
        },
        "runtime": {
            "backend": runtime_backend,
            "artifact_policy": "single_canonical_lossless",
            "repair_rounding_policy": "ceil",
            "qualified": False,
        },
        "cacheblend": {
            "base_commit": cacheblend_commit,
            "patch_mode": cacheblend_patch_mode,
            "patch_sha256": cacheblend_patch_sha256,
            "tree": cacheblend_tree,
        },
        "next_required_gate": "v7-A800-runtime-qualification",
    }
