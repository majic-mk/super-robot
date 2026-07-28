"""Run resumable, non-paper H1 jobs on the pinned CacheBlend/A800 stack."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from probekv.cacheblend_server_runtime import CacheBlendCaseRuntime
from probekv.experiment_jobs import (
    E1Job,
    E1Result,
    ResultStatus,
    resumable_e1_jobs,
)
from probekv.io import append_jsonl_fsync, atomic_write_json, sha256_file
from probekv.manifest import manifest_case_from_row, validate_manifest


def _read_jsonl(path: Path, loader):
    return [
        loader(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def _runtime_provenance(
    environment_path: Path,
    patch_path: Path,
    revision: str,
) -> dict:
    import torch
    import vllm

    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    required = {
        "cacheblend_commit",
        "cacheblend_patch_sha256",
        "cacheblend_tree",
    }
    missing = required - set(patch)
    if missing:
        raise ValueError("patch provenance missing: %s" % sorted(missing))
    return {
        "code_commit": _command("git", "rev-parse", "HEAD"),
        "environment_hash": sha256_file(environment_path),
        "model_revision": revision,
        "cacheblend_commit": patch["cacheblend_commit"],
        "cacheblend_patch_sha256": patch["cacheblend_patch_sha256"],
        "cacheblend_tree": patch["cacheblend_tree"],
        "vllm_version": str(vllm.__version__),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_uuid": _command(
            "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"
        ).splitlines()[0],
        "finished_at_utc": _now(),
    }


def _failure_result(job: E1Job, attempt: int, error: Exception) -> E1Result:
    message = "%s: %s" % (type(error).__name__, error)
    lowered = message.lower()
    status = ResultStatus.OOM if "out of memory" in lowered else ResultStatus.DATA_ERROR
    return E1Result(
        job_id=job.job_id,
        attempt=attempt,
        status=status,
        error_type=status.value.upper(),
        error_message=message[:2000],
        finished_at_utc=_now(),
        evidence_class="server_pilot",
        paper_evidence=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--patch-provenance", required=True)
    parser.add_argument(
        "--model", default="mistralai/Mistral-7B-Instruct-v0.3"
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--pass",
        dest="run_pass",
        choices=("primary", "anchors", "all"),
        default="primary",
    )
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_hours <= 0:
        raise ValueError("max-hours must be positive")
    session_start = time.monotonic()
    deadline = session_start + args.max_hours * 3600.0
    manifest_path = Path(args.manifest).resolve()
    jobs_path = Path(args.jobs).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / ("results-%s.jsonl" % args.run_pass)
    cases = _read_jsonl(manifest_path, manifest_case_from_row)
    validate_manifest(cases)
    if any(case.split != "pilot" for case in cases):
        raise ValueError("server pilot manifest contains a non-pilot split")
    case_by_id = {case.case_id: case for case in cases}
    jobs = _read_jsonl(jobs_path, E1Job.from_row)
    if args.run_pass == "primary":
        jobs = [job for job in jobs if job.reuse_layer == 5]
    elif args.run_pass == "anchors":
        jobs = [job for job in jobs if job.reuse_layer != 5]
    if any(job.split == "test" for job in jobs):
        raise ValueError("locked test jobs are prohibited in server_pilot")
    if any(job.case_id not in case_by_id for job in jobs):
        raise ValueError("jobs reference a case outside the pilot manifest")
    existing = (
        _read_jsonl(result_path, E1Result.from_row)
        if args.resume and result_path.exists()
        else []
    )
    if result_path.exists() and not args.resume:
        raise FileExistsError(
            "%s already exists; pass --resume or choose a new output" % result_path
        )
    pending, attempt_by_job = resumable_e1_jobs(jobs, existing)
    pending_by_case = {}
    for job in pending:
        pending_by_case.setdefault(job.case_id, []).append(job)
    ordered_case_ids = sorted(pending_by_case)
    if args.case_limit:
        ordered_case_ids = ordered_case_ids[: args.case_limit]

    environment_path = Path(args.environment).resolve()
    patch_path = Path(args.patch_provenance).resolve()
    provenance = _runtime_provenance(
        environment_path, patch_path, args.revision
    )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
    )
    llm = LLM(
        model=args.model,
        revision=args.revision,
        tokenizer=args.model,
        tokenizer_revision=args.revision,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )
    llm.set_tokenizer(tokenizer)

    completed_cases = 0
    appended_rows = 0
    case_durations = []
    for case_id in ordered_case_ids:
        if time.monotonic() >= deadline:
            break
        case_start = time.monotonic()
        case_jobs = pending_by_case[case_id]
        try:
            runtime = CacheBlendCaseRuntime(
                llm,
                tokenizer,
                SamplingParams,
                case_by_id[case_id],
                provenance,
                max_new_tokens=64,
            )
            rows = []
            source_ids = sorted({job.source_id for job in case_jobs})
            for source_id in source_ids:
                source_jobs = [
                    job for job in case_jobs if job.source_id == source_id
                ]
                rows.extend(
                    runtime.run_source_jobs(
                        source_id, source_jobs, attempt_by_job
                    )
                )
            del runtime
        except Exception as error:
            rows = [
                _failure_result(
                    job,
                    attempt_by_job.get(job.job_id, 0),
                    error,
                )
                for job in case_jobs
            ]
        appended_rows += append_jsonl_fsync(
            result_path, [row.to_row() for row in rows]
        )
        completed_cases += 1
        case_durations.append(time.monotonic() - case_start)
        gc.collect()
        torch.cuda.empty_cache()
        elapsed = time.monotonic() - session_start
        mean_case = sum(case_durations) / len(case_durations)
        remaining_cases = len(ordered_case_ids) - completed_cases
        summary = {
            "run_pass": args.run_pass,
            "assigned_jobs": len(jobs),
            "pending_jobs_at_start": len(pending),
            "completed_cases_this_run": completed_cases,
            "appended_rows_this_run": appended_rows,
            "elapsed_seconds": elapsed,
            "mean_case_seconds": mean_case,
            "projected_remaining_seconds": mean_case * remaining_cases,
            "deadline_reached": time.monotonic() >= deadline,
            "paper_evidence": False,
            "evidence_class": "server_pilot",
            "updated_at_utc": _now(),
        }
        atomic_write_json(output / "run_summary.json", summary)

    final_summary = json.loads(
        (output / "run_summary.json").read_text(encoding="utf-8")
    ) if completed_cases else {
        "run_pass": args.run_pass,
        "assigned_jobs": len(jobs),
        "pending_jobs_at_start": len(pending),
        "completed_cases_this_run": 0,
        "appended_rows_this_run": 0,
        "deadline_reached": time.monotonic() >= deadline,
        "paper_evidence": False,
        "evidence_class": "server_pilot",
        "updated_at_utc": _now(),
    }
    atomic_write_json(output / "run_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
