"""Run real-case H1 labels on the qualified ProbeKV v6 resumable data plane."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from probekv.experiment_jobs import E1Job, E1Result, ResultStatus
from probekv.h1_qualification import validate_h1_qualification_gate
from probekv.io import append_jsonl_fsync, atomic_write_json, sha256_file
from probekv.manifest import manifest_case_from_row, validate_manifest
from probekv.model_adapters import MISTRAL_SPEC, QWEN_SPEC
from probekv.v6_a800_executor import RealCacheBlendA800Executor
from probekv.v6_h1_runtime import V6H1CaseRuntime, V6H1CorrectnessError


def _rows(path: Path, loader):
    return [loader(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _command(*values: str) -> str:
    return subprocess.check_output(values, text=True).strip()


def _provenance(environment: Path, patch: dict, revision: str) -> dict:
    import torch
    import vllm

    return {
        "code_commit": _command("git", "rev-parse", "HEAD"),
        "environment_hash": sha256_file(environment),
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
    }


def _failure(job: E1Job, error: Exception, provenance: dict) -> E1Result:
    return E1Result(
        job_id=job.job_id,
        attempt=0,
        status=ResultStatus.DATA_ERROR,
        error_type=(
            "V6_R1_DENSE_EQUIVALENCE" if isinstance(error, V6H1CorrectnessError)
            else type(error).__name__.upper()
        ),
        error_message=("%s: %s" % (type(error).__name__, error))[:2000],
        code_commit=provenance["code_commit"],
        environment_hash=provenance["environment_hash"],
        model_revision=provenance["model_revision"],
        cacheblend_commit=provenance["cacheblend_commit"],
        cacheblend_patch_sha256=provenance["cacheblend_patch_sha256"],
        cacheblend_tree=provenance["cacheblend_tree"],
        vllm_version=provenance["vllm_version"],
        torch_version=provenance["torch_version"],
        cuda_version=provenance["cuda_version"],
        gpu_uuid=provenance["gpu_uuid"],
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_class="server_pilot",
        paper_evidence=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--qualification-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-key", choices=("mistral", "qwen"), required=True)
    parser.add_argument("--pass", dest="run_pass", choices=("primary", "anchors", "all"), default="primary")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest).resolve()
    jobs_path = Path(args.jobs).resolve()
    handoff = json.loads(Path(args.handoff).read_text(encoding="utf-8"))
    model_audit = json.loads(Path(args.model_audit).read_text(encoding="utf-8"))
    patch = json.loads(Path(args.patch_audit).read_text(encoding="utf-8"))
    qualification_path = Path(args.qualification_gate).resolve()
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    environment = Path(args.environment).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / ("results-%s.jsonl" % args.run_pass)
    gate_path = output / "gate.json"
    if results_path.exists() and not args.resume:
        raise FileExistsError("results exist; pass --resume or select a new output")
    code_commit = _command("git", "rev-parse", "HEAD")
    if _command("git", "status", "--porcelain"):
        raise ValueError("v6 H1 requires a clean ProbeKV worktree")
    if handoff.get("stage") != "v6_h1_model_data_handoff":
        raise ValueError("invalid v6 H1 data handoff")
    if handoff.get("paper_evidence") is not False or handoff.get("locked_test_accessed") is not False:
        raise ValueError("v6 H1 handoff must remain an unlocked server pilot")
    if handoff.get("ready_for_v6_h1_gpu_sentinel") is not True:
        raise ValueError("v6 H1 handoff is not sentinel-ready")
    if handoff.get("code_commit") != code_commit:
        raise ValueError("v6 H1 handoff was built by a different ProbeKV commit")
    if handoff.get("pilot_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("v6 H1 manifest differs from its handoff")
    if handoff.get("jobs_sha256") != sha256_file(jobs_path):
        raise ValueError("v6 H1 jobs differ from their handoff")
    if handoff.get("model_audit_sha256") != sha256_file(Path(args.model_audit)):
        raise ValueError("v6 H1 model audit differs from its handoff")
    if handoff.get("patch_audit_sha256") != sha256_file(Path(args.patch_audit)):
        raise ValueError("v6 H1 CacheBlend patch audit differs from its handoff")

    cases = _rows(manifest_path, manifest_case_from_row)
    validate_manifest(cases)
    if any(case.split != "pilot" for case in cases):
        raise ValueError("v6 H1 may only open pilot cases")
    case_by_id = {case.case_id: case for case in cases}
    jobs = _rows(jobs_path, E1Job.from_row)
    spec = MISTRAL_SPEC if args.model_key == "mistral" else QWEN_SPEC
    primary_layer = max(1, min(spec.num_layers - 1, round(spec.num_layers * 0.15)))
    if args.run_pass == "primary":
        jobs = [job for job in jobs if job.reuse_layer == primary_layer]
    elif args.run_pass == "anchors":
        jobs = [job for job in jobs if job.reuse_layer != primary_layer]
    if any(job.case_id not in case_by_id or job.split != "pilot" for job in jobs):
        raise ValueError("v6 H1 jobs escape the pilot manifest")
    if model_audit.get("model_id") != spec.model_id or model_audit.get("revision") != spec.revision:
        raise ValueError("model audit does not match the selected adapter")
    if handoff.get("model_id") != spec.model_id or handoff.get("model_revision") != spec.revision:
        raise ValueError("handoff does not match the selected adapter")
    if any(case.model_signature != "%s@%s" % (spec.model_id, spec.revision) for case in cases):
        raise ValueError("manifest was not tokenized for the selected model")

    # This is deliberately before provenance imports torch/vLLM and before the
    # executor constructs the model.  An invalid, stale or cross-model gate
    # must therefore consume no model-loading time or GPU memory.
    current_gpu_uuid = _command(
        "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"
    ).splitlines()[0]
    validate_h1_qualification_gate(
        qualification,
        code_commit=code_commit,
        model_id=spec.model_id,
        model_revision=spec.revision,
        adapter_name=spec.adapter_name,
        cacheblend_patch_sha256=patch["cacheblend_patch_sha256"],
        cacheblend_tree=patch["cacheblend_tree"],
        gpu_uuid=current_gpu_uuid,
    )

    existing = _rows(results_path, E1Result.from_row) if results_path.exists() else []
    completed_ids = {row.job_id for row in existing if row.status is ResultStatus.COMPLETED}
    groups = {}
    for job in jobs:
        groups.setdefault((job.case_id, job.source_id, job.reuse_layer), []).append(job)
    pending_by_case = {}
    for key, members in sorted(groups.items()):
        if all(job.job_id in completed_ids for job in members):
            continue
        pending_by_case.setdefault(key[0], []).append((key, tuple(members)))
    case_ids = sorted(pending_by_case)
    if args.case_limit:
        case_ids = case_ids[: args.case_limit]

    provenance = _provenance(environment, patch, spec.revision)
    executor = RealCacheBlendA800Executor(
        model_path=str(model_audit["snapshot_path"]),
        model_spec=spec,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        expected_cacheblend_tree=patch["cacheblend_tree"],
    )
    started = time.monotonic()
    deadline = started + args.max_hours * 3600.0
    completed_cases = 0
    completed_groups = 0
    appended = 0
    hard_failure = None
    for case_id in case_ids:
        if time.monotonic() >= deadline:
            break
        runtime = V6H1CaseRuntime(executor, case_by_id[case_id], provenance)
        for (_, source_id, _), members in pending_by_case[case_id]:
            if time.monotonic() >= deadline:
                break
            try:
                rows = runtime.run_group(source_id, members, {})
            except Exception as error:
                r1 = next(job for job in members if job.repair_ratio == 1.0)
                append_jsonl_fsync(results_path, [_failure(r1, error, provenance).to_row()])
                hard_failure = error
                break
            appended += append_jsonl_fsync(results_path, [row.to_row() for row in rows])
            completed_groups += 1
        if hard_failure is not None:
            break
        completed_cases += 1

    gate = {
        "schema_version": 1,
        "stage": "v6_h1_server_pilot",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "code_commit": provenance["code_commit"],
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "primary_reuse_layer": primary_layer,
        "manifest_sha256": sha256_file(manifest_path),
        "jobs_sha256": sha256_file(jobs_path),
        "qualification_gate_sha256": sha256_file(qualification_path),
        "qualification_gate_schema": qualification["schema_version"],
        "native_prefix_cache_qualified": qualification[
            "native_prefix_cache_qualified"
        ],
        "completed_cases_this_run": completed_cases,
        "completed_groups_this_run": completed_groups,
        "appended_rows_this_run": appended,
        "elapsed_seconds_this_run": time.monotonic() - started,
        "deadline_reached": time.monotonic() >= deadline,
        "r1_dense_equivalence_passed": hard_failure is None,
        "h1_scan_allowed": hard_failure is None,
        "failure": None if hard_failure is None else "%s: %s" % (type(hard_failure).__name__, hard_failure),
    }
    atomic_write_json(gate_path, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if hard_failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
