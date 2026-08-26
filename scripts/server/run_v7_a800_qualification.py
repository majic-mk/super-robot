"""Run and durably audit the real CacheBlend v7 A800 qualification matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from probekv.cacheblend_v6_online_engine import CacheBlendV7OnlineEngine
from probekv.io import append_jsonl_fsync, atomic_write_json, sha256_file
from probekv.model_adapters import MISTRAL_SPEC, QWEN_SPEC
from probekv.native_prefix_cache import evaluate_native_prefix_cache_audit
from probekv.v6_a800_executor import RealCacheBlendA800Executor
from probekv.v6_a800_jobs import V6A800Job
from probekv.v6_qualification_worker import (
    QualificationJobResult,
    validate_qualification_results,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path, loader):
    if not path.exists():
        return []
    return [
        loader(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _command(repo: Path, *values: str) -> str:
    return subprocess.check_output(values, cwd=str(repo), text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--job-manifest", required=True)
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-key", choices=("mistral", "qwen"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sentinel-only", action="store_true")
    parser.add_argument("--job-limit", type=int, default=0)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    jobs_path = Path(args.jobs).resolve()
    manifest_path = Path(args.job_manifest).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    runtime_audit_path = output / "runtime_audit.json"
    sentinel_path = output / "sentinel.json"
    prefix_path = output / "native_prefix_cache_audit.json"

    jobs = _jsonl(jobs_path, V6A800Job.from_row)
    manifest = _json(manifest_path)
    model_audit = _json(Path(args.model_audit).resolve())
    patch_audit = _json(Path(args.patch_audit).resolve())
    if manifest.get("protocol_version") != 7 or manifest.get("schema_version") != 3:
        raise ValueError("v7 qualification requires a schema-v3 manifest")
    if len(jobs) != 140 or manifest.get("jobs") != 140:
        raise ValueError("qualification requires the frozen 140-job matrix")
    if sha256_file(jobs_path) != manifest.get("jobs_sha256"):
        raise ValueError("qualification jobs differ from the frozen manifest")
    if model_audit.get("complete") is not True:
        raise ValueError("model audit is incomplete")
    if model_audit.get("revision") != manifest.get("model", {}).get("revision"):
        raise ValueError("model revision differs from the job manifest")
    if patch_audit.get("cacheblend_patch_sha256") != manifest.get(
        "cacheblend", {}
    ).get("patch_sha256"):
        raise ValueError("CacheBlend patch differs from the job manifest")
    code_commit = _command(repo, "git", "rev-parse", "HEAD")
    if code_commit != manifest.get("code_commit"):
        raise ValueError("checked-out code differs from the job manifest")
    if _command(repo, "git", "status", "--porcelain"):
        raise ValueError("qualification requires a clean ProbeKV worktree")

    spec = MISTRAL_SPEC if args.model_key == "mistral" else QWEN_SPEC
    if spec.model_id != model_audit.get("model_id"):
        raise ValueError("model-key adapter differs from the model audit")
    executor = RealCacheBlendA800Executor(
        model_path=str(model_audit["snapshot_path"]),
        model_spec=spec,
        expected_cacheblend_tree=str(patch_audit["cacheblend_tree"]),
        engine_class=CacheBlendV7OnlineEngine,
        protocol_version=7,
    )
    gpu_uuid = _command(
        repo, "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"
    ).splitlines()[0]
    try:
        prefix_observed = executor.run_native_prefix_cache_sentinel()
        prefix_error = None
    except Exception as error:
        prefix_error = "%s: %s" % (type(error).__name__, error)
        prefix_observed = {
            "paper_evidence": False,
            "locked_test_accessed": False,
            "model_num_layers": spec.num_layers,
            "runner_error": prefix_error,
        }
    prefix_observed.update(
        {
            "code_commit": code_commit,
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "adapter_name": spec.adapter_name,
            "cacheblend_patch_sha256": patch_audit["cacheblend_patch_sha256"],
            "cacheblend_tree": patch_audit["cacheblend_tree"],
            "gpu_uuid": gpu_uuid,
        }
    )
    prefix_audit = evaluate_native_prefix_cache_audit(
        prefix_observed, expected_layers=spec.num_layers
    )
    if prefix_error:
        prefix_audit["passed"] = False
        prefix_audit["failures"].append(prefix_error)
    atomic_write_json(prefix_path, prefix_audit)
    atomic_write_json(sentinel_path, executor.sentinel)
    if not prefix_audit["passed"]:
        print(json.dumps(prefix_audit, ensure_ascii=False, indent=2))
        return 1
    if args.sentinel_only:
        print(
            json.dumps(
                {"native_prefix_cache": prefix_audit, "r1": executor.sentinel},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    existing = _jsonl(results_path, QualificationJobResult.from_row) if args.resume else []
    if results_path.exists() and not args.resume:
        raise FileExistsError("results exist; pass --resume or choose another output")
    if tuple(row.job_id for row in existing) != tuple(
        job.job_id for job in jobs[: len(existing)]
    ):
        raise ValueError("existing results are not an immutable job prefix")
    if any(not row.passed for row in existing):
        raise RuntimeError("existing results contain a failed job; use a new output")
    pending = jobs[len(existing):]
    if args.job_limit:
        pending = pending[: args.job_limit]
    stopped = False
    for job in pending:
        try:
            result = executor.execute(job)
        except Exception as error:
            result = QualificationJobResult(
                job_id=job.job_id,
                passed=False,
                cuda_event_timing=False,
                gpu_ms=0.0,
                host_ms=0.0,
                r1_dense_token_ids_equal=False,
                teacher_forced_logit_relative_l2=float("inf"),
                canonical_source_digests_unchanged=False,
                artifact_digests_unchanged=False,
                absolute_union_mask_verified=False,
                error="%s: %s" % (type(error).__name__, error),
            )
            stopped = True
        append_jsonl_fsync(results_path, [result.to_row()])
        existing.append(result)
        print(json.dumps({"completed": len(existing), "planned": 140, "job_id": job.job_id}))
        if stopped:
            break
    complete = len(existing) == len(jobs) and not stopped
    if complete:
        validate_qualification_results(jobs, existing)
    capabilities = dict(executor.capabilities())
    audit = {
        "schema_version": 3,
        "protocol_version": 7,
        "stage": "v7_a800_runtime_qualification",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "runtime_backend": "cacheblend_v7_closed_loop",
        "concrete_engine_hook": True,
        "capabilities": capabilities,
        "code_commit": code_commit,
        "job_digest": manifest["job_digest"],
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "adapter_name": spec.adapter_name,
        "cacheblend_patch_sha256": patch_audit["cacheblend_patch_sha256"],
        "cacheblend_tree": patch_audit["cacheblend_tree"],
        "runtime_provenance": dict(executor.runtime_provenance),
        "gpu_uuid": gpu_uuid,
        "native_prefix_cache_audit_sha256": sha256_file(prefix_path),
        "single_artifact_policy_verified": capabilities.get(
            "single_lossless_bf16_artifact"
        ) is True,
        "max_artifacts_per_source_variant_observed": 1,
        "repair_rounding_policy": executor.repair_rounding_policy,
        "alignment_quantum": 16,
        "runtime_vllm_block_size": 16,
        "correctness": executor.sentinel,
        "all_job_artifact_digests_unchanged": (
            bool(existing)
            and all(row.artifact_digests_unchanged for row in existing)
        ),
        "jobs": {
            "planned": len(jobs),
            "completed": len(existing),
            "failed": sum(not row.passed for row in existing),
            "results_sha256": sha256_file(results_path) if results_path.exists() else None,
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(runtime_audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if stopped:
        return 1
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
