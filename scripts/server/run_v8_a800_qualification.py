"""Run the Profile-bound non-paper ProbeKV v8 A800 qualification."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from probekv.cacheblend_v6_online_engine import CacheBlendV8OnlineEngine
from probekv.io import append_jsonl_fsync, atomic_write_json, sha256_file
from probekv.model_adapters import MISTRAL_SPEC, QWEN_SPEC
from probekv.native_prefix_cache import evaluate_native_prefix_cache_audit
from probekv.v6_a800_executor import RealCacheBlendA800Executor
from probekv.v6_a800_jobs import V6A800Job
from probekv.v6_qualification_worker import QualificationJobResult, validate_qualification_results
from probekv.v8_profile import validate_frozen_selector_profile


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path, loader):
    if not path.exists():
        return []
    return [
        loader(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def command(repo: Path, *values: str) -> str:
    return subprocess.check_output(values, cwd=str(repo), text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--job-manifest", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--lock", default="configs/a800_server_lock_v8.json")
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
    manifest = load(Path(args.job_manifest).resolve())
    profile = load(Path(args.profile).resolve())
    lock = load(Path(args.lock).resolve())
    model_audit = load(Path(args.model_audit).resolve())
    patch_audit = load(Path(args.patch_audit).resolve())
    jobs = load_jsonl(jobs_path, V6A800Job.from_row)
    if manifest.get("protocol_version") != 8 or manifest.get("schema_version") != 4:
        raise ValueError("v8 qualification requires a schema-v4 manifest")
    validate_frozen_selector_profile(profile, model_key=args.model_key)
    if manifest.get("profile_sha256") != profile.get("profile_sha256"):
        raise ValueError("qualification manifest used another Profile")
    if lock.get("protocol_version") != 8 or lock.get("schema_version") != 4:
        raise ValueError("v8 qualification requires the schema-v4 server lock")
    if len(jobs) != 140 or manifest.get("jobs") != 140:
        raise ValueError("v8 qualification requires the frozen 140-job matrix")
    if sha256_file(jobs_path) != manifest.get("jobs_sha256"):
        raise ValueError("qualification jobs differ from the manifest")
    code_commit = command(repo, "git", "rev-parse", "HEAD")
    if code_commit != manifest.get("code_commit") or command(repo, "git", "status", "--porcelain"):
        raise ValueError("v8 qualification requires the exact clean checkout")
    if model_audit.get("complete") is not True:
        raise ValueError("model audit is incomplete")
    locked_model = lock["models"][args.model_key]
    validate_frozen_selector_profile(
        profile,
        model_key=args.model_key,
        policy=manifest.get("selection_execution_policy"),
        code_commit=code_commit,
        model_revision=locked_model["revision"],
        tokenizer_hash=model_audit.get("tokenizer_hash"),
        cacheblend_patch_sha256=patch_audit.get("cacheblend_patch_sha256"),
    )
    if (
        manifest.get("model", {}).get("model_id") != locked_model["model_id"]
        or manifest.get("model", {}).get("revision") != locked_model["revision"]
        or manifest.get("model", {}).get("adapter_name") != locked_model["adapter_name"]
    ):
        raise ValueError("qualification manifest differs from the locked model")
    if patch_audit.get("patch_mode") != lock["stack"]["cacheblend_patch_mode"]:
        raise ValueError("CacheBlend patch audit used another v8 mode")
    if patch_audit.get("cacheblend_patch_sha256") != manifest["cacheblend"]["patch_sha256"]:
        raise ValueError("CacheBlend patch differs from the qualification manifest")

    spec = MISTRAL_SPEC if args.model_key == "mistral" else QWEN_SPEC
    executor = RealCacheBlendA800Executor(
        model_path=str(model_audit["snapshot_path"]),
        model_spec=spec,
        expected_cacheblend_tree=str(patch_audit["cacheblend_tree"]),
        engine_class=CacheBlendV8OnlineEngine,
        protocol_version=8,
        selection_scratch_capacity_bytes=(
            int(lock["runtime"]["selection_scratch_capacity_mib"]) * 1024 * 1024
        ),
    )
    provenance = executor.runtime_provenance
    if not re.match(lock["gpu"]["name_regex"], provenance["gpu_name"]):
        raise RuntimeError("qualification GPU differs from the frozen A800")
    if provenance["compute_capability"] != [8, 0]:
        raise RuntimeError("qualification compute capability is not 8.0")
    expected_stack = lock["stack"]
    if not str(provenance["torch"]).startswith(expected_stack["pytorch"]):
        raise RuntimeError("PyTorch differs from the v8 server lock")
    for observed_key, lock_key in (("vllm", "vllm"), ("xformers", "xformers")):
        if provenance[observed_key] != expected_stack[lock_key]:
            raise RuntimeError("%s differs from the v8 server lock" % observed_key)
    capabilities = dict(executor.capabilities())
    for capability in lock["runtime"]["required_capabilities"]:
        if capabilities.get(capability) is not True:
            raise RuntimeError("v8 runtime capability is missing: %s" % capability)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    runtime_audit_path = output / "runtime_audit.json"
    prefix_path = output / "native_prefix_cache_audit.json"
    sentinel_path = output / "sentinel.json"
    gpu_uuid = command(repo, "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader").splitlines()[0]
    observed_prefix = executor.run_native_prefix_cache_sentinel()
    observed_prefix.update(
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
    prefix_audit = evaluate_native_prefix_cache_audit(observed_prefix, expected_layers=spec.num_layers)
    prefix_audit["native_prefix_cache_qualified"] = prefix_audit.get("passed") is True
    atomic_write_json(prefix_path, prefix_audit)
    sentinel = dict(executor.sentinel)
    atomic_write_json(sentinel_path, sentinel)
    if not prefix_audit["native_prefix_cache_qualified"]:
        return 1
    if args.sentinel_only:
        print(json.dumps({"prefix": prefix_audit, "r1": sentinel}, ensure_ascii=False, indent=2))
        return 0

    existing = load_jsonl(results_path, QualificationJobResult.from_row) if args.resume else []
    if results_path.exists() and not args.resume:
        raise FileExistsError("results exist; use --resume or another output")
    if tuple(item.job_id for item in existing) != tuple(item.job_id for item in jobs[:len(existing)]):
        raise ValueError("resume results are not an immutable successful prefix")
    if any(not item.passed for item in existing):
        raise RuntimeError("failed qualification output cannot be resumed")
    pending = jobs[len(existing):]
    if args.job_limit:
        pending = pending[:args.job_limit]
    stopped = False
    for job in pending:
        try:
            result = executor.execute(job)
        except Exception as error:
            result = QualificationJobResult(
                job_id=job.job_id, passed=False, cuda_event_timing=False,
                gpu_ms=0.0, host_ms=0.0, r1_dense_token_ids_equal=False,
                teacher_forced_logit_relative_l2=float("inf"),
                canonical_source_digests_unchanged=False,
                artifact_digests_unchanged=False,
                absolute_union_mask_verified=False,
                error="%s: %s" % (type(error).__name__, error),
            )
            stopped = True
        append_jsonl_fsync(results_path, [result.to_row()])
        existing.append(result)
        if stopped:
            break
    complete = len(existing) == 140 and not stopped
    if complete:
        validate_qualification_results(jobs, existing)
    correctness = {
        "r1_dense_token_ids_equal": sentinel.get("r1_dense_token_ids_equal") is True,
        "source_digest_unchanged": sentinel.get("canonical_source_digests_unchanged") is True,
        "artifact_digest_unchanged": sentinel.get("artifact_digests_unchanged") is True,
        "absolute_union_mask_verified": sentinel.get("absolute_union_mask_verified") is True,
        "completed_depth_hook_verified": sentinel.get("completed_depth_hook_verified") is True,
    }
    audit = {
        "schema_version": 4,
        "protocol_version": 8,
        "stage": "v8_a800_profile_bound_runtime_qualification",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "code_commit": code_commit,
        "model_revision": spec.revision,
        "model_id": spec.model_id,
        "adapter_name": spec.adapter_name,
        "tokenizer_hash": manifest["model"]["tokenizer_hash"],
        "cacheblend_patch_sha256": patch_audit["cacheblend_patch_sha256"],
        "cacheblend_tree": patch_audit["cacheblend_tree"],
        "profile_sha256": profile["profile_sha256"],
        "job_digest": manifest["job_digest"],
        "gpu_uuid": gpu_uuid,
        "cuda_event_timing": bool(existing) and all(item.cuda_event_timing for item in existing),
        "fake_timing": False,
        "runtime_backend": lock["runtime"]["backend"],
        "runtime_provenance": provenance,
        "capabilities": capabilities,
        "single_artifact_policy_verified": True,
        "selection_state_k_only_verified": True,
        "selection_state_separate_backing_verified": sentinel.get(
            "selection_state_separate_backing_verified"
        ) is True,
        "selection_scratch_peak_bytes": executor.selection_scratch_peak_bytes,
        "selection_scratch_capacity_bytes": executor.selection_scratch_capacity_bytes,
        "fixed_repair_ratio": 0.15,
        "runtime_vllm_block_size": prefix_audit.get("block_size"),
        "correctness": correctness,
        "selection_transfer": {
            key: int(sentinel.get(key, -1))
            for key in (
                "request_attributed_full_kv_bytes_transferred_for_selection",
                "request_attributed_nonwinner_full_kv_bytes_transferred",
                "request_attributed_full_kv_prefetch_before_source_freeze",
            )
        },
        "jobs": {
            "planned": 140,
            "completed": len(existing),
            "failed": sum(not item.passed for item in existing),
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
