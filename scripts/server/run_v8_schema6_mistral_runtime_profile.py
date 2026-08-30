"""Measure and freeze one Mistral schema-v6 RuntimeCostProfile on A800."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from probekv.cacheblend_v6_online_engine import CacheBlendV8OnlineEngine
from probekv.io import append_jsonl_fsync, atomic_write_json, sha256_file
from probekv.model_adapters import MISTRAL_SCHEMA6_SPEC
from probekv.v6_a800_executor import RealCacheBlendA800Executor
from probekv.v8_schema6_hbm import GIB, UnifiedHBMReservationManager
from probekv.v8_schema6_jobs import build_mistral_schema6_runtime_profile_jobs
from probekv.v8_schema6_profile import (
    build_schema6_runtime_cost_profile,
    make_measurement_cell,
    validate_schema6_runtime_cost_profile,
)
from probekv.v8_schema6_transfer import Schema6FullKVTransferAuthorizer
from probekv.v8_schema6_workspace import acquire_elastic_selection_workspace


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def command(repo: Path, *values: str) -> str:
    return subprocess.check_output(values, cwd=str(repo), text=True).strip()


def digest_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def require_cacheblend_runtime_source(
    cacheblend_root: Path, patch_audit: Mapping[str, Any]
) -> dict[str, Any]:
    root = cacheblend_root.resolve()
    package_root = (root / "vllm_blend" / "vllm").resolve()
    if command(root, "git", "rev-parse", "HEAD") != patch_audit.get(
        "cacheblend_commit"
    ):
        raise ValueError("CacheBlend runtime source has the wrong base commit")
    if command(root, "git", "write-tree") != patch_audit.get("cacheblend_tree"):
        raise ValueError("CacheBlend runtime source tree differs from patch audit")
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != package_root / "__init__.py":
        raise RuntimeError("vLLM does not resolve to the audited CacheBlend tree")
    return {"vllm_origin": str(Path(spec.origin).resolve())}


def profile_axes(kind: str, coordinates: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "comparison_batch": ("k", "token_count", "completed_depth", "backing_tier"),
        "selection_state_transfer": ("bytes", "tier", "batch_k"),
        "full_kv_tier_load": ("bytes", "source_tier", "layer_range"),
        "dense_remaining_joint": ("boundary_vector", "active_rows", "segment_count"),
        "repair": ("boundary", "token_count", "repair_count"),
        "union_mask_remaining": ("boundary_vector", "layer_active_rows"),
        "interference": ("copy_bytes", "overlap", "concurrency"),
        "scheduler_blocking": ("policy", "concurrency", "ready_resume_state"),
    }[kind]
    return {key: coordinates[key] for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sentinel-gate", required=True)
    parser.add_argument(
        "--lock", default="configs/a800_server_lock_v8_schema6_profile.json"
    )
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--cacheblend", required=True)
    parser.add_argument("--ssd-staging", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hourly-price-cny", required=True, type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest).resolve()
    sentinel_path = Path(args.sentinel_gate).resolve()
    manifest = load(manifest_path)
    sentinel = load(sentinel_path)
    lock = load((repo / args.lock).resolve())
    model_audit = load(Path(args.model_audit).resolve())
    patch_audit = load(Path(args.patch_audit).resolve())
    if (manifest.get("protocol_version"), manifest.get("schema_version")) != (8, 6):
        raise ValueError("Runtime Profile manifest is not v8 schema-v6")
    if sentinel.get("schema_v6_runtime_contract_passed") is not True:
        raise ValueError("Runtime Profile requires a passing schema-v6 sentinel")
    if sha256_file(sentinel_path) != manifest.get("sentinel_gate_sha256"):
        raise ValueError("Runtime Profile manifest binds another sentinel Gate")
    limits = manifest["limits"]
    if args.hourly_price_cny > float(limits["max_hourly_cny"]):
        raise RuntimeError("GPU hourly price exceeds the frozen limit")
    if args.hourly_price_cny * float(limits["max_session_hours"]) > float(
        limits["max_total_cny"]
    ):
        raise RuntimeError("GPU session would exceed the frozen total budget")
    code_commit = command(repo, "git", "rev-parse", "HEAD")
    if code_commit != manifest.get("code_commit") or command(
        repo, "git", "status", "--porcelain"
    ):
        raise ValueError("Runtime Profile requires its exact clean checkout")
    locked_model = lock["models"]["mistral"]
    if (
        model_audit.get("complete") is not True
        or model_audit.get("model_id") != locked_model["model_id"]
        or model_audit.get("revision") != locked_model["revision"]
        or model_audit.get("tokenizer_hash") != manifest.get("tokenizer_hash")
    ):
        raise ValueError("Mistral model audit differs from Runtime Profile manifest")
    if (
        patch_audit.get("patch_mode") != lock["stack"]["cacheblend_patch_mode"]
        or patch_audit.get("cacheblend_patch_sha256")
        != manifest.get("cacheblend_patch_sha256")
        or patch_audit.get("cacheblend_tree") != manifest.get("cacheblend_tree")
    ):
        raise ValueError("CacheBlend audit differs from Runtime Profile manifest")
    runtime_source_audit = require_cacheblend_runtime_source(
        Path(args.cacheblend), patch_audit
    )
    policy = str(manifest["selection_execution_policy"])
    jobs = build_mistral_schema6_runtime_profile_jobs(policy)
    if digest_payload(jobs) != manifest.get("jobs_sha256"):
        raise ValueError("Runtime Profile jobs differ from the frozen manifest")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "measurements.jsonl"
    existing = []
    if results_path.exists():
        if not args.resume:
            raise FileExistsError("Profile measurements exist; use --resume or a new output")
        existing = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if [row["job_id"] for row in existing] != [
        row["job_id"] for row in jobs[: len(existing)]
    ]:
        raise ValueError("resume measurements are not an immutable job prefix")
    if any(row.get("passed") is not True for row in existing):
        raise RuntimeError("failed Runtime Profile output cannot be resumed")

    import torch

    holder: dict[str, UnifiedHBMReservationManager] = {}

    def hbm_manager() -> UnifiedHBMReservationManager:
        manager = holder.get("manager")
        if manager is None:
            free_bytes, _ = torch.cuda.mem_get_info()
            manager = UnifiedHBMReservationManager(
                allocator_capacity_bytes=int(free_bytes), safety_bytes=4 * GIB
            )
            holder["manager"] = manager
        return manager

    active_workspace_bytes = {"value": 0}

    def workspace_capacity() -> int:
        if active_workspace_bytes["value"]:
            return active_workspace_bytes["value"]
        manager = holder.get("manager")
        if manager is not None:
            return manager.selector_lease_bytes
        free_bytes, _ = torch.cuda.mem_get_info()
        return max(0, int(free_bytes) - 4 * GIB)

    transfer_authorizer = Schema6FullKVTransferAuthorizer(
        hbm_manager_provider=hbm_manager
    )
    executor = RealCacheBlendA800Executor(
        model_path=str(model_audit["snapshot_path"]),
        model_spec=MISTRAL_SCHEMA6_SPEC,
        max_model_len=4096,
        gpu_memory_utilization=0.70,
        expected_cacheblend_tree=str(patch_audit["cacheblend_tree"]),
        engine_class=CacheBlendV8OnlineEngine,
        protocol_version=8,
        runtime_schema_version=6,
        selection_workspace_capacity_provider=workspace_capacity,
        full_kv_transfer_authorizer=transfer_authorizer,
    )
    hbm_manager()
    provenance = dict(executor.runtime_provenance)
    if not re.match(lock["gpu"]["name_regex"], provenance["gpu_name"]):
        raise RuntimeError("Runtime Profile GPU is not the frozen A800 80GB")
    if provenance["compute_capability"] != [8, 0]:
        raise RuntimeError("Runtime Profile requires compute capability 8.0")

    started = time.monotonic()
    max_seconds = float(limits["max_session_hours"]) * 3600
    ssd_staging = str(Path(args.ssd_staging).resolve())
    for job in jobs[len(existing) :]:
        if time.monotonic() - started >= max_seconds:
            break
        kind = str(job["kind"])
        coordinates = dict(job["coordinates"])
        row: dict[str, Any] = {
            "job_id": job["job_id"],
            "kind": kind,
            "coordinates": coordinates,
            "paper_evidence": False,
            "locked_test_accessed": False,
        }
        try:
            if kind == "comparison_batch":
                head_dim = int(executor.inner_model.layers[0].self_attn.head_dim)
                state_bytes = (
                    int(coordinates["token_count"])
                    * MISTRAL_SCHEMA6_SPEC.num_kv_heads
                    * head_dim
                    * 2
                )
                workspace = acquire_elastic_selection_workspace(
                    hbm_manager(),
                    owner_request_id=str(job["job_id"]),
                    segment_id="selection",
                    compared_k=int(coordinates["k"]),
                    current_state_bytes=state_bytes,
                    per_source_state_bytes=state_bytes,
                )
                if not workspace.one_shot:
                    raise RuntimeError("A800 full Profile expected one-shot K<=16 comparison")
                active_workspace_bytes["value"] = workspace.reserved_bytes
                try:
                    measurement = dict(
                        executor.measure_schema6_comparison_batch(
                            compared_k=int(coordinates["k"]),
                            token_count=int(coordinates["token_count"]),
                            completed_depth=int(coordinates["completed_depth"]),
                            warmups=int(job["warmups"]),
                            repeats=int(job["repeats"]),
                        )
                    )
                finally:
                    active_workspace_bytes["value"] = 0
                    hbm_manager().release(workspace.reservation.reservation_id)
            elif kind == "selection_state_transfer":
                if coordinates["tier"] == "pinned_cpu":
                    measurement = dict(
                        executor.measure_schema6_transfer(
                            bytes_count=int(coordinates["bytes"]),
                            warmups=int(job["warmups"]),
                            repeats=int(job["repeats"]),
                        )
                    )
                else:
                    measurement = dict(
                        executor.measure_schema6_ssd_staged_transfer(
                            bytes_count=int(coordinates["bytes"]),
                            staging_directory=ssd_staging,
                            warmups=int(job["warmups"]),
                            repeats=int(job["repeats"]),
                        )
                    )
            elif kind == "full_kv_tier_load":
                measurement = dict(
                    executor.measure_schema6_full_kv_load(
                        token_count=int(coordinates["token_count"]),
                        layer_range=tuple(coordinates["layer_range"]),
                        source_tier=str(coordinates["source_tier"]),
                        staging_directory=ssd_staging,
                        warmups=int(job["warmups"]),
                        repeats=int(job["repeats"]),
                    )
                )
            elif kind in {"dense_remaining_joint", "repair", "union_mask_remaining"}:
                if kind == "dense_remaining_joint":
                    segments = int(coordinates["segment_count"])
                    token_count = int(coordinates["active_rows"]) // segments
                    boundary = int(coordinates["boundary_vector"][0])
                    ratio = 1.0
                elif kind == "repair":
                    segments = 1
                    token_count = int(coordinates["token_count"])
                    boundary = int(coordinates["boundary"])
                    ratio = float(coordinates["repair_ratio"])
                else:
                    segments = int(coordinates["segment_count"])
                    token_count = 512
                    boundary = int(coordinates["boundary_vector"][0])
                    ratio = 0.15
                measurement = dict(
                    executor.measure_schema6_joint_operation(
                        segment_count=segments,
                        segment_token_count=token_count,
                        boundary=boundary,
                        repair_ratio=ratio,
                        warmups=int(job["warmups"]),
                        repeats=int(job["repeats"]),
                    )
                )
            elif kind == "interference":
                measurement = dict(
                    executor.measure_schema6_copy_interference(
                        copy_bytes=int(coordinates["copy_bytes"]),
                        overlap=bool(coordinates["overlap"]),
                        concurrency=int(coordinates["concurrency"]),
                        warmups=int(job["warmups"]),
                        repeats=int(job["repeats"]),
                    )
                )
            elif kind == "scheduler_blocking":
                measurement = dict(
                    executor.measure_schema6_scheduler_blocking(
                        policy=str(coordinates["policy"]),
                        concurrency=int(coordinates["concurrency"]),
                        ready_resume_state=str(coordinates["ready_resume_state"]),
                        warmups=int(job["warmups"]),
                        repeats=int(job["repeats"]),
                    )
                )
            else:
                raise RuntimeError("unknown Runtime Profile job category")
            samples = tuple(float(value) for value in measurement["measurements_ms"])
            if len(samples) != int(job["repeats"]) or any(value < 0 for value in samples):
                raise RuntimeError("Runtime Profile cell lacks its frozen timing samples")
            row["measurement"] = measurement
            row["passed"] = True
        except Exception as error:
            row["passed"] = False
            row["error"] = "%s: %s" % (type(error).__name__, error)
        append_jsonl_fsync(results_path, [row])
        existing.append(row)
        atomic_write_json(
            output / "resume_checkpoint.json",
            {
                "protocol_version": 8,
                "schema_version": 6,
                "completed": len(existing),
                "planned": len(jobs),
                "failed": sum(item.get("passed") is not True for item in existing),
                "elapsed_seconds_this_run": time.monotonic() - started,
                "paper_evidence": False,
            },
        )
        if row["passed"] is not True:
            break

    completed = len(existing)
    failed = sum(row.get("passed") is not True for row in existing)
    success = completed == len(jobs) and failed == 0
    gate: dict[str, Any] = {
        "protocol_version": 8,
        "schema_version": 6,
        "stage": "mistral_schema6_runtime_profile_measurement",
        "code_commit": code_commit,
        "manifest_sha256": sha256_file(manifest_path),
        "planned": len(jobs),
        "completed": completed,
        "failed": failed,
        "runtime_profile_measurements_complete": success,
        "runtime_cost_profile_frozen": False,
        "selector_profile_frozen": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "runtime_source_audit": runtime_source_audit,
        "full_kv_transfer_audit": transfer_authorizer.audit(),
    }
    if success:
        cells = [
            make_measurement_cell(
                str(row["kind"]),
                axes=profile_axes(str(row["kind"]), row["coordinates"]),
                measurements_ms=row["measurement"]["measurements_ms"],
                warmups=int(jobs[index]["warmups"]),
            )
            for index, row in enumerate(existing)
        ]
        hardware_signature = str(
            provenance.get("hardware_compatibility_signature")
            or digest_payload(
                {
                    key: provenance.get(key)
                    for key in (
                        "gpu_name",
                        "compute_capability",
                        "torch_version",
                        "cuda_version",
                        "vllm_version",
                    )
                }
            )
        )
        profile = build_schema6_runtime_cost_profile(
            model_key="mistral",
            policy=policy,
            code_commit=code_commit,
            cacheblend_patch_sha256=str(patch_audit["cacheblend_patch_sha256"]),
            gpu_uuid=str(provenance["gpu_uuid"]),
            hardware_compatibility_signature=hardware_signature,
            measurement_cells=cells,
            measurement_sha256=sha256_file(results_path),
            frozen=True,
        )
        validate_schema6_runtime_cost_profile(profile, require_frozen=True)
        atomic_write_json(output / "runtime_cost_profile.json", profile)
        gate["runtime_cost_profile_frozen"] = True
        gate["runtime_cost_profile_sha256"] = profile[
            "runtime_cost_profile_sha256"
        ]
        gate["gpu_uuid"] = provenance["gpu_uuid"]
    atomic_write_json(output / "runtime_profile_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
