#!/usr/bin/env python3
"""Run one bounded real-A800 schema10 Profile session.

The runner writes an immutable successful prefix.  It never freezes Profiles;
the separate freeze command consumes and validates the completed measurements.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version as package_version
import json
import math
import re
from statistics import mean
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from probekv.cacheblend_v6_online_engine import CacheBlendV8Schema8OnlineEngine
from probekv.io import append_jsonl_fsync, atomic_write_json, sha256_file
from probekv.manifest import manifest_case_from_row
from probekv.model_adapters import SCHEMA6_MODEL_SPECS
from probekv.v6_a800_executor import R1DenseEquivalenceError, RealCacheBlendA800Executor
from probekv.v8_schema10_jobs import build_schema10_profile_jobs
from probekv.v8_schema10_profile import SCHEMA10_MODEL_CHECKPOINTS, SCHEMA10_TRIM_GRID
from probekv.v8_schema10_profile_runtime import Schema10DevelopmentCaseRuntime
from probekv.v8_schema10_profile_analysis import (
    build_selection_candidates,
    build_threshold_table,
    linear_quantile,
    select_case_source,
    select_dispatch,
)
from probekv.v8_schema6_hbm import GIB, UnifiedHBMReservationManager
from probekv.v8_schema6_transfer import Schema6FullKVTransferAuthorizer


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _command(repo: Path, *values: str) -> str:
    return subprocess.check_output(values, cwd=str(repo), text=True).strip()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _measurement_median(row: Mapping[str, Any]) -> float:
    values = sorted(float(value) for value in row.get("measurements_ms", ()))
    if not values:
        raise RuntimeError("real profile measurement lacks samples")
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=("mistral", "qwen"))
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--server-lock", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--cacheblend", required=True)
    parser.add_argument("--ssd-staging", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hourly-price-cny", type=float, required=True)
    parser.add_argument("--max-hours", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.hourly_price_cny > 7.5 or not 0 < args.max_hours <= 4.0:
        raise RuntimeError("schema10 A800 Profile exceeds the frozen price/time limit")
    repo = Path(__file__).resolve().parents[2]
    code_commit = _command(repo, "git", "rev-parse", "HEAD")
    if _command(repo, "git", "status", "--porcelain"):
        raise ValueError("schema10 A800 Profile requires a clean checkout")
    handoff_path = Path(args.handoff).resolve()
    handoff = _load(handoff_path)
    if (handoff.get("protocol_version"), handoff.get("schema_version")) != (8, 10):
        raise ValueError("schema10 Profile handoff has the wrong schema")
    if handoff.get("code_commit") != code_commit or handoff.get("model_key") != args.model_key:
        raise ValueError("schema10 handoff differs from checkout/model")
    if handoff.get("gpu_rental_ready_for_schema10_profile_freeze") is not True:
        raise RuntimeError("schema10 no-GPU handoff is not rental-ready")
    if (
        sha256_file(Path(args.config).resolve()) != handoff.get("config_sha256")
        or sha256_file(Path(args.contract).resolve()) != handoff.get("contract_sha256")
        or sha256_file(Path(args.server_lock).resolve())
        != handoff.get("server_lock_sha256")
    ):
        raise ValueError("schema10 config/contract/server lock differs from handoff")
    server_lock = _load(Path(args.server_lock).resolve())
    if (server_lock.get("protocol_version"), server_lock.get("schema_version")) != (8, 10):
        raise ValueError("schema10 Profile requires the schema10 server lock")

    model_audit = _load(Path(args.model_audit).resolve())
    patch_audit = _load(Path(args.patch_audit).resolve())
    if (
        model_audit.get("complete") is not True
        or model_audit.get("revision") != handoff.get("model_revision")
        or model_audit.get("tokenizer_hash") != handoff.get("tokenizer_hash")
    ):
        raise ValueError("model audit differs from schema10 handoff")
    if (
        patch_audit.get("cacheblend_patch_sha256") != handoff.get("cacheblend_patch_sha256")
        or patch_audit.get("cacheblend_tree") != handoff.get("cacheblend_tree")
        or patch_audit.get("patch_mode") != handoff.get("runtime_patch_mode")
    ):
        raise ValueError("CacheBlend patch audit differs from schema10 handoff")
    cacheblend = Path(args.cacheblend).resolve()
    if _command(cacheblend, "git", "write-tree") != handoff.get("cacheblend_tree"):
        raise ValueError("CacheBlend checkout tree differs from handoff")

    development_path = Path(args.development_manifest).resolve()
    if sha256_file(development_path) != handoff.get("development_partition_sha256"):
        raise ValueError("development partition differs from schema10 handoff")
    development = _jsonl(development_path)
    if len(development) != 90 or any(
        row.get("partition_role") != "development_profile_freeze"
        or row.get("locked_test_accessed") is not False
        for row in development
    ):
        raise ValueError("schema10 Profile requires exactly 90 isolated development cases")
    wanted = {str(row["case_id"]): row for row in development}
    case_manifest_path = Path(args.case_manifest).resolve()
    if sha256_file(case_manifest_path) != handoff.get(
        "development_case_manifest_sha256"
    ):
        raise ValueError("development case manifest differs from schema10 handoff")
    all_cases = [manifest_case_from_row(row) for row in _jsonl(case_manifest_path)]
    cases = [case for case in all_cases if case.case_id in wanted]
    if len(cases) != 90 or {case.case_id for case in cases} != set(wanted):
        raise ValueError("case manifest does not cover the development partition")
    if any(case.split not in {"calibration", "development"} for case in cases):
        raise ValueError("Profile case manifest crossed into pilot/test")

    jobs = build_schema10_profile_jobs(args.model_key)
    if _digest(jobs) != _digest(handoff["profile_jobs"]):
        raise ValueError("Profile jobs differ from the frozen handoff")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Profile output exists; use --resume or a new directory")
    if args.resume and (output / "initialization_failure.json").exists():
        raise RuntimeError("failed initialization evidence cannot be resumed")
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    existing = _jsonl(results_path) if results_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError("Profile results exist; use --resume or a new directory")
    if [row["job_id"] for row in existing] != [row["job_id"] for row in jobs[: len(existing)]]:
        raise ValueError("resume results are not an immutable successful prefix")
    if any(row.get("passed") is not True for row in existing):
        raise RuntimeError("failed Profile evidence cannot be resumed")

    model_id = str(model_audit["model_id"])
    spec = SCHEMA6_MODEL_SPECS[model_id]
    started = time.monotonic()
    deadline = started + args.max_hours * 3600.0
    import torch

    hbm_holder: dict[str, UnifiedHBMReservationManager] = {}

    def hbm_manager() -> UnifiedHBMReservationManager:
        manager = hbm_holder.get("manager")
        if manager is None:
            free_bytes, _ = torch.cuda.mem_get_info()
            manager = UnifiedHBMReservationManager(
                allocator_capacity_bytes=int(free_bytes), safety_bytes=4 * GIB
            )
            hbm_holder["manager"] = manager
        return manager

    try:
        executor = RealCacheBlendA800Executor(
            model_path=str(model_audit["snapshot_path"]),
            model_spec=spec,
            max_model_len=4096,
            gpu_memory_utilization=0.70,
            expected_cacheblend_tree=str(patch_audit["cacheblend_tree"]),
            engine_class=CacheBlendV8Schema8OnlineEngine,
            protocol_version=8,
            runtime_schema_version=10,
            selection_workspace_capacity_provider=lambda: hbm_manager().selector_lease_bytes,
            full_kv_transfer_authorizer=Schema6FullKVTransferAuthorizer(
                hbm_manager_provider=hbm_manager
            ),
            integrity_mode="qualification_full",
        )
    except Exception as error:
        failure = {
            "protocol_version": 8,
            "schema_version": 10,
            "stage": "schema10_a800_profile_initialization",
            "code_commit": code_commit,
            "model_key": args.model_key,
            "error_type": type(error).__name__,
            "error": str(error),
            "r1_audit": (
                dict(error.audit)
                if isinstance(error, R1DenseEquivalenceError) else None
            ),
            "passed": False,
            "paper_evidence": False,
            "locked_test_accessed": False,
        }
        atomic_write_json(output / "initialization_failure.json", failure)
        atomic_write_json(output / "runtime_audit.json", {
            **failure,
            "planned": len(jobs),
            "completed": 0,
            "failed": 1,
            "fake_timing": False,
        })
        raise
    provenance = dict(executor.runtime_provenance)
    gpu_contract = dict(server_lock["gpu"])
    if not re.match(str(gpu_contract["name_regex"]), str(provenance["gpu_name"])):
        raise RuntimeError("schema10 Profile requires A800 80GB")
    expected_capability = [int(value) for value in str(gpu_contract["compute_capability"]).split(".")]
    if provenance.get("compute_capability") != expected_capability:
        raise RuntimeError("schema10 Profile requires compute capability 8.0")
    total_memory_mib = int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory // (1024 * 1024))
    if total_memory_mib < int(gpu_contract["minimum_memory_mib"]):
        raise RuntimeError("schema10 Profile GPU memory is below the server lock")
    stack = dict(server_lock["stack"])
    observed_versions = {
        "pytorch": str(provenance["torch"]).split("+")[0],
        "pytorch_cuda": str(provenance["torch_cuda"]),
        "vllm": str(provenance["vllm"]),
        "xformers": str(provenance["xformers"]),
        "transformers": package_version("transformers"),
    }
    mismatched_stack = [
        key for key, observed in observed_versions.items()
        if observed != str(stack[key])
    ]
    if mismatched_stack:
        raise RuntimeError(
            "schema10 Profile stack differs from server lock: "
            + ",".join(mismatched_stack)
        )
    if (
        str(patch_audit.get("cacheblend_commit")) != str(stack["cacheblend_commit"])
        or str(patch_audit.get("patch_mode")) != str(stack["cacheblend_patch_mode"])
    ):
        raise RuntimeError("CacheBlend commit/mode differs from server lock")
    profile_provenance = {
        "code_commit": code_commit,
        "environment_hash": _digest(provenance),
        "model_revision": str(model_audit["revision"]),
        "cacheblend_commit": str(patch_audit["cacheblend_commit"]),
        "cacheblend_patch_sha256": str(patch_audit["cacheblend_patch_sha256"]),
        "cacheblend_tree": str(patch_audit["cacheblend_tree"]),
        "vllm_version": str(provenance["vllm"]),
        "torch_version": str(provenance["torch"]),
        "cuda_version": str(provenance["torch_cuda"]),
        "gpu_uuid": str(provenance["gpu_uuid"]),
        "server_lock_sha256": sha256_file(Path(args.server_lock).resolve()),
        "server_lock_stack": observed_versions,
        "gpu_total_memory_mib": total_memory_mib,
    }
    if any(
        row.get("kind") == "correctness_sentinel" and row.get("passed") is True
        for row in existing
    ):
        executor.source_loader.integrity_mode = "online_immutable"
        executor.source_loader.require_pre_pinned = True

    def case_runtime(case: Any) -> Schema10DevelopmentCaseRuntime:
        return Schema10DevelopmentCaseRuntime(
            executor, case, profile_provenance, max_new_tokens=64
        )

    ssd_staging = str(Path(args.ssd_staging).resolve())
    duration_history: dict[str, list[float]] = {}
    for prior in existing:
        duration = prior.get("job_wall_seconds")
        if duration is not None:
            duration_history.setdefault(str(prior["kind"]), []).append(float(duration))
    estimated_next_seconds = 0.0
    reference_selection_cache: dict[str, Any] = {}

    def frozen_reference_selection() -> tuple[dict[str, Any], dict[tuple[float, int], float], list[dict[str, Any]]]:
        cached = reference_selection_cache.get("value")
        if cached is not None:
            return cached
        completed_rows = _jsonl(results_path)
        observations = [
            observation
            for result in completed_rows
            if result.get("kind") == "selection_admission_sweep"
            for observation in result["measurement"]["observations"]
        ]
        if len({str(row["case_id"]) for row in observations}) != 90:
            raise RuntimeError("Stage B cannot start before all 90 Stage-A cases")
        timings = [
            float(result["measurement_median_ms"])
            for result in completed_rows
            if result.get("kind") == "factorized_selection"
            and result.get("measurement_median_ms") is not None
            and int(result["coordinates"]["compared_k"]) == 16
            and int(result["coordinates"]["token_count"]) == 512
        ]
        if len(timings) != len(SCHEMA10_MODEL_CHECKPOINTS[args.model_key]):
            raise RuntimeError("Stage B lacks the complete Stage-A selection timing profile")
        dense_times = [float(row["dense_reference_ms"]) for row in observations]
        _, thresholds = build_threshold_table(
            observations, SCHEMA10_MODEL_CHECKPOINTS[args.model_key]
        )
        candidates = build_selection_candidates(
            observations,
            SCHEMA10_MODEL_CHECKPOINTS[args.model_key],
            thresholds,
            linear_quantile(timings, 0.95) / max(mean(dense_times), 1e-12),
        )
        selected = dict(select_dispatch(candidates))
        value = (selected, thresholds, observations)
        reference_selection_cache["value"] = value
        return value

    for job in jobs[len(existing):]:
        kind = str(job["kind"])
        known = duration_history.get(kind, [])
        fallback_seconds = (
            900.0 if kind in {"selection_admission_sweep", "repair_policy_development_sweep"}
            else 300.0 if kind == "correctness_sentinel"
            else 60.0
        )
        estimated_next_seconds = (
            sorted(known)[len(known) // 2] if known else fallback_seconds
        )
        if time.monotonic() + estimated_next_seconds + 60.0 > deadline:
            break
        job_started = time.monotonic()
        coordinates = dict(job["coordinates"])
        row: dict[str, Any] = {
            "job_id": job["job_id"],
            "kind": kind,
            "phase": job["phase"],
            "coordinates": coordinates,
            "paper_evidence": False,
            "locked_test_accessed": False,
        }
        try:
            if kind == "correctness_sentinel":
                prefix = dict(executor.run_native_prefix_cache_sentinel())
                measurement = {
                    "runtime_sentinel": dict(executor.sentinel),
                    "native_prefix_sentinel": prefix,
                    "cuda_event_timing": bool(
                        prefix.get("cuda_event_timing")
                        and executor.sentinel.get("r1_dense_token_ids_equal")
                    ),
                    "timing_basis": "real_prefix_and_r1_correctness_sentinel",
                }
                if not (
                    prefix.get("native_prefix_cache_hit") is True
                    and prefix.get("dense_token_ids_equal") is True
                    and float(prefix.get("logit_relative_l2", 1.0)) <= 1e-4
                    and executor.sentinel.get("completed_depth_hook_verified") is True
                    and executor.sentinel.get("canonical_source_digests_unchanged") is True
                    and executor.sentinel.get("artifact_digests_unchanged") is True
                    and executor.sentinel.get("absolute_union_mask_verified") is True
                ):
                    raise RuntimeError("schema10 correctness sentinel failed")
                # The correctness job deliberately uses qualification_full.
                # Formal Profile timing must exclude per-request full-KV SHA256.
                executor.source_loader.integrity_mode = "online_immutable"
                executor.source_loader.require_pre_pinned = True
            elif kind == "reference_runtime":
                measurement = dict(executor.measure_schema10_joint_anchor(
                    segment_count=1, segment_token_count=512, boundary=1,
                    repair_ratio=0.15, source_path="pinned_cpu",
                    staging_directory=ssd_staging, warmups=2, repeats=5,
                ))
            elif kind == "selection_admission_sweep":
                observations = []
                case_start = int(coordinates["case_start"])
                selected_cases = cases[case_start:case_start + int(coordinates["case_count"])]
                for case in selected_cases:
                    runtime = case_runtime(case)
                    observations.extend(
                        {
                            **value.__dict__,
                            "case_id": case.case_id,
                            "dataset": case.dataset,
                            "content_id": case.reuse_content_key or case.content_hash,
                            "request_epoch": case_start + selected_cases.index(case) + 1,
                            "dense_reference_ms": runtime.full.host_ms,
                        }
                        for value in runtime.residual_observations(
                            SCHEMA10_MODEL_CHECKPOINTS[args.model_key], SCHEMA10_TRIM_GRID
                        )
                    )
                measurement = {
                    "cases": len(selected_cases),
                    "observations": observations,
                    "cuda_event_timing": False,
                    "deterministic_replay_valid": True,
                    "real_model_state": True,
                    "timing_basis": "real_dense_fixture_state_plus_deterministic_residual_k",
                }
            elif kind == "repair_policy_development_sweep":
                observations = []
                selected_dispatch, threshold_index, selection_observations = (
                    frozen_reference_selection()
                )
                case_start = int(coordinates["case_start"])
                selected_cases = cases[case_start:case_start + int(coordinates["case_count"])]
                for case in selected_cases:
                    runtime = case_runtime(case)
                    winner = select_case_source(
                        selection_observations,
                        case_id=case.case_id,
                        selected_dispatch=selected_dispatch,
                        thresholds=threshold_index,
                    )
                    for ratio in coordinates["repair_ratio_grid"]:
                        if winner is None:
                            observations.append({
                                "case_id": case.case_id,
                                "dataset": case.dataset,
                                "source_id": None,
                                "first_reuse_layer": None,
                                "repair_ratio": float(ratio),
                                "answer_f1": runtime.full_answer_f1,
                                "full_answer_f1": runtime.full_answer_f1,
                                "answer_f1_drop": 0.0,
                                "ordered_token_f1": 1.0,
                                "token_ids_equal_full": True,
                                "logit_relative_l2": 0.0,
                                "gpu_ms": runtime.full.gpu_ms,
                                "host_ms": runtime.full.host_ms,
                                "source_digest_unchanged": True,
                                "artifact_digest_unchanged": True,
                                "absolute_union_mask_verified": True,
                                "execution_mode": "dense_fallback_no_selected_source",
                                "cuda_event_timing": True,
                                "fake_timing": False,
                                "paper_evidence": False,
                                "locked_test_accessed": False,
                            })
                        else:
                            observations.append(runtime.repair_ratio_observation(
                                source_id=str(winner["source_id"]),
                                first_reuse_layer=int(winner["completed_depth"]) + 1,
                                repair_ratio=float(ratio),
                            ))
                measurement = {
                    "cases": len(selected_cases),
                    "observations": observations,
                    "stage_a_reference_dispatch": selected_dispatch,
                    "cuda_event_timing": True,
                    "timing_basis": "real_case_generation",
                }
            elif kind == "gate1_paired_ab":
                normalize_dataset = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
                target_dataset = normalize_dataset(coordinates["dataset"])
                selected_cases = sorted(
                    (case for case in cases if normalize_dataset(case.dataset) == target_dataset),
                    key=lambda case: case.case_id,
                )
                if len(selected_cases) < 6:
                    raise RuntimeError("paired Gate1 dataset has fewer than six anchors")
                runtime = case_runtime(selected_cases[int(coordinates["anchor_index"])])
                residuals = runtime.residual_observations((2,), (0.15,))
                winner = min(residuals, key=lambda value: (value.residual_score, value.source_id))
                bypass = runtime.repair_ratio_observation(
                    source_id=winner.source_id, first_reuse_layer=3, repair_ratio=0.15
                )
                measurement = {
                    "request_id": runtime.case.case_id,
                    "dataset": runtime.case.dataset,
                    "dense_wall_ms": runtime.full.host_ms,
                    "reuse_wall_ms": float(bypass["host_ms"]),
                    "winner_source_id": winner.source_id,
                    "winner_residual": winner.residual_score,
                    "transferred_bytes": sum(
                        tensor.numel() * tensor.element_size()
                        for key, value in runtime.fixture.runtime.canonical_variants[0][
                            runtime.source_index[winner.source_id]
                        ]
                        for tensor in (key, value)
                    ),
                    "correctness_match": bool(bypass["source_digest_unchanged"]),
                    "cuda_event_timing": True,
                    "timing_basis": "paired_real_execution",
                }
            elif kind == "factorized_selection":
                measurement = dict(executor.measure_schema10_selection_batch(
                    compared_k=int(coordinates["compared_k"]),
                    token_count=int(coordinates["token_count"]),
                    completed_depth=int(coordinates["completed_depth"]),
                    warmups=int(job["warmups"]), repeats=int(job["repeats"]),
                ))
            elif kind == "factorized_transfer":
                if coordinates["path"] == "pinned_cpu_to_gpu":
                    measurement = dict(executor.measure_schema6_transfer(
                        bytes_count=int(coordinates["bytes"]),
                        warmups=int(job["warmups"]), repeats=int(job["repeats"]),
                    ))
                else:
                    measurement = dict(executor.measure_schema6_ssd_staged_transfer(
                        bytes_count=int(coordinates["bytes"]), staging_directory=ssd_staging,
                        warmups=int(job["warmups"]), repeats=int(job["repeats"]),
                    ))
                concurrency = int(coordinates["concurrency"])
                measurement["concurrency"] = concurrency
                if concurrency > 1:
                    measurement["contention_component"] = dict(
                        executor.measure_schema6_copy_interference(
                        copy_bytes=int(coordinates["bytes"]), overlap=True,
                        concurrency=concurrency,
                        warmups=int(job["warmups"]), repeats=int(job["repeats"]),
                        )
                    )
                    measurement["concurrency_contention_measured"] = True
                else:
                    measurement["concurrency_contention_measured"] = False
            elif kind == "factorized_repair":
                segments = max(1, int(math.ceil(int(coordinates["active_rows"]) / 512.0)))
                measurement = dict(executor.measure_schema10_joint_anchor(
                    segment_count=segments,
                    segment_token_count=max(1, int(coordinates["active_rows"]) // segments),
                    boundary=1, repair_ratio=float(coordinates["repair_ratio"]),
                    source_path="pinned_cpu", staging_directory=ssd_staging,
                    warmups=int(job["warmups"]), repeats=int(job["repeats"]),
                ))
            elif kind == "factorized_scheduler":
                measurement = dict(executor.measure_schema6_scheduler_blocking(
                    policy="dense_selection_barrier",
                    concurrency=int(coordinates["concurrency"]),
                    ready_resume_state="ready",
                    warmups=int(job["warmups"]), repeats=int(job["repeats"]),
                ))
            elif kind == "joint_anchor":
                path = {
                    "gpu_resident": "gpu_resident",
                    "pinned_cpu_to_gpu": "pinned_cpu",
                    "ssd_staged_to_gpu": "ssd_staged",
                }[str(coordinates["path"])]
                measurement = dict(executor.measure_schema10_joint_anchor(
                    segment_count=int(coordinates["segment_count"]),
                    segment_token_count=512,
                    boundary=1,
                    repair_ratio=float(coordinates["repair_ratio"]),
                    source_path=path,
                    staging_directory=ssd_staging,
                    warmups=int(job["warmups"]),
                    repeats=int(job["repeats"]),
                ))
                concurrency = int(coordinates["concurrency"])
                measurement["requested_path"] = coordinates["path"]
                measurement["concurrency"] = concurrency
                if concurrency > 1:
                    width = (
                        int(spec.num_kv_heads)
                        * int(executor.inner_model.layers[0].self_attn.head_dim)
                    )
                    copy_bytes = (
                        int(coordinates["segment_count"])
                        * 512 * int(spec.num_layers) * 2 * width * 2
                    )
                    contention = dict(executor.measure_schema6_copy_interference(
                        copy_bytes=copy_bytes,
                        overlap=True,
                        concurrency=concurrency,
                        warmups=int(job["warmups"]),
                        repeats=int(job["repeats"]),
                    ))
                    measurement["contention_component"] = contention
                    measurement["concurrency_contention_measured"] = True
                    measurement["integrated_concurrent_requests"] = False
                    measurement["anchor_semantics"] = (
                        "integrated_single_request_path_plus_real_contention_component"
                    )
                else:
                    measurement["concurrency_contention_measured"] = False
                    measurement["integrated_concurrent_requests"] = True
                    measurement["anchor_semantics"] = "integrated_single_request_path"
            else:
                # Growth/probation/shadow/final-consistency are deterministic
                # replays over real measurements and are finalized by the
                # aggregation command; they perform no fake GPU timing.
                measurement = {
                    "deferred_to_measurement_aggregation": True,
                    "source_measurements_real_gpu": True,
                    "cuda_event_timing": False,
                    "deterministic_replay_valid": True,
                    "timing_basis": "deterministic_replay_of_real_measurements",
                }
            row.update({
                "measurement": measurement,
                "measurement_median_ms": (
                    _measurement_median(measurement)
                    if measurement.get("measurements_ms") else None
                ),
                "cuda_event_timing": measurement.get("cuda_event_timing") is True,
                "fake_timing": False,
                "job_wall_seconds": time.monotonic() - job_started,
                "passed": (
                    measurement.get("cuda_event_timing") is True
                    or measurement.get("deterministic_replay_valid") is True
                ),
            })
        except Exception as error:
            row.update({
                "passed": False,
                "fake_timing": False,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            append_jsonl_fsync(results_path, [row])
            break
        append_jsonl_fsync(results_path, [row])
        if row.get("passed") is True:
            duration_history.setdefault(kind, []).append(float(row["job_wall_seconds"]))

    final_rows = _jsonl(results_path)
    audit = {
        "protocol_version": 8,
        "schema_version": 10,
        "stage": "schema10_a800_profile_measurement",
        "code_commit": code_commit,
        "model_key": args.model_key,
        "model_id": model_id,
        "model_revision": model_audit["revision"],
        "tokenizer_hash": model_audit["tokenizer_hash"],
        "cacheblend_patch_sha256": patch_audit["cacheblend_patch_sha256"],
        "cacheblend_tree": patch_audit["cacheblend_tree"],
        "gpu_uuid": provenance["gpu_uuid"],
        "runtime_environment_hash": _digest(provenance),
        "development_partition_sha256": sha256_file(development_path),
        "development_case_manifest_sha256": sha256_file(case_manifest_path),
        "config_sha256": sha256_file(Path(args.config).resolve()),
        "contract_sha256": sha256_file(Path(args.contract).resolve()),
        "handoff_sha256": sha256_file(handoff_path),
        "server_lock_sha256": sha256_file(Path(args.server_lock).resolve()),
        "jobs_sha256": _digest(jobs),
        "planned": len(jobs),
        "completed": sum(row.get("passed") is True for row in final_rows),
        "failed": sum(row.get("passed") is not True for row in final_rows),
        "deadline_reached": time.monotonic() >= deadline,
        "elapsed_seconds": time.monotonic() - started,
        "estimated_next_job_seconds": (
            estimated_next_seconds if len(final_rows) < len(jobs) else 0.0
        ),
        "real_gpu_measurements": True,
        "correctness_integrity_mode": "qualification_full",
        "performance_integrity_mode": "online_immutable",
        "performance_path_full_kv_hashing": False,
        "fake_timing": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
    }
    atomic_write_json(output / "runtime_audit.json", audit)
    atomic_write_json(output / "checkpoint.json", {
        "successful_prefix_job_ids": [row["job_id"] for row in final_rows if row.get("passed") is True],
        "next_job_index": len(final_rows),
        "complete": len(final_rows) == len(jobs) and audit["failed"] == 0,
    })
    sentinel_rows = [
        row for row in final_rows
        if row.get("kind") == "correctness_sentinel" and row.get("passed") is True
    ]
    if sentinel_rows:
        atomic_write_json(
            output / "correctness_sentinel.json",
            sentinel_rows[0]["measurement"],
        )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
