"""Run the bounded non-paper ProbeKV v8 schema-v6 Mistral A800 sentinel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from probekv.cacheblend_v6_online_engine import CacheBlendV8OnlineEngine
from probekv.contracts import KVLocation
from probekv.io import append_jsonl_fsync, atomic_write_json, sha256_file
from probekv.model_adapters import MISTRAL_SCHEMA6_SPEC
from probekv.native_prefix_cache import evaluate_native_prefix_cache_audit
from probekv.v6_a800_executor import RealCacheBlendA800Executor
from probekv.v6_a800_jobs import V6A800Job, V6A800JobKind
from probekv.v8_leases import V8LeaseManager, V8ReplicaResource
from probekv.v8_schema6_contracts import (
    CommitAxisState,
    Gate2AxisState,
    Gate3SubsetDecision,
    PlannerSnapshot,
)
from probekv.v8_schema6_hbm import GIB, UnifiedHBMReservationManager
from probekv.v8_schema6_jobs import build_mistral_schema6_sentinel_jobs
from probekv.v8_schema6_runtime import Schema6RequestController
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


def _qualification_job(
    row: Mapping[str, Any], *, kind: V6A800JobKind, segment_count: int,
    stored_variants: int = 1, compared_variants: int = 0,
    probe_layer: int = 5, repair_ratio: float = 0.0,
) -> V6A800Job:
    return V6A800Job(
        job_id=str(row["job_id"]), kind=kind, segment_count=segment_count,
        stored_variants=stored_variants, compared_variants=compared_variants,
        probe_layer=probe_layer, repair_ratio=repair_ratio,
        warmups=int(row["warmups"]), repeats=int(row["repeats"]),
        paper_evidence=False,
    )


def _exercise_state_contract(policy: str) -> Mapping[str, Any]:
    leases = V8LeaseManager()
    leases.register_source("source-c1", "artifact-c1", "mistral")
    leases.register_replica(
        V8ReplicaResource(
            "cpu-c1", "source-c1", "artifact-c1", KVLocation.PINNED_CPU,
            1, 1, 1024, True,
        )
    )
    hbm = UnifiedHBMReservationManager(
        allocator_capacity_bytes=4 * GIB + 1024 * 1024, safety_bytes=4 * GIB
    )
    controller = Schema6RequestController(
        request_id="schema6-sentinel", request_generation=1,
        ordered_segment_ids=("c1",), policy=policy,
        lease_manager=leases, hbm_manager=hbm,
    )
    controller.decision_ready("c1", "source-c1", 4)
    controller.gate1(
        "c1", passed=True, at_lmax=False, predicted_remaining_s=1.0
    )
    snap = PlannerSnapshot(1, 1, "scheduler-sentinel", hbm.epoch, "sparse-profile")
    controller.apply_gate2(
        {"c1": Gate2AxisState.DEFERRED.value}, snapshot=snap, current_snapshot=snap
    )
    controller.begin_winner_prefetch(
        "c1", artifact_id="artifact-c1", replica_id="cpu-c1",
        replica_generation=1, placement_epoch=1, target_hbm_bytes=1024,
        predicted_remaining_s=1.0, speculative_resource_admitted=True,
    )
    controller.mark_winner_ready("c1", actual_reuse_boundary=5)
    ready_while_deferred = (
        controller.records["c1"].gate2_state is Gate2AxisState.DEFERRED
    )
    snap2 = PlannerSnapshot(1, 1, "scheduler-sentinel", hbm.epoch, "sparse-profile")
    controller.apply_gate2(
        {"c1": Gate2AxisState.PROVISIONAL_REUSE.value},
        snapshot=snap2, current_snapshot=snap2,
    )
    if policy == "causal_commit_wait":
        # The one-Segment inventory is already causally closed.
        assert controller.selection_closed_for("c1")
    snap3 = PlannerSnapshot(1, 1, "scheduler-sentinel", hbm.epoch, "sparse-profile")
    controller.apply_gate3_subset(
        Gate3SubsetDecision(
            ("c1",), (), (), 1.0, 2.0, snap3, {"c1": "sentinel_accept"}
        ),
        current_snapshot=snap3,
    )
    return {
        "ready_while_deferred": ready_while_deferred,
        "lease_promoted_atomically": (
            leases.leases[str(controller.records["c1"].physical_lease_id)].purpose.value
            == "execution"
        ),
        "commit_state": controller.records["c1"].commit_state.value,
        "passed": (
            ready_while_deferred
            and controller.records["c1"].commit_state is CommitAxisState.REUSE_COMMIT
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock", default="configs/a800_server_lock_v8_schema6.json")
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hourly-price-cny", required=True, type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    manifest = load(Path(args.manifest).resolve())
    lock = load((repo / args.lock).resolve())
    model_audit = load(Path(args.model_audit).resolve())
    patch_audit = load(Path(args.patch_audit).resolve())
    if (manifest.get("protocol_version"), manifest.get("schema_version")) != (8, 6):
        raise ValueError("schema-v6 sentinel requires a v8 schema-v6 manifest")
    if (lock.get("protocol_version"), lock.get("schema_version")) != (8, 6):
        raise ValueError("schema-v6 sentinel requires the schema-v6 server lock")
    limits = manifest["limits"]
    if args.hourly_price_cny > float(limits["max_hourly_cny"]):
        raise RuntimeError("GPU hourly price exceeds the frozen 7.5 CNY limit")
    if args.hourly_price_cny * float(limits["max_session_hours"]) > float(limits["max_total_cny"]):
        raise RuntimeError("GPU session would exceed the frozen 30 CNY limit")
    jobs = build_mistral_schema6_sentinel_jobs()
    if digest_payload(jobs) != manifest.get("jobs_sha256"):
        raise ValueError("schema-v6 sentinel jobs differ from the manifest")
    code_commit = command(repo, "git", "rev-parse", "HEAD")
    if code_commit != manifest.get("code_commit") or command(repo, "git", "status", "--porcelain"):
        raise ValueError("schema-v6 sentinel requires the exact clean checkout")
    locked_model = lock["models"]["mistral"]
    if (
        model_audit.get("complete") is not True
        or model_audit.get("model_id") != locked_model["model_id"]
        or model_audit.get("revision") != locked_model["revision"]
        or model_audit.get("tokenizer_hash") != manifest["model"]["tokenizer_hash"]
    ):
        raise ValueError("Mistral model audit differs from the schema-v6 manifest")
    if (
        patch_audit.get("patch_mode") != lock["stack"]["cacheblend_patch_mode"]
        or patch_audit.get("cacheblend_patch_sha256")
        != manifest["cacheblend"]["patch_sha256"]
        or patch_audit.get("cacheblend_tree") != manifest["cacheblend"]["tree"]
    ):
        raise ValueError("CacheBlend patch audit differs from schema-v6 provenance")

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
        raise RuntimeError("schema-v6 sentinel GPU is not the frozen A800 80GB")
    if provenance["compute_capability"] != [8, 0]:
        raise RuntimeError("schema-v6 sentinel requires compute capability 8.0")
    capabilities = dict(executor.capabilities())
    missing = [
        key for key in lock["runtime"]["required_capabilities"]
        if capabilities.get(key) is not True
    ]
    if missing:
        raise RuntimeError("schema-v6 runtime capabilities are missing: %s" % missing)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    existing = []
    if results_path.exists():
        if not args.resume:
            raise FileExistsError("sentinel results exist; use --resume or a new output")
        existing = [
            json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if [row["job_id"] for row in existing] != [row["job_id"] for row in jobs[:len(existing)]]:
        raise ValueError("resume results are not an immutable job prefix")
    if any(row.get("passed") is not True for row in existing):
        raise RuntimeError("failed sentinel output cannot be resumed")

    started = time.monotonic()
    max_seconds = float(limits["max_session_hours"]) * 3600
    for job in jobs[len(existing):]:
        if time.monotonic() - started >= max_seconds:
            break
        kind = str(job["kind"])
        coordinates = dict(job["coordinates"])
        row: dict[str, Any] = {
            "job_id": job["job_id"], "kind": kind,
            "coordinates": coordinates, "paper_evidence": False,
            "locked_test_accessed": False,
        }
        try:
            if kind == "hardware_environment_gate":
                row["audit"] = {"runtime_provenance": provenance, "capabilities": capabilities}
            elif kind == "native_prefix_cache_sentinel":
                observed = executor.run_native_prefix_cache_sentinel()
                audit = evaluate_native_prefix_cache_audit(
                    observed, expected_layers=MISTRAL_SCHEMA6_SPEC.num_layers
                )
                if audit.get("passed") is not True:
                    raise RuntimeError("native Prefix Cache sentinel failed")
                row["audit"] = audit
                atomic_write_json(output / "native_prefix_cache_audit.json", audit)
            elif kind in {"k_hook_depth_sentinel", "r1_dense_equivalence_sentinel"}:
                row["audit"] = dict(executor.sentinel)
            elif kind == "cfo_eager_streaming_sentinel":
                row["audit"] = dict(
                    executor.run_cfo_eager_streaming_sentinel(
                        token_count=int(coordinates["token_count"])
                    )
                )
                if row["audit"].get("passed") is not True:
                    raise RuntimeError("CFO eager/streaming sentinel failed")
                atomic_write_json(output / "cfo_audit.json", row["audit"])
            elif kind == "comparison_batch":
                head_dim = int(executor.inner_model.layers[0].self_attn.head_dim)
                current_bytes = (
                    int(coordinates["token_count"])
                    * MISTRAL_SCHEMA6_SPEC.num_kv_heads * head_dim * 2
                )
                workspace = acquire_elastic_selection_workspace(
                    hbm_manager(), owner_request_id=str(job["job_id"]),
                    segment_id="selection", compared_k=int(coordinates["k"]),
                    current_state_bytes=current_bytes,
                    per_source_state_bytes=current_bytes,
                )
                active_workspace_bytes["value"] = workspace.reserved_bytes
                try:
                    measurement = dict(
                        executor.measure_schema6_comparison_batch(
                        compared_k=workspace.microbatch_k,
                        token_count=int(coordinates["token_count"]),
                        completed_depth=int(coordinates["completed_depth"]),
                        warmups=int(job["warmups"]), repeats=int(job["repeats"]),
                    )
                    )
                finally:
                    active_workspace_bytes["value"] = 0
                    hbm_manager().release(workspace.reservation.reservation_id)
                measurement.update(
                    {
                        "requested_k": int(coordinates["k"]),
                        "microbatch_k": workspace.microbatch_k,
                        "one_shot": workspace.one_shot,
                    }
                )
                row["measurement"] = measurement
            elif kind == "selection_state_transfer":
                head_dim = int(executor.inner_model.layers[0].self_attn.head_dim)
                bytes_count = (
                    int(coordinates["token_count"])
                    * MISTRAL_SCHEMA6_SPEC.num_kv_heads * head_dim * 2
                    * int(coordinates["batch_k"])
                )
                row["measurement"] = dict(
                    executor.measure_schema6_transfer(
                        bytes_count=bytes_count, warmups=int(job["warmups"]),
                        repeats=int(job["repeats"]),
                    )
                )
            elif kind == "full_kv_tier_load":
                measured = executor.execute(
                    _qualification_job(
                        job, kind=V6A800JobKind.MULTISOURCE_LOAD, segment_count=1
                    )
                )
                row["measurement"] = measured.to_row()
                row["path"] = "pinned_cpu_to_gpu_layerwise"
            elif kind in {"repair", "joint_gate2_gate3"}:
                segments = int(coordinates.get("segment_count", 1))
                ratio = float(coordinates.get("repair_ratio", 0.15))
                measured = executor.execute(
                    _qualification_job(
                        job, kind=V6A800JobKind.UNION_REPAIR,
                        segment_count=segments, probe_layer=5, repair_ratio=ratio,
                    )
                )
                row["measurement"] = measured.to_row()
                if kind == "joint_gate2_gate3":
                    row["state_audit"] = _exercise_state_contract(str(coordinates["policy"]))
            elif kind in {"winner_deferred_lease_promotion", "gate3_subset"}:
                row["state_audit"] = _exercise_state_contract(
                    str(coordinates.get("policy", "immediate_staggered_closed_loop"))
                )
            else:
                raise RuntimeError("unknown schema-v6 sentinel job: %s" % kind)
            row["passed"] = True
        except Exception as error:
            row["passed"] = False
            row["error"] = "%s: %s" % (type(error).__name__, error)
        append_jsonl_fsync(results_path, [row])
        existing.append(row)
        atomic_write_json(
            output / "resume_checkpoint.json",
            {
                "protocol_version": 8, "schema_version": 6,
                "completed": len(existing), "planned": len(jobs),
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
    gate = {
        "protocol_version": 8, "schema_version": 6,
        "stage": "mistral_schema6_a800_4h_sentinel",
        "code_commit": code_commit,
        "manifest_sha256": sha256_file(Path(args.manifest).resolve()),
        "planned": len(jobs), "completed": completed, "failed": failed,
        "schema_v6_runtime_contract_passed": success,
        "mistral_correctness_sentinel_passed": success,
        "cfo_measurement_pipeline_ready": success,
        "runtime_measurement_pipeline_ready": success,
        "runtime_cost_profile_frozen": False,
        "selector_profile_frozen": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "paper_evidence": False, "locked_test_accessed": False,
        "full_kv_transfer_audit": transfer_authorizer.audit(),
    }
    atomic_write_json(output / "sentinel_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
