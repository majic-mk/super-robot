"""Run the bounded non-paper schema-v8 Mistral A800 correctness sentinel.

This entry point is intentionally smaller than final qualification.  It proves
that the frozen d1/d2 barrier, real CacheBlend engine, lease/HBM authorizer,
native Prefix Cache and r=1 endpoint are connected before Profile work starts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from probekv.cacheblend_v6_online_engine import (
    CacheBlendV8Schema8OnlineEngine,
    integrity_mode_performs_full_digest,
)
from probekv.contracts import KVLocation
from probekv.io import atomic_write_json
from probekv.model_adapters import MISTRAL_SCHEMA8_SPEC
from probekv.runtime_source_audit import audit_v8_schema8_runtime_sources
from probekv.v6_a800_executor import RealCacheBlendA800Executor
from probekv.v8_schema6_hbm import GIB, UnifiedHBMReservationManager
from probekv.v8_schema6_transfer import Schema6FullKVTransferAuthorizer
from probekv.v8_schema6_contracts import PlannerSnapshot
from probekv.v8_schema6_planner import DeterministicJointTimelineEstimator
from probekv.v8_schema7_planner import FinalCommitPlanner
from probekv.v8_schema8_barrier import close_dense_selection_barrier
from probekv.v8_schema8_contracts import RepairRatioScope
from probekv.v8_schema8_planner import Gate1LocalPlan, Gate1MarginalLowerBound
from probekv.v8_schema8_repair import (
    JointRepairRatioCandidate,
    SegmentLayerRepairRatio,
    UniformIOBalanceDecision,
    choose_request_level_adaptive_ratio,
    validate_union_repair_ratio_plan,
)
from probekv.v8_schema8_runtime import Schema8BarrierRequestController
from probekv.v8_schema8_storage import Schema8TieredBackingManager
from probekv.v8_schema8_storage import Schema8TieredReplicaCoordinator
from probekv.v8_leases import ReplicaLifecycle, V8LeaseManager, V8ReplicaResource


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=str(repo), text=True).strip()


def _contract_sentinel_evidence() -> dict[str, bool]:
    all_d1 = close_dense_selection_barrier(
        segment_ids=("c1", "c2"),
        resolved_completed_depth_by_segment={"c1": 1, "c2": 1},
        source_frozen_segment_ids=("c1", "c2"),
        abstained_segment_ids=(),
    )
    mixed = close_dense_selection_barrier(
        segment_ids=("c1", "c2"),
        resolved_completed_depth_by_segment={"c1": 1, "c2": 2},
        source_frozen_segment_ids=("c1",),
        abstained_segment_ids=("c2",),
    )
    gate1 = Gate1LocalPlan(
        "winner", 1, 1, 2, 0.5,
        Gate1MarginalLowerBound(0.5, 0.5, 0.5), 3,
    )
    snapshot = PlannerSnapshot(1, 1, "schema8-sentinel", 1, "sparse-real")
    final = FinalCommitPlanner(
        DeterministicJointTimelineEstimator(
            base_future_ms=0,
            dense_cost_ms_by_segment={"c1": 40, "c2": 40},
            reuse_cost_ms_by_segment={"c1": 10, "c2": 50},
        )
    ).plan_ready_subset(
        inventory_segment_ids=("c1", "c2"),
        eligible_ready_segment_ids=("c1",),
        committed_segment_ids=(),
        actual_boundary_by_segment={"c1": 3},
        actual_sunk_ms=1,
        dense_reference_total_ms=100,
        snapshot=snapshot,
        current_snapshot=snapshot,
        union_mask_digest="schema8-sentinel-mask",
    )
    detached_leases = V8LeaseManager()
    for suffix in ("1", "2"):
        detached_leases.register_source(
            "detached-source-%s" % suffix,
            "detached-artifact-%s" % suffix,
            "model",
        )
        detached_leases.register_replica(
            V8ReplicaResource(
                "detached-cpu-%s" % suffix,
                "detached-source-%s" % suffix,
                "detached-artifact-%s" % suffix,
                KVLocation.PINNED_CPU, 1, 1, 1024, True,
            )
        )
    detached_hbm = UnifiedHBMReservationManager(
        allocator_capacity_bytes=8 * GIB, safety_bytes=4 * GIB
    )
    detached = Schema8BarrierRequestController(
        request_id="detached", request_generation=1,
        ordered_segment_ids=("c1", "c2"), lease_manager=detached_leases,
        hbm_manager=detached_hbm,
    )
    detached.decision_ready("c1", "detached-source-1", 1)
    detached.apply_gate1_plan(
        "c1",
        Gate1LocalPlan(
            "detached-source-1", 1, 1, 2, 0,
            Gate1MarginalLowerBound(0, 0.25, 0.25), 1,
        ),
        predicted_remaining_s=1,
    )
    detached_snapshot = PlannerSnapshot(
        1, 1, "detached", detached_hbm.epoch, "profile"
    )
    detached.apply_detached_preparation_admission(
        ("c1",), snapshot=detached_snapshot,
        current_snapshot=detached_snapshot,
    )
    detached.begin_winner_prefetch(
        "c1", artifact_id="detached-artifact-1",
        replica_id="detached-cpu-1", replica_generation=1,
        placement_epoch=1, target_hbm_bytes=1024, predicted_remaining_s=1,
    )
    detached.mark_winner_ready("c1", actual_reuse_boundary=3)
    detached_not_visible = bool(
        detached.barrier_decision is None and not detached.gate3_eligible("c1")
    )
    storage = Schema8TieredBackingManager(
        cpu_capacity_bytes=1024, ssd_capacity_bytes=1024
    )
    storage.register("a", size_bytes=1024)
    demotion = storage.register("b", size_bytes=1024)
    deletion = storage.register("c", size_bytes=1024)
    migration_leases = V8LeaseManager()
    migration_leases.register_source("migration", "artifact", "model")
    old_backing = V8ReplicaResource(
        "migration-cpu", "migration", "artifact", KVLocation.PINNED_CPU,
        1, 1, 1024, True,
    )
    migration_leases.register_replica(old_backing)
    migration_policy = Schema8TieredBackingManager(
        cpu_capacity_bytes=1024, ssd_capacity_bytes=1024
    )
    migration = Schema8TieredReplicaCoordinator(
        backing_manager=migration_policy, lease_manager=migration_leases
    )
    migration.register("migration", size_bytes=1024)
    destination = V8ReplicaResource(
        "migration-ssd", "migration", "artifact", KVLocation.SSD,
        1, 1, 1024, False, lifecycle=ReplicaLifecycle.ALLOCATING,
    )
    migration_ticket = migration.begin_backing_migration(destination)
    migration.finish_backing_migration(
        migration_ticket, copy_completed=True,
        source_logical_digest="schema8", destination_logical_digest="schema8",
    )
    scope_rows = {
        "uniform_fixed": (
            SegmentLayerRepairRatio("c1", 3, 3, 0.15),
            SegmentLayerRepairRatio("c2", 3, 3, 0.15),
        ),
        "shared_relative_schedule": (
            SegmentLayerRepairRatio("c1", 3, 3, 0.15),
            SegmentLayerRepairRatio("c2", 3, 3, 0.15),
        ),
        "per_segment_load_aware": (
            SegmentLayerRepairRatio("c1", 3, 3, 0.15),
            SegmentLayerRepairRatio("c2", 3, 3, 0.12),
        ),
        "request_layer_uniform_io_balanced": (
            SegmentLayerRepairRatio("c1", 3, 3, 0.30),
            SegmentLayerRepairRatio("c2", 3, 3, 0.30),
        ),
    }
    scope_pass = {}
    for name, rows in scope_rows.items():
        adaptive_decisions = ()
        if name == "per_segment_load_aware":
            adaptive_decisions = (
                choose_request_level_adaptive_ratio(
                    candidates=(
                        JointRepairRatioCandidate(
                            "joint-vector", 3,
                            (("c1", 0.15), ("c2", 0.12)), 1, 1, 0,
                        ),
                        JointRepairRatioCandidate(
                            "fixed15", 3,
                            (("c1", 0.15), ("c2", 0.15)), 1, 2, 0,
                        ),
                    ),
                    expected_segment_ids=("c1", "c2"),
                    repair_policy_profile_sha256="0" * 64,
                    runtime_cost_profile_sha256="0" * 64,
                ),
            )
        uniform_decisions = ()
        if name == "request_layer_uniform_io_balanced":
            uniform_decisions = (
                UniformIOBalanceDecision(
                    3, ("c1", "c2"), 0.30, 0.10, 0.15, 0.30,
                    10.0, 8.0, 0.0, "1" * 64, "2" * 64, "3" * 64,
                ),
            )
        scope_pass[name] = bool(validate_union_repair_ratio_plan(
            scope=RepairRatioScope(name), rows=rows,
            certified_floor=0.12,
            profile_frozen=name in {
                "per_segment_load_aware",
                "request_layer_uniform_io_balanced",
            },
            certified_ratio_candidates=(0.10, 0.12, 0.15, 0.30),
            adaptive_joint_decisions=adaptive_decisions,
            uniform_io_decisions=uniform_decisions,
        ))
    return {
        "d1_all_resolved_layer2_reuse": all_d1.first_selective_reuse_layer == 2,
        "d1_d2_dense_barrier_layer3_reuse": mixed.first_selective_reuse_layer == 3,
        "d1_detached_prefetch_not_execution_visible": detached_not_visible,
        "gate1_optimistic_marginal_feasibility": gate1.passed,
        "final_commit_joint_timeline": final.accepted_ready_segment_ids == ("c1",),
        "cpu_lru_demotion_to_ssd": any(row.action == "demote_cpu_to_ssd" for row in demotion),
        "ssd_lru_source_deletion": any(row.action == "evict_ssd_lru_source" for row in deletion),
        "single_backing_replica": len(storage.snapshot()["entries"]) == 2,
        "verified_backing_migration": bool(
            destination.is_backing
            and destination.lifecycle is ReplicaLifecycle.READY
            and old_backing.lifecycle is ReplicaLifecycle.DELETED
        ),
        **{"repair_ratio_scope:%s" % key: value for key, value in scope_pass.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--lock", default="configs/a800_server_lock_v8_schema8.json")
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hourly-price-cny", type=float, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    handoff = _load(Path(args.handoff).resolve())
    lock = _load((repo / args.lock).resolve())
    model_audit = _load(Path(args.model_audit).resolve())
    patch_audit = _load(Path(args.patch_audit).resolve())
    if (handoff.get("protocol_version"), handoff.get("schema_version")) != (8, 8):
        raise ValueError("schema-v8 sentinel requires a schema-v8 handoff")
    if (lock.get("protocol_version"), lock.get("schema_version")) != (8, 8):
        raise ValueError("schema-v8 sentinel requires the schema-v8 server lock")
    if handoff.get("gpu_rental_ready_for_schema8_sentinel") is not True:
        raise RuntimeError("schema-v8 no-GPU runtime audit did not pass")
    if args.hourly_price_cny > 7.5:
        raise RuntimeError("GPU hourly price exceeds the frozen limit")
    head = _git(repo, "rev-parse", "HEAD")
    if head != handoff.get("code_commit") or _git(repo, "status", "--porcelain"):
        raise ValueError("schema-v8 sentinel requires its exact clean checkout")
    source_audit = audit_v8_schema8_runtime_sources(repo)
    if source_audit.get("runtime_source_ready") is not True:
        raise RuntimeError("schema-v8 executable runtime source audit failed")
    if model_audit.get("complete") is not True:
        raise ValueError("model audit is incomplete")
    if model_audit.get("revision") != handoff.get("model_revision"):
        raise ValueError("model revision differs from handoff")
    if patch_audit.get("patch_mode") != handoff.get("runtime_patch_mode"):
        raise ValueError("CacheBlend patch mode differs from handoff")
    if patch_audit.get("cacheblend_tree") != handoff.get("cacheblend_tree"):
        raise ValueError("CacheBlend tree differs from handoff")

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

    def workspace_capacity() -> int:
        return hbm_manager().selector_lease_bytes

    executor = RealCacheBlendA800Executor(
        model_path=str(model_audit["snapshot_path"]),
        model_spec=MISTRAL_SCHEMA8_SPEC,
        expected_cacheblend_tree=str(patch_audit["cacheblend_tree"]),
        engine_class=CacheBlendV8Schema8OnlineEngine,
        protocol_version=8,
        runtime_schema_version=8,
        selection_workspace_capacity_provider=workspace_capacity,
        full_kv_transfer_authorizer=Schema6FullKVTransferAuthorizer(
            hbm_manager_provider=hbm_manager
        ),
        integrity_mode="qualification_full",
        require_pre_pinned=False,
    )
    capabilities = dict(executor.capabilities())
    missing = [
        name for name in lock["runtime"]["required_capabilities"]
        if capabilities.get(name) is not True
    ]
    if missing:
        raise RuntimeError("schema-v8 runtime capabilities are missing: %s" % missing)
    prefix = executor.run_native_prefix_cache_sentinel()
    cfo = executor.run_cfo_eager_streaming_sentinel(token_count=128)
    sentinel = dict(executor.sentinel)
    contract_evidence = _contract_sentinel_evidence()
    sentinel_rows = []
    for job in handoff["sentinel_jobs"]:
        kind = job["kind"]
        if kind == "native_prefix_cache_sentinel":
            job_passed = bool(prefix.get("native_prefix_cache_hit"))
        elif kind == "r1_dense_equivalence":
            job_passed = bool(sentinel.get("r1_dense_token_ids_equal"))
        elif kind == "integrity_online_no_full_digest":
            job_passed = not integrity_mode_performs_full_digest("online_immutable")
        elif kind == "repair_ratio_scope":
            job_passed = contract_evidence[
                "repair_ratio_scope:%s" % job["coordinates"]["scope"]
            ]
        else:
            job_passed = contract_evidence.get(kind, False)
        sentinel_rows.append({
            "job_id": job["job_id"], "kind": kind, "passed": job_passed,
            "paper_evidence": False, "locked_test_accessed": False,
        })
    passed = bool(
        prefix.get("native_prefix_cache_hit")
        and prefix.get("dense_token_ids_equal")
        and prefix.get("logit_relative_l2", 1.0) <= 1e-4
        and sentinel.get("r1_dense_token_ids_equal")
        and sentinel.get("canonical_source_digests_unchanged")
        and cfo.get("passed")
        and all(row["passed"] for row in sentinel_rows)
    )
    result = {
        "protocol_version": 8,
        "schema_version": 8,
        "stage": "schema8_a800_runtime_sentinel",
        "code_commit": head,
        "runtime_patch_mode": handoff["runtime_patch_mode"],
        "runtime_source_audit": source_audit,
        "capabilities": capabilities,
        "prefix": prefix,
        "constructor_sentinel": sentinel,
        "cfo": cfo,
        "sentinel_results": sentinel_rows,
        "runtime_measurement_jobs_planned": len(
            handoff["runtime_measurement_jobs"]
        ),
        "runtime_measurement_jobs_completed": 0,
        "runtime_measurement_status": "pending_profile_session",
        "schema8_runtime_contract_passed": passed,
        "selector_depth_profile_frozen": False,
        "repair_policy_profile_frozen": False,
        "runtime_cost_profile_frozen": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
    }
    atomic_write_json(Path(args.output).resolve(), result)
    if not passed:
        raise RuntimeError("schema-v8 A800 sentinel failed")
    print(json.dumps({"output": str(Path(args.output).resolve()), "passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
