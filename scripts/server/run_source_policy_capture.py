"""Explicit native dense-shadow observation job. Default validates only.

Requires a signed development partition and an audited native runtime manifest.
This is deliberately separate from online source selection/QA/performance.
"""
import argparse
import json
import re
from pathlib import Path
import subprocess
import time

from probekv.io import atomic_write_json
from probekv.v8_schema10_execution import digest_json


def read_verified(path, expected):
    import hashlib
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("input file digest mismatch: " + str(path))
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique)


def validate_plan(plan, partition):
    from probekv.v8_schema10_inventory import mandatory_suffix_positions, native_segment_inventory
    if (plan.get("kind") != "source_policy_native_capture_plan_v1"
            or digest_json({k:v for k,v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256")
            or plan.get("paper_evidence") is not False or plan.get("locked_test_accessed") is not False):
        raise ValueError("invalid immutable development plan")
    if plan.get("selection_path") not in {"d1_d2_rescue", "legacy_multicheckpoint"}:
        raise ValueError("full d1/d2 observer requires both checkpoints")
    if (type(plan.get("host_budget_bytes")) is not int or plan["host_budget_bytes"] <= 0
            or type(plan.get("global_byte_budget")) is not int or plan["global_byte_budget"] <= 0):
        raise ValueError("explicit storage and diagnostic host budgets required")
    sources = plan.get("source_requests", [])
    target = plan.get("target_request", {})
    if not isinstance(sources,list) or not 1 <= len(sources) <= 16:
        raise ValueError("1..16 preregistered historical requests required")
    if (partition.get("kind") != "source_policy_capture_partition_v1"
            or not re.fullmatch(r"[0-9a-f]{64}", str(partition.get("parent_development_partition_sha256", "")))
            or partition.get("locked_test_accessed") is not False):
        raise ValueError("development partition required")
    entries = partition.get("entries", [])
    by_id = {row["request_id"]: row for row in entries}
    if len(by_id) != len(entries):
        raise ValueError("duplicate partition request")
    groups = {}
    for entry in entries:
        if entry.get("role") not in {"fit", "validation", "development", "profile_freeze"} or not entry.get("content_group"):
            raise ValueError("invalid partition role/content group")
        old = groups.setdefault(entry["content_group"], entry["role"])
        if old != entry["role"]:
            raise ValueError("content group leaks across partition roles")
    requests = sources + [target]
    if len({q["request_id"] for q in requests}) != len(requests):
        raise ValueError("request identities must be unique")
    for q in requests:
        entry = by_id.get(q["request_id"])
        if (entry is None or entry.get("request_sha256") != digest_json(q)
                or q.get("partition_role") != entry["role"]
                or q.get("development_partition_digest") != partition.get("parent_development_partition_sha256")
                or q.get("locked_test_accessed") is not False
                or len(q.get("segments", [])) != 1
                or type(q.get("request_epoch")) is not int):
            raise ValueError("request is not an exact member of its frozen development partition")
        forbidden = {"correctness_repair_ratio", "teacher_token_ids", "capture_logits",
                     "capture_original_full_prefill", "force_nonpaper_measurement_admission"}
        if forbidden & q.keys():
            raise ValueError("capture plan cannot inject execution/diagnostic overrides")
        mandatory_suffix_positions(q)
        segment = q["segments"][0]
        native_segment_inventory({segment["segment_id"]: segment}, prompt_tokens=len(q["token_ids"]), cached_prefix_tokens=0)
        if [q["token_ids"][p] for p in segment["positions"]] != segment["token_ids"]:
            raise ValueError("Segment tokens differ from actual request span")
    target_segment = target["segments"][0]
    if len({by_id[q["request_id"]]["content_group"] for q in requests}) != 1:
        raise ValueError("exact-content cohort must share one partition content group")
    if any(a["request_epoch"] >= b["request_epoch"] for a,b in zip(requests, requests[1:])):
        raise ValueError("source requests must be in strictly increasing causal order")
    for q in sources:
        s = q["segments"][0]
        if (q["request_epoch"] >= target["request_epoch"] or s["token_ids"] != target_segment["token_ids"]
                or s["content_key"] != target_segment["content_key"]):
            raise ValueError("Source must be earlier exact-content history")


def validate_inputs(plan, partition, runtime):
    """Validate executable assets before CUDA/model loading, including dry runs."""
    from probekv.v8_schema10_native_factory import validate_native_attachment
    validate_plan(plan, partition)
    return validate_native_attachment(runtime, allow_unmeasured=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = read_verified(args.plan, args.plan_sha256)
    partition = read_verified(plan["partition_path"],plan["partition_file_sha256"])
    runtime = read_verified(plan["runtime_manifest_path"],plan["runtime_manifest_file_sha256"])
    validate_inputs(plan,partition,runtime)
    if not args.execute:
        print(json.dumps({"plan_valid":True,"model_loaded":False,"execution_requested":False,
                          "gpu_runtime_qualified":False}))
        return
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError("fresh capture output directory required")
    repo = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    if sha != runtime["binding"]["code_commit"] or subprocess.check_output(
            ["git","status","--porcelain","--untracked-files=no"],cwd=repo,text=True).strip():
        raise RuntimeError("native execution requires clean exact checkout")
    root.mkdir(parents=True)
    atomic_write_json(root/"input-plan.json",plan)
    backend = None
    started = time.perf_counter()
    try:
        from probekv.v8_schema10_native_factory import create_native_measurement_backend
        from probekv.source_policy_native_capture import capture_native_source_observation
        from probekv.source_policy_replay import replay_observation
        from probekv.v7_contracts import SourceVariantIdentity
        backend = create_native_measurement_backend(runtime)
        backend.reset(capacity=16,global_byte_budget=plan["global_byte_budget"])
        adapter = backend.adapters[plan["selection_path"]]
        sources = []
        for index,q in enumerate(plan["source_requests"]):
            s = q["segments"][0]
            capture = adapter.build_exact_dense_source(q,s["segment_id"])
            identity = SourceVariantIdentity(s["content_key"],digest_json(q["token_ids"][:s["positions"][0]]),
                digest_json(s["positions"]),q["request_id"]+":"+s["segment_id"],backend.provenance["model_signature"])
            source = backend.store.publish_exact_dense(identity,layers=capture["layers"],
                selection_states=capture["selection_states"],metadata=capture["source_metadata"],
                request_epoch=q["request_epoch"],whole_request_origin="exact_dense_full_prefill",materialization_reason="content_miss")
            sources.append(source.source_variant_id)
            atomic_write_json(root/("source-%02d.json"%index),{"source_variant_id":source.source_variant_id,
                "capture_audit":capture["capture_audit"],"request_sha256":digest_json(q)})
            del capture
        adapter.reset()  # source construction does not silently warm target Prefix
        report = capture_native_source_observation(backend,request=plan["target_request"],
            selection_path=plan["selection_path"],source_ids=sources,binding=runtime["binding"],
            host_budget_bytes=plan["host_budget_bytes"])
        atomic_write_json(root/"observation.json",report["observation"])
        atomic_write_json(root/"capture-report.json",report)
        atomic_write_json(root/"replay.json",replay_observation(report["observation"]))
        atomic_write_json(root/"completed.json",{"capture_complete":True,"source_count":len(sources),
            "whole_job_wall_seconds":time.perf_counter()-started,"gpu_runtime_qualified":False,
            "paper_evidence":False,"locked_test_accessed":False})
    except Exception as exc:
        atomic_write_json(root/"failed.json",{"type":type(exc).__name__,"error":str(exc),
            "whole_job_wall_seconds":time.perf_counter()-started,"gpu_runtime_qualified":False})
        raise


if __name__ == "__main__":
    main()
