"""Run actual Mistral/Qwen native correctness, before any cost/online gate.

Inputs are diagnostic token requests, not RAG/QA results.  Model loading is
behind --execute.  Nothing here qualifies a Profile or bypasses online gates.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

from probekv.canonical_segment import CanonicalSegment, SemanticBoundary
from probekv.model_adapters import SCHEMA6_MODEL_SPECS
from probekv.v7_contracts import SourceVariantIdentity
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_event_log import atomic_json
from probekv.v8_schema10_storage import file_digest
from probekv.v8_schema10_native_factory import create_native_measurement_backend
from probekv.v8_schema10_native_preflight import run_native_prefix_sentinel, run_native_k_hook_sentinel
from probekv.v8_schema10_native_correctness import execute_fixed_source_arm, run_combined_native_r1


RUNTIME_FILES = ("model_executor/models/llama.py", "model_executor/models/qwen2.py",
    "attention/backends/xformers.py", "model_executor/layers/layernorm.py",
    "worker/model_runner.py", "core/block_manager_v1.py", "sequence.py")


def diagnostic_requests(tokenizer, model_signature, tokenizer_hash, *, segment_tokens=128, prefetch_window=0):
    if not isinstance(segment_tokens, int) or not 128 <= segment_tokens <= 640:
        raise ValueError("diagnostic Segment length must stay inside the canonical 128..640 contract")
    def tokens(text, count):
        ids = tokenizer.encode(text * (count + 1), add_special_tokens=False)
        return ids[:count]
    prefix = tokens("A shared exact prefix describes the reference library. ", 256)
    dense = tokens("New context changes the requested information. ", 32)
    content = tokens("The canonical document records the capital and river of a city. ", segment_tokens)
    suffix = tokens("Now answer the question using the supplied document. ", 32)
    canonical = CanonicalSegment(0, 0, len(content), tuple(content), SemanticBoundary.TOKEN,
                                 "correctness-diagnostic-v1", "diagnostic-not-development-data")
    key = canonical.reuse_content_key(model_signature, tokenizer_hash)
    def request(name, head, tail, epoch, has_segment=True):
        start = len(head)
        ids = head + (content if has_segment else []) + tail
        segments = [{"segment_id": "C", "content_key": key, "token_ids": content,
                     "positions": list(range(start, start + len(content)))}] if has_segment else []
        return {"request_id": name, "request_epoch": epoch, "token_ids": ids, "segments": segments,
            "mandatory_suffix_positions": list(range(len(ids) - len(tail), len(ids))),
            "max_new_tokens": 32, "prefetch_window": int(prefetch_window),
            "evidence_class": "synthetic_correctness_diagnostic",
            "paper_evidence": False, "locked_test_accessed": False}
    target = request("native-target", prefix + dense, suffix, 10)
    warm_tail = tokens("Unrelated warmup continuation. ", 64)
    if warm_tail[0] == dense[0]:
        raise ValueError("diagnostic Prefix warmup must diverge after the exact prefix")
    return {"target": target, "warm": request("native-warm", prefix, warm_tail, 0, False),
        "source": request("native-source", tokens("Earlier historical setting. ", 64), suffix, 1),
        "teacher_token_ids": suffix[:31]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-audit", required=True)
    p.add_argument("--patch-audit", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--layer-controls", action="store_true", help="extra read-only Prefix numerical diagnosis")
    p.add_argument("--backing-tier", choices=("cpu", "ssd"), default="cpu")
    p.add_argument("--reuse-boundary", type=int, default=2)
    p.add_argument("--cost-probe", action="store_true",
                   help="also collect matched-Prefix dense/fixed15 online-immutable landmarks")
    p.add_argument("--segment-tokens", type=int, default=128,
                   help="canonical diagnostic Segment length (128..640)")
    p.add_argument("--prefetch-window", type=int, default=0,
                   help="diagnostic layerwise prefetch window; 0 keeps eager loading")
    p.add_argument("--skip-eager-cfo", action="store_true",
                   help="capture CFO metadata without the bounded eager reference; never marks CFO passed")
    p.add_argument("--hardware-trace", action="store_true",
                   help="separate instrumented fixed15 arm; never use profiler TTFT as performance evidence")
    p.add_argument("--defer-layer-timing", action="store_true",
                   help="opt-in audited 0013 patch: resolve timing after prefill, preserve layer dependency waits")
    args = p.parse_args()
    if args.hardware_trace and not args.cost_probe:
        p.error("--hardware-trace requires --cost-probe")
    root = Path(args.output).resolve()
    if root.exists():
        raise ValueError("correctness run needs a fresh output directory")
    repo = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip():
        raise RuntimeError("correctness needs a clean tracked checkout")
    audit_path, patch_path = Path(args.model_audit).resolve(), Path(args.patch_audit).resolve()
    audit, patch = json.loads(audit_path.read_text()), json.loads(patch_path.read_text())
    if not audit.get("complete") or not audit.get("files") or not audit.get("tokenizer_assets_sha256"):
        raise ValueError("complete current model asset audit required")
    spec = SCHEMA6_MODEL_SPECS[audit["model_id"]]
    if spec.revision != audit["revision"]:
        raise ValueError("model revision differs from frozen adapter")
    import importlib.util
    from transformers import AutoTokenizer
    package = Path(importlib.util.find_spec("vllm").origin).parent
    cb_root = package.parents[1]
    actual_tree = subprocess.check_output(["git", "write-tree"], cwd=cb_root, text=True).strip()
    subprocess.check_call(["git", "diff", "--quiet"], cwd=cb_root)
    if actual_tree != patch["cacheblend_tree"]:
        raise ValueError("installed CacheBlend index tree differs from the independent patch audit")
    tokenizer = AutoTokenizer.from_pretrained(audit["snapshot_path"], local_files_only=True)
    model = audit["model_id"] + "@" + audit["revision"]
    token_hash, patch_sha = audit["tokenizer_assets_sha256"], patch["cacheblend_patch_sha256"]
    config_sha = file_digest(Path(args.config))
    if args.prefetch_window < 0:
        raise ValueError("prefetch window must be non-negative")
    requests = diagnostic_requests(tokenizer, model, token_hash,
                                   segment_tokens=args.segment_tokens,
                                   prefetch_window=args.prefetch_window)
    for request_name in ("target", "warm", "source"):
        requests[request_name]["defer_layer_timing"] = args.defer_layer_timing
    if not args.skip_eager_cfo and len(requests["source"]["token_ids"]) > 512:
        raise ValueError("eager CFO reference requires total Source request <=512 tokens; "
                         "preregister --skip-eager-cfo for longer overlap-only diagnostics")
    if args.reuse_boundary - 1 not in spec.checkpoints:
        raise ValueError("diagnostic reuse boundary must follow a legal model checkpoint")
    numerical_policy = {"allow_bf16_reduced_precision_reduction": False,
                        "prefill_attention_kernel": "cutlass_mha", "fused_norm_max_rows": 128}
    plan_sha = digest_json({"requests": requests, "code": sha, "model": model, "patch": patch_sha,
                           "layer_controls": args.layer_controls, "numerical_execution_policy": numerical_policy,
                           "backing_tier": args.backing_tier, "reuse_boundary": args.reuse_boundary,
                           "cost_probe": args.cost_probe, "segment_tokens": args.segment_tokens,
                           "cost_probe_readiness_cells": ["streaming", "all_ready_reuse", "all_ready_dense"],
                           "prefetch_window": args.prefetch_window,
                           "hardware_trace": args.hardware_trace,
                           "eager_cfo_reference": not args.skip_eager_cfo})
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    binding = {"code_commit": sha, "patch_sha256": patch_sha, "config_sha256": config_sha,
        "model_signature": model, "model_revision": spec.revision, "tokenizer_hash": token_hash,
        "runtime_measurement_sha256": None, "global_byte_budget": 8 * 1024**3}
    provenance = {"model_id": audit["model_id"], "model_revision": spec.revision,
        "model_signature": model, "tokenizer_hash": token_hash, "code_commit": sha,
        "runtime_compatibility": digest_json([patch_sha, spec.adapter_name, "bf16-pre-rope-v1", numerical_policy])}
    runtime = {"model_path": audit["snapshot_path"], "model_key": audit["model_id"],
        "model_audit_path": str(audit_path), "model_audit_sha256": file_digest(audit_path),
        "source_provenance": provenance, "cost_provenance": {"model": model, "code": sha,
            "patch": patch_sha, "gpu": gpu, "config": config_sha, "timing_scope": "diagnostic_prequalification",
            "profile_binding_kind": "preregistered_measurement_plan", "measurement_plan_sha256": plan_sha,
            "runtime_profile": None, "numerical_execution_policy_sha256": digest_json(numerical_policy)},
        "numerical_execution_policy": numerical_policy,
        "allocator_capacity_bytes": 12 * 1024**3, "prefix_shadow_capacity_bytes": 256 * 1024**2,
        "max_model_len": 4096, "gpu_memory_utilization": .6, "storage_root": str(root / "store"),
        "cpu_backing_bytes": (6 * 1024**3 if args.backing_tier == "cpu" else
                              2 * 1024**3 + 256 * 1024**2 + 1024),
        "selector_parameters": {"source_residual_trim_ratio": .15,
            "thresholds": [[d, .25] for d in spec.checkpoints], "strong_margin": .6,
            "stable_margin": .3, "residual_band_relative_tolerance": .05},
        "sentinel_evidence_paths": {}, "repair_policy": "fixed_15", "integrity_mode": "qualification_full",
        "installed_runtime_source_files_sha256": {name: file_digest(package / name) for name in RUNTIME_FILES}}
    manifest = {"protocol_version": 8, "schema_version": 10, "stage": "native_correctness_diagnostic",
        "binding": binding, "native_runtime": runtime, "diagnostic_requests": requests,
        "layer_controls": args.layer_controls,
        "cost_probe": args.cost_probe,
        "diagnostic_segment_tokens": args.segment_tokens,
        "hardware_trace": args.hardware_trace,
        "defer_layer_timing": args.defer_layer_timing,
        "eager_cfo_reference": not args.skip_eager_cfo,
        "diagnostic_backing_tier": args.backing_tier, "diagnostic_reuse_boundary": args.reuse_boundary,
        "paper_evidence": False, "locked_test_accessed": False}
    manifest["manifest_sha256"] = digest_json(manifest)
    root.mkdir(parents=True)
    atomic_json(root / "manifest.json", manifest)
    if not args.execute:
        print(json.dumps({"manifest": str(root / "manifest.json"), "gpu_started": False}))
        return
    started = time.perf_counter_ns()
    try:
        backend = create_native_measurement_backend(manifest)
        backend.reset(capacity=16, global_byte_budget=binding["global_byte_budget"])
        adapter = backend.adapters["legacy_multicheckpoint"]
        with adapter.open_request(requests["target"], arrival_ns=time.perf_counter_ns()) as context:
            first = []
            output = context.finish(lambda: first.append(time.perf_counter_ns()))
            atomic_json(root / "dense_smoke.json", {**output, "first_token_ns": first,
                "origin": "real_cuda_execution", "fake_timing": False})
        adapter.reset()
        k = run_native_k_hook_sentinel(adapter, request=requests["target"], segment_id="C", completed_depths=adapter.depths)
        atomic_json(root / "k_hook.json", k)
        prefix = run_native_prefix_sentinel(adapter, warm_request=requests["warm"], request=requests["target"])
        atomic_json(root / "prefix.json", prefix)
        if args.layer_controls:
            from probekv.v8_schema10_prefix_numerics import run_prefix_layer_controls
            atomic_json(root / "prefix_layer_controls.json", run_prefix_layer_controls(adapter,
                request=requests["target"], warm_request=requests["warm"]))
        from probekv.v8_schema10_canonical import capture_exact_dense_source
        from probekv.v8_schema10_native_validation import validate_correctness_observation
        capture = capture_exact_dense_source(adapter, requests["source"], "C",
                                             eager_reference=not args.skip_eager_cfo)
        cfo = {**capture["capture_audit"]["cfo"], "origin": "real_cuda_execution",
               "fake_timing": False, "paper_evidence": False}
        atomic_json(root / "cfo.json", cfo)
        if not args.skip_eager_cfo:
            validate_correctness_observation("cfo", cfo)
        descriptor = requests["source"]["segments"][0]
        identity = SourceVariantIdentity(descriptor["content_key"],
            digest_json(requests["source"]["token_ids"][:descriptor["positions"][0]]),
            digest_json(descriptor["positions"]), "diagnostic-source-1", model)
        source = backend.store.publish_exact_dense(identity, layers=capture["layers"],
            selection_states=capture["selection_states"], metadata=capture["source_metadata"],
            request_epoch=1, whole_request_origin="exact_dense_full_prefill", materialization_reason="content_miss")
        atomic_json(root / "canonical_capture.json", capture["capture_audit"])
        obj = backend.store.objects[source.source_variant_id]
        expected_tier = "pinned_cpu" if args.backing_tier == "cpu" else "ssd"
        if obj.tier.value != expected_tier:
            raise RuntimeError("diagnostic Source did not enter preregistered backing tier")
        # Validate the requested transfer schedule itself, never substitute
        # eager correctness evidence for a windowed execution.
        correctness_target = dict(requests["target"])
        r1 = run_combined_native_r1(backend, request=correctness_target, warm_request=requests["warm"],
            source_id=source.source_variant_id, segment_id="C", teacher_token_ids=requests["teacher_token_ids"],
            output_dir=root / "combined-r1", boundary=args.reuse_boundary)
        requests["target"]["prefetch_window"] = int(args.prefetch_window)
        loader = adapter.loader
        if args.cost_probe:
            cost_root = root / "cost-probe"
            cost_root.mkdir()
            loader.integrity_mode = "online_immutable"
            native_dense_cost, _ = execute_fixed_source_arm(backend, request=requests["target"],
                warm_request=requests["warm"], verify_full_digests=False)
            dense_cost, _ = execute_fixed_source_arm(backend, request=requests["target"],
                warm_request=requests["warm"], diagnostic_completed_depth=args.reuse_boundary - 1,
                boundary=args.reuse_boundary, verify_full_digests=False)
            source_cost, _ = execute_fixed_source_arm(backend, request=requests["target"],
                warm_request=requests["warm"], source_id=source.source_variant_id, segment_id="C",
                boundary=args.reuse_boundary, repair_ratio=.15, verify_full_digests=False)
            source_all_ready, _ = execute_fixed_source_arm(backend, request=requests["target"],
                warm_request=requests["warm"], source_id=source.source_variant_id, segment_id="C",
                boundary=args.reuse_boundary, repair_ratio=.15, verify_full_digests=False,
                wait_all_source_layers=True)
            prepared_dense, _ = execute_fixed_source_arm(backend, request=requests["target"],
                warm_request=requests["warm"], source_id=source.source_variant_id, segment_id="C",
                boundary=args.reuse_boundary, repair_ratio=.15, verify_full_digests=False,
                wait_all_source_layers=True, commit_source=False)
            if (source_cost["integrity_verification_mode"] != "online_immutable"
                    or any(source_cost[k] is not None for k in (
                        "source_digest_before", "destination_digest", "source_digest_after"))):
                raise RuntimeError("cost probe performed per-request full-KV hashing")
            for name, row in (("native_dense_prefix", native_dense_cost),
                              ("dense_prefix", dense_cost), ("fixed15_source", source_cost),
                              ("fixed15_all_ready", source_all_ready), ("prepared_dense", prepared_dense)):
                row["raw_observation_sha256"] = digest_json(row)
                atomic_json(cost_root / (name + ".json"), row)
            atomic_json(cost_root / "summary.json", {
                "origin": "real_cuda_execution", "fake_timing": False,
                "timing_scope": "matched_prefix_completed_depth_to_first_token",
                "primary_baseline": "native_prefix_direct_forward",
                "native_dense_first_token_ms": native_dense_cost["first_token_host_ms"],
                "resumable_dense_first_token_ms": dense_cost["first_token_host_ms"],
                "dense_boundary_to_first_token_ms": dense_cost["boundary_to_first_token_ms"],
                "dense_boundary_to_first_token_cuda_ms": dense_cost["boundary_to_first_token_cuda_ms"],
                "fixed15_boundary_to_first_token_ms": source_cost["boundary_to_first_token_ms"],
                "fixed15_boundary_to_first_token_cuda_ms": source_cost["boundary_to_first_token_cuda_ms"],
                "fixed15_ready_to_first_token_ms": source_cost["ready_to_first_token_ms"],
                "fixed15_ready_to_first_token_cuda_ms": source_cost["ready_to_first_token_cuda_ms"],
                "winner_preparation_ms": source_cost["winner_preparation_ms"],
                "winner_preparation_cuda_ms": source_cost["winner_preparation_cuda_ms"],
                "repair_check_ms": source_cost["repair_check_ms"],
                "request_full_kv_digest_performed": False,
                "formal_profile_frozen": False, "paper_evidence": False})
        if args.hardware_trace:
            import torch
            from probekv.v8_schema10_hardware_overlap import summarize_hardware_overlap
            trace_root = root / "hardware-trace"
            trace_root.mkdir()
            loader.capture_hardware_trace = True
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    traced, _ = execute_fixed_source_arm(backend, request=requests["target"],
                        warm_request=requests["warm"], source_id=source.source_variant_id, segment_id="C",
                        boundary=args.reuse_boundary, repair_ratio=.15, verify_full_digests=False)
                trace_path = trace_root / "trace.json"
                profiler.export_chrome_trace(str(trace_path))
                summary = summarize_hardware_overlap(json.loads(trace_path.read_text()))
                summary.update(trace_sha256=file_digest(trace_path), code_commit=sha,
                               instrumented_timing_not_performance_evidence=True)
                atomic_json(trace_root / "summary.json", summary)
                atomic_json(trace_root / "instrumented_arm.json", traced)
            finally:
                loader.capture_hardware_trace = False
        if (backend.hbm.active_reserved_bytes or adapter.active is not None
                or any(slot.leased or slot.completion is not None and not slot.completion.query()
                       for slot in loader.pool.slots)):
            raise RuntimeError("completed native sentinel retained active execution resources")
        expected_path = "CPU_PINNED_TO_GPU" if args.backing_tier == "cpu" else "SSD_STAGED_TO_GPU"
        if not loader.events or any(e["path"] != expected_path or e["source_id"] != source.source_variant_id
                                    for e in loader.events):
            raise RuntimeError("physical transfer did not use only the frozen winner and declared tier")
        atomic_json(root / "transfer.json", {"origin": "real_cuda_execution", "fake_timing": False,
            "events": loader.events, "staging_peak_bytes": loader.pool.peak_bytes,
            "staging_capacity_bytes": loader.pool.capacity_bytes,
            "staging_slots": len(loader.pool.slots), "active_hbm_reserved_bytes": backend.hbm.active_reserved_bytes,
            "paper_evidence": False})
        atomic_json(root / "result.json", {"native_prefix_k_hook_r1_passed": True,
            "native_cfo_eager_streaming_passed": not args.skip_eager_cfo,
            "matched_prefix_cost_probe_passed": bool(args.cost_probe),
            "native_transfer_path": expected_path,
            "r1_observation_sha256": r1["raw_observation_sha256"], "gpu_runtime_qualified": False,
            "online_trace_execution_allowed": False, "paper_evidence": False})
        print(json.dumps({"native_prefix_k_hook_r1_passed": True, "output": str(root)}))
    except Exception as error:
        atomic_json(root / "failed.json", {"error_type": type(error).__name__, "error": str(error),
            "elapsed_host_ms": (time.perf_counter_ns() - started) / 1e6,
            "code_commit": sha, "paper_evidence": False})
        raise


if __name__ == "__main__":
    main()
