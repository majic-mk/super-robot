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
from probekv.v8_schema10_native_correctness import run_combined_native_r1


RUNTIME_FILES = ("model_executor/models/llama.py", "model_executor/models/qwen2.py",
    "attention/backends/xformers.py", "worker/model_runner.py", "core/block_manager_v1.py", "sequence.py")


def diagnostic_requests(tokenizer, model_signature, tokenizer_hash):
    def tokens(text, count):
        ids = tokenizer.encode(text * (count + 1), add_special_tokens=False)
        return ids[:count]
    prefix = tokens("A shared exact prefix describes the reference library. ", 256)
    dense = tokens("New context changes the requested information. ", 32)
    content = tokens("The canonical document records the capital and river of a city. ", 128)
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
            "max_new_tokens": 32, "evidence_class": "synthetic_correctness_diagnostic",
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
    args = p.parse_args()
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
    requests = diagnostic_requests(tokenizer, model, token_hash)
    numerical_policy = {"allow_bf16_reduced_precision_reduction": False}
    plan_sha = digest_json({"requests": requests, "code": sha, "model": model, "patch": patch_sha,
                           "layer_controls": args.layer_controls, "numerical_execution_policy": numerical_policy})
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
        "cpu_backing_bytes": 6 * 1024**3, "selector_parameters": {"source_residual_trim_ratio": .15,
            "thresholds": [[d, .25] for d in spec.checkpoints], "strong_margin": .6,
            "stable_margin": .3, "residual_band_relative_tolerance": .05},
        "sentinel_evidence_paths": {}, "repair_policy": "fixed_15", "integrity_mode": "qualification_full",
        "installed_runtime_source_files_sha256": {name: file_digest(package / name) for name in RUNTIME_FILES}}
    manifest = {"protocol_version": 8, "schema_version": 10, "stage": "native_correctness_diagnostic",
        "binding": binding, "native_runtime": runtime, "diagnostic_requests": requests,
        "layer_controls": args.layer_controls,
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
        capture = adapter.build_exact_dense_source(requests["source"], "C")
        descriptor = requests["source"]["segments"][0]
        identity = SourceVariantIdentity(descriptor["content_key"],
            digest_json(requests["source"]["token_ids"][:descriptor["positions"][0]]),
            digest_json(descriptor["positions"]), "diagnostic-source-1", model)
        source = backend.store.publish_exact_dense(identity, layers=capture["layers"],
            selection_states=capture["selection_states"], metadata=capture["source_metadata"],
            request_epoch=1, whole_request_origin="exact_dense_full_prefill", materialization_reason="content_miss")
        atomic_json(root / "canonical_capture.json", capture["capture_audit"])
        r1 = run_combined_native_r1(backend, request=requests["target"], warm_request=requests["warm"],
            source_id=source.source_variant_id, segment_id="C", teacher_token_ids=requests["teacher_token_ids"],
            output_dir=root / "combined-r1")
        atomic_json(root / "result.json", {"native_prefix_k_hook_r1_passed": True,
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
