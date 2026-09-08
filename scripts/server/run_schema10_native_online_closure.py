"""Build a provisional exact-shape cost table and run one real online request.

This is development evidence only.  It consumes raw matched-Prefix landmarks
from ``run_schema10_native_correctness.py --cost-probe`` and never freezes a
Runtime Profile or bypasses Gate1/FinalCommit.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

from probekv.model_adapters import SCHEMA6_MODEL_SPECS
from probekv.v7_contracts import SourceVariantIdentity
from probekv.v8_schema10_canonical import capture_exact_dense_source
from probekv.v8_schema10_cost_provider import EXECUTION_SHAPE_KEY, MeasurementKey
from probekv.v8_schema10_event_log import OnlineEventLog, atomic_json
from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_native_factory import create_native_backend
from probekv.v8_schema10_storage import file_digest


def _read_signed(path):
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = row.get("raw_observation_sha256")
    if claimed != digest_json({k: v for k, v in row.items() if k != "raw_observation_sha256"}):
        raise ValueError("raw cost observation digest differs: " + str(path))
    if row.get("origin") != "real_cuda_execution" or row.get("fake_timing") is not False:
        raise ValueError("cost observation is not real CUDA execution")
    return row


def _interval(start, end, cuda_ms, endpoint):
    if not all(isinstance(x, int) for x in (start, end)) or not start < end:
        raise ValueError("raw timing landmarks are missing or unordered")
    if not isinstance(cuda_ms, (int, float)) or cuda_ms < 0:
        raise ValueError("raw CUDA timing landmark is invalid")
    return {"warmup": False, "host_start_ns": start, "host_end_ns": end,
            "wall_endpoint_kind": endpoint, "cuda_timing_scope": endpoint,
            "cuda_ms": float(cuda_ms), "completion_ns": end}


def _row(category, query, provenance, sample, interval, **extra):
    if not isinstance(sample, (int, float)) or sample <= 0:
        raise ValueError("cost sample must be positive")
    row = {"category": category, "query": query, "provenance": provenance,
           "origin": "real_cuda_execution", "fake_timing": False,
           "warmup_excluded": True, "outlier_policy": "none",
           "samples_ms": [float(sample)], "cuda_samples_ms": [interval["cuda_ms"]],
           "raw_intervals": [interval], **extra}
    row["row_sha256"] = digest_json(row)
    return row


def build_cost_table(correctness_root, output_path):
    root = Path(correctness_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != digest_json({k: v for k, v in manifest.items()
                                                        if k != "manifest_sha256"}):
        raise ValueError("correctness manifest digest differs")
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    if (result.get("native_prefix_k_hook_r1_passed") is not True
            or result.get("matched_prefix_cost_probe_passed") is not True):
        raise ValueError("correctness/cost probe prerequisites did not pass")
    r1 = _read_signed(root / "combined-r1" / "r1.json")
    if r1.get("logit_relative_l2", 1) > 1e-4:
        raise ValueError("r=1 evidence no longer passes")
    dense = _read_signed(root / "cost-probe" / "dense_prefix.json")
    source = _read_signed(root / "cost-probe" / "fixed15_source.json")
    if (dense.get("cached_prefix_tokens") != source.get("cached_prefix_tokens")
            or source.get("diagnostic_repair_ratio") != .15
            or source.get("integrity_verification_mode") != "online_immutable"):
        raise ValueError("cost arms do not share the Prefix/fixed15 online contract")
    request = manifest["diagnostic_requests"]["target"]
    segment = request["segments"][0]
    spec = SCHEMA6_MODEL_SPECS[manifest["native_runtime"]["model_key"]]
    config = json.loads((Path(manifest["native_runtime"]["model_path"]) / "config.json").read_text())
    head_dim = config["hidden_size"] // spec.num_attention_heads
    prefix, prompt = dense["cached_prefix_tokens"], len(request["token_ids"])
    positions = list(segment["positions"])
    tier = "pinned_cpu" if manifest["diagnostic_backing_tier"] == "cpu" else "ssd"
    full_bytes = len(positions) * spec.num_kv_heads * head_dim * spec.num_layers * 4
    shape = {"prompt_tokens": prompt, "prefix_tokens": prefix, "positions": positions,
             "completed_depth": manifest["diagnostic_reuse_boundary"] - 1,
             "first_reuse_layer": manifest["diagnostic_reuse_boundary"],
             "num_layers": spec.num_layers, "dtype": "bfloat16", "kv_heads": spec.num_kv_heads,
             "head_dim": head_dim, "tier": tier, "bytes": full_bytes,
             "layout": "pre_rope_k_raw_v", "repair_ratio": .15,
             "timing_scope": "source_local_boundary_future"}
    provenance = manifest["native_runtime"]["cost_provenance"]
    identity = {"prompt_token_ids_sha256": digest_json(request["token_ids"]),
                "cached_prefix_tokens": prefix, "prefix_cache_mode": dense["prefix_cache_mode"],
                "timing_scope": "arrival_to_first_token", "sampling": dense["sampling_signature"]}
    dense_whole = _interval(dense["diagnostic_start_ns"], dense["first_token_ns"],
                            dense["first_token_cuda_ms"], "first_token")
    dense_future = _interval(dense["selection_boundary_ready_ns"], dense["first_token_ns"],
                             dense["boundary_to_first_token_cuda_ms"], "boundary_to_first_token")
    source_future = _interval(source["selection_boundary_ready_ns"], source["first_token_ns"],
                              source["boundary_to_first_token_cuda_ms"], "boundary_to_first_token")
    preparation = _interval(source["selection_boundary_ready_ns"], source["winner_source_ready_ns"],
                            source["winner_preparation_cuda_ms"], "winner_preparation")
    ready_future = _interval(source["winner_source_ready_ns"], source["first_token_ns"],
                             source["ready_to_first_token_cuda_ms"], "ready_to_first_token")
    source_query = lambda category: MeasurementKey(category, shape).query()
    support = source["repair_check_ms"]
    primitives = [
        _row("dense_reference", identity, provenance, dense["first_token_host_ms"], dense_whole),
        _row("source_local_dense", source_query("source_local_dense"), provenance,
             dense["boundary_to_first_token_ms"], dense_future),
        _row("source_local_marginal", source_query("source_local_marginal"), provenance,
             source["winner_preparation_ms"], preparation,
             component_observations_ms=[{"support_build": support,
                                         "visible_load": max(0., source["winner_preparation_ms"] - support),
                                         "repair": 0.0}],
             component_lower_ms={"support_build": support,
                                 "visible_load": max(0., source["winner_preparation_ms"] - support),
                                 "repair": 0.0},
             repair_lower_bound_semantics="nonnegative_zero; exact repair is in joint future"),
        _row("source_future", source_query("source_future"), provenance,
             source["boundary_to_first_token_ms"], source_future),
        _row("winner_visible_preparation", source_query("winner_visible_preparation"), provenance,
             source["winner_preparation_ms"], preparation)]
    depth = shape["completed_depth"]
    dense_masks = {str(layer): list(range(prefix, prompt))
                   for layer in range(depth + 1, spec.num_layers + 1)}
    observed_masks = {str(row["layer"]): row["expected_positions"] for row in source["layer_rows"]
                      if row["layer"] > depth}
    if set(observed_masks) != set(dense_masks):
        raise ValueError("fixed15 cost arm lacks every future-layer mask")
    dense_joint = MeasurementKey("joint_future", {
        "segments": [{"positions": positions, "execution": "dense", "boundary": None, "physical": {}}],
        "layer_active_positions": dense_masks, "completed_depth": depth, "num_layers": spec.num_layers,
        "prompt_tokens": prompt, "prefix_tokens": prefix, "sampling": identity["sampling"],
        "timing_scope": "boundary_to_first_token"}).query()
    ready_layers = source.get("winner_ready_layers")
    if not isinstance(ready_layers, list):
        raise ValueError("cost arm predates ready-layer audit")
    reuse_joint = MeasurementKey("joint_future", {
        "segments": [{"positions": positions, "execution": "reuse",
                      "boundary": shape["first_reuse_layer"],
                      "physical": {"tier": tier, "bytes": full_bytes,
                                   "ready_layers": ready_layers,
                                   "copy_in_flight": source["winner_copy_in_flight_at_commit_check"],
                                   "layout": "pre_rope_k_raw_v"}}],
        "layer_active_positions": observed_masks, "completed_depth": depth, "num_layers": spec.num_layers,
        "prompt_tokens": prompt, "prefix_tokens": prefix, "sampling": identity["sampling"],
        "timing_scope": "boundary_to_first_token"}).query()
    joints = [
        _row("joint_future", dense_joint, provenance, dense["boundary_to_first_token_ms"], dense_future,
             joint_future_wall_ms_samples=[dense["boundary_to_first_token_ms"]]),
        _row("joint_future", reuse_joint, provenance, source["ready_to_first_token_ms"], ready_future,
             joint_future_wall_ms_samples=[source["ready_to_first_token_ms"]])]
    payload = {"key_contract": EXECUTION_SHAPE_KEY, "provenance": provenance,
               "formal_profile_frozen": False, "rows": primitives, "joint_rows": joints,
               "source_correctness_manifest_sha256": manifest["manifest_sha256"],
               "raw_observations_sha256": digest_json([dense, source]),
               "paper_evidence": False}
    atomic_json(output_path, payload)
    return manifest, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    code = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=repo, text=True).strip():
        raise RuntimeError("online closure requires a clean tracked checkout")
    cost_path = output / "provisional_costs.json"
    base, costs = build_cost_table(args.correctness_root, cost_path)
    if base["binding"]["code_commit"] != code:
        raise ValueError("cost probe belongs to another code revision")
    cost_sha = file_digest(cost_path)
    manifest = deepcopy(base)
    manifest.update(stage="native_single_request_online_closure", paper_evidence=False,
                    locked_test_accessed=False)
    manifest["binding"]["runtime_measurement_sha256"] = cost_sha
    runtime = manifest["native_runtime"]
    runtime.update(cost_table_path=str(cost_path), cost_table_sha256=cost_sha,
                   integrity_mode="online_immutable", storage_root=str(output / "store"))
    manifest["manifest_sha256"] = digest_json({k: v for k, v in manifest.items()
                                                if k != "manifest_sha256"})
    atomic_json(output / "manifest.json", manifest)
    backend = create_native_backend(manifest)
    backend.reset(capacity=16, global_byte_budget=manifest["binding"]["global_byte_budget"])
    requests = manifest["diagnostic_requests"]
    adapter = backend.adapters["legacy_multicheckpoint"]
    capture = capture_exact_dense_source(adapter, requests["source"], "C", eager_reference=False)
    descriptor = requests["source"]["segments"][0]
    identity = SourceVariantIdentity(descriptor["content_key"],
        digest_json(requests["source"]["token_ids"][:descriptor["positions"][0]]),
        digest_json(descriptor["positions"]), "diagnostic-source-1",
        backend.provenance["model_signature"])
    source = backend.store.publish_exact_dense(identity, layers=capture["layers"],
        selection_states=capture["selection_states"], metadata=capture["source_metadata"],
        request_epoch=1, whole_request_origin="exact_dense_full_prefill",
        materialization_reason="content_miss")
    with adapter.open_request({**requests["warm"], "capture_original_full_prefill": True},
                              arrival_ns=time.perf_counter_ns()) as context:
        context.finish(lambda: None)
    initial = backend.snapshot(retain_backing=False)
    dispatch = {"selection_path": "legacy_multicheckpoint", "gate1_mode": "explicit_barrier",
                "selection_budget_policy": "end_to_end_aware"}
    event_binding = {**manifest["binding"], "dispatch": digest_json(dispatch),
                     "initial_state_sha256": digest_json(initial), "job_id": "mistral-online-closure"}
    backend.event_log = OnlineEventLog(output / "events.jsonl", binding=event_binding)
    request = requests["target"]
    outcome = backend.execute(request, dispatch, arrival_ns=time.perf_counter_ns())
    backend.finalize_request(request, outcome)
    atomic_json(output / "outcome.json", outcome)
    closed = bool(outcome.get("committed_source_variant_ids"))
    summary = {"code_commit": code, "source_variant_id": source.source_variant_id,
               "runtime_correctness_prerequisite_passed": True,
               "online_closed_loop_passed": closed,
               "execution_disposition": outcome.get("execution_disposition"),
               "actual_ttft_ms": outcome.get("request_ttft_ms"),
               "matched_dense_ttft_ms": outcome.get("matched_dense_ttft_ms"),
               "final_predicted_request_total_ms": outcome.get("final_predicted_request_total_ms"),
               "runtime_events": outcome.get("runtime_events"),
               "runtime_cost_profile_frozen": False, "gpu_runtime_qualified": False,
               "paper_evidence": False, "locked_test_accessed": False}
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
