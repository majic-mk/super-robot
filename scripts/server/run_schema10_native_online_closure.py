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


def validate_native_dense_reference(native, boundary_control):
    if (native.get("resumable_engine_used") is not False or native.get("source_id") is not None
            or native.get("diagnostic_completed_depth") != 0 or native.get("committed_segments")
            or native.get("whole_request_origin") != "native_prefix_dense_remaining"):
        raise ValueError("primary dense reference must execute direct native Prefix forward")
    for key in ("request_tokens_sha256", "cached_prefix_tokens", "prefix_cache_mode", "sampling_signature"):
        if key not in native or native[key] != boundary_control.get(key):
            raise ValueError("native/reference boundary control mismatch: " + key)


def ready_joint_sample(observation):
    """Future after preparation/repair-check, matching FinalCommit's state.

    The boundary-to-token sample also includes winner preparation, which is
    already in the online request's elapsed wall time. Keep that sample for
    Source-local prediction only, never add it again at FinalCommit.
    """
    interval = _interval(observation["winner_source_ready_ns"], observation["first_token_ns"],
                         observation["ready_to_first_token_cuda_ms"], "ready_to_first_token")
    sample = (interval["host_end_ns"] - interval["host_start_ns"]) / 1e6
    reported = observation.get("ready_to_first_token_ms")
    if not isinstance(reported, (int, float)) or abs(reported - sample) > 1e-6:
        raise ValueError("ready joint sample differs from raw timing landmarks")
    return sample, interval


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
    native_dense = _read_signed(root / "cost-probe" / "native_dense_prefix.json")
    validate_native_dense_reference(native_dense, dense)
    source = _read_signed(root / "cost-probe" / "fixed15_source.json")
    all_ready = _read_signed(root / "cost-probe" / "fixed15_all_ready.json")
    prepared_dense = _read_signed(root / "cost-probe" / "prepared_dense.json")
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
    dense_whole = _interval(native_dense["diagnostic_start_ns"], native_dense["first_token_ns"],
                            native_dense["first_token_cuda_ms"], "first_token")
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
        _row("dense_reference", identity, provenance, native_dense["first_token_host_ms"], dense_whole,
             baseline_execution="native_prefix_direct_forward"),
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
             joint_future_wall_ms_samples=[dense["boundary_to_first_token_ms"]])]

    def append_joint_if_new(row):
        """Keep exactly one row for each exact execution-shape digest."""
        digest = digest_json(row["query"])
        if digest not in {digest_json(existing["query"]) for existing in joints}:
            joints.append(row)
    # The streaming source arm is a real joint-future observation too.  Keep
    # its exact ready-layer/copy-in-flight state rather than discarding it and
    # leaving only the all-ready arm below.  Online closure commonly reaches
    # FinalCommit while later winner layers are still copying; an all-ready
    # row must not be (unsafely) reused for that shape.  This row is still
    # exact-support only: if the runtime reaches a different ready shape it
    # remains UNSUPPORTED and falls back to dense.
    partial_source_joint = deepcopy(reuse_joint)
    partial_source_joint["geometry"]["segments"][0]["physical"].update(
        ready_layers=source["winner_ready_layers"],
        copy_in_flight=source["winner_copy_in_flight_at_commit_check"])
    partial_source_joint["geometry"]["layer_active_positions"] = observed_masks
    partial_sample, partial_interval = ready_joint_sample(source)
    partial_row = _row("joint_future", partial_source_joint, provenance,
                       partial_sample, partial_interval,
                       joint_future_wall_ms_samples=[partial_sample])
    # A fully-ready source arm may have the same key as fixed15_all_ready;
    # retain one cell per exact execution shape, never duplicate it.
    append_joint_if_new(partial_row)
    for observation, commit in ((all_ready, True), (prepared_dense, False)):
        if (observation.get("diagnostic_wait_all_source_layers") is not True
                or observation.get("diagnostic_commit_source") is not commit
                or observation.get("winner_ready_layers") != list(range(1, spec.num_layers + 1))
                or observation.get("winner_copy_in_flight_at_commit_check") is not False
                or observation.get("cached_prefix_tokens") != prefix
                or observation.get("request_tokens_sha256") != dense.get("request_tokens_sha256")
                or observation.get("sampling_signature") != identity["sampling"]
                or observation.get("integrity_verification_mode") != "online_immutable"):
            raise ValueError("all-ready cost evidence does not match its actual execution")
        query = deepcopy(reuse_joint)
        segment_query = query["geometry"]["segments"][0]
        segment_query["physical"].update(ready_layers=observation["winner_ready_layers"], copy_in_flight=False)
        if not commit:
            segment_query.update(execution="dense", boundary=None)
        query["geometry"]["layer_active_positions"] = {
            str(row["layer"]): row["expected_positions"] for row in observation["layer_rows"]
            if row["layer"] > depth}
        if set(query["geometry"]["layer_active_positions"]) != set(dense_masks):
            raise ValueError("all-ready cost evidence is missing future execution layers")
        sample, interval = ready_joint_sample(observation)
        append_joint_if_new(_row("joint_future", query, provenance,
            sample, interval, joint_future_wall_ms_samples=[sample]))
    payload = {"key_contract": EXECUTION_SHAPE_KEY, "provenance": provenance,
               "formal_profile_frozen": False, "rows": primitives, "joint_rows": joints,
               "source_correctness_manifest_sha256": manifest["manifest_sha256"],
               "raw_observations_sha256": digest_json([native_dense, dense, source, all_ready, prepared_dense]),
               "paper_evidence": False}
    atomic_json(output_path, payload)
    return manifest, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replays", type=int, default=1,
                        help="independent identical-state replays; preserve cold and warm results")
    parser.add_argument("--selection-path", choices=("legacy_multicheckpoint", "d1_only", "d1_d2_rescue"),
                        default="legacy_multicheckpoint",
                        help="Source-selection dispatch; legacy is the default")
    parser.add_argument("--kv-layout-mode", choices=("legacy", "packed_slice"), default=None,
                        help="override the audited request composite layout")
    parser.add_argument("--prefetch-window", type=int, default=None,
                        help="override the audited layer prefetch window")
    parser.add_argument("--no-restore", action="store_true",
                        help="keep one live Pool/runtime across replays for amortization diagnostics")
    parser.add_argument("--gpu-hot-cache", action="store_true",
                        help="retain the winner GPU replica across the online replay (diagnostic only)")
    parser.add_argument("--disable-current-kv-cache", action="store_true",
                        help="same-SHA diagnostic control: repeat the current QKV projection at repair check")
    args = parser.parse_args()
    if not 1 <= args.replays <= 20:
        raise ValueError("closure replay count must be between 1 and 20")
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
    from probekv.cacheblend_patch import validate_native_patch_audit
    native_evidence = base["native_runtime"]
    patch_path = Path(native_evidence["patch_audit_path"])
    if file_digest(patch_path) != native_evidence["patch_audit_sha256"]:
        raise ValueError("cost probe patch audit digest differs")
    patch = json.loads(patch_path.read_text())
    validate_native_patch_audit(patch, repo / "patches/cacheblend/manifest.json",
                                deferred_timing=base.get("defer_layer_timing", False))
    if patch["cacheblend_patch_sha256"] != base["binding"]["patch_sha256"]:
        raise ValueError("cost probe patch provenance differs")
    cost_sha = file_digest(cost_path)
    manifest = deepcopy(base)
    manifest.update(stage="native_single_request_online_closure", paper_evidence=False,
                    locked_test_accessed=False, closure_replays=args.replays,
                    current_kv_observation_cache_enabled=not args.disable_current_kv_cache)
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
    adapter = backend.adapters[args.selection_path]
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
        # The warm request is an exact dense prefill.  Publish its native
        # Prefix blocks in the real block manager; without this explicit
        # publication a later request cannot obtain a fresh physical lease
        # for the same logical prefix (and would incorrectly report a
        # missing Prefix shadow).
        if getattr(context, "native", None) is not None:
            context.native.manager.mark_blocks_as_computed(context.native.group)
        # Fail with an explicit diagnostic if the exact warm-up did not
        # publish a logical Prefix shadow; otherwise the next replay would
        # silently look like a runtime miss.
        if not adapter.shadows.entries:
            raise RuntimeError("warm exact dense produced no persistent Prefix shadow")
    initial = backend.snapshot(retain_backing=True)
    dispatch = {"selection_path": args.selection_path, "gate1_mode": "explicit_barrier",
                "selection_budget_policy": "end_to_end_aware"}
    event_binding = {**manifest["binding"], "dispatch": digest_json(dispatch),
                     "initial_state_sha256": digest_json(initial), "job_id": "mistral-online-closure"}
    backend.event_log = OnlineEventLog(output / "events.jsonl", binding=event_binding)
    replay_summaries = []
    for replay in range(args.replays):
        if replay and not args.no_restore:
            backend.restore(initial)
        request = {**requests["target"], "request_id": requests["target"]["request_id"] + ":replay:" + str(replay),
                   "request_epoch": int(requests["target"].get("request_epoch", 10)) + replay,
                   "reuse_current_kv_observation": not args.disable_current_kv_cache,
                   "selection_cache_enabled": bool(replay > 0),
                   "use_gpu_hot_cache": bool(args.gpu_hot_cache),
                   "retain_gpu_hot_cache": bool(args.gpu_hot_cache)}
        if args.kv_layout_mode is not None:
            request["kv_layout_mode"] = args.kv_layout_mode
        if args.prefetch_window is not None:
            if args.prefetch_window < 0:
                raise ValueError("prefetch window must be non-negative")
            request["prefetch_window"] = args.prefetch_window
        outcome = backend.execute(request, dispatch, arrival_ns=time.perf_counter_ns())
        backend.finalize_request(request, outcome)
        atomic_json(output / ("outcome-%02d.json" % replay), outcome)
        replay_summaries.append({"replay": replay, "kernel_state": "cold_selector" if replay == 0 else "after_prior_replay",
            "source_committed": bool(outcome.get("committed_source_variant_ids")),
            "actual_ttft_ms": outcome.get("request_ttft_ms"),
            "matched_dense_ttft_ms": outcome.get("coverage_event", {}).get("matched_dense_ttft_ms"),
            "final_predicted_request_total_ms": outcome.get("final_predicted_request_total_ms"),
            "initial_pool_snapshot_sha256": outcome.get("initial_pool_snapshot_sha256")})
    backend.release_snapshot(initial)
    atomic_json(output / "outcome.json", outcome)
    atomic_json(output / "joint_query_audit.json", backend.costs.joint_query_audit)
    closed = bool(outcome.get("committed_source_variant_ids"))
    summary = {"code_commit": code, "source_variant_id": source.source_variant_id,
               "current_kv_observation_cache_enabled": not args.disable_current_kv_cache,
               "prefix_khook_r1_prerequisite_passed": True,
               "online_closed_loop_passed": closed,
               "execution_disposition": outcome.get("execution_disposition"),
               "actual_ttft_ms": outcome.get("request_ttft_ms"),
               "matched_dense_ttft_ms": outcome.get("coverage_event", {}).get("matched_dense_ttft_ms"),
               "final_predicted_request_total_ms": outcome.get("final_predicted_request_total_ms"),
               "runtime_events": outcome.get("runtime_events"),
               "joint_query_statuses": [{k: row.get(k) for k in ("status", "query_digest", "reason")}
                                        for row in backend.costs.joint_query_audit],
               "replay_summaries": replay_summaries,
               "runtime_cost_profile_frozen": False, "gpu_runtime_qualified": False,
               "paper_evidence": False, "locked_test_accessed": False}
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
