"""Bounded local CUDA engineering tests, never A800/model qualification.

No weights, datasets, server connection or formal Profile are used. Tests use
the project's actual comparison and physical staging code on synthetic tensors.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
from unittest.mock import patch

from probekv.io import atomic_write_json, sha256_file
from probekv.source_policy_development import cacheblend_pinned_value_scores
from probekv.source_policy_replay import observe_depth_k
from probekv.v8_schema10_execution import digest_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        print(json.dumps({"execution_requested": False, "a800_qualification_allowed": False}))
        return
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError("retain previous local evidence; choose a new directory")
    root.mkdir(parents=True)
    report = {"kind": "local_cuda_synthetic_primitives_v1", "seed": 20260726,
              "tests": [], "local_cuda_primitives_passed": False,
              "model_loaded": False, "a800_stack_tested": False,
              "gpu_runtime_qualified": False, "paper_evidence": False,
              "locked_test_accessed": False, "failures": []}
    repo = Path(__file__).resolve().parents[1]
    report["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    report["tracked_worktree_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip())
    report["source_files_sha256"] = {str(path.relative_to(repo)): sha256_file(path) for path in (
        Path(__file__).resolve(), repo / "src/probekv/source_policy_replay.py",
        repo / "src/probekv/source_policy_development.py", repo / "src/probekv/v8_schema10_staging.py",
        repo / "src/probekv/v8_schema10_layer_storage.py")}
    started = time.perf_counter()
    try:
        import torch
        from probekv.v8_schema10_staging import PhysicalPinnedStagingPool, PhysicalLayerwiseSourceLoader
        from probekv.v8_schema10_layer_storage import LayerFile, write_layer_replica
        from probekv.v8_schema10_storage import tensor_digest
        if not torch.cuda.is_available():
            raise RuntimeError("no local CUDA device")
        report.update(torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                      gpu_name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
        free, total = torch.cuda.mem_get_info()
        if free < 2**30:
            raise MemoryError("at least 1GiB free headroom required for local tests")
        torch.manual_seed(report["seed"])
        torch.cuda.reset_peak_memory_stats()
        current = torch.ones((512, 8, 128), dtype=torch.bfloat16, device="cuda")
        sources = {"s%02d" % i: current + (16-i)/32 for i in range(16)}
        original = observe_depth_k(current, sources, completed_depth=1)
        scores = {s: sum(row["normalized_k_drifts"])/512 for s, row in original["sources"].items()}
        if min(scores, key=scores.get) != "s15":
            raise RuntimeError("sixteenth Source did not win")
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        def batch(states):
            stack = torch.stack(states).float()
            return (stack-current.float()).square().sum((2,3)).sqrt()/current.float().square().sum((1,2)).sqrt().clamp_min(1e-12)
        begin.record()
        full = batch(list(sources.values()))
        end.record(); end.synchronize()
        split = torch.cat([batch(list(sources.values())[i:i+4]) for i in range(0,16,4)])
        torch.testing.assert_close(full, split, rtol=0, atol=0)
        report["tests"].append({"name":"source16_and_full_vs_microbatch", "passed":True,
                                "comparison_cuda_ms":begin.elapsed_time(end), "source_count":16})
        value = torch.ones_like(current)*2
        source_v = value-.5
        gpu_v = cacheblend_pinned_value_scores(value, source_v)
        cpu_v = cacheblend_pinned_value_scores(value.cpu(), source_v.cpu())
        torch.testing.assert_close(gpu_v.cpu(), cpu_v, rtol=0, atol=0)
        report["tests"].append({"name":"pinned_dtype_v_squared_l2_exact_control", "passed":True,
                                "fixed_a800_kernel_qualified":False})
        layers = tuple(tuple(torch.randn(128,8,32,dtype=torch.bfloat16).pin_memory() for _ in range(2)) for _ in range(4))
        expected = tensor_digest(t for pair in layers for t in pair)
        pool = PhysicalPinnedStagingPool(capacity_bytes=2*128*8*32*4)
        authorizations = []
        def authorize(**kw):
            expected_bytes = sum(t.numel() * t.element_size() for pair in layers for t in pair)
            if kw["source_id"] != "winner" or kw["bytes_required"] != expected_bytes:
                raise RuntimeError("incorrect local transfer identity/size")
            authorizations.append(kw)
        loader = PhysicalLayerwiseSourceLoader(pool, authorize=authorize, integrity_mode="online_immutable")
        with patch("probekv.v8_schema10_staging.tensor_digest", side_effect=AssertionError("online full hash forbidden")):
            ticket = loader.begin(segment_id="C",source_id="winner",canonical_layers=layers,
                segment_positions=range(128),expected_artifact_digest=expected,prefetch_window=1)
            for layer in range(1,5):
                loader.prefetch_pending(ticket,layer)
                ticket.layer_events[layer].wait(torch.cuda.current_stream())
                torch.testing.assert_close(ticket.layer_tensors[layer][0].cpu(), layers[layer-1][0],rtol=0,atol=0)
        torch.cuda.synchronize()
        if ticket.pending_layers or ticket.per_request_full_digest_verified:
            raise RuntimeError("incorrect online transfer lifecycle")
        report["tests"].append({"name":"actual_windowed_pinned_transfer_no_full_hash", "passed":True,
                                "layers":4,"authorization_count":len(authorizations),
                                "overlap_speedup_claimed":False})
        with tempfile.TemporaryDirectory(dir=root) as temp:
            path = Path(temp)/"source.layer"
            with path.open("wb") as stream:
                write_layer_replica(stream,layers)
            qualification = PhysicalLayerwiseSourceLoader(pool,authorize=authorize,integrity_mode="qualification_full")
            checked = qualification.begin(segment_id="C",source_id="winner",canonical_layers=LayerFile(path),
                segment_positions=range(128),expected_artifact_digest=expected)
            if not checked.source_digest_before == checked.source_digest_after == checked.destination_digest == expected:
                raise RuntimeError("staged Source/destination integrity differs")
            if not checked.per_request_full_digest_verified:
                raise RuntimeError("full verification missing")
        pool.clear()
        report["tests"].append({"name":"actual_file_pinned_gpu_double_buffer_integrity", "passed":True,
                                "staging_peak_bytes":pool.peak_bytes, "cold_ssd_performance_claimed":False})
        report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["local_cuda_primitives_passed"] = True
    except Exception as exc:
        report["failures"].append({"type":type(exc).__name__,"message":str(exc)})
        raise
    finally:
        report["wall_seconds"] = time.perf_counter()-started
        report["report_sha256"] = digest_json(report)
        atomic_write_json(root/"report.json",report)
        print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    main()
