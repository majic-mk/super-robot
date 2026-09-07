"""Generate an honest code-bound local checkpoint; never start GPU/server work."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from probekv.cacheblend_patch import patch_files_for_mode, combined_patch_sha256
from probekv.model_adapters import SCHEMA6_MODEL_SPECS
from probekv.v8_schema10_event_log import atomic_json
from probekv.v8_schema10_storage import file_digest


REPO = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-source-tree", help="optional separately replayed local CacheBlend source tree; never an SSH target")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to((REPO / "artifacts").resolve()) or output.exists():
        raise ValueError("use a new output directory inside artifacts; never overwrite evidence")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("commit tracked changes before preparing an exact-SHA handoff")
    commit = git("rev-parse", "HEAD")
    output.mkdir(parents=True)
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONIOENCODING": "utf-8"}
    configs = sorted((REPO / "configs").glob("local_system*.json"))
    commands = [[sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                [sys.executable, "scripts/validate_contract.py"],
                *[[sys.executable, "-m", "probekv.cli", "--config", str(p.relative_to(REPO))] for p in configs],
                ["git", "diff", "--check"]]
    results = []
    test_count = skipped = None
    for index, command in enumerate(commands):
        run = subprocess.run(command, cwd=REPO, env=env, text=True, encoding="utf-8",
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        # Generated validation output, not a source file or claimed GPU sample.
        log = output / f"validation-{index:02d}.log"
        log.write_text(run.stdout, encoding="utf-8")
        results.append({"command": command, "exit_code": run.returncode,
                        "log": log.name, "sha256": file_digest(log)})
        if index == 1:
            count = re.search(r"Ran (\d+) tests", run.stdout)
            skip = re.search(r"OK \(skipped=(\d+)\)", run.stdout)
            test_count = int(count[1]) if count else None
            skipped = int(skip[1]) if skip else 0
        print(json.dumps({"validation": index, "exit_code": run.returncode}), flush=True)
    failures = [r["command"] for r in results if r["exit_code"]]
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain", "--untracked-files=no"):
        failures.append(["checkout_changed_during_validation"])
    lock_path = REPO / "configs/a800_server_lock_v8_schema10.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest_path = REPO / "patches/cacheblend/manifest.json"
    patches = patch_files_for_mode(manifest_path, lock["stack"]["cacheblend_patch_mode"])
    native_audit = None
    if args.native_source_tree:
        tree = Path(args.native_source_tree).resolve()
        files = ("model_executor/models/llama.py", "model_executor/models/qwen2.py",
                 "attention/backends/xformers.py", "worker/model_runner.py",
                 "core/block_manager_v1.py", "sequence.py")
        root = tree / "vllm_blend/vllm"
        hashes = {}
        for relative in files:
            path = root / relative
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            hashes[relative] = file_digest(path)
        native_audit = {"scope": "local_source_syntax_and_file_identity_only", "path": str(tree),
            "installed_runtime_source_files_sha256": hashes, "syntax_passed": True,
            "gpu_behavior_verified": False, "server_environment_verified": False}
        atomic_json(output / "native_source_audit.json", native_audit)
    source_files = sorted((REPO / "src/probekv").glob("v8_schema10_*.py"))
    report = {"stage": "single_request_local_backend_checkpoint", "protocol_version": 8, "schema_version": 10,
        "code_commit": commit, "branch": git("branch", "--show-current"),
        "tracked_checkout_clean": not git("status", "--porcelain", "--untracked-files=no"),
        "tests_run": test_count, "tests_skipped": skipped, "validations": results,
        "single_request_backend_local_tests_passed": not failures and test_count is not None,
        "config_sha256": {str(p.relative_to(REPO)): file_digest(p) for p in configs},
        "server_lock_sha256": file_digest(lock_path), "cacheblend_base": lock["stack"]["cacheblend_commit"],
        "cacheblend_patch_sha256": combined_patch_sha256(patches),
        "native_source_audit": native_audit,
        "native_source_module_sha256": {str(p.relative_to(REPO)): file_digest(p) for p in source_files},
        "local_environment": {"python": sys.version, "executable": sys.executable,
                              "is_frozen_server_environment": False},
        "models": [{"model_id": s.model_id, "revision": s.revision, "legacy_checkpoints": s.checkpoints,
                    "tokenizer_assets_sha256": None, "snapshot_audited": False} for s in SCHEMA6_MODEL_SPECS.values()],
        "pending": ["complete combined Prefix+r1/mask operations and staged cost dispatcher; isolated Prefix/K-hook operators now implemented but not GPU-tested",
                    "full native-context CPU harness and failure-path integration audit before declaring source ready",
                    "new-SHA development trace manifests and actual tokenizer/snapshot audits",
                    "instance and budget confirmation before GPU execution"],
        "native_runtime_source_ready": False,
        "online_trace_execution_allowed": False,
        "artifact_preparation_ready": False, "ready_for_single_request_gpu_sentinel": False,
        "single_request_runtime_sentinel_passed": False, "formal_profile_bundle_frozen": False,
        "integrated_concurrency_qualified": False, "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False, "paper_evidence": False, "locked_test_accessed": False,
        "server_modified": False, "gpu_started": False, "failures": failures}
    atomic_json(output / "handoff.json", report)
    print(json.dumps({"handoff": str(output / "handoff.json"), "code_commit": commit,
                      "local_tests_passed": report["single_request_backend_local_tests_passed"],
                      "ready_for_gpu": False}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__": raise SystemExit(main())
