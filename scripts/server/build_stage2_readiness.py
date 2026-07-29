"""Freeze and validate the no-GPU handoff into the Stage-2 A800 run."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from probekv.io import atomic_write_json, sha256_file


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_stage2_audits(
    manifest: Dict[str, Any],
    cb_jobs: Dict[str, Any],
    h1_jobs: Dict[str, Any],
    cb0: Dict[str, Any],
) -> None:
    expected_datasets = {"MuSiQue", "2WikiMultiHopQA", "HotPotQA"}
    if manifest.get("cases") != 150:
        raise ValueError("Stage-2 manifest must contain exactly 150 cases")
    if set(manifest.get("datasets", {})) != expected_datasets:
        raise ValueError("Stage-2 manifest dataset identity mismatch")
    for dataset, counts in manifest["datasets"].items():
        if counts != {"cases": 50, "natural": 25, "controlled": 25}:
            raise ValueError("%s is not balanced 25 natural / 25 controlled" % dataset)
    if not manifest.get("all_split_pilot") or manifest.get("locked_test_accessed"):
        raise ValueError("Stage-2 manifest must be pilot-only and test-blind")
    if manifest.get("paper_evidence") is not False:
        raise ValueError("Stage-2 pilot cannot be paper evidence")
    if cb_jobs.get("jobs") != 252:
        raise ValueError("CB1-CB3 job manifest must contain exactly 252 jobs")
    if h1_jobs.get("jobs") != 9720:
        raise ValueError("H1 job manifest must contain exactly 9,720 jobs")
    if h1_jobs.get("layer_job_counts") != {
        "5": 5400,
        "3": 1080,
        "7": 1080,
        "10": 1080,
        "13": 1080,
    }:
        raise ValueError("H1 layer distribution does not match the frozen plan")
    if not cb0.get("passed") or cb0.get("cached_outputs") != 10 or cb0.get("full_outputs") != 10:
        raise ValueError("patched CB0 gate is not complete")
    manifest_sha = manifest.get("manifest_sha256")
    if cb_jobs.get("manifest_sha256") != manifest_sha:
        raise ValueError("CB jobs were not built from the frozen manifest")
    if h1_jobs.get("manifest_sha256") != manifest_sha:
        raise ValueError("H1 jobs were not built from the frozen manifest")


def build_readiness(
    repo: Path,
    manifest_audit_path: Path,
    cb_jobs_audit_path: Path,
    h1_jobs_audit_path: Path,
    cb0_gate_path: Path,
    patch_path: Path,
    prepared_audit_paths: Iterable[Path],
    gpu_present: bool,
) -> Dict[str, Any]:
    manifest = _read_json(manifest_audit_path)
    cb_jobs = _read_json(cb_jobs_audit_path)
    h1_jobs = _read_json(h1_jobs_audit_path)
    cb0 = _read_json(cb0_gate_path)
    patch = _read_json(patch_path)
    validate_stage2_audits(manifest, cb_jobs, h1_jobs, cb0)
    manifest_file = manifest_audit_path.with_name("h1_pilot_cases.jsonl")
    cb_jobs_file = cb_jobs_audit_path.with_name("cb_gate_jobs.jsonl")
    h1_jobs_file = h1_jobs_audit_path.with_name("jobs.jsonl")
    actual_manifest_sha = sha256_file(manifest_file)
    actual_cb_jobs_sha = sha256_file(cb_jobs_file)
    actual_h1_jobs_sha = sha256_file(h1_jobs_file)
    if manifest["manifest_sha256"] != actual_manifest_sha:
        raise ValueError("frozen manifest hash does not match its audit")
    if cb_jobs.get("jobs_sha256") != actual_cb_jobs_sha:
        raise ValueError("CB job hash does not match its audit")
    if h1_jobs.get("jobs_sha256") != actual_h1_jobs_sha:
        raise ValueError("H1 job hash does not match its audit")
    expected_base = "b72d7945e6d6306f12be66520196e0f081fa2b0c"
    if patch.get("cacheblend_commit") != expected_base:
        raise ValueError("CacheBlend patch provenance has the wrong base")
    for key in ("cacheblend_patch_sha256", "cacheblend_tree"):
        if not patch.get(key):
            raise ValueError("CacheBlend patch provenance is missing %s" % key)
    if cb0.get("cacheblend_commit", expected_base) != expected_base:
        raise ValueError("CB0 and ProbeKV patch use different CacheBlend bases")
    if (
        cb0.get("cacheblend_tree")
        and cb0["cacheblend_tree"] != patch["cacheblend_tree"]
    ):
        raise ValueError("CB0 and ProbeKV patch trees do not match")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
    ).strip()
    clean = not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(repo), text=True
    ).strip()
    if not clean:
        raise ValueError("ProbeKV repository is dirty")

    prepared = []
    for path in prepared_audit_paths:
        audit = _read_json(path)
        prepared.append(
            {
                "dataset": audit["dataset_argument"],
                "cases": audit["cases"],
                "normalized_examples_scanned": audit["normalized_examples_scanned"],
                "documents_scanned": audit["documents_scanned"],
                "raw_input": audit["raw_input"],
                "raw_input_sha256": audit["raw_input_sha256"],
                "official_source_revision": audit["official_source_revision"],
                "audit_path": str(path),
                "audit_sha256": sha256_file(path),
            }
        )

    return {
        "stage": "stage2_no_gpu_readiness",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "git_clean": clean,
        "gpu_present": gpu_present,
        "artifact_ready_for_gpu": True,
        "gpu_rental_ready": True,
        "gpu_runtime_ready": bool(gpu_present),
        "gpu_runtime_complete": False,
        "next_required_gates": ["CB1", "CB2", "CB3", "H1-pilot"],
        "manifest": {
            "cases": manifest["cases"],
            "datasets": manifest["datasets"],
            "sha256": actual_manifest_sha,
            "locked_test_accessed": manifest["locked_test_accessed"],
        },
        "cb_gate_jobs": {
            "jobs": cb_jobs["jobs"],
            "sha256": actual_cb_jobs_sha,
            "audit_sha256": sha256_file(cb_jobs_audit_path),
        },
        "h1_jobs": {
            "jobs": h1_jobs["jobs"],
            "layer_job_counts": h1_jobs["layer_job_counts"],
            "sha256": actual_h1_jobs_sha,
            "audit_sha256": sha256_file(h1_jobs_audit_path),
        },
        "cb0_patched": cb0,
        "cacheblend_patch": patch,
        "prepared_datasets": prepared,
        "paper_evidence": False,
        "evidence_class": "server_pilot",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--manifest-audit", required=True)
    parser.add_argument("--cb-jobs-audit", required=True)
    parser.add_argument("--h1-jobs-audit", required=True)
    parser.add_argument("--cb0-gate", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--prepared-audit", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_readiness(
        Path(args.repo).resolve(),
        Path(args.manifest_audit).resolve(),
        Path(args.cb_jobs_audit).resolve(),
        Path(args.h1_jobs_audit).resolve(),
        Path(args.cb0_gate).resolve(),
        Path(args.patch).resolve(),
        [Path(value).resolve() for value in args.prepared_audit],
        gpu_present=Path("/dev/nvidia0").exists(),
    )
    atomic_write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
