"""Freeze all pre-Profile v8 task lists without inventing GPU evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file, write_jsonl
from probekv.v8_a800_jobs import build_v8_preprofile_manifest
from probekv.v8_profile import (
    V8_PROFILE_DATASETS,
    build_profile_freeze_contract,
    selector_profile_candidates,
)


POLICIES = ("causal_commit_wait", "immediate_staggered_closed_loop")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/a800_server_lock_v8.json")
    parser.add_argument("--mistral-model-audit", required=True)
    parser.add_argument("--qwen-model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--development-partition", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    code_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=str(repo), text=True
    ).strip()
    if subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=str(repo), text=True
    ).strip():
        raise ValueError("v8 no-GPU manifests require a clean frozen checkout")
    lock = load(Path(args.lock).resolve())
    audits = {
        "mistral": load(Path(args.mistral_model_audit).resolve()),
        "qwen": load(Path(args.qwen_model_audit).resolve()),
    }
    patch = load(Path(args.patch_audit).resolve())
    development_path = Path(args.development_partition).resolve()
    development_rows = [
        json.loads(line)
        for line in development_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = {
        dataset: sum(row.get("source_dataset") == dataset for row in development_rows)
        for dataset in V8_PROFILE_DATASETS
    }
    if counts != {dataset: 30 for dataset in V8_PROFILE_DATASETS}:
        raise ValueError("development partition requires exactly 30 cases per dataset")
    if any(
        row.get("partition_role") != "development_profile_freeze"
        or row.get("locked_test_accessed") is not False
        for row in development_rows
    ):
        raise ValueError("development partition crossed its frozen role boundary")
    development_sha = sha256_file(development_path)
    profile_contract = build_profile_freeze_contract(
        code_commit=code_commit,
        development_partition_sha256=development_sha,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "profile_freeze_contract.json", profile_contract)

    sentinels = []
    microbench = []
    profile_tasks = []
    for model_key in ("mistral", "qwen"):
        model = lock["models"][model_key]
        audit = audits[model_key]
        if audit.get("complete") is not True:
            raise ValueError("%s model audit is incomplete" % model_key)
        for policy in POLICIES:
            manifest = build_v8_preprofile_manifest(
                code_commit=code_commit,
                model_id=model["model_id"],
                model_revision=model["revision"],
                tokenizer_hash=audit["tokenizer_hash"],
                adapter_name=model["adapter_name"],
                selection_execution_policy=policy,
                checkpoint_depths=model["completed_depths"],
                cacheblend_patch_sha256=patch["cacheblend_patch_sha256"],
                cacheblend_tree=patch["cacheblend_tree"],
                profile_freeze_contract_sha256=profile_contract[
                    "profile_freeze_contract_sha256"
                ],
                development_partition_sha256=development_sha,
            )
            atomic_write_json(
                output / ("preprofile_%s_%s.json" % (model_key, policy)),
                manifest,
            )
            for kind in ("native_prefix_cache", "completed_depth_k_hook", "r1_dense_equivalence"):
                sentinels.append({
                    "task_id": "%s-%s-%s" % (model_key, policy, kind),
                    "model_key": model_key,
                    "selection_execution_policy": policy,
                    "kind": kind,
                    "code_commit": code_commit,
                    "paper_evidence": False,
                    "locked_test_accessed": False,
                })
            for depth in (0, *model["completed_depths"]):
                for tier in ("pinned_cpu", "ssd"):
                    for compared_k in (1, 2, 4, 8, 16):
                        microbench.append({
                            "task_id": "%s-%s-d%d-%s-k%d" % (
                                model_key, policy, depth, tier, compared_k
                            ),
                            "model_key": model_key,
                            "selection_execution_policy": policy,
                            "completed_depth": depth,
                            "tier": tier,
                            "comparison_destination": "gpu_ephemeral_scratch",
                            "compared_k": compared_k,
                            "selection_state": "exact_bfloat16_pre_rope_k",
                            "full_kv_transfer_for_selection": False,
                            "cuda_event_timing_required": True,
                            "paper_evidence": False,
                        })
            candidates = selector_profile_candidates(model_key, policy)
            candidates_sha = hashlib.sha256(
                json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for row in development_rows:
                profile_tasks.append({
                    "task_id": "%s-%s-%s-%s" % (
                        model_key, policy, row["source_dataset"], row["case_id"]
                    ),
                    "model_key": model_key,
                    "selection_execution_policy": policy,
                    "dataset": row["source_dataset"],
                    "case_id": row["case_id"],
                    "profile_freeze_partition_id": row["profile_freeze_partition_id"],
                    "candidate_profiles_sha256": candidates_sha,
                    "candidate_profile_count": len(candidates),
                    "capture_all_checkpoint_states_once": True,
                    "offline_evaluate_all_profiles": True,
                    "split_role": "development_profile_freeze_only",
                    "probability_calibration": False,
                    "paper_evidence": False,
                    "locked_test_accessed": False,
                })
    write_jsonl(output / "sentinel_tasks.jsonl", sentinels)
    write_jsonl(output / "selection_microbench_tasks.jsonl", microbench)
    write_jsonl(output / "profile_freeze_tasks.jsonl", profile_tasks)
    atomic_write_json(
        output / "h1_offline_diagnostic_plan.json",
        {
            "schema_version": 5,
            "protocol_version": 8,
            "code_commit": code_commit,
            "paper_evidence": False,
            "locked_test_accessed": False,
            "cases": 150,
            "sources_per_case": 4,
            "ratios": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0],
            "legacy_schema4_ratio_point": 0.16,
            "mistral": {"h1_primary_completed_depth": 4, "first_reused_layer_1based": 5},
            "qwen": {"h1_primary_completed_depth": 3, "first_reused_layer_1based": 4},
            "primary_rows": 5400,
            "anchor_cases": 30,
            "anchor_rows": 4320,
            "total_rows": 9720,
            "data_builder": "scripts/server/prepare_v6_h1_model_data.py --protocol-version 8",
            "runner": "scripts/server/run_v8_h1_pilot.py",
            "full_h1_started": False,
        },
    )
    summary = {
        "schema_version": 5,
        "protocol_version": 8,
        "code_commit": code_commit,
        "sentinel_tasks": len(sentinels),
        "microbenchmark_tasks": len(microbench),
        "profile_freeze_tasks": len(profile_tasks),
        "development_partition_sha256": development_sha,
        "profile_freeze_contract_sha256": profile_contract[
            "profile_freeze_contract_sha256"
        ],
        "final_qualification_manifests_expected": 4,
        "qualification_jobs_per_model_policy": 140,
        "qualification_jobs_total": 560,
        "profile_bound_qualification_generated": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
    }
    atomic_write_json(output / "task_manifest_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
