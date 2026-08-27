"""Re-tokenize audited train data and freeze one model-specific v6 H1 pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file


DATASETS = {
    "musique": "MuSiQue",
    "2wiki": "2WikiMultiHopQA",
    "hotpotqa": "HotPotQA",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit-root", required=True)
    parser.add_argument("--model-audit", required=True)
    parser.add_argument("--patch-audit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--contract", default="configs/experiment_contract.yaml")
    parser.add_argument("--server-lock", default="configs/a800_server_lock.json")
    parser.add_argument("--protocol-version", type=int, choices=(6, 7, 8), default=6)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--per-dataset", type=int, default=50)
    parser.add_argument("--natural-target", type=int, default=25)
    parser.add_argument("--max-controlled-cases", type=int, default=250)
    parser.add_argument("--max-corpus-repeat-cases", type=int, default=250)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    code_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=str(repo), text=True
    ).strip()
    if subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=str(repo), text=True
    ).strip():
        raise ValueError("H1 data handoff requires a clean ProbeKV worktree")
    source_root = Path(args.source_audit_root).resolve()
    model_audit_path = Path(args.model_audit).resolve()
    patch_audit_path = Path(args.patch_audit).resolve()
    config_path = Path(args.config).resolve()
    contract_path = Path(args.contract).resolve()
    server_lock_path = Path(args.server_lock).resolve()
    for required in (patch_audit_path, config_path, contract_path, server_lock_path):
        if not required.is_file():
            raise ValueError("missing frozen H1 input: %s" % required)
    model = json.loads(model_audit_path.read_text(encoding="utf-8"))
    experiment_config = json.loads(config_path.read_text(encoding="utf-8"))
    if model.get("complete") is not True:
        raise ValueError("model snapshot audit is incomplete")
    snapshot = Path(model["snapshot_path"]).resolve()
    if not snapshot.is_dir():
        raise ValueError("audited model snapshot is unavailable")
    model_signature = "%s@%s" % (model["model_id"], model["revision"])
    output = Path(args.output).resolve()
    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)

    manifests = []
    source_rows = []
    for key in DATASETS:
        audit_path = source_root / key / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        raw = Path(audit["raw_input"]).resolve()
        if sha256_file(raw) != audit["raw_input_sha256"]:
            raise ValueError("%s raw train file differs from its audit" % key)
        target = prepared / key
        command = [
            sys.executable,
            str(repo / "scripts" / "prepare_rag_data.py"),
            "--dataset", key,
            "--input", str(raw),
            "--output", str(target),
            "--tokenizer", str(snapshot),
            "--model-signature", model_signature,
            "--protocol-version", str(args.protocol_version),
            "--tokenizer-signature", str(model["tokenizer_hash"]),
            "--source-url", str(audit["official_source_url"]),
            "--source-revision", str(audit["official_source_revision"]),
            "--license", str(audit["dataset_license"]),
            "--construction", "both",
            "--seed", str(args.seed),
            "--streaming-pilot",
            "--max-controlled-cases", str(args.max_controlled_cases),
            "--max-corpus-repeat-cases", str(args.max_corpus_repeat_cases),
        ]
        subprocess.check_call(command, cwd=str(repo))
        manifest = target / "cases.jsonl"
        manifests.append(manifest)
        source_rows.append({
            "dataset": DATASETS[key],
            "source_audit": str(audit_path),
            "source_audit_sha256": sha256_file(audit_path),
            "raw_input_sha256": sha256_file(raw),
            "prepared_manifest_sha256": sha256_file(manifest),
        })

    manifest_output = output / "manifest"
    command = [
        sys.executable,
        str(repo / "scripts" / "build_h1_pilot_manifest.py"),
    ]
    for manifest in manifests:
        command.extend(("--dataset-manifest", str(manifest)))
    command.extend((
        "--output", str(manifest_output),
        "--per-dataset", str(args.per_dataset),
        "--natural-target", str(args.natural_target),
        "--seed", str(args.seed),
        "--model-revision", str(model["revision"]),
    ))
    subprocess.check_call(command, cwd=str(repo))

    jobs_output = output / "jobs"
    subprocess.check_call([
        sys.executable,
        str(repo / "scripts" / "build_e1_jobs.py"),
        "--manifest", str(manifest_output / "h1_pilot_cases.jsonl"),
        "--config", str(config_path),
        "--output", str(jobs_output),
        "--splits", "pilot",
        "--anchor-fraction", "0.20",
    ], cwd=str(repo))

    handoff = {
        "schema_version": (5 if args.protocol_version == 8 else (3 if args.protocol_version == 7 else 1)),
        "protocol_version": args.protocol_version,
        "stage": "%s_h1_model_data_handoff" % (
            ("v8" if args.protocol_version == 8 else ("v7" if args.protocol_version == 7 else "v6"))
        ),
        "paper_evidence": False,
        "locked_test_accessed": False,
        "code_commit": code_commit,
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "tokenizer_hash": model["tokenizer_hash"],
        "model_audit_sha256": sha256_file(model_audit_path),
        "patch_audit_sha256": sha256_file(patch_audit_path),
        "config_sha256": sha256_file(config_path),
        "contract_sha256": sha256_file(contract_path),
        "server_lock_sha256": sha256_file(server_lock_path),
        "sources": source_rows,
        "pilot_manifest": str(manifest_output / "h1_pilot_cases.jsonl"),
        "pilot_manifest_sha256": sha256_file(manifest_output / "h1_pilot_cases.jsonl"),
        "jobs": str(jobs_output / "jobs.jsonl"),
        "jobs_sha256": sha256_file(jobs_output / "jobs.jsonl"),
        # Keep the historical v6 handoff field literal and readable by old
        # static auditors; v7 receives a distinct field below.
        "ready_for_v6_h1_gpu_sentinel": True,
    }
    if args.protocol_version == 7:
        handoff["ready_for_v6_h1_gpu_sentinel"] = False
        handoff["ready_for_v7_h1_gpu_sentinel"] = True
    elif args.protocol_version == 8:
        handoff["ready_for_v6_h1_gpu_sentinel"] = False
        handoff["ready_for_v8_h1_gpu_sentinel"] = True
        primary_layer = int(experiment_config["first_reused_layer_1based"])
        if int(experiment_config["h1_primary_completed_depth"]) != primary_layer - 1:
            raise ValueError("v8 H1 completed-depth/reused-layer contract is inconsistent")
        handoff["h1_primary_completed_depth"] = primary_layer - 1
        handoff["first_reused_layer_1based"] = primary_layer
    atomic_write_json(output / "handoff.json", handoff)
    print(json.dumps(handoff, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
