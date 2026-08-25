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
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--per-dataset", type=int, default=50)
    parser.add_argument("--natural-target", type=int, default=25)
    parser.add_argument("--max-controlled-cases", type=int, default=250)
    parser.add_argument("--max-corpus-repeat-cases", type=int, default=250)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    source_root = Path(args.source_audit_root).resolve()
    model_audit_path = Path(args.model_audit).resolve()
    model = json.loads(model_audit_path.read_text(encoding="utf-8"))
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
        "--config", str(Path(args.config).resolve()),
        "--output", str(jobs_output),
        "--splits", "pilot",
        "--anchor-fraction", "0.20",
    ], cwd=str(repo))

    handoff = {
        "schema_version": 1,
        "stage": "v6_h1_model_data_handoff",
        "paper_evidence": False,
        "locked_test_accessed": False,
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "tokenizer_hash": model["tokenizer_hash"],
        "model_audit_sha256": sha256_file(model_audit_path),
        "config_sha256": sha256_file(Path(args.config).resolve()),
        "sources": source_rows,
        "pilot_manifest": str(manifest_output / "h1_pilot_cases.jsonl"),
        "pilot_manifest_sha256": sha256_file(manifest_output / "h1_pilot_cases.jsonl"),
        "jobs": str(jobs_output / "jobs.jsonl"),
        "jobs_sha256": sha256_file(jobs_output / "jobs.jsonl"),
        "ready_for_v6_h1_gpu_sentinel": True,
    }
    atomic_write_json(output / "handoff.json", handoff)
    print(json.dumps(handoff, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
