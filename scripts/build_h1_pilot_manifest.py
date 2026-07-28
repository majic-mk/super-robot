"""Combine audited dataset manifests into the frozen 150-case H1 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file, write_jsonl
from probekv.manifest import manifest_case_from_row, manifest_digest
from probekv.pilot_manifest import pilot_manifest_audit, select_h1_pilot


def _read_cases(path: Path):
    return [
        manifest_case_from_row(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-manifest",
        action="append",
        required=True,
        help="prepared cases.jsonl; pass once for each dataset",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-dataset", type=int, default=50)
    parser.add_argument("--natural-target", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()

    inputs = [Path(value).resolve() for value in args.dataset_manifest]
    cases = []
    input_rows = []
    for path in inputs:
        source_audit_path = path.parent / "audit.json"
        if not source_audit_path.is_file():
            raise ValueError("dataset manifest is missing sibling audit.json")
        source_audit = json.loads(
            source_audit_path.read_text(encoding="utf-8")
        )
        required_source_fields = (
            "official_source_url",
            "official_source_revision",
            "dataset_license",
            "raw_input_sha256",
        )
        if any(
            not source_audit.get(field)
            or str(source_audit[field]).startswith("unspecified")
            for field in required_source_fields
        ):
            raise ValueError(
                "pilot dataset audit does not freeze official provenance"
            )
        if source_audit.get("source_split") != "train":
            raise ValueError("H1 pilot may only use an official train split")
        loaded = _read_cases(path)
        cases.extend(loaded)
        input_rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "cases": len(loaded),
                "datasets": sorted({case.dataset for case in loaded}),
                "source_audit": str(source_audit_path),
                "source_audit_sha256": sha256_file(source_audit_path),
                "official_source_url": source_audit["official_source_url"],
                "official_source_revision": source_audit[
                    "official_source_revision"
                ],
                "dataset_license": source_audit["dataset_license"],
            }
        )
    pilot = select_h1_pilot(
        cases,
        per_dataset=args.per_dataset,
        natural_target=args.natural_target,
        seed=args.seed,
    )
    expected_datasets = {"MuSiQue", "2WikiMultiHopQA", "HotpotQA"}
    observed_datasets = {case.dataset for case in pilot}
    if observed_datasets != expected_datasets:
        raise ValueError(
            "pilot requires exactly %s; observed %s"
            % (sorted(expected_datasets), sorted(observed_datasets))
        )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "h1_pilot_cases.jsonl"
    write_jsonl(manifest_path, [case.to_row() for case in pilot])
    audit = pilot_manifest_audit(pilot)
    audit.update(
        {
            "seed": args.seed,
            "per_dataset": args.per_dataset,
            "natural_target": args.natural_target,
            "model_revision": args.model_revision,
            "inputs": input_rows,
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_digest": manifest_digest(pilot),
            "locked_test_accessed": False,
        }
    )
    atomic_write_json(output / "h1_pilot_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
