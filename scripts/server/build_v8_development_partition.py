"""Freeze the 90-case profile-only partition without opening H1 or locked test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from probekv.io import write_jsonl


DATASETS = ("musique", "2wikimultihopqa", "hotpotqa")
SEED = 20260726


def _read(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rank(dataset: str, case_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{dataset}:{case_id}".encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for dataset in DATASETS:
        parser.add_argument("--%s" % dataset, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = []
    seen_groups = set()
    for dataset in DATASETS:
        rows = _read(Path(getattr(args, dataset)).resolve())
        allowed = []
        for row in rows:
            role = str(row.get("split", row.get("split_role", ""))).lower()
            if "locked" in role or role in {"test", "h1_pilot"}:
                continue
            case_id = str(row.get("case_id", ""))
            group = str(row.get("document_id", row.get("content_hash", case_id)))
            if not case_id or not group or group in seen_groups:
                continue
            allowed.append((_rank(dataset, case_id), case_id, group, role or "train"))
        chosen = sorted(allowed)[:30]
        if len(chosen) != 30:
            raise ValueError("%s lacks 30 isolated development cases" % dataset)
        for _, case_id, group, source_split in chosen:
            seen_groups.add(group)
            result.append(
                {
                    "source_dataset": dataset,
                    "source_split": source_split,
                    "profile_freeze_partition_id": "v8-dev-20260726",
                    "case_id": case_id,
                    "group_id": group,
                    "partition_role": "development_profile_freeze",
                    "paper_evidence": False,
                    "locked_test_accessed": False,
                }
            )
    write_jsonl(Path(args.output).resolve(), result)
    print(json.dumps({"cases": len(result), "per_dataset": 30}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
