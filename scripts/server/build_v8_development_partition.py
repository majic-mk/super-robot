"""Freeze the 90-case profile-only partition without opening H1 or locked test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from probekv.io import write_jsonl


DATASETS = ("musique", "2wikimultihopqa", "hotpotqa")
SEED = 20260726
DEVELOPMENT_SOURCE_ROLES = frozenset(
    {"calibration", "development", "development_profile_freeze"}
)


def _read(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rank(dataset: str, case_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{dataset}:{case_id}".encode("utf-8")).hexdigest()


def _group(row: dict) -> str:
    case_id = str(row.get("case_id", ""))
    return str(
        row.get(
            "group_id",
            row.get("document_id", row.get("content_hash", case_id)),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    for dataset in DATASETS:
        parser.add_argument("--%s" % dataset, action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = []
    seen_groups = set()
    for dataset in DATASETS:
        input_rows = [
            _read(Path(path).resolve()) for path in getattr(args, dataset)
        ]
        indexed = [
            {str(row.get("case_id", "")): row for row in rows if row.get("case_id")}
            for rows in input_rows
        ]
        common_case_ids = set(indexed[0])
        for rows_by_id in indexed[1:]:
            common_case_ids.intersection_update(rows_by_id)
        allowed = []
        for case_id in common_case_ids:
            rows = [rows_by_id[case_id] for rows_by_id in indexed]
            roles = {
                str(row.get("split", row.get("split_role", ""))).lower()
                for row in rows
            }
            # The source manifests already carry a group-isolated calibration
            # partition.  Do not silently sample from train: H1 pilot cases are
            # also drawn from train and would otherwise be eligible for both
            # system-profile selection and the subsequent H1 diagnostic.
            if not roles or not roles.issubset(DEVELOPMENT_SOURCE_ROLES):
                continue
            groups = {_group(row) for row in rows}
            if len(groups) != 1:
                raise ValueError(
                    "%s case %s has model-dependent experiment group" %
                    (dataset, case_id)
                )
            group = next(iter(groups))
            if not group or group in seen_groups:
                continue
            allowed.append(
                (_rank(dataset, case_id), case_id, group, sorted(roles)[0])
            )
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
