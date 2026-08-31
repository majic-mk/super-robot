"""Freeze the three schema-v8 Profiles from preregistered real-GPU evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from probekv.io import atomic_write_json
from probekv.v8_schema8_profile import (
    Schema8ProfileProvenance,
    build_repair_policy_profile_v8,
    build_runtime_cost_profile_v8,
    build_selection_depth_profile_v8,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(profile: Any) -> dict[str, Any]:
    return {**dict(profile.payload()), "profile_sha256": profile.profile_sha256}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-evidence", required=True)
    parser.add_argument("--repair-evidence", required=True)
    parser.add_argument("--runtime-evidence", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--cacheblend-patch-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-hash", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    selection_path = Path(args.selection_evidence).resolve()
    repair_path = Path(args.repair_evidence).resolve()
    runtime_path = Path(args.runtime_evidence).resolve()
    selection_raw = _load(selection_path)
    repair_raw = _load(repair_path)
    runtime_raw = _load(runtime_path)
    for name, row in (
        ("selection", selection_raw),
        ("repair", repair_raw),
        ("runtime", runtime_raw),
    ):
        if (row.get("protocol_version"), row.get("schema_version")) != (8, 8):
            raise ValueError("%s evidence is not schema-v8" % name)
        if row.get("real_gpu_measurements") is not True or row.get("fake_timing") is not False:
            raise ValueError("%s Profile requires real GPU measurements" % name)
        if row.get("gpu_uuid") != args.gpu_uuid:
            raise ValueError("%s evidence belongs to another GPU" % name)

    common = dict(
        code_commit=args.code_commit,
        cacheblend_patch_sha256=args.cacheblend_patch_sha256,
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_hash=args.tokenizer_hash,
        gpu_uuid=args.gpu_uuid,
        frozen=True,
    )
    selection = build_selection_depth_profile_v8(
        provenance=Schema8ProfileProvenance(
            profile_kind="selection_depth",
            measurement_sha256=_sha(selection_path),
            **common,
        ),
        allowed_completed_depths=tuple(selection_raw["allowed_completed_depths"]),
        source_score_trim_ratio=float(selection_raw["source_score_trim_ratio"]),
    )
    repair = build_repair_policy_profile_v8(
        provenance=Schema8ProfileProvenance(
            profile_kind="repair_policy",
            measurement_sha256=_sha(repair_path),
            **common,
        ),
        policy=str(repair_raw["policy"]),
        scope=str(repair_raw["scope"]),
        certified_floor=float(repair_raw["certified_floor"]),
        shared_ratio_by_age={
            int(key): float(value)
            for key, value in repair_raw["shared_ratio_by_age"].items()
        },
        no_reentry_oracle_recall=float(repair_raw["no_reentry_oracle_recall"]),
        minimum_no_reentry_recall=float(repair_raw["minimum_no_reentry_recall"]),
        adaptive_candidate_templates=tuple(
            str(value)
            for value in repair_raw.get("adaptive_candidate_templates", ())
        ),
        timing_equivalence_absolute_ms=float(
            repair_raw.get("timing_equivalence_absolute_ms", 0.02)
        ),
        timing_equivalence_relative=float(
            repair_raw.get("timing_equivalence_relative", 0.01)
        ),
    )
    runtime = build_runtime_cost_profile_v8(
        provenance=Schema8ProfileProvenance(
            profile_kind="runtime_cost",
            measurement_sha256=_sha(runtime_path),
            **common,
        ),
        category_measurements={
            key: tuple(value)
            for key, value in runtime_raw["category_measurements"].items()
        },
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "selection_depth_profile.json", _row(selection))
    atomic_write_json(output / "repair_policy_profile.json", _row(repair))
    atomic_write_json(output / "runtime_cost_profile.json", _row(runtime))
    print(json.dumps({
        "selection_depth_profile_sha256": selection.profile_sha256,
        "repair_policy_profile_sha256": repair.profile_sha256,
        "runtime_cost_profile_sha256": runtime.profile_sha256,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
