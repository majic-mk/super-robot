#!/usr/bin/env python3
"""Create the dual-model schema10 Profile-freeze Gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from probekv.io import atomic_write_json, sha256_file


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mistral-bundle", required=True)
    parser.add_argument("--qwen-bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "mistral": Path(args.mistral_bundle).resolve(),
        "qwen": Path(args.qwen_bundle).resolve(),
    }
    bundles = {key: _load(path) for key, path in paths.items()}
    for key, bundle in bundles.items():
        if (bundle.get("protocol_version"), bundle.get("schema_version")) != (8, 10):
            raise ValueError(f"{key} Profile bundle has the wrong schema")
        if bundle.get("stage") != "schema10_profile_bundle_frozen":
            raise ValueError(f"{key} Profile bundle is not frozen")
        if bundle.get("ready_for_schema10_runtime_qualification") is not True:
            raise ValueError(f"{key} Profile bundle is not qualification-ready")
        if bundle.get("quality_tail_rate_1pct_certified") is not False:
            raise ValueError("Profile evidence cannot certify the 1% quality tail")
        if bundle.get("real_cuda_timing") is not True or bundle.get("fake_timing") is not False:
            raise ValueError(f"{key} Profile bundle lacks real CUDA timing")
        if bundle.get("operational_coverage_causal") is not True:
            raise ValueError(f"{key} Profile bundle coverage is not causal")
        if bundle.get("paper_evidence") is not False or bundle.get("locked_test_accessed") is not False:
            raise ValueError("Profile bundle crossed the non-paper boundary")
        claimed = str(bundle.get("profile_bundle_sha256", ""))
        unsigned = dict(bundle)
        unsigned.pop("profile_bundle_sha256", None)
        observed = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if claimed != observed:
            raise ValueError(f"{key} Profile bundle digest differs")
    if bundles["mistral"]["code_commit"] != bundles["qwen"]["code_commit"]:
        raise ValueError("dual-model Profile bundles use different code commits")
    if bundles["mistral"]["model_id"] == bundles["qwen"]["model_id"]:
        raise ValueError("dual-model Profile bundles must use distinct models")
    combined = {
        "protocol_version": 8,
        "schema_version": 10,
        "stage": "schema10_dual_model_profile_freeze_gate",
        "code_commit": bundles["mistral"]["code_commit"],
        "mistral_profile_bundle_frozen": True,
        "qwen_profile_bundle_frozen": True,
        "profile_freeze_order_verified": all(
            bundle.get("profile_freeze_order_verified") is True
            for bundle in bundles.values()
        ),
        "operational_coverage_causal": all(
            bundle.get("operational_coverage_causal") is True
            for bundle in bundles.values()
        ),
        "real_cuda_timing": True,
        "fake_timing": False,
        "bundles": {
            key: {
                "path_sha256": sha256_file(paths[key]),
                "profile_bundle_sha256": bundles[key]["profile_bundle_sha256"],
                "model_id": bundles[key]["model_id"],
                "gpu_uuid": bundles[key]["gpu_uuid"],
            }
            for key in ("mistral", "qwen")
        },
        "ready_for_schema10_runtime_qualification": True,
        "runtime_qualification_jobs_per_model": 140,
        "runtime_qualification_jobs_total": 280,
        "minimum_quality_certification_requests_per_model": 300,
        "quality_tail_rate_1pct_certified": False,
        "gpu_runtime_qualified": False,
        "h1_h2_execution_allowed": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "failures": [],
    }
    combined["gate_sha256"] = hashlib.sha256(
        json.dumps(combined, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    atomic_write_json(Path(args.output).resolve(), combined)
    print(json.dumps(combined, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
