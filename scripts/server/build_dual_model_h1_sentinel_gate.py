"""Build the non-paper stop gate after both model H1 sentinels pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json, sha256_file
from probekv.model_adapters import MISTRAL_SPEC, QWEN_SPEC


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_model(name: str, qualification: dict, h1: dict) -> list[str]:
    failures = []
    expected = MISTRAL_SPEC if name == "mistral" else QWEN_SPEC
    if qualification.get("model_id") != expected.model_id:
        failures.append("%s qualification used the wrong model" % name)
    if qualification.get("model_revision") != expected.revision:
        failures.append("%s qualification used the wrong revision" % name)
    if qualification.get("adapter_name") != expected.adapter_name:
        failures.append("%s qualification used the wrong adapter" % name)
    if qualification.get("schema_version") != 2:
        failures.append("%s qualification is not schema v2" % name)
    for key in (
        "native_prefix_cache_qualified",
        "gpu_runtime_qualified",
        "h1_h2_execution_allowed",
    ):
        if qualification.get(key) is not True:
            failures.append("%s qualification lacks %s" % (name, key))
    if qualification.get("failures") not in ([], ()):
        failures.append("%s qualification contains failures" % name)
    if h1.get("paper_evidence") is not False:
        failures.append("%s H1 sentinel is not marked non-paper" % name)
    if h1.get("code_commit") != qualification.get("code_commit"):
        failures.append("%s H1 sentinel used another code commit" % name)
    if h1.get("model_id") != qualification.get("model_id"):
        failures.append("%s H1 sentinel used another model" % name)
    if int(h1.get("completed_cases_this_run", -1)) != 1:
        failures.append("%s H1 sentinel did not complete one case" % name)
    if int(h1.get("completed_groups_this_run", -1)) != 4:
        failures.append("%s H1 sentinel did not complete four Sources" % name)
    if int(h1.get("appended_rows_this_run", -1)) != 36:
        failures.append("%s H1 sentinel did not append 36 ratio rows" % name)
    if h1.get("r1_dense_equivalence_passed") is not True:
        failures.append("%s H1 r=1 dense equivalence failed" % name)
    if h1.get("h1_scan_allowed") is not True or h1.get("failure") is not None:
        failures.append("%s H1 sentinel did not authorize scanning" % name)
    return failures


def build_joint_gate(rows: dict, *, hourly_price: float = 0.0) -> dict:
    failures = _validate_model(
        "mistral", rows["mistral_qualification"], rows["mistral_h1"]
    )
    failures += _validate_model(
        "qwen", rows["qwen_qualification"], rows["qwen_h1"]
    )
    commits = {
        rows["mistral_qualification"].get("code_commit"),
        rows["qwen_qualification"].get("code_commit"),
    }
    if len(commits) != 1 or None in commits:
        failures.append("dual-model gates do not share one frozen code commit")
    estimates = {}
    for name in ("mistral", "qwen"):
        seconds = float(rows[name + "_h1"].get("elapsed_seconds_this_run", 0.0))
        if seconds <= 0:
            failures.append("%s H1 sentinel lacks an elapsed-time estimate" % name)
            continue
        main_hours = seconds * 150 / 3600.0
        anchor_hours = seconds * 30 * 4 / 3600.0
        estimates[name] = {
            "single_case_seconds": seconds,
            "primary_150_case_gpu_hours": main_hours,
            "anchor_30_case_gpu_hours": anchor_hours,
            "estimated_total_gpu_hours": main_hours + anchor_hours,
            "estimated_cost": (main_hours + anchor_hours) * hourly_price,
        }
    ready = not failures
    return {
        "schema_version": 1,
        "stage": "dual_model_h1_sentinel_gate",
        "code_commit": next(iter(commits)) if len(commits) == 1 else None,
        "mistral_prefix_qualified": rows["mistral_qualification"].get(
            "native_prefix_cache_qualified"
        ) is True,
        "mistral_runtime_qualified": rows["mistral_qualification"].get(
            "gpu_runtime_qualified"
        ) is True,
        "mistral_h1_sentinel_passed": not _validate_model(
            "mistral", rows["mistral_qualification"], rows["mistral_h1"]
        ),
        "qwen_prefix_qualified": rows["qwen_qualification"].get(
            "native_prefix_cache_qualified"
        ) is True,
        "qwen_runtime_qualified": rows["qwen_qualification"].get(
            "gpu_runtime_qualified"
        ) is True,
        "qwen_h1_sentinel_passed": not _validate_model(
            "qwen", rows["qwen_qualification"], rows["qwen_h1"]
        ),
        "ready_for_full_h1_pilot": ready,
        "full_h1_started": False,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "estimates": estimates,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mistral-qualification", required=True)
    parser.add_argument("--mistral-h1-gate", required=True)
    parser.add_argument("--qwen-qualification", required=True)
    parser.add_argument("--qwen-h1-gate", required=True)
    parser.add_argument("--hourly-price", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "mistral_qualification": Path(args.mistral_qualification).resolve(),
        "mistral_h1": Path(args.mistral_h1_gate).resolve(),
        "qwen_qualification": Path(args.qwen_qualification).resolve(),
        "qwen_h1": Path(args.qwen_h1_gate).resolve(),
    }
    rows = {key: _json(path) for key, path in paths.items()}
    output = build_joint_gate(rows, hourly_price=args.hourly_price)
    output["input_sha256"] = {
        key: sha256_file(path) for key, path in paths.items()
    }
    atomic_write_json(Path(args.output).resolve(), output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["ready_for_full_h1_pilot"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
