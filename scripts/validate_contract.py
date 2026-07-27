"""Cross-check the frozen YAML contract and enumerate expensive matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from probekv.matrix import main_rag_matrix, profile_matrix
from probekv.statistics import minimum_zero_violation_trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract_path = Path(args.contract)
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    errors = []
    invariants = contract["invariants"]
    if invariants["canonical_source_origin"] != "full_prefill":
        errors.append("canonical source origin must be full_prefill")
    if invariants["promote_selective_repair"]:
        errors.append("selective repair promotion must be false")
    if invariants["online_kmax"] != 4:
        errors.append("online Kmax must be 4")
    if max(invariants["offline_k"]) != 8:
        errors.append("offline K ablation must include 8")
    if max(contract["repair_ratios"]) != 1.0 or min(contract["repair_ratios"]) != 0.0:
        errors.append("repair grid must cover [0, 1]")
    tail_minimum = minimum_zero_violation_trials(
        contract["quality"]["tail_violation_upper_bound"], 0.95
    )
    pooled_cases = sum(dataset["locked_test"] for dataset in contract["datasets"]["rag"])
    if contract["quality"]["tail_gate_scope"] == "pooled_three_rag_datasets_per_model":
        if pooled_cases < tail_minimum:
            errors.append("pooled tail gate has insufficient cases")
    matrix_counts = {
        "main_rag_cells_before_replays": sum(1 for _ in main_rag_matrix()),
        "profile_cells_without_ssd": sum(1 for _ in profile_matrix(False)),
        "profile_cells_with_ssd": sum(1 for _ in profile_matrix(True)),
        "minimum_cases_for_zero_violation_exact_95pct_upper_1pct": tail_minimum,
        "pooled_rag_cases_per_primary_model": pooled_cases,
    }
    result = {
        "contract": str(contract_path.resolve()),
        "valid": not errors,
        "errors": errors,
        "matrix_counts": matrix_counts,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
