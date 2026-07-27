"""Create diagnostic plots for the local E1/E2 software-validation run.

These plots validate the reporting path only. They are watermarked as
synthetic and must never be used as empirical paper evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required; install requirements/analysis.txt") from error

    source = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    decisions = read_jsonl(source / "decisions.jsonl")
    labels = read_jsonl(source / "safe_budget_labels.jsonl")
    if not decisions or not labels:
        raise ValueError("input directory does not contain a completed local E1/E2 run")

    primary_layer = json.loads(
        (source / "summary.json").read_text(encoding="utf-8")
    )["primary_reuse_layer"]
    by_case = {}
    for row in labels:
        if row["reuse_layer"] == primary_layer and row["split"] == "test":
            by_case.setdefault(row["case_id"], []).append(row["safe_repair_ratio"])
    spreads = [max(values) - min(values) for values in by_case.values()]

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].hist(spreads, bins=10, color="#3B82F6", edgecolor="white")
    axes[0].axvline(0.10, color="#DC2626", linestyle="--", linewidth=1)
    axes[0].set_title("Source safe-ratio spread")
    axes[0].set_xlabel("max(r_safe) - min(r_safe)")
    axes[0].set_ylabel("Synthetic test cases")

    counts = Counter(
        "abstain" if row["abstained"] else str(row["probe_layer"])
        for row in decisions
    )
    keys = sorted((key for key in counts if key != "abstain"), key=int)
    if "abstain" in counts:
        keys.append("abstain")
    axes[1].bar(keys, [counts[key] for key in keys], color="#10B981")
    axes[1].set_title("Dynamic probe exit")
    axes[1].set_xlabel("Probe layer / outcome")

    probe = [row["normalized_regret_with_full_fallback"] for row in decisions]
    cachecraft = []
    for row in decisions:
        costs = row["actual_costs_ms"]
        oracle = min(costs.values())
        worst = max(costs.values())
        denominator = worst - oracle
        chosen = costs[row["cachecraft_source"]]
        cachecraft.append(0.0 if denominator <= 0 else (chosen - oracle) / denominator)
    axes[2].boxplot([cachecraft, probe], labels=["Metadata", "ProbeKV"])
    axes[2].set_title("Normalized regret")
    axes[2].set_ylim(-0.05, 1.05)

    figure.suptitle("SYNTHETIC LOCAL VALIDATION — NOT PAPER EVIDENCE", fontsize=11)
    figure.tight_layout()
    figure.savefig(output / "local_e1e2_diagnostics.png", dpi=180)
    figure.savefig(output / "local_e1e2_diagnostics.pdf")
    print(str((output / "local_e1e2_diagnostics.png").resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
