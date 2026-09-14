"""Aggregate native four-source capture observations without claiming reuse gains."""
import argparse
import glob
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.v8_schema10_execution import digest_json


def _winner(scores):
    if not isinstance(scores, dict) or not scores:
        return None
    return min(scores, key=lambda key: (float(scores[key]), str(key)))


def aggregate(inputs):
    rows = []
    for path in sorted(Path(p) for p in inputs):
        replay_path = path / "replay.json" if path.is_dir() else path
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        if replay.get("gpu_execution_allowed") is not False:
            raise ValueError("capture replay must remain diagnostic-only")
        cells = [c for c in replay.get("cells", [])
                 if c.get("reference_trim_ratio") == 0.15 and c.get("depth2_keep_fraction") == 1.0]
        if not cells:
            raise ValueError(f"missing full-cohort rho=0.15 cell: {replay_path}")
        cell = cells[0]
        d1, d2 = cell.get("depth1_scores"), cell.get("depth2_scores")
        if not isinstance(d1, dict) or not isinstance(d2, dict) or set(d1) != set(d2):
            raise ValueError(f"incomplete source score set: {replay_path}")
        rows.append({
            "capture_dir": str(replay_path.parent),
            "observation_sha256": replay.get("observation_sha256"),
            "source_count": len(d1),
            "d1_winner": _winner(d1),
            "d2_winner": _winner(d2),
            "d1_scores": d1,
            "d2_scores": d2,
            "d1_d2_agree": _winner(d1) == _winner(d2),
            "margin": cell.get("margin"),
            "qa_passed": cell.get("qa_passed"),
            "production_admission_allowed": cell.get("production_admission_allowed"),
        })
    if not rows:
        raise ValueError("at least one capture is required")
    d1_winners = {r["d1_winner"] for r in rows}
    d2_winners = {r["d2_winner"] for r in rows}
    report = {
        "kind": "multisource_selection_complementarity_report_v1",
        "evidence_origin": "real_cuda_selection_state_capture",
        "capture_count": len(rows),
        "source_count_per_capture": sorted({r["source_count"] for r in rows}),
        "distinct_d1_winners": len(d1_winners),
        "distinct_d2_winners": len(d2_winners),
        "d1_d2_disagreement_count": sum(not r["d1_d2_agree"] for r in rows),
        "d1_d2_disagreement_rate": sum(not r["d1_d2_agree"] for r in rows) / len(rows),
        "multisource_opportunity_observed": all(r["source_count"] >= 2 for r in rows),
        "multisource_complementarity_observed": len(d1_winners) > 1 or len(d2_winners) > 1,
        "reuse_gain_measured": False,
        "qa_and_final_commit_pending": True,
        "rows": rows,
        "paper_evidence": False,
        "locked_test_accessed": False,
    }
    report["report_sha256"] = digest_json(report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = aggregate(args.inputs)
    atomic_write_json(Path(args.output), report)
    print(json.dumps({k: report[k] for k in (
        "capture_count", "source_count_per_capture", "distinct_d1_winners",
        "distinct_d2_winners", "d1_d2_disagreement_rate", "reuse_gain_measured",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
