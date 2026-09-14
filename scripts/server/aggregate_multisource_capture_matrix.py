"""Aggregate native four-source capture observations without claiming reuse gains."""
import argparse
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.v8_schema10_execution import digest_json
from probekv.source_policy_replay import replay_observation


def _winner(scores):
    if not isinstance(scores, dict) or not scores:
        return None
    return min(scores, key=lambda key: (float(scores[key]), str(key)))


def aggregate(inputs):
    rows = []
    seen_observations = set()
    for path in sorted(Path(p) for p in inputs):
        replay_path = path / "replay.json" if path.is_dir() else path
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        observation = json.loads((replay_path.parent / "observation.json").read_text(encoding="utf-8"))
        # A signed summary alone cannot establish native provenance. Validate
        # the raw observation and recompute the deterministic policy replay.
        if digest_json(replay) != digest_json(replay_observation(observation)):
            raise ValueError("replay differs from its raw observation")
        observation_id = observation["observation_sha256"]
        if observation_id in seen_observations:
            raise ValueError("duplicate observation cannot count as another request")
        seen_observations.add(observation_id)
        if replay.get("gpu_execution_allowed") is not False:
            raise ValueError("capture replay must remain diagnostic-only")
        cells = [c for c in replay.get("cells", [])
                 if c.get("reference_trim_ratio") == 0.15 and c.get("depth2_keep_fraction") == 1.0]
        if len(cells) != 1:
            raise ValueError(f"missing full-cohort rho=0.15 cell: {replay_path}")
        cell = cells[0]
        d1, d2 = cell.get("depth1_scores"), cell.get("depth2_scores")
        if not isinstance(d1, dict) or not isinstance(d2, dict) or set(d1) != set(d2):
            raise ValueError(f"incomplete source score set: {replay_path}")
        rows.append({
            "capture_dir": str(replay_path.parent),
            "evidence_origin": observation["evidence_origin"],
            "request_id": replay["provenance"]["request_id"],
            "pool_comparison_key": digest_json({
                "source_ids": sorted(d1),
                "model_signature": replay["provenance"]["model_signature"],
                "tokenizer_signature": replay["provenance"]["tokenizer_signature"],
                "code_commit": replay["provenance"]["code_commit"],
                "patch_sha256": replay["provenance"]["patch_sha256"],
                "config_sha256": replay["provenance"]["config_sha256"],
                "partition": replay["provenance"]["development_partition_digest"],
            }),
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
    pool_groups = {}
    for row in rows:
        pool_groups.setdefault(row["pool_comparison_key"], []).append(row)
    within_pool_ranking_changes = any(
        len({r["request_id"] for r in group}) > 1
        and (len({r["d1_winner"] for r in group}) > 1
             or len({r["d2_winner"] for r in group}) > 1)
        for group in pool_groups.values())
    origins = {row["evidence_origin"] for row in rows}
    report = {
        "kind": "multisource_selection_complementarity_report_v2",
        "evidence_origin": (next(iter(origins)) if len(origins) == 1 else "mixed_diagnostic"),
        "comparison_execution_device": "cpu",
        "capture_count": len(rows),
        "source_count_per_capture": sorted({r["source_count"] for r in rows}),
        "distinct_d1_winners": len(d1_winners),
        "distinct_d2_winners": len(d2_winners),
        "d1_d2_disagreement_count": sum(not r["d1_d2_agree"] for r in rows),
        "d1_d2_disagreement_rate": sum(not r["d1_d2_agree"] for r in rows) / len(rows),
        "multisource_opportunity_observed": all(r["source_count"] >= 2 for r in rows),
        "candidate_pool_count": len(pool_groups),
        "same_pool_cross_request_rank_change_observed": within_pool_ranking_changes,
        "multisource_complementarity_observed": None,
        "complementarity_status": "PENDING_SAME_POOL_QA_AND_NET_GAIN",
        "global_distinct_winners_are_complementarity_evidence": False,
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
