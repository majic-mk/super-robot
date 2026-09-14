"""Build a signed four-source development capture plan from audited corpus-repeat cases.

The generated requests are deterministic tokenizer reconstructions of one frozen
development case and its historical contexts.  They are for controlled
multi-source opportunity validation only; they never touch the locked test set.
"""
import argparse
import hashlib
import json
from pathlib import Path

from probekv.io import atomic_write_json
from probekv.v8_schema10_execution import digest_json


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode(tokenizer, text):
    return list(tokenizer.encode(text, add_special_tokens=False))


def _request(tokenizer, request_id, epoch, prefix_text, segment_text, segment_ids,
             content_key, suffix_text, group, parent_digest):
    prefix_ids = _encode(tokenizer, prefix_text)
    suffix_ids = _encode(tokenizer, suffix_text)
    token_ids = prefix_ids + list(segment_ids) + suffix_ids
    start = len(prefix_ids)
    positions = list(range(start, start + len(segment_ids)))
    return {
        "request_id": request_id,
        "request_epoch": int(epoch),
        "token_ids": token_ids,
        "segments": [{"segment_id": "C", "content_key": content_key,
                       "token_ids": list(segment_ids), "positions": positions}],
        "mandatory_suffix_positions": list(range(len(token_ids) - len(suffix_ids), len(token_ids))),
        "max_new_tokens": 32,
        "prefetch_window": 1,
        "evidence_class": "development_multisource_capture",
        "partition_role": "development",
        "content_group": group,
        "development_partition_digest": parent_digest,
        "paper_evidence": False,
        "locked_test_accessed": False,
        "source_context_prefix": hashlib.sha256(prefix_text.encode()).hexdigest(),
        "segment_text_digest": hashlib.sha256(segment_text.encode()).hexdigest(),
    }


def build_plan(cases_path, partition_path, runtime_manifest_path, tokenizer_path,
               output_path, group_id=None, case_index=0):
    from transformers import AutoTokenizer

    output_path = Path(output_path)
    rows = [json.loads(line) for line in Path(cases_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    cohorts = [r for r in rows if r.get("regime") == "corpus-repeat" and len(r.get("sources", [])) >= 4]
    if group_id is not None:
        cohorts = [r for r in cohorts if r.get("group_id") == group_id]
    if not cohorts:
        raise ValueError("no audited corpus-repeat case with at least four historical sources")
    case = cohorts[int(case_index)]
    sources = case["sources"][:4]
    parent_digest = _sha(Path(partition_path))
    runtime_sha = _sha(Path(runtime_manifest_path))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=False)
    segment_text = case["segment_text"]
    segment_ids = case["segment_token_ids"]
    content_key = case["reuse_content_key"]
    suffix = "\nQuestion: " + case.get("question", "") + "\nAnswer:"
    requests = []
    for epoch, source in enumerate(sources, 1):
        requests.append(_request(
            tokenizer, case["case_id"] + "::source::" + source["origin_example_id"], epoch,
            source["historical_context"], segment_text, segment_ids, content_key, suffix,
            case["group_id"], parent_digest))
    requests.append(_request(
        tokenizer, case["case_id"] + "::target", 5, case.get("current_context", ""),
        segment_text, segment_ids, content_key, suffix, case["group_id"], parent_digest))
    entries = [{"request_id": q["request_id"], "request_sha256": digest_json(q),
                "content_group": case["group_id"], "role": q["partition_role"],
                "paper_evidence": False, "locked_test_accessed": False} for q in requests]
    partition = {"kind": "source_policy_capture_partition_v1", "parent_development_partition_sha256": parent_digest,
                 "locked_test_accessed": False, "paper_evidence": False, "entries": entries}
    partition_path_out = Path(output_path).with_name(Path(output_path).stem + "-partition.json")
    atomic_write_json(partition_path_out, partition)
    partition_sha = _sha(partition_path_out)
    plan = {"kind": "source_policy_native_capture_plan_v1", "selection_path": "legacy_multicheckpoint",
            "source_requests": requests[:4], "target_request": requests[4],
            "partition_path": str(partition_path_out), "partition_file_sha256": partition_sha,
            "runtime_manifest_path": str(runtime_manifest_path), "runtime_manifest_file_sha256": runtime_sha,
            "host_budget_bytes": 8 * 1024**3, "global_byte_budget": 8 * 1024**3,
            "development_cohort": case["group_id"], "case_id": case["case_id"],
            "paper_evidence": False, "locked_test_accessed": False}
    plan["plan_sha256"] = digest_json(plan)
    atomic_write_json(output_path, plan)
    return {"plan": str(output_path), "partition": str(partition_path_out), "case_id": case["case_id"],
            "group_id": case["group_id"], "source_count": 4, "plan_sha256": plan["plan_sha256"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--runtime-manifest", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--group-id")
    p.add_argument("--case-index", type=int, default=0)
    args = p.parse_args()
    print(json.dumps(build_plan(args.cases, args.partition, args.runtime_manifest,
                                args.tokenizer, args.output, args.group_id, args.case_index),
                      sort_keys=True))


if __name__ == "__main__":
    main()
