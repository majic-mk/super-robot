"""Lossless QA request construction for development Source-oracle experiments.

This helper does not authorize a partition or turn corpus pseudotime into real
request chronology. Callers must audit group membership before executing it.
"""
from .rag_data import render_preceding_context, segment_text
from .v8_schema10_execution import digest_json


def build_quality_request(example, case, encode, *, request_id, request_epoch,
                          partition_role, partition_digest, max_model_len,
                          max_new_tokens=32):
    if partition_role not in {"fit", "validation"}:
        raise ValueError("explicit audited fit/validation role required")
    if not example.answers or not partition_digest or not request_id:
        raise ValueError("QA answers and partition provenance required")
    if request_epoch < 1 or max_new_tokens < 1:
        raise ValueError("positive epoch and generation budget required")
    matches = [(i, d) for i, d in enumerate(example.documents)
               if d.document_id == case["target_document_id"]]
    if len(matches) != 1:
        raise ValueError("exactly one shared document occurrence required")
    index, document = matches[0]
    left = list(case.get("canonical_parent_left_token_ids", []))
    shared = list(case["segment_token_ids"])
    right = list(case.get("canonical_parent_right_token_ids", []))
    if not shared or list(encode(segment_text(document))) != left + shared + right:
        raise ValueError("canonical document token reconstruction mismatch")
    # Keep every preceding/following document, including supporting evidence.
    # The canonical document's exact token slice is not retokenized in context.
    prefix = list(encode(render_preceding_context(example.documents[:index]))) + left
    suffix = right + list(encode(render_preceding_context(example.documents[index + 1:])))
    suffix += list(encode("\nQuestion: " + example.question + "\nAnswer:"))
    tokens = prefix + shared + suffix
    if len(tokens) + max_new_tokens > max_model_len:
        raise ValueError("full QA context exceeds model limit; truncation forbidden")
    start = len(prefix)
    return {
        "request_id": request_id, "request_epoch": request_epoch,
        "token_ids": tokens,
        "segments": [{"segment_id": "C", "content_key": case["reuse_content_key"],
                      "token_ids": shared, "positions": list(range(start, start + len(shared)))}],
        "mandatory_suffix_positions": list(range(start + len(shared), len(tokens))),
        "question": example.question, "answers": list(example.answers),
        "max_new_tokens": max_new_tokens, "prefetch_window": 1,
        "partition_role": partition_role, "content_group": case["group_id"],
        "development_partition_digest": partition_digest,
        "origin_example_id": example.example_id,
        "origin_example_digest": digest_json(example.to_row()),
        "document_ids": [d.document_id for d in example.documents],
        "full_qa_context_preserved": True,
        "paper_evidence": False, "locked_test_accessed": False,
    }
