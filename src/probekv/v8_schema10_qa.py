"""Recomputable QA evidence, never a hand-filled passed flag."""
from .metrics import best_answer_f1, normalize_answer
from .v8_schema10_execution import digest_json

SCORER_VERSION = "probekv-hotpot-normalized-answer-f1-v1"


def answer_evidence(token_ids, *, tokenizer, request):
    answer = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    references = request.get("answers")
    refs = list(references) if references else []
    dense = request.get("matched_dense_answer_evidence")
    contract = request.get("quality_contract")
    score = best_answer_f1(answer, refs) if refs else None
    exact = any(normalize_answer(answer) == normalize_answer(ref) for ref in refs) if refs else None
    passed = None
    if dense is not None:
        unsigned = {k: v for k, v in dense.items() if k != "evidence_sha256"}
        if (digest_json(unsigned) != dense.get("evidence_sha256")
                or dense.get("scorer_version") != SCORER_VERSION
                or dense.get("request_tokens_sha256") != digest_json(request["token_ids"])
                or dense.get("references") != refs):
            raise ValueError("matched dense QA evidence differs from request/scorer")
        if score is not None and contract is not None:
            limit = contract["max_answer_f1_drop"]
            if not 0 <= limit <= 1:
                raise ValueError("invalid preregistered QA tolerance")
            passed = score >= best_answer_f1(dense["answer"], refs) - limit
    row = {"token_ids": list(token_ids), "answer": answer, "references": refs,
        "request_tokens_sha256": digest_json(request["token_ids"]), "scorer_version": SCORER_VERSION,
        "answer_f1": score, "answer_exact_match": exact, "quality_passed": passed,
        "matched_dense_evidence_sha256": dense.get("evidence_sha256") if dense else None,
        "quality_contract": contract}
    row["evidence_sha256"] = digest_json(row)
    return {"token_ids": list(token_ids), "answer": answer, "quality_passed": passed, "qa_evidence": row}


def validate_answer_evidence(row, *, request):
    claimed = row.get("evidence_sha256")
    if digest_json({k: v for k, v in row.items() if k != "evidence_sha256"}) != claimed:
        raise ValueError("corrupt QA evidence")
    if row["scorer_version"] != SCORER_VERSION or row["request_tokens_sha256"] != digest_json(request["token_ids"]):
        raise ValueError("QA request/scorer mismatch")
    refs = list(request.get("answers") or [])
    if row["references"] != refs:
        raise ValueError("QA references differ")
    score = best_answer_f1(row["answer"], refs) if refs else None
    if row["answer_f1"] != score:
        raise ValueError("QA score does not follow raw answer")
    expected = None
    dense = request.get("matched_dense_answer_evidence")
    contract = request.get("quality_contract")
    if dense is not None:
        unsigned = {k: v for k, v in dense.items() if k != "evidence_sha256"}
        if (digest_json(unsigned) != dense.get("evidence_sha256") or dense.get("references") != refs
                or dense.get("request_tokens_sha256") != row["request_tokens_sha256"]
                or dense.get("scorer_version") != SCORER_VERSION):
            raise ValueError("bad matched dense QA evidence")
        if contract is not None and score is not None:
            expected = score >= best_answer_f1(dense["answer"], refs) - contract["max_answer_f1_drop"]
    if (row["quality_passed"] != expected or row.get("quality_contract") != contract
            or row.get("matched_dense_evidence_sha256") != (dense.get("evidence_sha256") if dense else None)):
        raise ValueError("QA pass flag differs from raw paired answers")
    return score
