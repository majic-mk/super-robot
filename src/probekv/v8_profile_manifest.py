from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Callable, Sequence, Tuple

from .canonical_segment import canonicalize_token_ids
from .manifest import ManifestCase, token_content_hash, validate_manifest


def canonicalize_v8_profile_cases(
    cases: Sequence[ManifestCase],
    *,
    tokenizer_signature: str,
    document_revision: str,
    decode: Callable[[Sequence[int]], str],
) -> Tuple[ManifestCase, ...]:
    """Upgrade an already model-tokenized development manifest to v8 Segments.

    This transformation is intentionally restricted to development evidence.
    It never retokenizes content and therefore preserves the exact model input
    identity while adding deterministic canonical Segment provenance.
    """

    if not tokenizer_signature or not document_revision:
        raise ValueError("v8 profile canonicalization requires frozen provenance")
    output = []
    for case in cases:
        if case.split not in {"calibration", "development"}:
            raise ValueError("v8 profile canonicalization cannot access pilot/test rows")
        parent_tokens = tuple(int(value) for value in case.segment_token_ids)
        for segment in canonicalize_token_ids(
            parent_tokens,
            tokenizer_signature=tokenizer_signature,
            document_revision=document_revision,
        ):
            provenance = hashlib.sha256(
                (
                    "%s|%s|%s|%d|%d|%d"
                    % (
                        case.case_id,
                        case.document_id,
                        document_revision,
                        segment.ordinal,
                        segment.token_start,
                        segment.token_end,
                    )
                ).encode("utf-8")
            ).hexdigest()
            output.append(
                replace(
                    case,
                    case_id="%s-c%03d" % (case.case_id, segment.ordinal),
                    segment_text=decode(segment.token_ids),
                    segment_token_ids=segment.token_ids,
                    content_hash=token_content_hash(segment.token_ids),
                    protocol_version=8,
                    canonicalizer_signature=segment.canonicalizer_signature,
                    segment_provenance_id=provenance,
                    reuse_content_key=segment.reuse_content_key(
                        case.model_signature, tokenizer_signature
                    ),
                    canonical_parent_content_hash=token_content_hash(parent_tokens),
                    canonical_parent_left_token_ids=parent_tokens[: segment.token_start],
                    canonical_parent_right_token_ids=parent_tokens[segment.token_end :],
                )
            )
    if output:
        validate_manifest(output)
    if len({case.case_id for case in output}) != len(output):
        raise RuntimeError("canonical profile case identifiers are not unique")
    return tuple(output)


__all__ = ["canonicalize_v8_profile_cases"]
