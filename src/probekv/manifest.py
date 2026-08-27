from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .data import assert_group_isolation, deterministic_group_split


CONTROLLED_REGIMES = (
    "high-prefix/same-order",
    "high-prefix/different-order",
    "low-prefix/same-order",
    "low-prefix/different-order",
)


def token_content_hash(token_ids: Sequence[int]) -> str:
    """Hash the model-tokenized segment, not its display text.

    Token IDs are serialized as canonical JSON to avoid platform-dependent
    integer widths.  A production manifest must be rebuilt when the tokenizer
    revision changes.
    """
    if not token_ids:
        raise ValueError("segment token_ids must not be empty")
    payload = json.dumps(
        [int(token) for token in token_ids], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ManifestSource:
    source_id: str
    historical_context: str
    context_id: str
    origin_example_id: str = ""
    preceding_document_ids: Tuple[str, ...] = ()
    regime: str = "unspecified"


@dataclass(frozen=True)
class ManifestCase:
    case_id: str
    dataset: str
    document_id: str
    group_id: str
    split: str
    regime: str
    model_signature: str
    segment_text: str
    segment_token_ids: Tuple[int, ...]
    content_hash: str
    current_context: str
    sources: Tuple[ManifestSource, ...]
    current_suffix_context: str = ""
    question: str = ""
    answers: Tuple[str, ...] = ()
    construction: str = "normalized_input"
    target_document_id: str = ""
    protocol_version: int = 0
    canonicalizer_signature: str = ""
    segment_provenance_id: str = ""
    reuse_content_key: str = ""
    canonical_parent_content_hash: str = ""
    # v7 preserves the complete parent RAG token sequence when one retrieved
    # document is canonicalized into several independently reusable Segments.
    canonical_parent_left_token_ids: Tuple[int, ...] = ()
    canonical_parent_right_token_ids: Tuple[int, ...] = ()

    def validate(self, online_kmax: Optional[int] = None) -> None:
        if self.split not in {"train", "calibration", "test", "pilot"}:
            raise ValueError("unsupported split: %s" % self.split)
        if not self.case_id or not self.dataset or not self.document_id:
            raise ValueError("case, dataset and document identifiers are required")
        if self.content_hash != token_content_hash(self.segment_token_ids):
            raise ValueError("content_hash does not match segment_token_ids")
        effective_kmax = (
            int(online_kmax)
            if online_kmax is not None
            else (16 if self.protocol_version in {7, 8} else 4)
        )
        if not 1 <= len(self.sources) <= effective_kmax:
            raise ValueError("source count exceeds online Kmax")
        source_ids = [source.source_id for source in self.sources]
        context_ids = [source.context_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source identifiers must be unique")
        if len(context_ids) != len(set(context_ids)):
            raise ValueError("historical contexts must be independently identified")
        contexts = [source.historical_context for source in self.sources]
        if len(contexts) != len(set(contexts)):
            raise ValueError("historical preceding contexts must be distinct")
        if self.current_context in contexts:
            raise ValueError(
                "current context must not be reused as a historical source context"
            )
        if self.protocol_version in {7, 8} and not all(
            (
                self.canonicalizer_signature,
                self.segment_provenance_id,
                self.reuse_content_key,
                self.canonical_parent_content_hash,
            )
        ):
            raise ValueError("v7/v8 manifest requires canonical Segment provenance")
        if self.protocol_version in {7, 8} and self.canonical_parent_content_hash != (
            token_content_hash(
                self.canonical_parent_left_token_ids
                + self.segment_token_ids
                + self.canonical_parent_right_token_ids
            )
        ):
            raise ValueError("v7/v8 canonical parent token sequence is incomplete")

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["segment_token_ids"] = list(self.segment_token_ids)
        row["canonical_parent_left_token_ids"] = list(
            self.canonical_parent_left_token_ids
        )
        row["canonical_parent_right_token_ids"] = list(
            self.canonical_parent_right_token_ids
        )
        row["sources"] = [asdict(source) for source in self.sources]
        return row


def manifest_digest(cases: Sequence[ManifestCase]) -> str:
    rows = [case.to_row() for case in sorted(cases, key=lambda item: item.case_id)]
    payload = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(
    cases: Sequence[ManifestCase], online_kmax: Optional[int] = None
) -> None:
    if not cases:
        raise ValueError("manifest must contain at least one case")
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("case identifiers must be unique")
    group_to_splits: Dict[str, List[str]] = {}
    for case in cases:
        case.validate(online_kmax)
        isolation_keys = (
            "group:%s" % case.group_id,
            "content:%s" % case.content_hash,
            "document:%s:%s" % (case.dataset, case.document_id),
        )
        for key in isolation_keys:
            group_to_splits.setdefault(key, []).append(case.split)
    assert_group_isolation(group_to_splits)


def case_from_mapping(
    raw: Mapping[str, Any],
    model_signature: str,
    seed: int = 20260726,
) -> ManifestCase:
    """Build a strict manifest row from already-tokenized input.

    Required input fields are case_id, dataset, document_id, segment_text,
    segment_token_ids, current_context and historical_contexts.  The latter can
    contain strings or objects with context_id and text fields.
    """
    token_ids = tuple(int(token) for token in raw["segment_token_ids"])
    content_hash = token_content_hash(token_ids)
    group_id = str(
        raw.get(
            "group_id",
            "%s:%s:%s"
            % (raw["dataset"], raw["document_id"], content_hash),
        )
    )
    split = str(raw.get("split") or deterministic_group_split(group_id, seed))
    sources = []
    for index, source in enumerate(raw["historical_contexts"]):
        if isinstance(source, Mapping):
            context_id = str(source.get("context_id", "ctx-%d" % index))
            text = str(source["text"])
        else:
            context_id = "ctx-%d" % index
            text = str(source)
        sources.append(ManifestSource("s%d" % index, text, context_id))
    result = ManifestCase(
        case_id=str(raw["case_id"]),
        dataset=str(raw["dataset"]),
        document_id=str(raw["document_id"]),
        group_id=group_id,
        split=split,
        regime=str(raw.get("regime", "natural")),
        model_signature=model_signature,
        segment_text=str(raw["segment_text"]),
        segment_token_ids=token_ids,
        content_hash=content_hash,
        current_context=str(raw["current_context"]),
        sources=tuple(sources),
        current_suffix_context=str(raw.get("current_suffix_context", "")),
    )
    result.validate()
    return result


def manifest_case_from_row(raw: Mapping[str, Any]) -> ManifestCase:
    """Rehydrate a previously audited manifest row without re-splitting it."""
    sources = tuple(
        ManifestSource(
            source_id=str(source["source_id"]),
            historical_context=str(source["historical_context"]),
            context_id=str(source["context_id"]),
            origin_example_id=str(source.get("origin_example_id", "")),
            preceding_document_ids=tuple(
                str(value) for value in source.get("preceding_document_ids", ())
            ),
            regime=str(source.get("regime", "unspecified")),
        )
        for source in raw["sources"]
    )
    result = ManifestCase(
        case_id=str(raw["case_id"]),
        dataset=str(raw["dataset"]),
        document_id=str(raw["document_id"]),
        group_id=str(raw["group_id"]),
        split=str(raw["split"]),
        regime=str(raw["regime"]),
        model_signature=str(raw["model_signature"]),
        segment_text=str(raw["segment_text"]),
        segment_token_ids=tuple(int(token) for token in raw["segment_token_ids"]),
        content_hash=str(raw["content_hash"]),
        current_context=str(raw["current_context"]),
        sources=sources,
        current_suffix_context=str(raw.get("current_suffix_context", "")),
        question=str(raw.get("question", "")),
        answers=tuple(str(answer) for answer in raw.get("answers", ())),
        construction=str(raw.get("construction", "normalized_input")),
        target_document_id=str(raw.get("target_document_id", "")),
        protocol_version=int(raw.get("protocol_version", 0)),
        canonicalizer_signature=str(raw.get("canonicalizer_signature", "")),
        segment_provenance_id=str(raw.get("segment_provenance_id", "")),
        reuse_content_key=str(raw.get("reuse_content_key", "")),
        canonical_parent_content_hash=str(
            raw.get("canonical_parent_content_hash", "")
        ),
        canonical_parent_left_token_ids=tuple(
            int(value)
            for value in raw.get("canonical_parent_left_token_ids", ())
        ),
        canonical_parent_right_token_ids=tuple(
            int(value)
            for value in raw.get("canonical_parent_right_token_ids", ())
        ),
    )
    result.validate()
    return result


def synthetic_manifest(
    cases: int,
    seed: int,
    model_signature: str = "synthetic-reference-v1",
    online_kmax: int = 4,
) -> List[ManifestCase]:
    """Create deterministic, balanced fixtures for pipeline validation only."""
    if cases < 3:
        raise ValueError("at least three cases are required to cover all splits")
    if not 1 <= online_kmax <= 4:
        raise ValueError("online_kmax must be in [1, 4]")
    desired_splits = ("train", "train", "train", "train", "train", "calibration", "calibration", "test", "test", "test")
    result: List[ManifestCase] = []
    nonce = 0
    for index in range(cases):
        desired = desired_splits[index % len(desired_splits)]
        segment_text = "Synthetic repeated segment %04d" % index
        token_ids = tuple(1000 + index * 7 + offset for offset in range(16 + index % 8))
        content_hash = token_content_hash(token_ids)
        while True:
            group_id = "fixture:%04d:%06d:%s" % (index, nonce, content_hash[:12])
            nonce += 1
            if deterministic_group_split(group_id, seed) == desired:
                break
        sources = tuple(
            ManifestSource(
                source_id="s%d" % source_index,
                historical_context="Synthetic historical context %d for case %d"
                % (source_index, index),
                context_id="ctx-%04d-%d" % (index, source_index),
            )
            for source_index in range(online_kmax)
        )
        result.append(
            ManifestCase(
                case_id="fixture-%04d" % index,
                dataset="controlled-fixture",
                document_id="doc-%04d" % index,
                group_id=group_id,
                split=desired,
                regime=CONTROLLED_REGIMES[index % len(CONTROLLED_REGIMES)],
                model_signature=model_signature,
                segment_text=segment_text,
                segment_token_ids=token_ids,
                content_hash=content_hash,
                current_context="Synthetic current context for case %d" % index,
                sources=sources,
            )
        )
    validate_manifest(result, online_kmax)
    return result
