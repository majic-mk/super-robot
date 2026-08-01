from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .data import assert_group_isolation
from .manifest import ManifestSource, token_content_hash
from .model_signature import validate_v6_model_signature
from .v6_contracts import RegionKind, RegionSpec


@dataclass(frozen=True)
class SegmentManifest:
    segment_id: str
    order: int
    document_id: str
    token_start: int
    token_end: int
    segment_text: str
    segment_token_ids: Tuple[int, ...]
    content_hash: str
    sources: Tuple[ManifestSource, ...]

    def validate(self, max_sources: int = 16) -> None:
        if not self.segment_id or not self.document_id:
            raise ValueError("segment and document IDs are required")
        if self.order < 0 or self.token_start < 0 or self.token_end <= self.token_start:
            raise ValueError("invalid segment order or span")
        if len(self.segment_token_ids) != self.token_end - self.token_start:
            raise ValueError("segment token IDs do not match span")
        if self.content_hash != token_content_hash(self.segment_token_ids):
            raise ValueError("segment content hash mismatch")
        if len(self.sources) > max_sources:
            raise ValueError("segment has more than 16 historical variants")
        source_ids = [source.source_id for source in self.sources]
        context_ids = [source.context_id for source in self.sources]
        contexts = [source.historical_context for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("segment source IDs must be unique")
        if len(context_ids) != len(set(context_ids)):
            raise ValueError("segment historical context IDs must be unique")
        if len(contexts) != len(set(contexts)):
            raise ValueError("segment historical contexts must be independent")


@dataclass(frozen=True)
class RequestManifestCase:
    case_id: str
    dataset: str
    group_id: str
    split: str
    model_signature: str
    token_ids: Tuple[int, ...]
    regions: Tuple[RegionSpec, ...]
    segments: Tuple[SegmentManifest, ...]
    exact_prefix_tokens: int = 0
    mandatory_suffix_tokens: int = 0
    question: str = ""
    answers: Tuple[str, ...] = ()
    current_context_id: str = ""

    def __post_init__(self) -> None:
        if not self.current_context_id:
            object.__setattr__(self, "current_context_id", self.case_id)

    def validate(self, max_sources_per_segment: int = 16) -> None:
        if self.split not in {"train", "calibration", "test", "pilot"}:
            raise ValueError("unsupported split")
        if not self.case_id or not self.dataset or not self.group_id:
            raise ValueError("case manifest identifiers are required")
        if not self.model_signature or not self.token_ids:
            raise ValueError("model signature and request tokens are required")
        validate_v6_model_signature(self.model_signature)
        cursor = 0
        for region in self.regions:
            if region.start != cursor:
                raise ValueError("manifest regions must be contiguous")
            cursor = region.end
        if cursor != len(self.token_ids):
            raise ValueError("manifest regions do not cover the request")
        if self.exact_prefix_tokens:
            prefixes = [
                region for region in self.regions
                if region.kind is RegionKind.PREFIX_EXACT
            ]
            if len(prefixes) != 1 or prefixes[0].end != self.exact_prefix_tokens:
                raise ValueError("manifest exact prefix mismatch")
        if self.mandatory_suffix_tokens:
            suffixes = [
                region for region in self.regions
                if region.kind is RegionKind.MANDATORY_SUFFIX
            ]
            if len(suffixes) != 1 or suffixes[0].token_count != (
                self.mandatory_suffix_tokens
            ):
                raise ValueError("manifest mandatory suffix mismatch")
        if [segment.order for segment in self.segments] != list(
            range(len(self.segments))
        ):
            raise ValueError("manifest segments must be densely ordered")
        by_id = {segment.segment_id: segment for segment in self.segments}
        if len(by_id) != len(self.segments):
            raise ValueError("manifest segment IDs must be unique")
        reuse_regions = [
            region for region in self.regions
            if region.kind is RegionKind.REUSE_CANDIDATE
        ]
        if {region.segment_id for region in reuse_regions} != set(by_id):
            raise ValueError("manifest reuse regions and segments disagree")
        for segment in self.segments:
            segment.validate(max_sources_per_segment)
            current_context_id = self.current_context_id or self.case_id
            if any(
                source.context_id == current_context_id
                for source in segment.sources
            ):
                raise ValueError(
                    "current context cannot enter historical Source candidates"
                )
            region = next(
                item for item in reuse_regions
                if item.segment_id == segment.segment_id
            )
            if (region.start, region.end) != (
                segment.token_start,
                segment.token_end,
            ):
                raise ValueError("manifest segment span mismatch")

    def to_row(self) -> Dict[str, Any]:
        self.validate()
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "group_id": self.group_id,
            "split": self.split,
            "model_signature": self.model_signature,
            "token_ids": list(self.token_ids),
            "regions": [
                {
                    **asdict(region),
                    "kind": region.kind.value,
                }
                for region in self.regions
            ],
            "segments": [
                {
                    **asdict(segment),
                    "segment_token_ids": list(segment.segment_token_ids),
                    "sources": [asdict(source) for source in segment.sources],
                }
                for segment in self.segments
            ],
            "exact_prefix_tokens": self.exact_prefix_tokens,
            "mandatory_suffix_tokens": self.mandatory_suffix_tokens,
            "question": self.question,
            "answers": list(self.answers),
            "current_context_id": self.current_context_id,
        }


def validate_request_manifest(cases: Sequence[RequestManifestCase]) -> None:
    if not cases:
        raise ValueError("request manifest must not be empty")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("request manifest case IDs must be unique")
    isolation = {}
    for case in cases:
        case.validate()
        keys = ["group:%s" % case.group_id]
        for segment in case.segments:
            keys.extend(
                (
                    "content:%s" % segment.content_hash,
                    "document:%s:%s" % (case.dataset, segment.document_id),
                )
            )
        for key in keys:
            isolation.setdefault(key, []).append(case.split)
    assert_group_isolation(isolation)


def request_manifest_digest(cases: Sequence[RequestManifestCase]) -> str:
    validate_request_manifest(cases)
    rows = [case.to_row() for case in sorted(cases, key=lambda item: item.case_id)]
    payload = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def request_case_from_mapping(raw: Mapping[str, Any]) -> RequestManifestCase:
    regions = tuple(
        RegionSpec(
            region_id=str(item["region_id"]),
            kind=RegionKind(str(item["kind"])),
            start=int(item["start"]),
            end=int(item["end"]),
            segment_id=(
                None
                if item.get("segment_id") is None
                else str(item["segment_id"])
            ),
        )
        for item in raw["regions"]
    )
    segments = []
    for item in raw["segments"]:
        sources = tuple(
            ManifestSource(
                source_id=str(source["source_id"]),
                historical_context=str(source["historical_context"]),
                context_id=str(source["context_id"]),
                origin_example_id=str(source.get("origin_example_id", "")),
                preceding_document_ids=tuple(
                    str(value)
                    for value in source.get("preceding_document_ids", ())
                ),
                regime=str(source.get("regime", "unspecified")),
            )
            for source in item.get("sources", ())
        )
        segments.append(
            SegmentManifest(
                segment_id=str(item["segment_id"]),
                order=int(item["order"]),
                document_id=str(item["document_id"]),
                token_start=int(item["token_start"]),
                token_end=int(item["token_end"]),
                segment_text=str(item["segment_text"]),
                segment_token_ids=tuple(
                    int(value) for value in item["segment_token_ids"]
                ),
                content_hash=str(item["content_hash"]),
                sources=sources,
            )
        )
    case = RequestManifestCase(
        case_id=str(raw["case_id"]),
        dataset=str(raw["dataset"]),
        group_id=str(raw["group_id"]),
        split=str(raw["split"]),
        model_signature=str(raw["model_signature"]),
        token_ids=tuple(int(value) for value in raw["token_ids"]),
        regions=regions,
        segments=tuple(segments),
        exact_prefix_tokens=int(raw.get("exact_prefix_tokens", 0)),
        mandatory_suffix_tokens=int(
            raw.get("mandatory_suffix_tokens", 0)
        ),
        question=str(raw.get("question", "")),
        answers=tuple(str(value) for value in raw.get("answers", ())),
        current_context_id=str(
            raw.get("current_context_id", raw["case_id"])
        ),
    )
    case.validate()
    return case
