from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from .data import deterministic_group_split
from .manifest import (
    ManifestCase,
    ManifestSource,
    token_content_hash,
    validate_manifest,
)


TokenEncoder = Callable[[str], Sequence[int]]


@dataclass(frozen=True)
class RAGDocument:
    document_id: str
    title: str
    text: str
    supporting: bool
    position: int


@dataclass(frozen=True)
class RAGExample:
    dataset: str
    example_id: str
    question: str
    answers: Tuple[str, ...]
    documents: Tuple[RAGDocument, ...]

    def validate(self) -> None:
        if not self.example_id or not self.question:
            raise ValueError("example id and question are required")
        if not self.documents:
            raise ValueError("at least one document is required")
        identifiers = [document.document_id for document in self.documents]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("document ids must be unique within an example")

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["answers"] = list(self.answers)
        row["documents"] = [asdict(document) for document in self.documents]
        return row


def _text_digest(title: str, text: str) -> str:
    normalized = "%s\n%s" % (title.strip(), text.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sentences_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(str(sentence).strip() for sentence in value if str(sentence).strip())
    raise ValueError("document text/sentences must be a string or sequence")


def _answers(raw: Mapping[str, Any]) -> Tuple[str, ...]:
    candidates = raw.get("answers", raw.get("answer", ()))
    if isinstance(candidates, Mapping):
        candidates = candidates.get("text", ())
    if isinstance(candidates, str):
        candidates = [candidates]
    values = []
    for value in candidates or ():
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    aliases = raw.get("answer_aliases", ()) or ()
    if isinstance(aliases, str):
        aliases = [aliases]
    for alias in aliases:
        text = str(alias).strip()
        if text and text not in values:
            values.append(text)
    return tuple(values)


def _supporting_titles(raw: Mapping[str, Any]) -> set:
    supporting = raw.get("supporting_facts", ()) or ()
    if isinstance(supporting, Mapping):
        title_values = supporting.get("title", ())
        if isinstance(title_values, str):
            title_values = [title_values]
        return {str(title) for title in title_values}
    titles = set()
    for item in supporting:
        if isinstance(item, Mapping):
            title = item.get("title")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            title = item[0] if item else None
        else:
            title = None
        if title is not None:
            titles.add(str(title))
    return titles


def _context_documents(
    raw_context: Any, supporting_titles: set
) -> Tuple[RAGDocument, ...]:
    pairs = []
    if isinstance(raw_context, Mapping):
        titles = raw_context.get("title", raw_context.get("titles", ()))
        texts = raw_context.get(
            "sentences", raw_context.get("text", raw_context.get("paragraphs", ()))
        )
        if isinstance(titles, str):
            titles = [titles]
        if isinstance(texts, str):
            texts = [texts]
        pairs = list(zip(titles, texts))
    elif isinstance(raw_context, Sequence) and not isinstance(raw_context, (str, bytes)):
        for item in raw_context:
            if isinstance(item, Mapping):
                title = item.get("title", item.get("name", ""))
                text = item.get(
                    "sentences", item.get("text", item.get("paragraph_text", ""))
                )
                pairs.append((title, text))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
                pairs.append((item[0], item[1]))
    if not pairs:
        raise ValueError("unrecognized or empty context structure")
    documents = []
    seen = set()
    for position, (title_value, text_value) in enumerate(pairs):
        title = str(title_value).strip()
        text = _sentences_to_text(text_value)
        identifier = _text_digest(title, text)
        if identifier in seen:
            continue
        seen.add(identifier)
        documents.append(
            RAGDocument(
                identifier,
                title,
                text,
                title in supporting_titles,
                position,
            )
        )
    return tuple(documents)


def normalize_hotpotqa(raw: Mapping[str, Any]) -> RAGExample:
    supporting = _supporting_titles(raw)
    result = RAGExample(
        "HotPotQA",
        str(raw.get("id", raw.get("_id", ""))),
        str(raw["question"]).strip(),
        _answers(raw),
        _context_documents(raw["context"], supporting),
    )
    result.validate()
    return result


def normalize_2wiki(raw: Mapping[str, Any]) -> RAGExample:
    supporting = _supporting_titles(raw)
    result = RAGExample(
        "2WikiMultiHopQA",
        str(raw.get("id", raw.get("_id", ""))),
        str(raw["question"]).strip(),
        _answers(raw),
        _context_documents(raw["context"], supporting),
    )
    result.validate()
    return result


def normalize_musique(raw: Mapping[str, Any]) -> RAGExample:
    paragraphs = raw.get("paragraphs", raw.get("context", ()))
    documents = []
    seen = set()
    for position, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, Mapping):
            raise ValueError("MuSiQue paragraphs must be objects")
        title = str(paragraph.get("title", "")).strip()
        text = _sentences_to_text(
            paragraph.get("paragraph_text", paragraph.get("text", ""))
        )
        identifier = _text_digest(title, text)
        if identifier in seen:
            continue
        seen.add(identifier)
        documents.append(
            RAGDocument(
                identifier,
                title,
                text,
                bool(paragraph.get("is_supporting", paragraph.get("supporting", False))),
                int(paragraph.get("idx", position)),
            )
        )
    result = RAGExample(
        "MuSiQue",
        str(raw.get("id", raw.get("_id", ""))),
        str(raw["question"]).strip(),
        _answers(raw),
        tuple(documents),
    )
    result.validate()
    return result


def normalize_example(dataset: str, raw: Mapping[str, Any]) -> RAGExample:
    key = dataset.lower().replace("_", "").replace("-", "")
    if key in {"hotpot", "hotpotqa"}:
        return normalize_hotpotqa(raw)
    if key in {"2wiki", "2wikimultihopqa", "twowiki"}:
        return normalize_2wiki(raw)
    if key in {"musique"}:
        return normalize_musique(raw)
    raise ValueError("unsupported dataset: %s" % dataset)


def load_raw_records(path: Path) -> List[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        raise ValueError("input dataset is empty")
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if stripped.startswith("["):
                raise
        else:
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, Mapping):
                for key in ("data", "examples", "records"):
                    if isinstance(parsed.get(key), list):
                        return parsed[key]
                return [parsed]
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("invalid JSONL at line %d" % line_number) from error
        if not isinstance(row, Mapping):
            raise ValueError("JSONL rows must be objects")
        rows.append(row)
    return rows


def iter_raw_records(path: Path) -> Iterator[Mapping[str, Any]]:
    """Stream a JSON array or JSONL file without retaining the raw dataset."""
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(4096)
    stripped = prefix.lstrip()
    if not stripped:
        raise ValueError("input dataset is empty")
    if stripped.startswith("["):
        try:
            import ijson
        except ImportError as error:
            raise RuntimeError(
                "ijson is required to stream JSON-array datasets"
            ) from error
        with path.open("rb") as handle:
            for row in ijson.items(handle, "item"):
                if not isinstance(row, Mapping):
                    raise ValueError("JSON array rows must be objects")
                yield row
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "invalid streaming JSONL at line %d" % line_number
                ) from error
            if not isinstance(row, Mapping):
                raise ValueError("JSONL rows must be objects")
            yield row


def segment_text(document: RAGDocument) -> str:
    return "\n[Repeated document]\nTitle: %s\n%s\n" % (
        document.title,
        document.text,
    )


def render_preceding_context(documents: Sequence[RAGDocument]) -> str:
    """Render only chunks causally preceding the repeated segment.

    Cache-Craft places the user question after the retrieved chunks.  The
    question is therefore retained separately on ``ManifestCase`` and must not
    be copied into either the current or historical prefix of C.
    """
    chunks = []
    for index, document in enumerate(documents):
        chunks.append(
            "\n[Context %d]\nTitle: %s\n%s\n"
            % (index + 1, document.title, document.text)
        )
    return "".join(chunks)


def _target_document(example: RAGExample) -> RAGDocument:
    return next(
        (document for document in example.documents if document.supporting),
        example.documents[0],
    )


def build_controlled_cases(
    examples: Sequence[RAGExample],
    encoder: TokenEncoder,
    model_signature: str,
    seed: int = 20260726,
    max_cases: int = 0,
) -> List[ManifestCase]:
    """Build transparent pair-regime controls from real dataset documents."""
    document_pool = {}
    for example in examples:
        for document in example.documents:
            document_pool.setdefault(document.document_id, document)
    # Freeze one deterministic fallback order for the whole dataset.  The old
    # implementation re-sorted the complete document pool for every example,
    # making full-train construction quadratic.  Only the first few documents
    # not already present in an example are needed, so a single global order
    # preserves deterministic sampling while reducing the common case to a
    # short prefix scan.
    supplement_order = sorted(
        document_pool.values(),
        key=lambda document: hashlib.sha256(
            ("%d:supplement:%s" % (seed, document.document_id)).encode("utf-8")
        ).hexdigest(),
    )
    results = []
    for example in examples:
        target = _target_document(example)
        in_example = [
            document
            for document in example.documents
            if document.document_id != target.document_id
        ]
        seen_documents = {target.document_id}
        others = []
        for document in in_example:
            if document.document_id not in seen_documents:
                others.append(document)
                seen_documents.add(document.document_id)
        supplement_count = max(0, 5 - len(others))
        if supplement_count:
            for document in supplement_order:
                if document.document_id in seen_documents:
                    continue
                others.append(document)
                seen_documents.add(document.document_id)
                if len(others) >= 5:
                    break
        if len(others) < 5:
            continue
        in_example_ids = {
            document.document_id for document in in_example
        }
        ordered = sorted(
            others,
            key=lambda document: (
                0 if document.document_id in in_example_ids else 1,
                hashlib.sha256(
                    (
                        "%d:%s:%s"
                        % (seed, example.example_id, document.document_id)
                    ).encode("utf-8")
                ).hexdigest(),
            ),
        )[:5]
        current_documents = tuple(ordered)
        plans = (
            ("high-prefix/same-order", tuple(ordered[:4])),
            ("low-prefix/same-order", tuple(ordered[:1])),
            ("high-prefix/different-order", tuple(reversed(ordered[:3]))),
            ("low-prefix/different-order", tuple(reversed(ordered[:2]))),
        )
        repeated = segment_text(target)
        token_ids = tuple(int(token) for token in encoder(repeated))
        content_hash = token_content_hash(token_ids)
        # Use the exact tokenized segment group for every construction so a
        # controlled and corpus-repeat view of the same C can never diverge
        # into different splits.
        group_id = "%s:%s" % (example.dataset, content_hash)
        current_context = render_preceding_context(current_documents)
        sources = tuple(
            ManifestSource(
                "s%d" % index,
                render_preceding_context(documents),
                "%s:controlled:%d" % (example.example_id, index),
                example.example_id,
                tuple(document.document_id for document in documents),
                regime,
            )
            for index, (regime, documents) in enumerate(plans)
        )
        source_token_lengths = [
            len(tuple(encoder(source.historical_context))) for source in sources
        ]
        current_token_length = len(tuple(encoder(current_context)))
        if len(set(source_token_lengths)) != len(source_token_lengths):
            continue
        if current_token_length in set(source_token_lengths):
            continue
        case = ManifestCase(
            case_id="%s:controlled" % example.example_id,
            dataset=example.dataset,
            document_id=target.document_id,
            group_id=group_id,
            split=deterministic_group_split(group_id, seed),
            regime="mixed-controlled",
            model_signature=model_signature,
            segment_text=repeated,
            segment_token_ids=token_ids,
            content_hash=content_hash,
            current_context=current_context,
            sources=sources,
            question=example.question,
            answers=example.answers,
            construction="controlled_document_order",
            target_document_id=target.document_id,
        )
        case.validate()
        results.append(case)
        if max_cases and len(results) >= max_cases:
            break
    validate_manifest(results) if results else None
    return results


@dataclass(frozen=True)
class RetrievalEvent:
    event_id: str
    example: RAGExample
    target: RAGDocument
    preceding: Tuple[RAGDocument, ...]
    context: str
    token_ids: Tuple[int, ...]
    content_hash: str


def _stable_rank(seed: int, namespace: str, value: str) -> int:
    return int(
        hashlib.sha256(
            ("%d:%s:%s" % (seed, namespace, value)).encode("utf-8")
        ).hexdigest(),
        16,
    )


def build_streaming_pilot_cases(
    example_factory: Callable[[], Iterable[RAGExample]],
    encoder: TokenEncoder,
    model_signature: str,
    seed: int = 20260726,
    max_controlled_cases: int = 250,
    max_corpus_repeat_cases: int = 250,
) -> Tuple[List[ManifestCase], List[RAGExample], Dict[str, int]]:
    """Construct bounded pilot candidates while scanning the full dataset.

    The first pass counts exact documents and retains a deterministic bounded
    pool for controlled construction.  A second pass keeps only the five
    lowest-rank distinct contexts for document ids observed at least five
    times.  Raw examples are never accumulated in memory.
    """
    if max_controlled_cases < 0 or max_corpus_repeat_cases < 0:
        raise ValueError("streaming case limits must be non-negative")
    controlled_pool_limit = max(
        64,
        max_controlled_cases,
        max_controlled_cases * 4,
    )
    controlled_heap = []
    document_counts: Dict[str, int] = {}
    examples_scanned = 0
    documents_scanned = 0
    for ordinal, example in enumerate(example_factory()):
        examples_scanned += 1
        score = _stable_rank(seed, "controlled-example", example.example_id)
        entry = (-score, example.example_id, ordinal, example)
        if len(controlled_heap) < controlled_pool_limit:
            heapq.heappush(controlled_heap, entry)
        elif score < -controlled_heap[0][0]:
            heapq.heapreplace(controlled_heap, entry)
        for document in example.documents:
            documents_scanned += 1
            current = document_counts.get(document.document_id, 0)
            if current < 5:
                document_counts[document.document_id] = current + 1

    sampled_examples = [
        entry[3]
        for entry in sorted(
            controlled_heap,
            key=lambda entry: (-entry[0], entry[1], entry[2]),
        )
    ]
    controlled = build_controlled_cases(
        sampled_examples,
        encoder,
        model_signature,
        seed=seed,
        max_cases=max_controlled_cases,
    )
    repeated_ids = [
        document_id
        for document_id, count in document_counts.items()
        if count >= 5
    ]
    repeat_pool_limit = max(
        max_corpus_repeat_cases,
        max_corpus_repeat_cases * 4,
    )
    selected_repeat_ids = set(
        sorted(
            repeated_ids,
            key=lambda document_id: (
                _stable_rank(seed, "repeat-document", document_id),
                document_id,
            ),
        )[:repeat_pool_limit]
    )
    del document_counts

    event_buckets: Dict[
        str, Dict[str, Tuple[int, RetrievalEvent]]
    ] = {document_id: {} for document_id in selected_repeat_ids}
    segment_cache: Dict[str, Tuple[str, Tuple[int, ...], str]] = {}
    for example in example_factory():
        for position, target in enumerate(example.documents):
            if target.document_id not in selected_repeat_ids:
                continue
            preceding = tuple(
                example.documents[max(0, position - 5) : position]
            )
            context = render_preceding_context(preceding)
            event_id = "%s:%s" % (example.example_id, target.document_id)
            cached_segment = segment_cache.get(target.document_id)
            if cached_segment is None:
                repeated = segment_text(target)
                token_ids = tuple(int(token) for token in encoder(repeated))
                cached_segment = (
                    repeated,
                    token_ids,
                    token_content_hash(token_ids),
                )
                segment_cache[target.document_id] = cached_segment
            _, token_ids, content_hash = cached_segment
            event = RetrievalEvent(
                event_id,
                example,
                target,
                preceding,
                context,
                token_ids,
                content_hash,
            )
            rank = _stable_rank(seed, "repeat-event", event_id)
            bucket = event_buckets[target.document_id]
            existing = bucket.get(context)
            if existing is None or rank < existing[0]:
                bucket[context] = (rank, event)
            if len(bucket) > 5:
                worst_context = max(
                    bucket, key=lambda value: (bucket[value][0], value)
                )
                del bucket[worst_context]

    corpus_candidates = []
    for document_id, bucket in event_buckets.items():
        if len(bucket) < 5:
            continue
        distinct = [
            item[1]
            for item in sorted(
                bucket.values(),
                key=lambda item: (item[0], item[1].event_id),
            )
        ]
        historical = distinct[:4]
        current = distinct[4]
        content_hash = current.content_hash
        group_id = "%s:%s" % (current.example.dataset, content_hash)
        sources = tuple(
            ManifestSource(
                "s%d" % index,
                event.context,
                event.event_id,
                event.example.example_id,
                tuple(
                    document.document_id
                    for document in event.preceding
                ),
                "corpus-repeat",
            )
            for index, event in enumerate(historical)
        )
        case = ManifestCase(
            case_id="%s:corpus-repeat" % current.event_id,
            dataset=current.example.dataset,
            document_id=document_id,
            group_id=group_id,
            split=deterministic_group_split(group_id, seed),
            regime="corpus-repeat",
            model_signature=model_signature,
            segment_text=segment_text(current.target),
            segment_token_ids=current.token_ids,
            content_hash=content_hash,
            current_context=current.context,
            sources=sources,
            question=current.example.question,
            answers=current.example.answers,
            construction="corpus_repeat_pseudotime",
            target_document_id=document_id,
        )
        case.validate()
        corpus_candidates.append(case)
    corpus = sorted(
        corpus_candidates,
        key=lambda case: (case.content_hash, case.case_id),
    )[:max_corpus_repeat_cases]
    cases = controlled + corpus
    validate_manifest(cases) if cases else None
    return (
        cases,
        sampled_examples,
        {
            "normalized_examples_scanned": examples_scanned,
            "documents_scanned": documents_scanned,
            "controlled_pool_examples": len(sampled_examples),
            "repeat_document_ids_at_least_five": len(repeated_ids),
            "repeat_document_ids_examined": len(selected_repeat_ids),
        },
    )


def build_corpus_repeat_cases(
    examples: Sequence[RAGExample],
    encoder: TokenEncoder,
    model_signature: str,
    seed: int = 20260726,
    max_preceding_documents: int = 5,
    max_cases: int = 0,
) -> List[ManifestCase]:
    """Build corpus-derived repeats without claiming a production time trace.

    Stable hash order is a pseudo-time used only to choose four historical
    occurrences and one current occurrence of the exact same tokenized segment.
    """
    groups: Dict[str, List[RetrievalEvent]] = {}
    segment_cache: Dict[str, Tuple[str, Tuple[int, ...], str]] = {}
    for example in examples:
        for position, target in enumerate(example.documents):
            cached_segment = segment_cache.get(target.document_id)
            if cached_segment is None:
                repeated = segment_text(target)
                token_ids = tuple(int(token) for token in encoder(repeated))
                content_hash = token_content_hash(token_ids)
                cached_segment = (repeated, token_ids, content_hash)
                segment_cache[target.document_id] = cached_segment
            repeated, token_ids, content_hash = cached_segment
            preceding = tuple(example.documents[max(0, position - max_preceding_documents) : position])
            context = render_preceding_context(preceding)
            event = RetrievalEvent(
                "%s:%s" % (example.example_id, target.document_id),
                example,
                target,
                preceding,
                context,
                token_ids,
                content_hash,
            )
            groups.setdefault(content_hash, []).append(event)

    cases = []
    for content_hash, events in sorted(groups.items()):
        ordered = sorted(
            events,
            key=lambda event: hashlib.sha256(
                ("%d:%s" % (seed, event.event_id)).encode("utf-8")
            ).hexdigest(),
        )
        distinct = []
        seen_contexts = set()
        for event in ordered:
            if event.context in seen_contexts:
                continue
            seen_contexts.add(event.context)
            distinct.append(event)
        if len(distinct) < 5:
            continue
        historical = distinct[:4]
        current = distinct[4]
        group_id = "%s:%s" % (current.example.dataset, content_hash)
        sources = tuple(
            ManifestSource(
                "s%d" % index,
                event.context,
                event.event_id,
                event.example.example_id,
                tuple(document.document_id for document in event.preceding),
                "corpus-repeat",
            )
            for index, event in enumerate(historical)
        )
        case = ManifestCase(
            case_id="%s:corpus-repeat" % current.event_id,
            dataset=current.example.dataset,
            document_id=current.target.document_id,
            group_id=group_id,
            split=deterministic_group_split(group_id, seed),
            regime="corpus-repeat",
            model_signature=model_signature,
            segment_text=segment_text(current.target),
            segment_token_ids=current.token_ids,
            content_hash=content_hash,
            current_context=current.context,
            sources=sources,
            question=current.example.question,
            answers=current.example.answers,
            construction="corpus_repeat_pseudotime",
            target_document_id=current.target.document_id,
        )
        case.validate()
        cases.append(case)
        if max_cases and len(cases) >= max_cases:
            break
    validate_manifest(cases) if cases else None
    return cases


def construction_audit(
    examples: Sequence[RAGExample], cases: Sequence[ManifestCase]
) -> Dict[str, Any]:
    source_regimes: Dict[str, int] = {}
    for case in cases:
        for source in case.sources:
            source_regimes[source.regime] = source_regimes.get(source.regime, 0) + 1
    return {
        "normalized_examples": len(examples),
        "cases": len(cases),
        "split_counts": {
            split: sum(case.split == split for case in cases)
            for split in ("train", "calibration", "test")
        },
        "construction_counts": {
            construction: sum(case.construction == construction for case in cases)
            for construction in sorted({case.construction for case in cases})
        },
        "source_regime_counts": source_regimes,
        "unique_content_hashes": len({case.content_hash for case in cases}),
        "paper_evidence": False,
    }
