"""Deterministic, offset-aware development chunker; legacy v1 stays unchanged.

Compute at ingestion/manifest preparation and persist endpoints. Never choose a
different chunker because the same document happened to arrive under load.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
import re

from .canonical_segment import CanonicalSegment, SemanticBoundary
from .v8_schema10_execution import digest_json


@dataclass(frozen=True)
class SemanticWindowConfig:
    target_tokens: int = 512
    search_window_tokens: int = 100
    min_tail_tokens: int = 128
    alignment_quantum: int = 16
    policy: str = "paragraph_sentence_clause_v2"

    def __post_init__(self):
        if (type(self.target_tokens) is not int or self.target_tokens < 2
                or type(self.search_window_tokens) is not int
                or not 0 <= self.search_window_tokens < self.target_tokens
                or type(self.min_tail_tokens) is not int
                or not 1 <= self.min_tail_tokens <= self.target_tokens
                or type(self.alignment_quantum) is not int or self.alignment_quantum < 1
                or self.policy not in {"paragraph_sentence_clause_v2", "fixed_tokens_v1"}):
            raise ValueError("invalid frozen semantic-window config")

    def signature(self, tokenizer_signature):
        if not tokenizer_signature:
            raise ValueError("tokenizer signature is required")
        return digest_json({"chunker": asdict(self), "tokenizer_signature": tokenizer_signature,
                            "offset_contract": "whole_document_character_offsets_v1"})


def punctuation_token_boundaries(text, offsets):
    """Map punctuation to exact token endpoints without decoding each prefix.

    Rules are deterministic punctuation heuristics, not a linguistic guarantee.
    Decimal points/common English abbreviations are not sentence boundaries.
    Never cut inside a tokenizer span (including overlapping byte-token spans).
    """
    spans = tuple(tuple(pair) for pair in offsets)
    previous_start = previous_end = 0
    for pair in spans:
        if (len(pair) != 2 or any(type(v) is not int for v in pair)
                or not 0 <= pair[0] < pair[1] <= len(text)
                or pair[0] < previous_start or pair[1] < previous_end):
            raise ValueError("valid whole-text offsets without special/empty tokens required")
        previous_start, previous_end = pair
    ends = [e for _, e in spans]
    boundaries = {}
    priority = {SemanticBoundary.PARAGRAPH: 3, SemanticBoundary.SENTENCE: 2,
                SemanticBoundary.CLAUSE: 1}
    def publish(char_end, kind):
        index = bisect_left(ends, char_end)
        if index == len(spans):
            return
        token_end = ends[index]
        # A token may include closing punctuation/whitespace, but not the next word.
        if text[char_end:token_end].strip(' \t\r\n\"\'”’)]}」』'):
            return
        index = bisect_right(ends, token_end)
        if index < len(spans) and spans[index][0] < token_end:
            return
        if index not in boundaries or priority[kind] > priority[boundaries[index]]:
            boundaries[index] = kind
    for match in re.finditer(r'(?:\r?\n[ \t]*){2,}', text):
        # Whitespace gaps need not belong to any token. Cut after the preceding
        # token only if the next token begins beyond the paragraph separator.
        left = bisect_right(ends, match.start())
        if left and (left == len(spans) or spans[left][0] >= match.end()):
            boundaries[left] = SemanticBoundary.PARAGRAPH
        else:
            publish(match.end(), SemanticBoundary.PARAGRAPH)
    closing = re.escape('\"\'”’)]}」』')
    punctuation_pattern = r'[。！？!?]+[' + closing + r']*|\.[' + closing + r']*|[，,；;]'
    for match in re.finditer(punctuation_pattern, text):
        punctuation = match.group()[0]
        if punctuation == '.':
            prefix = text[max(0, match.start() - 24):match.start()]
            last = re.search(r'([\w.]+)$', prefix)
            word = last.group(1).lower() if last else ''
            if (word in {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs', 'etc', 'e.g', 'i.e'}
                    or len(word) == 1 and word.isalpha()
                    or match.end() < len(text) and not text[match.end()].isspace()
                    or match.start() and text[match.start() - 1] == '.'):
                continue
        kind = SemanticBoundary.CLAUSE if punctuation in ',，;；' else SemanticBoundary.SENTENCE
        publish(match.end(), kind)
    return boundaries


def segment_document_tokens(token_ids, *, text, offsets, tokenizer_signature,
                            document_revision, config=SemanticWindowConfig()):
    """Return a canonical partition and explicit forced-cut audit.

    If no paragraph/sentence/clause exists in the window, hard token fallback
    is recorded. Finite length bounds and never splitting a long sentence
    cannot both be guaranteed. No padding, runtime timing switch, or retokenize.
    """
    tokens = tuple(token_ids)
    if (not document_revision or any(type(t) is not int or t < 0 for t in tokens)
            or not isinstance(text, str)):
        raise ValueError("exact integer tokens and document provenance required")
    signature = config.signature(tokenizer_signature)
    if config.policy == "paragraph_sentence_clause_v2":
        if offsets is None or len(offsets) != len(tokens):
            raise ValueError("semantic mode requires tokenizer offsets for every exact token")
        boundaries = punctuation_token_boundaries(text, offsets)
    else:
        boundaries = {}
    result, cuts, start = [], [], 0
    kinds = (SemanticBoundary.PARAGRAPH, SemanticBoundary.SENTENCE, SemanticBoundary.CLAUSE)
    # Build once: no whole-document scan for every output chunk.
    positions_by_kind = {kind: sorted(p for p, k in boundaries.items() if k == kind) for kind in kinds}
    while start < len(tokens):
        target = start + config.target_tokens
        high = min(len(tokens), target + config.search_window_tokens)
        kind, reason = SemanticBoundary.TOKEN, "document_end"
        if config.policy == "fixed_tokens_v1":
            end = min(target, len(tokens))
            reason = "fixed_token_policy" if end < len(tokens) else reason
        elif len(tokens) <= high:
            end = len(tokens)
        else:
            low = max(start + 1, target - config.search_window_tokens)
            end = None
            for candidate_kind in kinds:
                points = positions_by_kind[candidate_kind]
                candidates = points[bisect_left(points, low):bisect_right(points, high)]
                # Prefer avoiding a tiny final tail when the same semantic class allows it.
                balanced = [p for p in candidates if len(tokens) - p >= config.min_tail_tokens]
                candidates = balanced or candidates
                if candidates:
                    end = min(candidates, key=lambda p: (abs(p - target),
                        (-(p - start)) % config.alignment_quantum, -p))
                    kind, reason = candidate_kind, candidate_kind.value
                    break
            if end is None:
                end, reason = target, "no_punctuation_in_window_forced_token_cut"
        result.append(CanonicalSegment(len(result), start, end, tokens[start:end], kind,
                                       signature, document_revision))
        cuts.append({"token_end": end, "token_count": end - start, "reason": reason,
                     "sentence_integrity_guaranteed": False,
                     "forced_token_cut": reason in {"fixed_token_policy", "no_punctuation_in_window_forced_token_cut"}})
        start = end
    audit = {"policy": config.policy, "config": asdict(config), "canonicalizer_signature": signature,
             "document_revision": document_revision, "tokenizer_signature": tokenizer_signature,
             "text_digest": digest_json(text), "token_digest": digest_json(tokens),
             "offset_digest": digest_json(offsets), "cuts": cuts,
             "runtime_timing_switch_allowed": False, "padding": False,
             "gpu_runtime_qualified": False, "paper_evidence": False}
    audit["manifest_digest"] = digest_json(audit)
    return tuple(result), audit
