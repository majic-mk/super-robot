from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple


class SemanticBoundary(str, Enum):
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    STRUCTURAL = "structural"
    TOKEN = "token"


@dataclass(frozen=True)
class CanonicalizerConfig:
    version: str = "semantic_block_v1"
    target_tokens: int = 512
    min_tokens: int = 128
    max_tokens: int = 640
    alignment_quantum: int = 16
    search_window_tokens: int = 64
    alignment_policy: str = "soft"
    tail_policy: str = "semantic_rebalance"
    padding: bool = False

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("canonicalizer version is required")
        if not 0 < self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("invalid canonical token bounds")
        if self.alignment_quantum <= 0 or self.search_window_tokens < 0:
            raise ValueError("invalid canonical alignment parameters")
        if self.alignment_policy != "soft":
            raise ValueError("v7 requires soft alignment")
        if self.tail_policy != "semantic_rebalance":
            raise ValueError("v7 requires semantic tail rebalancing")
        if self.padding:
            raise ValueError("canonical Segments must not be padded")

    def signature(self, tokenizer_signature: str) -> str:
        if not tokenizer_signature:
            raise ValueError("tokenizer signature is required")
        payload = {**asdict(self), "tokenizer_signature": tokenizer_signature}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "canonicalizer_v1:%s" % hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class CanonicalSegment:
    ordinal: int
    token_start: int
    token_end: int
    token_ids: Tuple[int, ...]
    boundary_kind: SemanticBoundary
    canonicalizer_signature: str
    document_revision: str

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    def reuse_content_key(
        self, model_math_signature: str, tokenizer_signature: str
    ) -> str:
        """Exact mathematical identity; chunker provenance is deliberately absent."""
        payload = {
            "domain": "probekv-v7-reuse-content",
            "model_math_signature": model_math_signature,
            "tokenizer_signature": tokenizer_signature,
            "token_count": self.token_count,
            "token_ids": list(self.token_ids),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_retrieved_text(value: str) -> str:
    """Deterministic provenance normalization without semantic rewriting."""
    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _candidate_score(
    *,
    length: int,
    remaining: int,
    kind: SemanticBoundary,
    config: CanonicalizerConfig,
) -> Tuple[float, int]:
    semantic = {
        SemanticBoundary.PARAGRAPH: 4.0,
        SemanticBoundary.STRUCTURAL: 3.5,
        SemanticBoundary.SENTENCE: 3.0,
        SemanticBoundary.TOKEN: 0.0,
    }[kind]
    length_penalty = abs(length - config.target_tokens) / config.target_tokens
    waste = (
        config.alignment_quantum - (length % config.alignment_quantum)
    ) % config.alignment_quantum
    alignment_penalty = waste / config.alignment_quantum
    tail_penalty = 0.0
    if 0 < remaining < config.min_tokens:
        tail_penalty = 5.0 * (config.min_tokens - remaining) / config.min_tokens
    score = semantic - 1.5 * length_penalty - 0.35 * alignment_penalty - tail_penalty
    # Stable tie-break: prefer the endpoint closer to target, then the longer one.
    tie = -abs(length - config.target_tokens) * 100000 + length
    return score, tie


def canonicalize_token_ids(
    token_ids: Sequence[int],
    *,
    tokenizer_signature: str,
    document_revision: str,
    semantic_boundaries: Optional[Mapping[int, SemanticBoundary]] = None,
    config: CanonicalizerConfig = CanonicalizerConfig(),
) -> Tuple[CanonicalSegment, ...]:
    """Partition exact tokens using deterministic semantic-soft alignment.

    Boundary keys are absolute exclusive token endpoints. The final endpoint is
    always legal. The function never pads or drops token content.
    """
    if not tokenizer_signature or not document_revision:
        raise ValueError("tokenizer signature and document revision are required")
    tokens = tuple(int(token) for token in token_ids)
    if any(token < 0 for token in tokens):
        raise ValueError("token IDs must be non-negative")
    if not tokens:
        return ()
    boundaries = {
        int(position): SemanticBoundary(kind)
        for position, kind in (semantic_boundaries or {}).items()
        if 0 < int(position) <= len(tokens)
    }
    signature = config.signature(tokenizer_signature)
    result = []
    start = 0
    ordinal = 0
    while start < len(tokens):
        remaining_total = len(tokens) - start
        if remaining_total <= config.max_tokens:
            end = len(tokens)
            kind = boundaries.get(end, SemanticBoundary.TOKEN)
        else:
            low = start + config.min_tokens
            high = min(start + config.max_tokens, len(tokens))
            target = start + config.target_tokens
            window_low = max(low, target - config.search_window_tokens)
            window_high = min(high, target + config.search_window_tokens)
            candidates = set(range(window_low, window_high + 1))
            candidates.update(
                position for position in boundaries if low <= position <= high
            )
            # Alignment is a candidate source, never a hard boundary.
            first_aligned = (
                (window_low + config.alignment_quantum - 1)
                // config.alignment_quantum
            ) * config.alignment_quantum
            candidates.update(
                range(first_aligned, window_high + 1, config.alignment_quantum)
            )
            end = max(
                candidates,
                key=lambda position: _candidate_score(
                    length=position - start,
                    remaining=len(tokens) - position,
                    kind=boundaries.get(position, SemanticBoundary.TOKEN),
                    config=config,
                ),
            )
            kind = boundaries.get(end, SemanticBoundary.TOKEN)
        segment_tokens = tokens[start:end]
        result.append(
            CanonicalSegment(
                ordinal=ordinal,
                token_start=start,
                token_end=end,
                token_ids=segment_tokens,
                boundary_kind=kind,
                canonicalizer_signature=signature,
                document_revision=document_revision,
            )
        )
        start = end
        ordinal += 1
    if tuple(token for segment in result for token in segment.token_ids) != tokens:
        raise RuntimeError("canonicalizer changed token content")
    return tuple(result)
