"""Snapshot-validated cache for repeat-request Source decisions.

The cache deliberately stores only selection evidence.  It never bypasses
FinalCommit, replica leasing, or cost support checks.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class SelectionCacheKey:
    token_digest: str
    model_signature: str
    tokenizer_hash: str
    prefix_digest: str
    dispatch: str
    pool_generation: int
    source_state_digest: str

@dataclass(frozen=True)
class SelectionCacheEntry:
    key: SelectionCacheKey
    selected_source_variant_id: Optional[str]
    completed_depth: int
    residual_score: Optional[float]
    evidence_digest: str

class SelectionResultCache:
    """Small in-memory cache; callers may persist entries by digest later."""
    def __init__(self) -> None:
        self._entries: Dict[SelectionCacheKey, SelectionCacheEntry] = {}

    def put(self, entry: SelectionCacheEntry) -> None:
        if entry.key is None or entry.completed_depth < 1:
            raise ValueError("invalid selection cache entry")
        self._entries[entry.key] = entry

    def get(self, key: SelectionCacheKey) -> Optional[SelectionCacheEntry]:
        return self._entries.get(key)

    def invalidate_pool(self, pool_generation: int) -> None:
        self._entries = {k: v for k, v in self._entries.items()
                         if k.pool_generation == pool_generation}

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def key_from_request(*, token_digest: str, model_signature: str,
                         tokenizer_hash: str, prefix_digest: str,
                         dispatch: str, pool_generation: int,
                         source_state_digest: str) -> SelectionCacheKey:
        """Canonical key constructor used by online adapters."""
        return SelectionCacheKey(token_digest, model_signature, tokenizer_hash,
                                 prefix_digest, dispatch, int(pool_generation),
                                 source_state_digest)
