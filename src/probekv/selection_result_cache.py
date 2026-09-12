"""Snapshot-validated cache for repeat-request Source decisions.

The cache deliberately stores only selection evidence.  It never bypasses
FinalCommit, replica leasing, or cost support checks.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional
import hashlib
import json

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
    candidate_set_digest: str = ""
    gate1_evidence_digest: str = ""
    repair_support_digest: str = ""
    planner_snapshot_digest: str = ""

    def admissible_for(self, *, candidate_set_digest: str,
                       gate1_evidence_digest: str,
                       repair_support_digest: str,
                       planner_snapshot_digest: str) -> bool:
        """Check all non-cost evidence before using a cached winner."""
        return (self.candidate_set_digest == candidate_set_digest and
                self.gate1_evidence_digest == gate1_evidence_digest and
                self.repair_support_digest == repair_support_digest and
                self.planner_snapshot_digest == planner_snapshot_digest)

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

    @staticmethod
    def digest_key(key: SelectionCacheKey) -> str:
        payload = json.dumps(key.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
