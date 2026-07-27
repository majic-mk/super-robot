from __future__ import annotations

from typing import Dict, Iterable, List

from .contracts import HistoricalSource


class SourceStore:
    """Read-only canonical source registry.

    A repeated segment C may be conditioned by A, B, or E, but S1/S2/S3 must
    each be produced by their own full prefill. Reusing S1 to construct S2
    would carry A's influence and violates this registry's admission rule.
    """

    def __init__(self, online_kmax: int = 4) -> None:
        if online_kmax < 1:
            raise ValueError("online_kmax must be positive")
        self.online_kmax = online_kmax
        self._by_hash: Dict[str, Dict[str, HistoricalSource]] = {}

    def register(self, source: HistoricalSource) -> None:
        source.validate_canonical()
        bucket = self._by_hash.setdefault(source.content_hash, {})
        if source.source_id in bucket:
            if bucket[source.source_id] != source:
                raise ValueError("source_id collision with different metadata")
            return
        if len(bucket) >= self.online_kmax:
            raise ValueError("online Kmax=%d exceeded" % self.online_kmax)
        bucket[source.source_id] = source

    def register_many(self, sources: Iterable[HistoricalSource]) -> None:
        for source in sources:
            self.register(source)

    def candidates(self, content_hash: str) -> List[HistoricalSource]:
        return list(self._by_hash.get(content_hash, {}).values())

    def get(self, content_hash: str, source_id: str) -> HistoricalSource:
        return self._by_hash[content_hash][source_id]
