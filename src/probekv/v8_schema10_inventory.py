"""Canonical identity is independent of native Prefix execution ownership."""
from __future__ import annotations

from dataclasses import dataclass


def mandatory_suffix_positions(request):
    """Native sampling must retain the real final prompt row, not last active C."""
    n = len(request["token_ids"])
    suffix = tuple(request.get("mandatory_suffix_positions", (n - 1,)))
    if (not n or not suffix or suffix[0] < 0
            or suffix != tuple(range(suffix[0], n))):
        raise ValueError("mandatory suffix must be a nonempty terminal span")
    if any(set(s["positions"]).intersection(suffix) for s in request["segments"]):
        raise ValueError("canonical reuse Segment cannot include mandatory suffix")
    return suffix


@dataclass(frozen=True)
class SegmentOwnership:
    segment_id: str
    canonical_positions: tuple[int, ...]
    prefix_positions: tuple[int, ...]
    remaining_positions: tuple[int, ...]
    disposition: str

    @property
    def comparison_eligible(self):
        return self.disposition == "NONPREFIX_CANDIDATE"


def native_segment_inventory(segments, *, prompt_tokens, cached_prefix_tokens):
    if not 0 <= cached_prefix_tokens <= prompt_tokens:
        raise ValueError("invalid native Prefix token count")
    occupied, inventory = set(), {}
    for sid, segment in segments.items():
        positions = tuple(segment["positions"])
        if (not positions or tuple(sorted(set(positions))) != positions
                or len(positions) != len(segment["token_ids"])
                or any(p < 0 or p >= prompt_tokens for p in positions)
                or occupied.intersection(positions)):
            raise ValueError("invalid or overlapping canonical Segment positions")
        occupied.update(positions)
        prefix = tuple(p for p in positions if p < cached_prefix_tokens)
        rest = tuple(p for p in positions if p >= cached_prefix_tokens)
        disposition = "PREFIX_EXACT" if not rest else "DENSE_PREFIX_TAIL" if prefix else "NONPREFIX_CANDIDATE"
        inventory[sid] = SegmentOwnership(sid, positions, prefix, rest, disposition)
    return inventory
