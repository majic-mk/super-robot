from __future__ import annotations

from typing import Sequence, Tuple


def absolute_causal_rows(
    query_positions: Sequence[int], key_count: int
) -> Tuple[Tuple[bool, ...], ...]:
    """Return the causal rows for arbitrary absolute query positions.

    ``True`` means that the query may attend to the key.  This CPU reference is
    intentionally independent of xFormers and is used to verify the geometry
    of the CacheBlend patch for non-contiguous repaired tokens.
    """
    if key_count <= 0:
        raise ValueError("key_count must be positive")
    positions = tuple(int(position) for position in query_positions)
    if len(positions) != len(set(positions)):
        raise ValueError("query positions must be unique")
    if any(position < 0 or position >= key_count for position in positions):
        raise ValueError("query position is outside the KV sequence")
    return tuple(
        tuple(key_position <= query_position for key_position in range(key_count))
        for query_position in positions
    )


def bottom_right_causal_rows(
    query_count: int, key_count: int
) -> Tuple[Tuple[bool, ...], ...]:
    """Reference the bottom-right triangular semantics used by the old patch."""
    if query_count < 0 or key_count <= 0 or query_count > key_count:
        raise ValueError("invalid bottom-right mask dimensions")
    offset = key_count - query_count
    return absolute_causal_rows(
        tuple(offset + index for index in range(query_count)),
        key_count,
    )
