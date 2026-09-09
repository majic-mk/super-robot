"""Immutable, diagnostic-only contract for matched resident repair backends.

This is not an online admission token. Native Prefix and continuation handoff
remain unsupported by the normal-loop adapter until independently qualified.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .v8_schema10_execution import digest_json


@dataclass(frozen=True)
class ResidentRepairPlan:
    source_id: str
    source_digest: str
    request_tokens_sha256: str
    boundary: int
    ratio: float
    segment_positions: tuple[int, ...]
    repair_positions: tuple[int, ...]
    prompt_tokens: int
    suffix_tokens: int
    cached_prefix_tokens: int = 0

    def __post_init__(self):
        if (not self.source_id or not self.source_digest or not self.request_tokens_sha256
                or self.cached_prefix_tokens != 0 or self.boundary < 2
                or not math.isfinite(self.ratio) or not 0 < self.ratio <= 1):
            raise ValueError("invalid resident repair binding or unsupported Prefix")
        p, r = self.segment_positions, self.repair_positions
        if not isinstance(p, tuple) or not isinstance(r, tuple):
            raise ValueError("repair plan positions must be immutable tuples")
        if (not p or p != tuple(range(p[0], p[-1] + 1)) or p[0] < 0
                or not 0 < self.suffix_tokens < self.prompt_tokens
                or p[-1] >= self.prompt_tokens - self.suffix_tokens
                or r != tuple(sorted(set(r))) or not set(r) <= set(p)
                or len(r) != min(len(p), math.ceil(len(p) * self.ratio))):
            raise ValueError("invalid repair support/mandatory dense ownership")

    @property
    def mask_digest(self):
        return digest_json(self.repair_positions)

    @property
    def active_positions(self):
        return tuple(sorted((set(range(self.prompt_tokens)) - set(self.segment_positions))
                            | set(self.repair_positions)))

    def assert_binding(self, *, source_id, token_ids, positions, boundary, ratio,
                       cached_prefix_tokens):
        if (source_id != self.source_id or digest_json(token_ids) != self.request_tokens_sha256
                or tuple(positions) != self.segment_positions or boundary != self.boundary
                or ratio != self.ratio or cached_prefix_tokens != self.cached_prefix_tokens):
            raise ValueError("stale or mismatched resident repair plan")

    def assert_execution(self, repair_positions, active_positions):
        if (tuple(repair_positions) != self.repair_positions
                or tuple(active_positions) != self.active_positions):
            raise RuntimeError("repair backend changed frozen mask or dense ownership")
