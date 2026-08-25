from __future__ import annotations

import itertools
from typing import Dict, Iterable, Iterator, Mapping, Sequence


def cartesian_rows(dimensions: Mapping[str, Sequence[object]]) -> Iterator[Dict[str, object]]:
    names = tuple(dimensions)
    values = [tuple(dimensions[name]) for name in names]
    if any(not dimension for dimension in values):
        raise ValueError("matrix dimensions must be non-empty")
    for combination in itertools.product(*values):
        yield dict(zip(names, combination))


def main_rag_matrix() -> Iterator[Dict[str, object]]:
    return cartesian_rows(
        {
            "model": (
                "Qwen/Qwen2.5-7B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
            ),
            "dataset": ("MuSiQue", "2WikiMultiHopQA", "HotPotQA"),
            "k": (1, 2, 4),
            "gamma": (0.6, 0.7, 0.8, 0.9),
            "concurrency": (1, 2, 4, 8, 16),
        }
    )


def profile_matrix(include_ssd: bool = False) -> Iterator[Dict[str, object]]:
    tiers = ("gpu", "pinned_cpu", "ssd") if include_ssd else ("gpu", "pinned_cpu")
    return cartesian_rows(
        {
            "segment_tokens": (256, 512, 1024, 2048, 4096, 8192),
            "concurrency": (1, 2, 4, 8, 16),
            "repair_ratio": (0.0, 0.05, 0.10, 0.16, 0.20, 0.30, 0.50, 0.75, 1.0),
            "tier": tiers,
            "k": (1, 2, 4),
        }
    )
