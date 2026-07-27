from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence


SPLITS = ("train", "calibration", "test")


def deterministic_group_split(
    group_id: str,
    seed: int,
    train_fraction: float = 0.50,
    calibration_fraction: float = 0.20,
) -> str:
    if train_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("test fraction must be positive")
    digest = hashlib.sha256(("%d:%s" % (seed, group_id)).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2 ** 64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + calibration_fraction:
        return "calibration"
    return "test"


def assert_group_isolation(group_to_splits: Mapping[str, Sequence[str]]) -> None:
    leaked = {
        group: sorted(set(splits))
        for group, splits in group_to_splits.items()
        if len(set(splits)) > 1
    }
    if leaked:
        raise ValueError("group leakage detected: %r" % leaked)


def assert_locked_test(
    threshold_fit_splits: Iterable[str], test_label_accessed: bool
) -> None:
    fit_splits = set(threshold_fit_splits)
    if "test" in fit_splits or test_label_accessed:
        raise ValueError("locked test labels must remain invisible until final evaluation")
    if not fit_splits.issubset({"train", "calibration", "pilot"}):
        raise ValueError("unknown fit split")
