from __future__ import annotations

import collections
import re
import string
from typing import Iterable, Sequence


def normalize_answer(text: str) -> str:
    """HotpotQA-style answer normalization shared by all three pilot datasets."""

    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def token_f1_text(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    common = collections.Counter(predicted) & collections.Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / float(len(predicted))
    recall = overlap / float(len(expected))
    return 2.0 * precision * recall / (precision + recall)


def best_answer_f1(prediction: str, answers: Iterable[str]) -> float:
    values = [token_f1_text(prediction, answer) for answer in answers]
    return max(values) if values else 0.0


def token_id_f1(prediction: Sequence[int], reference: Sequence[int]) -> float:
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0
    common = collections.Counter(prediction) & collections.Counter(reference)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / float(len(prediction))
    recall = overlap / float(len(reference))
    return 2.0 * precision * recall / (precision + recall)
