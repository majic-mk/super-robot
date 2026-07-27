from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def grouped_paired_bootstrap(
    differences_by_group: Mapping[str, Sequence[float]],
    iterations: int = 10_000,
    seed: int = 20_260_726,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    groups = sorted(differences_by_group)
    if not groups or any(not differences_by_group[group] for group in groups):
        raise ValueError("every group must contain observations")
    flattened = [
        float(value)
        for group in groups
        for value in differences_by_group[group]
    ]
    estimate = statistics.mean(flattened)
    randomizer = random.Random(seed)
    samples: List[float] = []
    for _ in range(iterations):
        sampled_groups = [randomizer.choice(groups) for _ in groups]
        sample_values: List[float] = []
        for group in sampled_groups:
            values = differences_by_group[group]
            sample_values.extend(
                randomizer.choice(values) for _ in range(len(values))
            )
        samples.append(statistics.mean(sample_values))
    alpha = 1.0 - confidence
    return ConfidenceInterval(
        estimate,
        percentile(samples, alpha / 2.0),
        percentile(samples, 1.0 - alpha / 2.0),
    )


def paired_hodges_lehmann(differences: Sequence[float]) -> float:
    if not differences:
        raise ValueError("differences must not be empty")
    values = [float(value) for value in differences]
    walsh = [
        (values[left] + values[right]) / 2.0
        for left in range(len(values))
        for right in range(left, len(values))
    ]
    return statistics.median(walsh)


def holm_bonferroni(p_values: Mapping[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    decisions: Dict[str, bool] = {name: False for name in p_values}
    still_rejecting = True
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        threshold = alpha / (total - index)
        if still_rejecting and value <= threshold:
            decisions[name] = True
        else:
            still_rejecting = False
    return decisions


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired rankings need at least two values")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.mean(left_ranks)
    right_mean = statistics.mean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left_ranks))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right_ranks))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def kendall_tau(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired rankings need at least two values")
    concordant = 0
    discordant = 0
    tied_left = 0
    tied_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            delta_left = left[first] - left[second]
            delta_right = right[first] - right[second]
            if delta_left == 0 and delta_right == 0:
                tied_left += 1
                tied_right += 1
            elif delta_left == 0:
                tied_left += 1
            elif delta_right == 0:
                tied_right += 1
            elif delta_left * delta_right > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_left)
        * (concordant + discordant + tied_right)
    )
    return (concordant - discordant) / denominator if denominator else 0.0


def _binomial_cdf(observed: int, trials: int, probability: float) -> float:
    return sum(
        math.comb(trials, index)
        * (probability ** index)
        * ((1.0 - probability) ** (trials - index))
        for index in range(observed + 1)
    )


def clopper_pearson_upper_bound(
    successes: int, trials: int, confidence: float = 0.95
) -> float:
    """Exact one-sided Clopper-Pearson binomial upper confidence bound."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if successes == trials:
        return 1.0
    alpha = 1.0 - confidence
    if successes == 0:
        return 1.0 - alpha ** (1.0 / trials)
    lower = successes / float(trials)
    upper = 1.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if _binomial_cdf(successes, trials, middle) > alpha:
            lower = middle
        else:
            upper = middle
    return upper


def minimum_zero_violation_trials(
    upper_limit: float = 0.01, confidence: float = 0.95
) -> int:
    if not 0.0 < upper_limit < 1.0:
        raise ValueError("upper_limit must be in (0, 1)")
    trials = 1
    while clopper_pearson_upper_bound(0, trials, confidence) > upper_limit:
        trials += 1
    return trials
