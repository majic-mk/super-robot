from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Dict, List, Sequence, Set

from .manifest import ManifestCase, validate_manifest


def _pilot_order(case: ManifestCase, seed: int) -> str:
    return hashlib.sha256(
        ("%d:%s:%s" % (seed, case.dataset, case.case_id)).encode("utf-8")
    ).hexdigest()


def _is_natural(case: ManifestCase) -> bool:
    return case.construction == "corpus_repeat_pseudotime"


def _is_controlled(case: ManifestCase) -> bool:
    return case.construction == "controlled_document_order"


def _take_unique(
    candidates: Sequence[ManifestCase],
    count: int,
    used_content_hashes: Set[str],
) -> List[ManifestCase]:
    selected = []
    for case in candidates:
        if case.content_hash in used_content_hashes:
            continue
        selected.append(case)
        used_content_hashes.add(case.content_hash)
        if len(selected) == count:
            break
    return selected


def select_h1_pilot(
    cases: Sequence[ManifestCase],
    per_dataset: int = 50,
    natural_target: int = 25,
    seed: int = 20260726,
) -> List[ManifestCase]:
    """Select a deterministic train-only H1 pilot and relabel it as pilot.

    Natural corpus-repeat cases are targeted first because they are scarcer.
    Any shortage is filled with controlled cases without reusing the same C.
    """

    if per_dataset <= 0:
        raise ValueError("per_dataset must be positive")
    if not 0 <= natural_target <= per_dataset:
        raise ValueError("natural_target must be in [0, per_dataset]")
    datasets: Dict[str, List[ManifestCase]] = {}
    for case in cases:
        case.validate()
        if case.split != "train":
            continue
        if len(case.sources) != 4:
            continue
        datasets.setdefault(case.dataset, []).append(case)
    if not datasets:
        raise ValueError("no train cases are available for the pilot")

    selected_all = []
    for dataset, members in sorted(datasets.items()):
        natural = sorted(
            (case for case in members if _is_natural(case)),
            key=lambda case: _pilot_order(case, seed),
        )
        controlled = sorted(
            (case for case in members if _is_controlled(case)),
            key=lambda case: _pilot_order(case, seed),
        )
        used: Set[str] = set()
        selected = _take_unique(natural, natural_target, used)
        controlled_target = per_dataset - len(selected)
        selected.extend(_take_unique(controlled, controlled_target, used))
        if len(selected) < per_dataset:
            selected_ids = {case.case_id for case in selected}
            fallback = sorted(
                (
                    case
                    for case in natural + controlled
                    if case.case_id not in selected_ids
                ),
                key=lambda case: _pilot_order(case, seed + 1),
            )
            selected.extend(
                _take_unique(fallback, per_dataset - len(selected), used)
            )
        if len(selected) != per_dataset:
            raise ValueError(
                "%s has only %d eligible unique train cases; %d required"
                % (dataset, len(selected), per_dataset)
            )
        selected_all.extend(replace(case, split="pilot") for case in selected)

    validate_manifest(selected_all)
    return sorted(selected_all, key=lambda case: (case.dataset, case.case_id))


def pilot_manifest_audit(cases: Sequence[ManifestCase]) -> Dict[str, object]:
    validate_manifest(cases)
    datasets = sorted({case.dataset for case in cases})
    return {
        "cases": len(cases),
        "datasets": {
            dataset: {
                "cases": sum(case.dataset == dataset for case in cases),
                "natural": sum(
                    case.dataset == dataset and _is_natural(case)
                    for case in cases
                ),
                "controlled": sum(
                    case.dataset == dataset and _is_controlled(case)
                    for case in cases
                ),
            }
            for dataset in datasets
        },
        "all_split_pilot": all(case.split == "pilot" for case in cases),
        "unique_content_hashes": len({case.content_hash for case in cases}),
        "paper_evidence": False,
        "evidence_class": "server_pilot",
    }
